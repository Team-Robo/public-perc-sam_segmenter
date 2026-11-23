#pragma once
#include "common.cuh"

// Image encoders need window partitioning and Hiera pooling, absent from the
// pinned upstream CUDA backend. These preserve GGML's contiguous tensor layout.
static __global__ void sam_win_part(const float *input,float *output,int channels,int width,int height,
    int window,int window_columns,int64_t count) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  int c=i%channels;int64_t rest=i/channels;
  int x=rest%window;rest/=window;int y=rest%window;int partition=rest/window;
  x+=(partition%window_columns)*window;y+=(partition/window_columns)*window;
  output[i]=x<width&&y<height?input[(int64_t(y)*width+x)*channels+c]:0.f;
}
static __global__ void sam_win_unpart(const float *input,float *output,int channels,int width,int window,
    int window_columns,int64_t count) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  int c=i%channels;int64_t rest=i/channels;int x=rest%width,y=rest/width;
  int partition=(y/window)*window_columns+x/window;
  output[i]=input[((int64_t(partition)*window+y%window)*window+x%window)*channels+c];
}
static void ggml_cuda_sam_window(ggml_backend_cuda_context &ctx,ggml_tensor *dst,bool partition) {
  const auto *src=dst->src[0];int64_t count=ggml_nelements(dst);
  if(partition){
    const auto *parameters=reinterpret_cast<const int32_t*>(dst->op_params);
    sam_win_part<<<(count+255)/256,256,0,ctx.stream()>>>(static_cast<const float*>(src->data),static_cast<float*>(dst->data),
      int(src->ne[0]),int(src->ne[1]),int(src->ne[2]),parameters[2],parameters[0],count);
  }else{
    int window=reinterpret_cast<const int32_t*>(dst->op_params)[0];
    sam_win_unpart<<<(count+255)/256,256,0,ctx.stream()>>>(static_cast<const float*>(src->data),static_cast<float*>(dst->data),
      int(dst->ne[0]),int(dst->ne[1]),window,int((dst->ne[1]+window-1)/window),count);
  }
}
template<class T>static __global__ void sam_pool_1d(const T *input,float *output,int width,int output_width,
    int kernel,int stride,int padding,bool maximum,int64_t count){
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
  int row=i/output_width,position=i%output_width;int start=position*stride-padding,n=0;
  float value=maximum?-FLT_MAX:0.f;
  for(int k=0;k<kernel;++k){int col=start+k;if(col<0||col>=width)continue;
    float v=float(input[int64_t(row)*width+col]);value=maximum?fmaxf(value,v):value+v;++n;
  }
  output[i]=maximum?value:(n?value/n:0.f);
}
static void ggml_cuda_sam_pool(ggml_backend_cuda_context &ctx,ggml_tensor *dst){
  const auto *src=dst->src[0];const auto *p=reinterpret_cast<const int32_t*>(dst->op_params);int64_t count=ggml_nelements(dst);
  if(src->type==GGML_TYPE_F16)sam_pool_1d<<<(count+255)/256,256,0,ctx.stream()>>>(static_cast<const half*>(src->data),static_cast<float*>(dst->data),int(src->ne[0]),int(dst->ne[0]),p[1],p[2],p[3],p[0]==GGML_OP_POOL_MAX,count);
  else sam_pool_1d<<<(count+255)/256,256,0,ctx.stream()>>>(static_cast<const float*>(src->data),static_cast<float*>(dst->data),int(src->ne[0]),int(dst->ne[0]),p[1],p[2],p[3],p[0]==GGML_OP_POOL_MAX,count);
}

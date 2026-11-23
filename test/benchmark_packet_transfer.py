"""Isolated SAM mask-transfer/packet benchmark; no model weights or ROS calls."""
import argparse
import json,time
from pathlib import Path
import numpy as np
import torch
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--iterations',type=int,default=35)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
if args.iterations<1:parser.error('iterations must be positive')
if not torch.cuda.is_available():raise SystemExit('CUDA is required')
h,w,n=480,848,32
masks=torch.ones((n,1,h,w),device='cuda',dtype=torch.float32);masks[::5]=0
scores=torch.linspace(0,1,n,device='cuda');boxes=torch.tensor([[0,0,w,h]]*n,device='cuda',dtype=torch.float32)
threshold=.7

def before():
 m=masks.detach().cpu().numpy();b=boxes.detach().float().cpu().numpy();s=scores.detach().float().cpu().numpy().reshape(-1)
 output=[]
 for row,box,score in zip(m,b,s):
  if not np.isfinite(score) or score<threshold:continue
  instance=row.reshape(h,w).astype(bool)
  if not instance.any():continue
  output.append((instance.astype(np.uint8)*255,box.tolist(),float(score)))
 return output

def after():
 s=scores.float();eligible=torch.isfinite(s)&(s.double()>=threshold)&masks.reshape(n,h*w).any(dim=1).bool()
 i=torch.nonzero(eligible,as_tuple=True)[0]
 m=masks.index_select(0,i).bool().to(torch.uint8).mul_(255).cpu().numpy()
 b=boxes.index_select(0,i).cpu().numpy();s=s.index_select(0,i).cpu().numpy()
 return [(row.reshape(h,w),box.tolist(),float(score)) for row,box,score in zip(m,b,s)]
a,b=before(),after();assert len(a)==len(b)
for left,right in zip(a,b):
 np.testing.assert_array_equal(left[0],right[0]);assert left[1:]==right[1:]
report={'device':torch.cuda.get_device_name(),'workload':{'instances':n,'height':h,'width':w,'accepted_instances':len(b),'before_mask_transfer_bytes':n*h*w*4,'after_mask_transfer_bytes':len(b)*h*w},'exact_parity':True}
for label,fn in [('before',before),('after',after)]:
 for _ in range(10):fn()
 times=[]
 for _ in range(args.iterations):
  torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
 report[label]={'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95))}
text=json.dumps(report,indent=2)
print(text)
if args.output:
 with args.output.open('x',encoding='utf-8') as stream:stream.write(text+'\n')

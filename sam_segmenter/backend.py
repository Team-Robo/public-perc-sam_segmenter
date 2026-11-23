"""Native SAM3 image inference; no ROS imports and no implicit downloads."""
from pathlib import Path
from contextlib import nullcontext

import numpy as np

from .targets import parse_targets


class Sam3Backend:
    def __init__(self, checkpoint, device='cuda', confidence=0.5, bpe_path='', release_cuda_cache=False):
        if not Path(checkpoint).is_file():
            raise ValueError('Set sam3_checkpoint_path to an existing, locally provisioned SAM3 checkpoint')
        if device not in ('cuda', 'cpu'):
            raise ValueError('sam3_device must be cuda or cpu')
        if not 0 <= confidence <= 1:
            raise ValueError('sam3_confidence_threshold must be between 0 and 1')
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('SAM3 CUDA is unavailable; check the model environment and NVIDIA driver')
        self.torch = torch
        self.device = device
        self.confidence = confidence
        self.release_cuda_cache = release_cuda_cache
        model = build_sam3_image_model(checkpoint_path=str(Path(checkpoint).resolve()),
                                      device=device, load_from_HF=False,
                                      bpe_path=bpe_path or None)
        self.processor = Sam3Processor(model, device=device, confidence_threshold=confidence)

    def memory_stats(self):
        if self.device != 'cuda':
            return {}
        return dict(cuda_allocated_bytes=self.torch.cuda.memory_allocated(),
                    cuda_reserved_bytes=self.torch.cuda.memory_reserved(),
                    cuda_peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=self.torch.cuda.max_memory_reserved())

    def segment(self, image, targets):
        from PIL import Image
        targets = parse_targets(','.join(targets))
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('SAM3 input must be an H x W x 3 uint8 RGB image')
        height, width = image.shape[:2]
        instance_masks = []
        detections = []
        precision = (self.torch.autocast('cuda', dtype=self.torch.bfloat16)
                     if self.device == 'cuda' else nullcontext())
        with self.torch.inference_mode(), precision:
            state = self.processor.set_image(Image.fromarray(image))
            try:
                for target in targets:
                    self.processor.reset_all_prompts(state)
                    result = self.processor.set_text_prompt(state=state, prompt=target)
                    masks = result['masks'].detach()
                    boxes = result['boxes'].detach().float()
                    scores = result['scores'].detach().float().reshape(-1)
                    if len(masks) != len(boxes) or len(scores) != len(boxes):
                        raise RuntimeError('SAM3 returned inconsistent instance arrays')
                    if self.device == 'cuda':
                        # Compare float32 output scores in double precision, matching
                        # NumPy's scalar comparison against the Python threshold.
                        eligible = self.torch.isfinite(scores) & (scores.double() >= self.confidence)
                        if masks.numel() != len(masks) * height * width:
                            if eligible.any().item():
                                raise RuntimeError('SAM3 returned an invalid mask shape')
                            continue
                        eligible &= masks.reshape(len(masks), height * width).any(dim=1).bool()
                        indices = self.torch.nonzero(eligible, as_tuple=True)[0]
                        count = indices.numel()
                        if len(instance_masks) + count > 32 or (len(instance_masks) + count) * height * width > 64*1024*1024:
                            raise RuntimeError('SAM3 instance packet exceeds bounds')
                        # Transfer only accepted, nonempty instances and compress
                        # masks on-device; mask pixels still use exactly 0/255.
                        masks = masks.index_select(0, indices).bool().to(self.torch.uint8).mul_(255).cpu().numpy()
                        boxes = boxes.index_select(0, indices).cpu().numpy()
                        scores = scores.index_select(0, indices).cpu().numpy()
                    else:
                        # CPU tensors already share NumPy storage. Preserve the
                        # low-allocation host path instead of adding compaction.
                        masks = masks.numpy()
                        boxes = boxes.numpy()
                        scores = scores.numpy()
                    for instance, box, score in zip(masks, boxes, scores):
                        if self.device == 'cuda':
                            # Already validated/nonempty and normalized on device.
                            instance = instance.reshape(height, width)
                        else:
                            if not np.isfinite(score) or score < self.confidence:
                                continue
                            instance = instance.reshape(height, width).astype(bool)
                            if not instance.any():
                                continue
                            instance = instance.astype(np.uint8) * 255
                        box = np.asarray(box, dtype=float)
                        if box.shape != (4,) or not np.isfinite(box).all():
                            raise RuntimeError('SAM3 returned an invalid box')
                        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
                        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
                        if len(instance_masks) >= 32 or (len(instance_masks)+1)*height*width > 64*1024*1024:
                            raise RuntimeError('SAM3 instance packet exceeds bounds')
                        instance_masks.append(instance)
                        detections.append({'label': target, 'score': float(score), 'box': box.tolist()})
            finally:
                del state
                if hasattr(self.processor, 'reset_state'):
                    try:
                        self.processor.reset_state()
                    except Exception:
                        pass
                if self.release_cuda_cache and self.device == 'cuda':
                    self.torch.cuda.empty_cache()
                if self.release_cuda_cache:
                    import gc
                    gc.collect()
        if len(instance_masks) > 32 or len(instance_masks)*height*width > 64*1024*1024:
            raise RuntimeError('SAM3 instance packet exceeds bounds')
        masks = np.stack(instance_masks) if instance_masks else np.zeros((0,height,width),np.uint8)
        return masks, detections

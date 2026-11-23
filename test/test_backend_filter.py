"""Check device-side packet filtering against the former host-side contract."""
from contextlib import nullcontext
from pathlib import Path
import sys
from unittest.mock import Mock
import numpy as np
import pytest

torch=pytest.importorskip('torch')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sam_segmenter.backend import Sam3Backend


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('threshold',[.5,.50000001])
def test_device_filter_keeps_masks_order_float_scores_and_threshold_boundaries(device,threshold):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('No CUDA')
    rng=np.random.default_rng(31)
    masks=rng.integers(0,3,(7,1,8,9)).astype(np.float32)
    masks[4]=0;masks[5]=np.nan
    boxes=np.array([[0,0,9,8]]*7,np.float32)
    scores=np.array([.9,.5,.1,np.nan,.95,.8,np.inf],np.float32)
    result={name:torch.as_tensor(value,device=device) for name,value in
            [('masks',masks),('boxes',boxes),('scores',scores)]}
    backend=Sam3Backend.__new__(Sam3Backend)
    backend.torch=torch;backend.device=device;backend.confidence=threshold;backend.release_cuda_cache=False
    backend.processor=Mock()
    backend.processor.set_text_prompt.return_value=result
    actual,detections=backend.segment(np.zeros((8,9,3),np.uint8),['cup'])
    indices=[i for i,score in enumerate(scores) if np.isfinite(score) and score>=threshold and masks[i].astype(bool).any()]
    expected=np.stack([masks[i].reshape(8,9).astype(bool).astype(np.uint8)*255 for i in indices])
    np.testing.assert_array_equal(actual,expected)
    assert [d['score'] for d in detections]==[float(scores[i]) for i in indices]
    assert all(d['label']=='cup' and d['box']==[0.,0.,9.,8.] for d in detections)


def test_all_rejected_and_packet_limit():
    backend=Sam3Backend.__new__(Sam3Backend)
    backend.torch=torch;backend.device='cpu';backend.confidence=.5;backend.release_cuda_cache=False
    backend.processor=Mock()
    result=dict(masks=torch.zeros((0,1,8,9)),boxes=torch.zeros((0,4)),scores=torch.zeros(0))
    backend.processor.set_text_prompt.return_value=result
    masks,detections=backend.segment(np.zeros((8,9,3),np.uint8),['cup'])
    assert masks.shape==(0,8,9) and detections==[]
    backend.processor.set_text_prompt.return_value=dict(masks=torch.ones((33,1,8,9)),boxes=torch.zeros((33,4)),scores=torch.ones(33))
    with pytest.raises(RuntimeError,match='bounds'):
        backend.segment(np.zeros((8,9,3),np.uint8),['cup'])

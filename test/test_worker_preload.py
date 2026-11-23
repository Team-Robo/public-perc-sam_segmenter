"""Exercise eager/deferred model-worker lifecycle using a real framed subprocess."""
from pathlib import Path
import sys
import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from sam_segmenter.worker_client import WorkerClient

PROGRAM = r'''
import sys
sys.path.insert(0, sys.argv[1])
from sam_segmenter.protocol import encode, read_packet
import numpy as np
out = sys.stdout.buffer
out.write(encode({'ready': True})); out.flush()
while True:
    try:
        request, arrays = read_packet(sys.stdin.buffer.read)
    except EOFError:
        break
    h, w = arrays['image'].shape[:2]
    out.write(encode({'id': request['id'], 'detections': []}, masks=np.zeros((0,h,w), np.uint8)))
    out.flush()
'''


@pytest.mark.parametrize('preload', [False, True])
def test_startup_policy_reuses_worker_and_close_prevents_restart(preload):
    client = WorkerClient('unused', command=[sys.executable, '-u', '-c', PROGRAM, str(SOURCE)],
                          preload=preload, startup_timeout=5)
    try:
        assert client.ready is preload
        assert (client.process is not None) is preload
        client.ensure_started()
        process = client.process
        for _ in range(2):
            masks, detections = client.segment(np.zeros((8,8,3), np.uint8), ['cup'], 5)
            assert masks.shape == (0,8,8) and detections == []
            assert client.process is process
    finally:
        client.close()
    assert process.poll() is not None
    with pytest.raises(RuntimeError, match='restart'):
        client.ensure_started()


def test_failed_deferred_startup_closes_worker_instead_of_reusing_broken_stream():
    client = WorkerClient('unused', command=[sys.executable, '-c', 'pass'], preload=False,
                          startup_timeout=1)
    with pytest.raises(RuntimeError, match='exited'):
        client.ensure_started()
    assert not client.ready and client.process.poll() is not None
    with pytest.raises(RuntimeError, match='restart'):
        client.ensure_started()

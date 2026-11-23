#!/usr/bin/env python3
"""SAM3 worker entrypoint. stdout is exclusively the framed protocol."""
import argparse
import contextlib
from pathlib import Path
import sys

# Executed by a different Python minor version; expose only this ROS-free package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sam_segmenter.protocol import encode, read_packet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--bpe-path', default='')
    parser.add_argument('--release-cuda-cache', action='store_true')
    args = parser.parse_args()
    output = sys.stdout.buffer

    def send(metadata, **arrays):
        output.write(encode(metadata, **arrays))
        output.flush()

    def read(size):
        result = bytearray()
        while len(result) < size:
            data = sys.stdin.buffer.read(size - len(result))
            if not data:
                raise EOFError()
            result.extend(data)
        return bytes(result)

    try:
        with contextlib.redirect_stdout(sys.stderr):
            from sam_segmenter.backend import Sam3Backend
            backend = Sam3Backend(args.checkpoint, args.device, args.confidence, args.bpe_path, args.release_cuda_cache)
        send({'ready': True, 'memory': backend.memory_stats()})
    except Exception as exc:
        send({'ready': False, 'error': str(exc)})
        return 1
    while True:
        try:
            request, arrays = read_packet(read)
        except EOFError:
            return 0
        try:
            with contextlib.redirect_stdout(sys.stderr):
                masks, detections = backend.segment(arrays['image'], request['targets'])
            send({'id': request['id'], 'detections': detections, 'memory': backend.memory_stats()}, masks=masks)
        except Exception as exc:
            send({'id': request.get('id'), 'error': str(exc)})
        finally:
            if 'arrays' in locals():
                del arrays
            if 'masks' in locals():
                del masks
            if 'detections' in locals():
                del detections
            if args.release_cuda_cache:
                import gc
                gc.collect()


if __name__ == '__main__':
    sys.exit(main())

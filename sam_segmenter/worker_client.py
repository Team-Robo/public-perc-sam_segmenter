"""One model worker, one bounded request, and wall-clock pipe deadlines."""
import os
from pathlib import Path
import selectors
import subprocess
import sys
import threading
import time

import numpy as np

from .protocol import encode, read_packet


class WorkerClient:
    def __init__(self, checkpoint, python='', device='cuda', confidence=0.5,
                 startup_timeout=120.0, bpe_path='', *, command=None, preload=True, release_cuda_cache=False):
        self.lock = threading.Lock()
        self.close_lock = threading.RLock()
        self.request_id = 0
        self.ready = False
        self.memory_stats = {}
        command = command or [os.path.expanduser(python) if python else sys.executable, '-u',
                              str(Path(__file__).with_name('worker.py')),
                              '--checkpoint', checkpoint, '--device', device,
                              '--confidence', str(confidence), '--bpe-path', bpe_path]
        if release_cuda_cache:
            command = [*command, '--release-cuda-cache']
        self.command = command
        self.startup_timeout = startup_timeout
        self.closed = False
        self.process = None
        if preload:
            self.ensure_started()

    def ensure_started(self):
        with self.close_lock:
            if self.closed:
                raise RuntimeError('SAM3 worker is unavailable; restart the node')
            if not self.ready:
                self._start()

    def _start(self):
        env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                   HF_HUB_OFFLINE='1')
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=None, env=env, bufsize=0, start_new_session=True)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        try:
            deadline = time.monotonic() + self.startup_timeout
            status, _ = read_packet(lambda size: self._read(size, deadline))
            if not status.get('ready'):
                raise RuntimeError('SAM3 worker startup failed: ' + status.get('error', 'unknown error'))
            self.ready = True
            self.memory_stats = status.get('memory', {})
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _wait(fd, event, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('SAM3 worker deadline exceeded')
        with selectors.DefaultSelector() as selector:
            selector.register(fd, event)
            if not selector.select(remaining):
                raise TimeoutError('SAM3 worker deadline exceeded')

    def _read(self, size, deadline):
        result = bytearray()
        fd = self.process.stdout.fileno()
        while len(result) < size:
            self._wait(fd, selectors.EVENT_READ, deadline)
            try:
                data = os.read(fd, min(size - len(result), 1024 * 1024))
            except BlockingIOError:
                continue
            if not data:
                raise RuntimeError('SAM3 worker exited before responding')
            result.extend(data)
        return bytes(result)

    def segment(self, image, targets, timeout):
        with self.lock:
            self.ensure_started()
            if not self.ready:
                raise RuntimeError('SAM3 worker is unavailable; restart the node')
            deadline = time.monotonic() + timeout
            self.request_id += 1
            try:
                packet = memoryview(encode({'id': self.request_id, 'targets': list(targets)}, image=image))
                fd = self.process.stdin.fileno()
                while packet:
                    self._wait(fd, selectors.EVENT_WRITE, deadline)
                    try:
                        packet = packet[os.write(fd, packet[:1024 * 1024]):]
                    except BlockingIOError:
                        continue
                response, arrays = read_packet(lambda size: self._read(size, deadline))
                if response.get('id') != self.request_id:
                    raise RuntimeError('SAM3 worker returned an uncorrelated response')
                if 'error' in response:
                    raise RuntimeError(response['error'])
                self.memory_stats = response.get('memory', {})
                mask = arrays['masks']
                if mask.dtype != np.uint8 or mask.ndim != 3 or mask.shape[1:] != image.shape[:2] or len(mask) != len(response['detections']) or len(mask) > 32:
                    raise RuntimeError('SAM3 worker returned an invalid mask')
                if not np.isin(mask, [0, 255]).all():
                    raise RuntimeError('SAM3 mask must contain only 0 and 255')
                return mask, response['detections']
            except BaseException:
                # A timed-out or broken stream can never satisfy a later request.
                self.close()
                raise

    def close(self):
        with self.close_lock:
            self._close()

    def _close(self):
        self.closed = True
        self.ready = False
        process = getattr(self, 'process', None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            for pipe in (process.stdin, process.stdout):
                if pipe and not pipe.closed:
                    pipe.close()

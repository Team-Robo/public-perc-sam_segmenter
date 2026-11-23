"""Length-delimited metadata/NumPy payloads on a private subprocess pipe."""
import io
import json
import struct

import numpy as np

MAX_PACKET = 64 * 1024 * 1024


def encode(metadata, **arrays):
    header = json.dumps(metadata, allow_nan=False).encode('utf-8')
    payload = io.BytesIO()
    if arrays:
        np.savez(payload, **arrays)
    data = payload.getvalue()
    if len(header) > 65536 or len(data) > MAX_PACKET:
        raise ValueError('SAM worker packet exceeds its size bound')
    return struct.pack('!II', len(header), len(data)) + header + data


def decode(header, payload):
    metadata = json.loads(header)
    arrays = {}
    if payload:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    return metadata, arrays


def read_packet(read):
    lengths = read(8)
    header_size, payload_size = struct.unpack('!II', lengths)
    if header_size > 65536 or payload_size > MAX_PACKET:
        raise ValueError('SAM worker packet exceeds its size bound')
    return decode(read(header_size), read(payload_size))

"""Minimal safetensors reader (header JSON + memmap); no torch, no safetensors package.

File = u64 header length | JSON header {name: {dtype, shape, data_offsets}} | raw little-endian data.
BF16 tensors are exposed as uint16 views (blocks.to_f32 handles GGML_BF16), F16/F32 as their dtype.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .blocks import GGML_BF16, GGML_F16, GGML_F32

DTYPES = {"BF16": (np.uint16, GGML_BF16), "F16": (np.float16, GGML_F16), "F32": (np.float32, GGML_F32)}


class SafetensorsFile:
    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        self.base = 8 + n
        self.meta = hdr.pop("__metadata__", {})
        self.header: dict[str, dict] = hdr
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")

    def names(self) -> list[str]:
        return list(self.header)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(int(x) for x in self.header[name]["shape"])

    def ggml_type(self, name: str) -> int:
        return DTYPES[self.header[name]["dtype"]][1]

    def array(self, name: str) -> np.ndarray:
        """Zero-copy view of one tensor (uint16 for BF16)."""
        h = self.header[name]
        dt, _ = DTYPES[h["dtype"]]
        a, b = h["data_offsets"]
        shape = tuple(int(x) for x in h["shape"])
        buf = self.mm[self.base + a : self.base + b]
        arr = buf.view(dt)
        assert arr.size == int(np.prod(shape)), (name, arr.size, shape)
        return arr.reshape(shape)


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Test helper: write {name: array} (float32 / float16 / uint16-as-BF16) as a safetensors file."""
    rev = {np.dtype(np.uint16): "BF16", np.dtype(np.float16): "F16", np.dtype(np.float32): "F32"}
    hdr = {}
    off = 0
    blobs = []
    for name, arr in tensors.items():
        raw = np.ascontiguousarray(arr).tobytes()
        hdr[name] = {"dtype": rev[arr.dtype], "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def bf16_from_f32(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even f32 -> BF16 bits (uint16)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16)

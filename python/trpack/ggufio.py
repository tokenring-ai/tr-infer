"""Sharded GGUF access on top of gguf-py's GGUFReader (memmap, no dequant)."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gguf import GGUFReader


@dataclass
class TensorRef:
    name: str
    ggml_type: int
    dims: tuple[int, ...]  # numpy order (outermost first); last dim = K (row length in elements)
    data: np.ndarray  # memmap: float array or uint8 [..., row_bytes]
    n_bytes: int
    shard: Path


def discover_shards(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.glob("*.gguf") if "mmproj" not in p.name)
    if "-of-" in path.name:
        prefix = path.name.rsplit("-of-", 1)[0].rsplit("-", 1)[0]
        return sorted(path.parent.glob(f"{prefix}-*-of-*.gguf"))
    return [path]


class GgufModel:
    def __init__(self, path: Path):
        self.shards = discover_shards(path)
        if not self.shards:
            raise FileNotFoundError(f"no GGUF shards at {path}")
        self.readers = [GGUFReader(str(p)) for p in self.shards]
        self.kv: dict = {}
        for r in self.readers:
            for k, f in r.fields.items():
                if k.startswith("tokenizer.ggml.") and k.split(".")[-1] in ("tokens", "scores", "token_type", "merges"):
                    continue
                if k in self.kv:
                    continue
                try:
                    self.kv[k] = f.contents()
                except Exception:
                    pass
        self.tensors: dict[str, TensorRef] = {}
        for r, p in zip(self.readers, self.shards):
            for t in r.tensors:
                if t.name in self.tensors:
                    continue
                dims = tuple(int(x) for x in reversed(t.shape.tolist()))
                self.tensors[t.name] = TensorRef(t.name, int(t.tensor_type), dims, t.data, int(t.n_bytes), p)

    def tokenizer_lists(self):
        r = self.readers[0]
        f = r.fields
        def strs(key):
            fld = f[key]
            return [bytes(fld.parts[i]).decode("utf-8", "replace") for i in fld.data]
        def ints(key):
            fld = f[key]
            return [int(fld.parts[i][0]) for i in fld.data]
        return strs("tokenizer.ggml.tokens"), ints("tokenizer.ggml.token_type"), strs("tokenizer.ggml.merges")

    def config(self) -> dict:
        arch = self.kv.get("general.architecture", "")
        cfg = {"architecture": arch}
        for k, v in self.kv.items():
            if k.startswith(f"{arch}.") or k in (
                "general.name", "general.size_label", "general.file_type",
                "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.eos_token_id",
                "tokenizer.ggml.bos_token_id", "tokenizer.ggml.padding_token_id", "tokenizer.ggml.add_bos_token",
            ):
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                cfg[k] = v
        return cfg


def shard_hashes(shards: list[Path], fast: bool = False, cache: Path | None = None) -> list[dict]:
    cache = cache or Path(os.path.expanduser("~/.cache/trpack/shard-hashes.json"))
    db = {}
    if cache.exists():
        try:
            db = json.loads(cache.read_text())
        except Exception:
            db = {}
    out = []
    for p in shards:
        st = p.stat()
        key = f"{p.resolve()}:{st.st_size}:{st.st_mtime_ns}:{'fast' if fast else 'full'}"
        if key not in db:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                left = (16 << 20) if fast else st.st_size
                while left > 0:
                    chunk = f.read(min(left, 64 << 20))
                    if not chunk:
                        break
                    h.update(chunk)
                    left -= len(chunk)
            db[key] = h.hexdigest()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(db, indent=1))
        out.append({"file": p.name, "bytes": st.st_size, "sha256": db[key], "fast": fast})
    return out

"""Vision encoder (llama.cpp mmproj GGUF, projector `qwen3vl_merger`) -> overlay pack.

The encoder is a SigLIP-style ViT (hidden 1152, 27 layers, 16 heads of 72, GELU MLP 4304, LayerNorm
with bias) over 16x16 patches, with a learned 48x48 position table, 2-D rope on q/k, and a 2x2 merger
MLP (4608 -> 4608 -> 2560) into the language model's hidden size. The engine runs it tensor-parallel
over the tiles: tile t owns heads 2t, 2t+1 (its slice of q/k/v rows and the matching K columns of
attn_out), a 1/8 slice of the MLP hidden rows, and 1/8 of the merger hidden. Matrices are stored as
bf16 in the AMX strip layout (`bf16_strips`: 8-bit weights cost ~8 % output error on real photos,
bf16 0.4 %); norms, biases and the position table stay f32.

Slices whose width is not a multiple of the codec block are zero-padded at the source:
attn_out's K slice 144 -> 160 per tile (the activation rows carry 16 zero lanes), the MLP hidden
4304 -> 4352 (544 per tile; padded rows have zero weights and bias, so their GELU output is 0 and
the padded down columns are zero too). The merger hidden 4608 splits into 576 per tile exactly.
The two temporal patch-embedding kernels are summed (a still image is the same frame twice).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import FORMAT_VERSION
from .blocks import GGML_F32, to_f32
from .ggufio import GgufModel, TensorRef
from .mtp import LazyTensors, _Dims, _fast_sha
from .plan import REPL, Spec, cols, rows, rows_ranges
from .writer import layout, resolve, write_spec

VISION_PLAN_VERSION = 2


def align_up(v: int, a: int) -> int:
    return (v + a - 1) // a * a


class VisionModel:
    """The mmproj tensors renamed, padded and reshaped as the engine expects (all f32 sources)."""

    def __init__(self, tensors: dict[str, TensorRef], kv: dict, n_tiles: int):
        self.src = tensors
        self.n_tiles = n_tiles
        g = lambda k, d=None: kv.get(f"clip.vision.{k}", d)
        if kv.get("clip.projector_type") != "qwen3vl_merger":
            raise SystemExit(f"unsupported projector {kv.get('clip.projector_type')!r} (need qwen3vl_merger)")
        T = n_tiles
        self.hidden = int(g("embedding_length"))
        self.n_layer = int(g("block_count"))
        self.n_head = int(g("attention.head_count"))
        self.n_ff = int(g("feed_forward_length"))
        self.patch = int(g("patch_size"))
        self.merge = int(g("spatial_merge_size", 2))
        self.proj = int(g("projection_dim"))
        self.eps = float(g("attention.layer_norm_epsilon", 1e-6))
        self.n_pos = int(tensors["v.position_embd.weight"].dims[0])
        if self.n_head % T or self.hidden % self.n_head:
            raise SystemExit(f"{self.n_head} heads do not split over {T} tiles")
        self.head_dim = self.hidden // self.n_head
        self.heads_per_tile = self.n_head // T
        self.dl = self.heads_per_tile * self.head_dim  # q (or k, v) width per tile
        self.opad = align_up(self.dl, 32)  # attn_out K slice per tile
        self.fp = align_up(-(-self.n_ff // T), 32)  # MLP hidden per tile
        self.n_ff_pad = self.fp * T
        self.mh = self.hidden * self.merge * self.merge  # merger hidden (4608)
        if self.mh % (32 * T):
            raise SystemExit(f"merger hidden {self.mh} does not split into 32-multiples over {T} tiles")
        self.mp = self.mh // T
        if self.hidden % 32 or (self.patch * self.patch * 3) % 32:
            raise SystemExit("hidden and patch vector must be multiples of 32")
        self.config = {
            "clip.projector_type": "qwen3vl_merger",
            "clip.vision.embedding_length": self.hidden,
            "clip.vision.block_count": self.n_layer,
            "clip.vision.attention.head_count": self.n_head,
            "clip.vision.feed_forward_length": self.n_ff,
            "clip.vision.patch_size": self.patch,
            "clip.vision.spatial_merge_size": self.merge,
            "clip.vision.projection_dim": self.proj,
            "clip.vision.attention.layer_norm_epsilon": self.eps,
            "clip.vision.image_mean": [float(x) for x in g("image_mean", [0.5, 0.5, 0.5])],
            "clip.vision.image_std": [float(x) for x in g("image_std", [0.5, 0.5, 0.5])],
            "vision.n_pos": self.n_pos,
            "vision.attn_out_k_pad": self.opad,
            "vision.ff_per_tile": self.fp,
            "vision.merge_per_tile": self.mp,
        }
        self.builders: dict[str, callable] = {}
        self.dims: dict[str, tuple] = {}
        self._map()
        self.tensors = LazyTensors(self.builders)
        self.types = {n: GGML_F32 for n in self.builders}

    def _f32(self, name: str) -> np.ndarray:
        r = self.src[name]
        return to_f32(np.asarray(r.data), r.ggml_type).reshape(r.dims)

    def _ref(self, name: str, dims: tuple, build):
        self.dims[name] = tuple(int(d) for d in dims)
        self.builders[name] = lambda: TensorRef(name, GGML_F32, self.dims[name], build().astype(np.float32, copy=False), int(np.prod(self.dims[name])) * 4, Path("mmproj"))

    def _map(self):
        H, F, T = self.hidden, self.n_ff, self.n_tiles
        pv = self.patch * self.patch * 3
        # patch embedding: both temporal kernels on the same frame -> one kernel
        self._ref("v.patch_embd.weight", (H, pv), lambda: (self._f32("v.patch_embd.weight").reshape(H, pv) + self._f32("v.patch_embd.weight.1").reshape(H, pv)))
        self._ref("v.patch_embd.bias", (H,), lambda: self._f32("v.patch_embd.bias"))
        self._ref("v.position_embd.weight", (self.n_pos, H), lambda: self._f32("v.position_embd.weight"))
        for n in ("v.post_ln.weight", "v.post_ln.bias"):
            self._ref(n, (H,), (lambda n: lambda: self._f32(n))(n))
        for il in range(self.n_layer):
            p = f"v.blk.{il}."
            for n in ("ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias", "attn_qkv.bias", "attn_out.bias", "ffn_down.bias"):
                self._ref(p + n, self.src[p + n].dims, (lambda n: lambda: self._f32(n))(p + n))
            self._ref(p + "attn_qkv.weight", (3 * H, H), (lambda n: lambda: self._f32(n))(p + "attn_qkv.weight"))
            # attn_out [H, H]: K padded to opad per tile
            self._ref(p + "attn_out.weight", (H, self.opad * T), (lambda n: lambda: self._pad_cols(self._f32(n), self.dl, self.opad))(p + "attn_out.weight"))
            self._ref(p + "ffn_up.weight", (self.n_ff_pad, H), (lambda n: lambda: self._pad_rows(self._f32(n), -(-F // T), self.fp))(p + "ffn_up.weight"))
            self._ref(p + "ffn_up.bias", (self.n_ff_pad,), (lambda n: lambda: self._pad_rows(self._f32(n), -(-F // T), self.fp))(p + "ffn_up.bias"))
            self._ref(p + "ffn_down.weight", (H, self.n_ff_pad), (lambda n: lambda: self._pad_cols(self._f32(n), -(-F // T), self.fp))(p + "ffn_down.weight"))
        self._ref("mm.0.weight", (self.mh, self.mh), lambda: self._f32("mm.0.weight"))
        self._ref("mm.0.bias", (self.mh,), lambda: self._f32("mm.0.bias"))
        self._ref("mm.2.weight", (self.proj, self.mh), lambda: self._f32("mm.2.weight"))
        self._ref("mm.2.bias", (self.proj,), lambda: self._f32("mm.2.bias"))

    def _pad_cols(self, w: np.ndarray, per: int, per_pad: int) -> np.ndarray:
        """Split the last axis into n_tiles chunks of `per` (last one may be short) and pad each to `per_pad`."""
        T = self.n_tiles
        out = np.zeros(w.shape[:-1] + (per_pad * T,), np.float32)
        for t in range(T):
            s = per * t
            ln = max(0, min(per, w.shape[-1] - s))
            out[..., per_pad * t : per_pad * t + ln] = w[..., s : s + ln]
        return out

    def _pad_rows(self, w: np.ndarray, per: int, per_pad: int) -> np.ndarray:
        T = self.n_tiles
        out = np.zeros((per_pad * T,) + w.shape[1:], np.float32)
        for t in range(T):
            s = per * t
            ln = max(0, min(per, w.shape[0] - s))
            out[per_pad * t : per_pad * t + ln] = w[s : s + ln]
        return out

    def specs(self) -> dict[str, list[Spec]]:
        T, H = self.n_tiles, self.hidden
        dl, opad, fp, mp = self.dl, self.opad, self.fp, self.mp
        # bf16 strips for every matrix: 8-bit per-32 weights cost ~8 % relative error on the encoder's
        # output (outlier tokens > 100 %), bf16 0.4 %; the encoder is 450 M parameters, so 2 bytes each
        tq = lambda name, tile, shard, layer=None: Spec(name, tile, "bf16_strips", shard, layer=layer)
        f32 = lambda name, tile, shard, layer=None: Spec(name, tile, "f32", shard, layer=layer)
        out: dict[str, list[Spec]] = {}
        out["vision_in"] = [
            tq("v.patch_embd.weight", None, rows(0, H)),
            f32("v.patch_embd.bias", None, REPL),
            f32("v.position_embd.weight", None, REPL),
        ]
        # one layout task for all layers: the 2 MiB per-task alignment would otherwise cost more than a layer
        out["vision_layers"] = specs = []
        for il in range(self.n_layer):
            p = f"v.blk.{il}."
            specs.extend(f32(p + n, None, REPL, il) for n in ("ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias", "attn_out.bias", "ffn_down.bias"))
            for t in range(T):
                qkv = [(dl * t, dl), (H + dl * t, dl), (2 * H + dl * t, dl)]
                specs.append(tq(p + "attn_qkv.weight", t, rows_ranges(qkv), il))
                specs.append(f32(p + "attn_qkv.bias", t, rows_ranges(qkv), il))
                specs.append(tq(p + "attn_out.weight", t, cols(opad * t, opad), il))
                specs.append(tq(p + "ffn_up.weight", t, rows(fp * t, fp), il))
                specs.append(f32(p + "ffn_up.bias", t, rows(fp * t, fp), il))
                specs.append(tq(p + "ffn_down.weight", t, cols(fp * t, fp), il))
        specs = [f32("v.post_ln.weight", None, REPL), f32("v.post_ln.bias", None, REPL), f32("mm.2.bias", None, REPL)]
        for t in range(T):
            specs.append(tq("mm.0.weight", t, rows(mp * t, mp)))
            specs.append(f32("mm.0.bias", t, rows(mp * t, mp)))
            specs.append(tq("mm.2.weight", t, cols(mp * t, mp)))
        out["vision_out"] = specs
        return out


def pack_vision(mmproj: Path, out_dir: Path, n_tiles: int = 8, force: bool = False, log=print, model: VisionModel | None = None, source_hash: str | None = None) -> Path:
    if str(out_dir).startswith("/tmp"):
        raise SystemExit("refusing to write a pack under /tmp (tmpfs)")
    if model is None:
        g = GgufModel(Path(mmproj))
        if g.kv.get("general.architecture") != "clip":
            raise SystemExit(f"{mmproj}: not an mmproj (architecture {g.kv.get('general.architecture')!r})")
        model = VisionModel(g.tensors, g.kv, n_tiles)
        source_hash = _fast_sha(Path(mmproj))
    specs_by_task = model.specs()
    for specs in specs_by_task.values():
        for s in specs:
            resolve(s, _Dims(model.dims[s.name]))
    sizes = layout(specs_by_task, n_tiles)
    identity = {
        "format_version": FORMAT_VERSION,
        "vision_plan_version": VISION_PLAN_VERSION,
        "n_tiles": n_tiles,
        # ints and strings only: the engine re-hashes the identity from its own JSON canonicalisation
        "config_sha": hashlib.sha256(json.dumps(model.config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16],
        "source": source_hash,
    }
    ident_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        old = json.loads(manifest_path.read_text())
        if old.get("hash") == ident_hash:
            log(f"already packed: {out_dir}")
            return out_dir
    if manifest_path.exists():
        manifest_path.unlink()
    for f, sz in sizes.items():
        with open(out_dir / f, "wb") as fh:
            fh.truncate(sz)
    log(f"vision pack -> {out_dir}  files: " + ", ".join(f"{f} {sz / 2**20:.0f} MiB" for f, sz in sizes.items()))
    fds = {f: os.open(out_dir / f, os.O_WRONLY) for f in sizes}
    t0 = time.time()
    written = 0
    order = sorted((s for specs in specs_by_task.values() for s in specs), key=lambda s: (s.name, s.tile if s.tile is not None else -1))
    last = None
    for s in order:
        if last is not None and last != s.name:
            model.tensors.drop(last)
        write_spec(fds, s, model.tensors[s.name])
        written += s.nbytes
        last = s.name
    for fd in fds.values():
        os.fsync(fd)
        os.close(fd)
    manifest = {
        "format_version": FORMAT_VERSION,
        "hash": ident_hash,
        "identity": identity,
        "overlay_of": None,
        "source_gguf": str(mmproj),
        "shards": [],
        "config": model.config,
        "chat_template": "",
        "n_tiles": n_tiles,
        "layer_ids": [],
        "n_expert_packed": 0,
        "n_vocab": 0,
        "files": [{"name": f, "bytes": sz} for f, sz in sizes.items()],
        "tokenizer": "",
        "tensors": [s.to_json() for specs in specs_by_task.values() for s in specs],
    }
    tmp = out_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.rename(manifest_path)
    log(f"done: {written / 2**20:.0f} MiB written in {time.time() - t0:.0f} s; manifest {manifest_path}")
    return out_dir

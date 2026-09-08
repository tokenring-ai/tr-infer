"""Layout + parallel writer for pack v2: node{N}.bin, shared.bin, tokenizer.json, manifest.json."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from . import FORMAT_VERSION
from .blocks import BITS, CONVERTERS, FLOAT_TYPES, GGML_IQ4_NL, float_rows_bits, rows_to_groups, to_f32
from .codec import matrix_bytes, pack_tq, pad_rows
from .ggufio import GgufModel, shard_hashes
from .plan import Planner, Spec
from .tokenizer import build_tokenizer_json, dumps

LAYER_ALIGN = 2 << 20
SEC_ALIGN = 64
PLE_NAME = "per_layer_token_embd.weight"


def align_up(v, a):
    return (v + a - 1) // a * a


def _row_ranges(spec: Spec, dims):
    sh = spec.shard
    if sh["kind"] == "rows":
        return [tuple(r) for r in sh["ranges"]]
    return [(0, dims[-2] if len(dims) >= 2 else dims[0])]


def resolve(spec: Spec, ref) -> None:
    """Fill rows/k/shape/nbytes for a spec from the GGUF tensor dims."""
    dims = ref.dims
    if spec.kind == "iq4nl_rows":
        spec.rows = spec.shard["len"]
        spec.row_bytes = dims[-1] // 32 * 18
        spec.nbytes = spec.rows * spec.row_bytes
        spec.shape = [spec.rows, dims[-1]]
        return
    if spec.kind == "bf16_strips":
        K = dims[-1]
        sh = spec.shard
        if sh["kind"] == "rows":
            spec.rows = sum(r[1] for r in sh["ranges"])
            spec.k = K
        elif sh["kind"] == "cols":
            spec.rows = dims[-2]
            spec.k = sh["len"]
        else:
            raise ValueError(sh)
        if spec.k % 32:
            raise ValueError(f"{spec.name}: bf16 strips need k % 32 == 0 (k={spec.k})")
        spec.stride = pad_rows(spec.rows) * spec.k * 2
        spec.nbytes = spec.stride
        spec.shape = [spec.rows, spec.k]
        return
    if spec.kind == "f32":
        sh = spec.shard
        if sh["kind"] == "replicate":
            shape = list(dims)
        elif sh["kind"] == "rows":
            shape = [sum(r[1] for r in sh["ranges"])] + list(dims[1:])
        elif sh["kind"] == "cols_ranges":
            shape = list(dims[:-1]) + [sum(r[1] for r in sh["ranges"])]
        elif sh["kind"] == "cols":
            shape = list(dims[:-1]) + [sh["len"]]
        else:
            raise ValueError(sh)
        spec.shape = shape
        spec.nbytes = int(np.prod(shape)) * 4
        return
    # tq
    K = dims[-1]
    nrows = dims[-2]
    sh = spec.shard
    if sh["kind"] == "rows":
        spec.rows = sum(r[1] for r in sh["ranges"])
        spec.k = K
    elif sh["kind"] == "cols":
        spec.rows = nrows
        spec.k = sh["len"]
    elif sh["kind"] == "cols_ranges":
        spec.rows = nrows
        spec.k = sum(r[1] for r in sh["ranges"])
    else:
        raise ValueError(sh)
    spec.stride = matrix_bytes(spec.rows, spec.k, spec.bits, spec.kb)
    spec.nbytes = spec.stride * spec.count
    if spec.count > 1:
        assert dims[0] >= spec.count, (spec.name, dims, spec.count)
    spec.shape = [spec.count, spec.rows, spec.k] if spec.count > 1 else [spec.rows, spec.k]


def layout(specs_by_task: dict, n_tiles: int) -> dict[str, int]:
    """Assign file/offset to every spec, task by task (layers 2 MiB aligned). Returns file sizes."""
    off = {f"node{t}.bin": 0 for t in range(n_tiles)}
    off["shared.bin"] = 0
    for task, specs in specs_by_task.items():
        for f in off:
            off[f] = align_up(off[f], LAYER_ALIGN)
        for s in specs:
            f = "shared.bin" if s.tile is None else f"node{s.tile}.bin"
            s.file = f
            s.offset = align_up(off[f], SEC_ALIGN)
            off[f] = s.offset + s.nbytes
    return {f: align_up(v, LAYER_ALIGN) for f, v in off.items()}


# ---------------------------------------------------------------------------
# conversion of one spec -> bytes

_FULL_CACHE: dict = {}  # (name, bits, kb) -> (q, d, m) of the whole tensor; cols shards of experts reuse it across tiles


def convert_tq(spec: Spec, ref) -> np.ndarray:
    raw = np.asarray(ref.data)
    K = ref.dims[-1]
    conv = CONVERTERS[ref.ggml_type]
    src_bits = BITS[ref.ggml_type]
    if ref.ggml_type in FLOAT_TYPES and spec.bits != src_bits:
        # float source: quantise straight to the target width (no 8-bit intermediate)
        conv = (lambda t, b: (lambda raw, K: float_rows_bits(raw, K, t, b)))(ref.ggml_type, spec.bits)
        src_bits = spec.bits
    if spec.count > 1:
        raw2 = raw.reshape(ref.dims[0], ref.dims[-2], raw.shape[-1])[: spec.count]
    else:
        raw2 = raw.reshape(1, ref.dims[-2], raw.shape[-1])
    sh = spec.shard
    if sh["kind"] == "rows":
        sel = np.concatenate([raw2[:, s : s + l] for s, l in sh["ranges"]], axis=1)
        flat = np.ascontiguousarray(sel).reshape(-1, raw.shape[-1])
        q, d, m = conv(flat, K)
        q, d, m = rows_to_groups(q, d, m, spec.kb)
    else:  # cols / cols_ranges: convert the whole rows then slice K in kb groups
        key = (spec.name, spec.bits, spec.kb)
        if key in _FULL_CACHE:
            q, d, m = _FULL_CACHE[key]
        else:
            flat = np.ascontiguousarray(raw2).reshape(-1, raw.shape[-1])
            q, d, m = conv(flat, K)
            q, d, m = rows_to_groups(q, d, m, spec.kb)
            if spec.count > 1:
                _FULL_CACHE.clear()
                _FULL_CACHE[key] = (q, d, m)
        ranges = sh["ranges"] if sh["kind"] == "cols_ranges" else [[sh["start"], sh["len"]]]
        for s, l in ranges:
            assert s % spec.kb == 0 and l % spec.kb == 0, (spec.name, s, l, spec.kb)
        q = np.concatenate([q[:, s : s + l] for s, l in ranges], axis=1)
        d = np.concatenate([d[:, s // spec.kb : (s + l) // spec.kb] for s, l in ranges], axis=1)
        m = np.concatenate([m[:, s // spec.kb : (s + l) // spec.kb] for s, l in ranges], axis=1)
    if spec.bits != src_bits:
        q, d, m = requant(q, d, m, spec.bits)
    rows_per = q.shape[0] // spec.count
    assert rows_per == spec.rows, (spec.name, rows_per, spec.rows)
    if spec.count > 1 and pad_rows(spec.rows) != spec.rows:
        raise ValueError(f"{spec.name}: per-item rows {spec.rows} must be a multiple of 16 when count>1")
    out = pack_tq(np.ascontiguousarray(q), np.ascontiguousarray(d), np.ascontiguousarray(m), spec.bits, spec.kb)
    assert out.nbytes == spec.nbytes, (spec.name, out.nbytes, spec.nbytes)
    return out


def requant(q, d, m, bits):
    """Lossy: re-quantize per group to `bits` (used only by --down-q8-to-q5)."""
    kb = q.shape[1] // d.shape[1]
    w = q.astype(np.float32).reshape(q.shape[0], -1, kb) * d.astype(np.float32)[:, :, None] + m.astype(np.float32)[:, :, None]
    lo, hi = w.min(-1), w.max(-1)
    qmax = (1 << bits) - 1
    dd = (hi - lo) / qmax
    dd = np.where(dd == 0, 1e-8, dd)
    qq = np.clip(np.rint((w - lo[:, :, None]) / dd[:, :, None]), 0, qmax).astype(np.uint8)
    return qq.reshape(q.shape), dd.astype(np.float16), lo.astype(np.float16)


def convert_bf16_strips(spec: Spec, ref) -> np.ndarray:
    """Float rows -> bf16 (round to nearest even) in AMX B-tile strip order: [S][K/2][16][2]."""
    from .safetensors import bf16_from_f32

    w = to_f32(np.asarray(ref.data), ref.ggml_type).reshape(ref.dims[-2], ref.dims[-1])
    sh = spec.shard
    if sh["kind"] == "rows":
        w = np.concatenate([w[s : s + l] for s, l in sh["ranges"]], axis=0)
    else:
        w = w[:, sh["start"] : sh["start"] + sh["len"]]
    rows, K = w.shape
    rp = pad_rows(rows)
    if rp != rows:
        w = np.concatenate([w, np.zeros((rp - rows, K), np.float32)], axis=0)
    b = bf16_from_f32(np.ascontiguousarray(w, dtype=np.float32)).reshape(rp // 16, 16, K // 2, 2)
    out = np.ascontiguousarray(b.transpose(0, 2, 1, 3)).reshape(-1)
    assert out.nbytes == spec.nbytes, (spec.name, out.nbytes, spec.nbytes)
    return out.view(np.uint8)


def convert_f32(spec: Spec, ref) -> np.ndarray:
    arr = to_f32(np.asarray(ref.data), ref.ggml_type).reshape(ref.dims)
    sh = spec.shard
    if sh["kind"] == "rows":
        arr = np.concatenate([arr[s : s + l] for s, l in sh["ranges"]], axis=0)
    elif sh["kind"] == "cols_ranges":
        arr = np.concatenate([arr[..., s : s + l] for s, l in sh["ranges"]], axis=-1)
    elif sh["kind"] == "cols":
        arr = arr[..., sh["start"] : sh["start"] + sh["len"]]
    out = np.ascontiguousarray(arr, dtype=np.float32)
    assert out.nbytes == spec.nbytes, (spec.name, out.shape, spec.shape)
    return out.view(np.uint8).reshape(-1)


def write_spec(fds: dict, spec: Spec, ref) -> None:
    if spec.kind == "iq4nl_rows":
        raw = np.asarray(ref.data).reshape(-1)
        start = spec.shard["start"] * spec.row_bytes
        total = spec.nbytes
        done = 0
        chunk = 256 << 20
        while done < total:
            n = min(chunk, total - done)
            os.pwrite(fds[spec.file], np.ascontiguousarray(raw[start + done : start + done + n]).tobytes(), spec.offset + done)
            done += n
        return
    data = convert_f32(spec, ref) if spec.kind == "f32" else convert_bf16_strips(spec, ref) if spec.kind == "bf16_strips" else convert_tq(spec, ref)
    os.pwrite(fds[spec.file], data.tobytes(), spec.offset)


# ---------------------------------------------------------------------------
# worker process

_G = {}


def _worker_init(gguf_path: str, out_dir: str, files: list[str]):
    _G["model"] = GgufModel(Path(gguf_path))
    _G["fds"] = {f: os.open(os.path.join(out_dir, f), os.O_WRONLY) for f in files}


def _worker_task(args):
    task_name, specs = args
    model, fds = _G["model"], _G["fds"]
    t0 = time.time()
    nbytes = 0
    for s in specs:
        try:
            write_spec(fds, s, model.tensors[s.name])
        except Exception:
            import traceback
            raise RuntimeError(f"spec {s.name} tile {s.tile} shard {s.shard}:\n" + traceback.format_exc())
        nbytes += s.nbytes
    return task_name, nbytes, time.time() - t0


# ---------------------------------------------------------------------------

def pack(gguf_path: Path, out_dir: Path, n_tiles: int = 8, layers: list[int] | None = None, n_experts: int | None = None,
         procs: int = 6, fast_hash: bool = False, down_q8_to_q5: bool = False, skip_ple: bool = False, force: bool = False,
         log=print) -> Path:
    if str(out_dir).startswith("/tmp"):
        raise SystemExit("refusing to write a pack under /tmp (tmpfs)")
    model = GgufModel(gguf_path)
    cfg = model.config()
    arch = cfg["architecture"]
    n_layer = cfg[f"{arch}.block_count"]
    layer_ids = list(range(n_layer)) if layers is None else list(layers)
    types = {n: t.ggml_type for n, t in model.tensors.items()}
    planner = Planner(cfg, n_tiles, types, down_q8_to_q5=down_q8_to_q5)
    if n_experts is not None:
        planner.n_expert = n_experts
    n_vocab = model.tensors["token_embd.weight"].dims[0]

    specs_by_task: dict[str, list[Spec]] = {}
    for il in layer_ids:
        specs_by_task[f"layer{il}"] = planner.layer_specs(il)
    specs_by_task["head"] = planner.head_specs() + planner.vocab_specs(n_vocab)
    if not skip_ple and PLE_NAME in model.tensors:
        specs_by_task["ple"] = planner.ple_specs(model.tensors[PLE_NAME].dims[0])
    for specs in specs_by_task.values():
        for s in specs:
            resolve(s, model.tensors[s.name])
            if s.count > 1 and n_experts is not None:
                pass
    sizes = layout(specs_by_task, n_tiles)

    shards = shard_hashes(model.shards, fast=fast_hash)
    identity = {
        "format_version": FORMAT_VERSION,
        "plan_version": 4,  # v2: periodic k-head broadcast; v3: hc_down K-split; v4: QSA indexer tensors
        "n_tiles": n_tiles,
        "layer_ids": layer_ids,
        "n_expert_packed": planner.n_expert,
        "down_q8_to_q5": down_q8_to_q5,
        "skip_ple": skip_ple or PLE_NAME not in model.tensors,
        "shards": [s["sha256"] for s in shards],
    }
    ident_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

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
    total_bytes = sum(sizes.values())
    log(f"pack -> {out_dir}  files: " + ", ".join(f"{f} {sz / 2**30:.2f} GiB" for f, sz in sizes.items()))

    # tokenizer
    toks, ttypes, merges = model.tokenizer_lists()
    (out_dir / "tokenizer.json").write_text(dumps(build_tokenizer_json(toks, ttypes, merges, cfg.get("tokenizer.ggml.pre", "qwen35"))), encoding="utf-8")

    tasks = sorted(specs_by_task.items(), key=lambda kv: -sum(s.nbytes for s in kv[1]))  # big first
    t0 = time.time()
    written = 0
    ctx = mp.get_context("forkserver")
    with ctx.Pool(procs, initializer=_worker_init, initargs=(str(gguf_path), str(out_dir), list(sizes))) as pool:
        for name, nb, dt in pool.imap_unordered(_worker_task, tasks):
            written += nb
            log(f"  {name:10s} {nb / 2**30:6.2f} GiB in {dt:6.1f} s   total {written / 2**30:7.2f} GiB  elapsed {time.time() - t0:6.0f} s")
    for f in sizes:
        fd = os.open(out_dir / f, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)

    manifest = {
        "format_version": FORMAT_VERSION,
        "hash": ident_hash,
        "identity": identity,
        "source_gguf": str(gguf_path),
        "shards": shards,
        "config": cfg,
        "chat_template": model.kv.get("tokenizer.chat_template", ""),
        "n_tiles": n_tiles,
        "layer_ids": layer_ids,
        "n_expert_packed": planner.n_expert,
        "n_vocab": n_vocab,
        "files": [{"name": f, "bytes": sz} for f, sz in sizes.items()],
        "tokenizer": "tokenizer.json",
        "tensors": [s.to_json() for specs in specs_by_task.values() for s in specs],
    }
    tmp = out_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.rename(manifest_path)
    log(f"done: {written / 2**30:.2f} GiB written in {time.time() - t0:.0f} s; manifest {manifest_path}")
    return out_dir

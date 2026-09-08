"""MTP (multi-token prediction) draft head: BF16 safetensors -> overlay pack.

The draft head is one full-attention decoder layer (packed as `blk.{il}.*`, il = block_count) plus
its own output mixer and two input projections. It shares `token_embd` / `output.weight` with the
base pack; the overlay directory references the base pack by hash (`overlay_of`).

HF names (`mtp.*`) map onto the GGUF names the engine already knows; Gemma norms get +1 exactly as
llama.cpp's converter does for `*norm.weight`; `index_qk_proj` splits into indexer q and k.
Experts are quantised round-to-nearest to 4-bit (gate/up) and 5-bit (down), the dense projections
to 8-bit, mirroring the base pack's Q4_K / Q5_1 / Q8_0 mix.
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
from .ggufio import TensorRef
from .plan import Planner, Spec
from .safetensors import SafetensorsFile
from .writer import layout, resolve, write_spec

MTP_PLAN_VERSION = 1


class LazyTensors(dict):
    """name -> TensorRef, materialised on first access (expert stacks are 1.7 GB each)."""

    def __init__(self, builders: dict):
        super().__init__()
        self._b = builders
        for k in builders:
            super().__setitem__(k, None)

    def __getitem__(self, k):
        v = super().__getitem__(k)
        if v is None:
            v = self._b[k]()
            super().__setitem__(k, v)
        return v

    def drop(self, k):
        super().__setitem__(k, None)


class MtpModel:
    """The safetensors MTP tensors, renamed and reshaped as the engine expects."""

    def __init__(self, path: Path, il: int, n_expert: int):
        self.st = SafetensorsFile(path)
        self.il = il
        self.n_expert = n_expert
        self.builders: dict[str, tuple[int, tuple[int, ...], callable]] = {}
        self._map()
        self.tensors = LazyTensors({n: (lambda n=n, b=b: self._ref(n, b)) for n, (_, _, b) in self.builders.items()})
        self.types = {n: t for n, (t, _, _) in self.builders.items()}
        self.dims = {n: d for n, (_, d, _) in self.builders.items()}

    def _ref(self, name, build):
        ty, dims, _ = self.builders[name]
        data = build()
        return TensorRef(name, ty, dims, data, int(data.nbytes), self.st.path)

    # ---- mapping -------------------------------------------------------
    def _map(self):
        st, p = self.st, f"blk.{self.il}."
        L = "mtp.layers.0."

        def view(engine, hf):
            self.builders[engine] = (st.ggml_type(hf), st.shape(hf), lambda hf=hf: st.array(hf))

        def norm_plus1(engine, hf):
            def b(hf=hf):
                return to_f32(st.array(hf), st.ggml_type(hf)).reshape(-1) + 1.0
            self.builders[engine] = (GGML_F32, (st.shape(hf)[-1],), b)

        def f32(engine, hf, shape=None):
            def b(hf=hf, shape=shape):
                a = to_f32(st.array(hf), st.ggml_type(hf))
                return a.reshape(shape) if shape else a
            self.builders[engine] = (GGML_F32, tuple(shape) if shape else st.shape(hf), b)

        # attention
        view(p + "attn_q.weight", L + "self_attn.q_proj.weight")
        view(p + "attn_k.weight", L + "self_attn.k_proj.weight")
        view(p + "attn_v.weight", L + "self_attn.v_proj.weight")
        view(p + "attn_output.weight", L + "self_attn.o_proj.weight")
        norm_plus1(p + "attn_q_norm.weight", L + "self_attn.q_norm.weight")
        norm_plus1(p + "attn_k_norm.weight", L + "self_attn.k_norm.weight")
        # QSA indexer: one projection feeds q (idx_heads*idx_dim rows) and k (idx_dim rows)
        qk = L + "self_attn.indexer.index_qk_proj.weight"
        n_k = st.shape(L + "self_attn.indexer.k_layernorm.weight")[0]
        n_q = st.shape(qk)[0] - n_k
        self.builders[p + "indexer.q_proj.weight"] = (st.ggml_type(qk), (n_q, st.shape(qk)[1]), lambda: st.array(qk)[:n_q])
        self.builders[p + "indexer.k_proj.weight"] = (st.ggml_type(qk), (n_k, st.shape(qk)[1]), lambda: st.array(qk)[n_q:])
        norm_plus1(p + "indexer.q_norm.weight", L + "self_attn.indexer.q_layernorm.weight")
        norm_plus1(p + "indexer.k_norm.weight", L + "self_attn.indexer.k_layernorm.weight")
        # hyper-connections (per branch)
        for which, hf in (("hc_attn", "attn_hyper_connection"), ("hc_ffn", "mlp_hyper_connection")):
            norm_plus1(p + f"{which}_norm.weight", L + f"{hf}.hc_norm.weight")
            f32(p + f"{which}_inject.weight", L + f"{hf}.block_inject_weight.weight")
            view(p + f"{which}_down.weight", L + f"{hf}.input_mix_weight_down.weight")
            view(p + f"{which}_up.weight", L + f"{hf}.input_mix_weight_up.weight")
        # MoE
        f32(p + "ffn_gate_inp.weight", L + "mlp.gate.weight")
        f32(p + "ffn_gate_inp_shexp.weight", L + "mlp.shared_expert_gate.weight", shape=(st.shape(L + "mlp.shared_expert_gate.weight")[-1],))
        view(p + "ffn_gate_shexp.weight", L + "mlp.shared_expert.gate_proj.weight")
        view(p + "ffn_up_shexp.weight", L + "mlp.shared_expert.up_proj.weight")
        view(p + "ffn_down_shexp.weight", L + "mlp.shared_expert.down_proj.weight")
        for w in ("gate", "up", "down"):
            names = [L + f"mlp.experts.{e}.{w}_proj.weight" for e in range(self.n_expert)]
            for n in names:
                if n not in st.header:
                    raise KeyError(f"missing expert tensor {n}")
            shape = (self.n_expert,) + st.shape(names[0])
            self.builders[p + f"ffn_{w}_exps.weight"] = (st.ggml_type(names[0]), shape, lambda names=names: np.stack([st.array(n) for n in names]))
        # draft-head specifics
        norm_plus1("mtp_hc_norm.weight", "mtp.hyper_connection_mixer.hc_norm.weight")
        view("mtp_hc_down.weight", "mtp.hyper_connection_mixer.input_mix_weight_down.weight")
        view("mtp_hc_up.weight", "mtp.hyper_connection_mixer.input_mix_weight_up.weight")
        view("mtp_fc_embedding.weight", "mtp.fc_embedding.weight")
        view("mtp_fc_hidden.weight", "mtp.fc_hidden.weight")
        norm_plus1("mtp_enorm.weight", "mtp.pre_fc_norm_embedding.weight")
        norm_plus1("mtp_hnorm.weight", "mtp.pre_fc_norm_hidden.weight")

    def unmapped(self) -> list[str]:
        """`mtp.*` tensors in the file that the mapping does not consume."""
        used = {"mtp.layers.0.self_attn.indexer.index_qk_proj.weight"}
        L = "mtp.layers.0."
        used |= {L + f"mlp.experts.{e}.{w}_proj.weight" for e in range(self.n_expert) for w in ("gate", "up", "down")}
        used |= {
            L + "self_attn.q_proj.weight", L + "self_attn.k_proj.weight", L + "self_attn.v_proj.weight", L + "self_attn.o_proj.weight",
            L + "self_attn.q_norm.weight", L + "self_attn.k_norm.weight",
            L + "self_attn.indexer.q_layernorm.weight", L + "self_attn.indexer.k_layernorm.weight",
            L + "mlp.gate.weight", L + "mlp.shared_expert_gate.weight",
            L + "mlp.shared_expert.gate_proj.weight", L + "mlp.shared_expert.up_proj.weight", L + "mlp.shared_expert.down_proj.weight",
            "mtp.hyper_connection_mixer.hc_norm.weight", "mtp.hyper_connection_mixer.input_mix_weight_down.weight", "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
            "mtp.fc_embedding.weight", "mtp.fc_hidden.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
        }
        for hf in ("attn_hyper_connection", "mlp_hyper_connection"):
            used |= {L + f"{hf}.{n}" for n in ("hc_norm.weight", "block_inject_weight.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight")}
        return sorted(n for n in self.st.names() if n.startswith("mtp.") and n not in used)


def _fast_sha(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(16 << 20))
    h.update(f"{st.st_size}".encode())
    return h.hexdigest()


def pack_mtp(st_path: Path, base_dir: Path, out_dir: Path, force: bool = False, expert_bits: tuple[int, int, int] = (4, 4, 5), log=print) -> Path:
    if str(out_dir).startswith("/tmp"):
        raise SystemExit("refusing to write a pack under /tmp (tmpfs)")
    base = json.loads((Path(base_dir) / "manifest.json").read_text())
    cfg = base["config"]
    arch = cfg["architecture"]
    n_tiles = base["n_tiles"]
    il = cfg[f"{arch}.block_count"]
    n_expert = cfg[f"{arch}.expert_count"]
    model = MtpModel(Path(st_path), il, n_expert)
    if model.unmapped():
        raise SystemExit(f"unmapped mtp tensors: {model.unmapped()[:8]} ...")
    bits = {f"blk.{il}.ffn_gate_exps.weight": expert_bits[0], f"blk.{il}.ffn_up_exps.weight": expert_bits[1], f"blk.{il}.ffn_down_exps.weight": expert_bits[2]}
    planner = Planner(cfg, n_tiles, model.types, bits_override=bits)
    specs_by_task = {f"layer{il}": planner.layer_specs(il, recurrent=False), "mtp": planner.mtp_specs()}
    for specs in specs_by_task.values():
        for s in specs:
            resolve(s, _Dims(model.dims[s.name]))
    sizes = layout(specs_by_task, n_tiles)

    identity = {
        "format_version": FORMAT_VERSION,
        "mtp_plan_version": MTP_PLAN_VERSION,
        "overlay_of": base["hash"],
        "n_tiles": n_tiles,
        "layer_ids": [il],
        "n_expert_packed": n_expert,
        "expert_bits": list(expert_bits),
        "source": _fast_sha(Path(st_path)),
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
    log(f"mtp pack -> {out_dir}  files: " + ", ".join(f"{f} {sz / 2**20:.0f} MiB" for f, sz in sizes.items()))
    fds = {f: os.open(out_dir / f, os.O_WRONLY) for f in sizes}
    t0 = time.time()
    written = 0
    # experts last so their stacks (and the writer's full-row cache) are alive as briefly as possible
    order = sorted((s for specs in specs_by_task.values() for s in specs), key=lambda s: (s.count > 1, s.name, s.tile if s.tile is not None else -1))
    last = None
    for s in order:
        if last is not None and last != s.name and last.endswith("_exps.weight"):
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
        "overlay_of": base["hash"],
        "source_gguf": str(st_path),
        "shards": [],
        "config": cfg,
        "chat_template": "",
        "n_tiles": n_tiles,
        "layer_ids": [il],
        "n_expert_packed": n_expert,
        "n_vocab": base["n_vocab"],
        "files": [{"name": f, "bytes": sz} for f, sz in sizes.items()],
        "tokenizer": "",
        "tensors": [s.to_json() for specs in specs_by_task.values() for s in specs],
    }
    tmp = out_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.rename(manifest_path)
    log(f"done: {written / 2**20:.0f} MiB written in {time.time() - t0:.0f} s; manifest {manifest_path}")
    return out_dir


class _Dims:
    def __init__(self, dims):
        self.dims = tuple(dims)

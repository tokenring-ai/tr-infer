"""Vision overlay packer: padding/sharding of a tiny synthetic mmproj and a round trip through the codec."""
import json
from pathlib import Path

import numpy as np
import pytest

from trpack.blocks import GGML_F16, GGML_F32
from trpack.ggufio import TensorRef
from trpack.vision import VisionModel, pack_vision

T = 8
H, NH, NL, FF, P, PROJ, NPOS = 64, 8, 2, 100, 16, 96, 9  # ff 100 -> 13 per tile -> padded 32


def tiny_mmproj(seed=0):
    rng = np.random.default_rng(seed)
    t = {}

    def add(name, shape, ty=GGML_F16):
        a = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        data = a.astype(np.float16) if ty == GGML_F16 else a
        t[name] = TensorRef(name, ty, tuple(shape), data, data.nbytes, Path("x"))
        return a

    add("v.patch_embd.weight", (H, 3, P, P))
    add("v.patch_embd.weight.1", (H, 3, P, P))
    add("v.patch_embd.bias", (H,), GGML_F32)
    add("v.position_embd.weight", (NPOS, H), GGML_F32)
    add("v.post_ln.weight", (H,), GGML_F32)
    add("v.post_ln.bias", (H,), GGML_F32)
    for il in range(NL):
        p = f"v.blk.{il}."
        for n in ("ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias", "attn_out.bias", "ffn_down.bias"):
            add(p + n, (H,), GGML_F32)
        add(p + "attn_qkv.weight", (3 * H, H))
        add(p + "attn_qkv.bias", (3 * H,), GGML_F32)
        add(p + "attn_out.weight", (H, H))
        add(p + "ffn_up.weight", (FF, H))
        add(p + "ffn_up.bias", (FF,), GGML_F32)
        add(p + "ffn_down.weight", (H, FF))
    add("mm.0.weight", (4 * H, 4 * H))
    add("mm.0.bias", (4 * H,), GGML_F32)
    add("mm.2.weight", (PROJ, 4 * H))
    add("mm.2.bias", (PROJ,), GGML_F32)
    kv = {
        "clip.projector_type": "qwen3vl_merger",
        "clip.vision.embedding_length": H,
        "clip.vision.block_count": NL,
        "clip.vision.attention.head_count": NH,
        "clip.vision.feed_forward_length": FF,
        "clip.vision.patch_size": P,
        "clip.vision.spatial_merge_size": 2,
        "clip.vision.projection_dim": PROJ,
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
    }
    return t, kv


def f32(ref):
    return np.asarray(ref.data).astype(np.float32).reshape(ref.dims)


def test_padding_and_pack(tmp_path):
    src, kv = tiny_mmproj()
    m = VisionModel(src, kv, T)
    assert (m.dl, m.opad, m.fp, m.n_ff_pad, m.mp) == (8, 32, 32, 256, 32)
    # patch kernels summed, attn_out K padded per tile, ffn padded rows/cols
    pe = m.tensors["v.patch_embd.weight"].data
    assert np.allclose(pe, f32(src["v.patch_embd.weight"]).reshape(H, -1) + f32(src["v.patch_embd.weight.1"]).reshape(H, -1))
    ao = m.tensors["v.blk.0.attn_out.weight"].data
    assert ao.shape == (H, 32 * T)
    ref = f32(src["v.blk.0.attn_out.weight"])
    for t in range(T):
        assert np.array_equal(ao[:, 32 * t : 32 * t + 8], ref[:, 8 * t : 8 * t + 8])
        assert not ao[:, 32 * t + 8 : 32 * (t + 1)].any()
    up = m.tensors["v.blk.1.ffn_up.weight"].data
    assert up.shape == (256, H)
    per = -(-FF // T)  # 13
    for t in range(T):
        ln = max(0, min(per, FF - per * t))
        assert np.array_equal(up[32 * t : 32 * t + ln], f32(src["v.blk.1.ffn_up.weight"])[per * t : per * t + ln])
        assert not up[32 * t + ln : 32 * (t + 1)].any()
    dn = m.tensors["v.blk.1.ffn_down.weight"].data
    assert dn.shape == (H, 256) and np.array_equal(dn[:, :13], f32(src["v.blk.1.ffn_down.weight"])[:, :13]) and not dn[:, 13:32].any()

    alt = tmp_path / "vis"
    pack_vision(Path("synthetic"), alt, n_tiles=T, model=m, source_hash="0" * 16, log=lambda *a: None)
    man = json.loads((alt / "manifest.json").read_text())
    assert man["overlay_of"] is None and man["config"]["clip.vision.projection_dim"] == PROJ
    assert man["config"]["vision.attn_out_k_pad"] == 32 and man["config"]["vision.ff_per_tile"] == 32
    by = {}
    for e in man["tensors"]:
        by.setdefault(e["name"], {})[e["tile"]] = e

    def read(e):
        with open(alt / e["file"], "rb") as f:
            f.seek(e["offset"])
            return np.frombuffer(f.read(e["nbytes"]), np.uint8)

    def unstrip(buf, rows, k):
        b = np.frombuffer(buf, np.uint16).reshape(-1, k // 2, 16, 2).transpose(0, 2, 1, 3).reshape(-1, k)[:rows]
        return (b.astype(np.uint32) << 16).view(np.float32)

    e = by["v.blk.0.attn_qkv.weight"][3]
    assert e["kind"] == "bf16_strips" and e["rows"] == 24 and e["k"] == H and e["nbytes"] == 32 * H * 2
    w = unstrip(read(e), e["rows"], e["k"])
    qkv = f32(src["v.blk.0.attn_qkv.weight"])
    ref = np.concatenate([qkv[24:32], qkv[H + 24 : H + 32], qkv[2 * H + 24 : 2 * H + 32]])
    assert np.abs(w - ref).max() <= np.abs(ref).max() * 2**-8 + 1e-6
    e = by["v.blk.0.attn_qkv.bias"][3]
    b = np.frombuffer(read(e), np.float32)
    bb = f32(src["v.blk.0.attn_qkv.bias"])
    assert np.array_equal(b, np.concatenate([bb[24:32], bb[H + 24 : H + 32], bb[2 * H + 24 : 2 * H + 32]]))
    e = by["v.blk.1.ffn_down.weight"][7]
    assert e["k"] == 32 and e["rows"] == H
    w = unstrip(read(e), H, 32)
    assert not w[:, 9:].any()  # tile 7 holds cols 91..100 (9 real) + zero padding
    assert np.abs(w[:, :9] - f32(src["v.blk.1.ffn_down.weight"])[:, 91:100]).max() <= 0.5 * 2**-8 + 1e-6
    e = by["mm.2.weight"][2]
    assert e["k"] == 32 and e["rows"] == PROJ
    e = by["v.position_embd.weight"][None]
    pos = np.frombuffer(read(e), np.float32).reshape(NPOS, H)
    assert np.array_equal(pos, f32(src["v.position_embd.weight"]))
    e = by["v.patch_embd.weight"][None]
    assert e["file"] == "shared.bin" and e["rows"] == H and e["k"] == 3 * P * P


def test_rejects_other_projectors():
    src, kv = tiny_mmproj()
    kv["clip.projector_type"] = "mlp"
    with pytest.raises(SystemExit):
        VisionModel(src, kv, T)

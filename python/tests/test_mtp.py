"""MTP overlay packer: safetensors reader, RTN quantiser, name mapping, end-to-end tiny pack."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from trpack.blocks import GGML_BF16, float_rows_bits, rows_to_groups, to_f32
from trpack.codec import pack_tq, unpack_tq
from trpack.safetensors import SafetensorsFile, bf16_from_f32, write_safetensors

REAL_ST = Path("/opt/llm/Qwen3.8-Flash-Next-W4A16-AutoRound/model_extra_tensors.safetensors")


def test_safetensors_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 8)).astype(np.float32)
    b = bf16_from_f32(rng.standard_normal((4, 16)).astype(np.float32))
    write_safetensors(tmp_path / "t.safetensors", {"a": a, "b": b})
    st = SafetensorsFile(tmp_path / "t.safetensors")
    assert st.names() == ["a", "b"]
    assert np.array_equal(st.array("a"), a)
    assert np.array_equal(st.array("b"), b)
    assert st.ggml_type("b") == GGML_BF16
    f = to_f32(st.array("b"), GGML_BF16)
    assert f.shape == (4, 16) and np.all(np.isfinite(f))


def test_bf16_rounding():
    x = np.array([1.0, 1.0 + 2**-8, 1.0 + 3 * 2**-9, -2.5, 1e-3], np.float32)
    back = to_f32(bf16_from_f32(x), GGML_BF16)
    assert np.all(np.abs(back - x) <= np.abs(x) * 2**-8 + 1e-12)


@pytest.mark.parametrize("bits", [4, 5, 8])
def test_float_rows_bits_error_bound(bits):
    rng = np.random.default_rng(2)
    w = rng.standard_normal((48, 64)).astype(np.float32) * 0.05
    q, d, m = float_rows_bits(bf16_from_f32(w), 64, GGML_BF16, bits)
    assert q.max() < (1 << bits)
    buf = pack_tq(q, d, m, bits, 32)
    back = unpack_tq(buf, 48, 64, bits, 32)
    src = to_f32(bf16_from_f32(w), GGML_BF16)
    rng_ = (src.reshape(48, 2, 32).max(-1) - src.reshape(48, 2, 32).min(-1))
    step = np.repeat(rng_ / ((1 << bits) - 1), 32, axis=1)
    assert np.all(np.abs(back - src) <= step * 0.5 + np.abs(src) * 2e-3 + 1e-6)


# ---- tiny synthetic MTP checkpoint --------------------------------------------------------
H, HC, LR, T = 128, 4, 32, 8
NH, NKV, HD = 8, 2, 32
IH, ID = 4, 128
NE, NFF = 16, 128
NGRAM_CFG = {
    "architecture": "qwen4exp", "qwen4exp.block_count": 2, "qwen4exp.embedding_length": H, "qwen4exp.attention.head_count": NH,
    "qwen4exp.attention.head_count_kv": NKV, "qwen4exp.attention.key_length": HD, "qwen4exp.expert_count": NE,
    "qwen4exp.expert_feed_forward_length": NFF, "qwen4exp.expert_shared_feed_forward_length": NFF, "qwen4exp.hyper_connection.count": HC,
    "qwen4exp.hyper_connection.low_rank": LR, "qwen4exp.ssm.state_size": 16, "qwen4exp.ssm.group_count": 8, "qwen4exp.ssm.time_step_rank": 16,
    "qwen4exp.full_attention_interval": 4, "qwen4exp.attention.indexer.head_count": IH, "qwen4exp.attention.indexer.key_length": ID,
}


def tiny_mtp(path: Path, seed=3) -> dict:
    rng = np.random.default_rng(seed)
    bf = lambda *shape: bf16_from_f32(rng.standard_normal(shape).astype(np.float32) * 0.1)
    L = "mtp.layers.0."
    t = {
        L + "self_attn.q_proj.weight": bf(NH * HD * 2, H), L + "self_attn.k_proj.weight": bf(NKV * HD, H), L + "self_attn.v_proj.weight": bf(NKV * HD, H),
        L + "self_attn.o_proj.weight": bf(H, NH * HD), L + "self_attn.q_norm.weight": bf(HD), L + "self_attn.k_norm.weight": bf(HD),
        L + "self_attn.indexer.index_qk_proj.weight": bf(IH * ID + ID, H), L + "self_attn.indexer.q_layernorm.weight": bf(ID), L + "self_attn.indexer.k_layernorm.weight": bf(ID),
        L + "mlp.gate.weight": bf(NE, H), L + "mlp.shared_expert_gate.weight": bf(1, H),
        L + "mlp.shared_expert.gate_proj.weight": bf(NFF, H), L + "mlp.shared_expert.up_proj.weight": bf(NFF, H), L + "mlp.shared_expert.down_proj.weight": bf(H, NFF),
        "mtp.hyper_connection_mixer.hc_norm.weight": bf(HC * H), "mtp.hyper_connection_mixer.input_mix_weight_down.weight": bf(LR, HC * H),
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight": bf(HC * H, LR),
        "mtp.fc_embedding.weight": bf(H, H), "mtp.fc_hidden.weight": bf(H, H), "mtp.pre_fc_norm_embedding.weight": bf(H), "mtp.pre_fc_norm_hidden.weight": bf(HC * H),
    }
    for hf in ("attn_hyper_connection", "mlp_hyper_connection"):
        t[L + f"{hf}.hc_norm.weight"] = bf(HC * H)
        t[L + f"{hf}.block_inject_weight.weight"] = bf(HC, HC * H)
        t[L + f"{hf}.input_mix_weight_down.weight"] = bf(LR, HC * H)
        t[L + f"{hf}.input_mix_weight_up.weight"] = bf(HC * H, LR)
    for e in range(NE):
        t[L + f"mlp.experts.{e}.gate_proj.weight"] = bf(NFF, H)
        t[L + f"mlp.experts.{e}.up_proj.weight"] = bf(NFF, H)
        t[L + f"mlp.experts.{e}.down_proj.weight"] = bf(H, NFF)
    write_safetensors(path, t)
    return t


def test_mapping_and_tiny_pack(tmp_path):
    from trpack.mtp import MtpModel, pack_mtp

    src = tiny_mtp(tmp_path / "mtp.safetensors")
    m = MtpModel(tmp_path / "mtp.safetensors", 2, NE)
    assert m.unmapped() == []
    names = set(m.types)
    for n in ("blk.2.attn_q.weight", "blk.2.indexer.q_proj.weight", "blk.2.indexer.k_proj.weight", "blk.2.hc_ffn_inject.weight", "blk.2.ffn_down_exps.weight",
              "mtp_hc_norm.weight", "mtp_hc_down.weight", "mtp_hc_up.weight", "mtp_fc_embedding.weight", "mtp_fc_hidden.weight", "mtp_enorm.weight", "mtp_hnorm.weight"):
        assert n in names, n
    assert len(names) == 33
    assert m.dims["blk.2.ffn_gate_exps.weight"] == (NE, NFF, H)
    assert m.dims["blk.2.ffn_down_exps.weight"] == (NE, H, NFF)
    assert m.dims["blk.2.indexer.q_proj.weight"] == (IH * ID, H)
    assert m.dims["blk.2.ffn_gate_inp_shexp.weight"] == (H,)
    # Gemma +1 and the qk split
    qn = m.tensors["blk.2.attn_q_norm.weight"].data
    assert np.allclose(qn, to_f32(src["mtp.layers.0.self_attn.q_norm.weight"], GGML_BF16) + 1)
    kp = m.tensors["blk.2.indexer.k_proj.weight"].data
    assert np.array_equal(kp, src["mtp.layers.0.self_attn.indexer.index_qk_proj.weight"][IH * ID:])

    # a fake base pack manifest with the tiny config
    base = tmp_path / "base"
    base.mkdir()
    (base / "manifest.json").write_text(json.dumps({"hash": "deadbeefdeadbeef", "config": NGRAM_CFG, "n_tiles": T, "n_vocab": 1000}))
    out = tmp_path / "pack-mtp"
    pack_mtp(tmp_path / "mtp.safetensors", base, out, log=lambda *a: None)
    man = json.loads((out / "manifest.json").read_text())
    assert man["overlay_of"] == "deadbeefdeadbeef" and man["layer_ids"] == [2]
    per_tile = {}
    for e in man["tensors"]:
        per_tile.setdefault(e["tile"], []).append(e)
    assert set(per_tile) == set(range(T)) | {None}
    # re-read one tensor of each kind and compare with the source
    def read(entry):
        with open(out / entry["file"], "rb") as f:
            f.seek(entry["offset"])
            return np.frombuffer(f.read(entry["nbytes"]), np.uint8)
    e = next(x for x in per_tile[3] if x["name"] == "blk.2.attn_q.weight")
    w = unpack_tq(read(e), e["rows"], e["k"], e["bits"], e["kb"])
    ref = to_f32(src["mtp.layers.0.self_attn.q_proj.weight"], GGML_BF16)[3 * (NH // T) * HD * 2:(3 + 1) * (NH // T) * HD * 2]
    assert np.abs(w - ref).max() < 2e-3 * np.abs(ref).max() + 1e-3
    e = next(x for x in per_tile[5] if x["name"] == "blk.2.ffn_down_exps.weight")
    assert e["bits"] == 5 and e["count"] == NE and e["kb"] == 16
    buf = read(e)
    w = unpack_tq(buf[: e["stride"]], e["rows"], e["k"], 5, 16)  # expert 0, this tile's column slice
    ref = to_f32(src["mtp.layers.0.mlp.experts.0.down_proj.weight"], GGML_BF16)[:, 5 * (NFF // T):(5 + 1) * (NFF // T)]
    assert np.abs(w - ref).max() < (ref.max() - ref.min()) / 31 + 1e-3
    e = next(x for x in per_tile[0] if x["name"] == "blk.2.ffn_gate_exps.weight")
    assert e["bits"] == 4
    e = next(x for x in per_tile[None] if x["name"] == "mtp_hnorm.weight")
    v = read(e).view(np.float32)
    assert np.allclose(v, to_f32(src["mtp.pre_fc_norm_hidden.weight"], GGML_BF16) + 1)
    e = next(x for x in per_tile[7] if x["name"] == "mtp_fc_hidden.weight")
    w = unpack_tq(read(e), e["rows"], e["k"], e["bits"], e["kb"])
    ref = to_f32(src["mtp.fc_hidden.weight"], GGML_BF16)[7 * (H // T):(7 + 1) * (H // T)]
    assert np.abs(w - ref).max() < 2e-3 * np.abs(ref).max() + 1e-3
    # idempotent
    pack_mtp(tmp_path / "mtp.safetensors", base, out, log=lambda *a: None)


@pytest.mark.skipif(not REAL_ST.exists(), reason="checkpoint not present")
def test_real_checkpoint_mapping_complete():
    from trpack.mtp import MtpModel

    m = MtpModel(REAL_ST, 48, 512)
    assert m.unmapped() == []
    assert m.dims["blk.48.attn_q.weight"] == (12288, 2560)
    assert m.dims["blk.48.indexer.q_proj.weight"] == (512, 2560) and m.dims["blk.48.indexer.k_proj.weight"] == (128, 2560)
    assert m.dims["blk.48.ffn_down_exps.weight"] == (512, 2560, 640)
    assert m.dims["mtp_hnorm.weight"] == (10240,)

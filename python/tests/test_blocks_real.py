"""Lossless-ness of block converters vs gguf-py dequantize on real tensors (needs the GGUF)."""
import os
from pathlib import Path

import numpy as np
import pytest
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as T

from trpack.blocks import CONVERTERS, BITS, rows_to_groups
from trpack.codec import pack_tq, unpack_tq
from trpack.ggufio import GgufModel

GGUF = Path(os.environ.get("TRPACK_GGUF", "/opt/llm/Qwen3.8-Flash-Next-UD-Q4_K_XL"))
pytestmark = pytest.mark.skipif(not GGUF.exists(), reason="model not present")


@pytest.fixture(scope="module")
def model():
    return GgufModel(GGUF)


@pytest.mark.parametrize("name,nrows", [
    ("blk.0.ffn_gate_exps.weight", 3),   # Q4_K, [512, 640, 2560]
    ("blk.0.ffn_down_exps.weight", 3),   # Q5_1, [512, 2560, 640]
    ("blk.0.attn_qkv.weight", 64),       # Q8_0
    ("blk.0.hc_attn_down.weight", 40),   # Q8_0 [320, 10240]
])
def test_converter_matches_gguf_dequant(model, name, nrows):
    t = model.tensors[name]
    conv = CONVERTERS[t.ggml_type]
    raw = np.asarray(t.data)
    K = t.dims[-1]
    flat = raw.reshape(-1, raw.shape[-1])
    sub = flat[:nrows] if raw.ndim == 2 else raw[0, :nrows]  # a few rows (of expert 0 for 3-D)
    q, d, m = conv(np.ascontiguousarray(sub), K)
    ref = dequantize(np.ascontiguousarray(sub), T(t.ggml_type)).reshape(-1, K)
    w = q.astype(np.float32) * np.repeat(d.astype(np.float32), 32, axis=1) + np.repeat(m.astype(np.float32), 32, axis=1)
    err = np.abs(w - ref).max()
    scale = np.abs(ref).max()
    assert err <= 2e-3 * scale, f"{name}: max abs err {err} vs scale {scale}"
    assert q.max() < (1 << BITS[t.ggml_type])
    # through the codec too, with kb=16 for the down proj
    kb = 16 if "down_exps" in name else 32
    q2, d2, m2 = rows_to_groups(q, d, m, kb)
    buf = pack_tq(q2, d2, m2, BITS[t.ggml_type], kb)
    w2 = unpack_tq(buf, q.shape[0], K, BITS[t.ggml_type], kb)
    assert np.array_equal(w2, w)


def test_q5k_layer_exists(model):
    names = [n for n, t in model.tensors.items() if t.ggml_type == 13]
    assert names, "expected Q5_K tensors"
    t = model.tensors[names[0]]
    raw = np.asarray(t.data)
    sub = raw[0, :4]
    q, d, m = CONVERTERS[13](np.ascontiguousarray(sub), t.dims[-1])
    ref = dequantize(np.ascontiguousarray(sub), T(13)).reshape(-1, t.dims[-1])
    w = q.astype(np.float32) * np.repeat(d.astype(np.float32), 32, axis=1) + np.repeat(m.astype(np.float32), 32, axis=1)
    assert np.abs(w - ref).max() <= 2e-3 * np.abs(ref).max()

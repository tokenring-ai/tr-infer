"""Lossless conversion of GGUF block quants to (q uint8, d f16, m f16) per 32-element group.

w = d * q + m for every element; q in [0, 2^bits). For Q8_0 the signed int8 is offset by 128 and the
offset folded into m, so all codecs share the unsigned-weight kernel.
Only d*sc products are rounded to f16 (rel err <= 2^-11), everything else is exact.
"""
from __future__ import annotations

import numpy as np

GGML_Q4_K, GGML_Q5_K, GGML_Q5_1, GGML_Q8_0, GGML_IQ4_NL, GGML_F32, GGML_F16, GGML_BF16 = 12, 13, 7, 8, 20, 0, 1, 30
BITS = {GGML_Q4_K: 4, GGML_Q5_K: 5, GGML_Q5_1: 5, GGML_Q8_0: 8, GGML_BF16: 8, GGML_F16: 8, GGML_F32: 8}
ROW_BYTES_PER_ELEM = {GGML_Q4_K: 144 / 256, GGML_Q5_K: 176 / 256, GGML_Q5_1: 24 / 32, GGML_Q8_0: 34 / 32, GGML_IQ4_NL: 18 / 32}


def _scale_min_k(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """scales: uint8 [..., 12] -> (sc, mn) uint8 [..., 8] (identical to gguf-py Q4_K.get_scale_min)."""
    s = scales.reshape(*scales.shape[:-1], 3, 4)
    d, m, m_d = s[..., 0, :], s[..., 1, :], s[..., 2, :]
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc, mn


def q4k_rows(raw: np.ndarray, K: int):
    """raw: uint8 [rows, K//256*144]."""
    rows = raw.shape[0]
    nsb = K // 256
    b = raw.reshape(rows, nsb, 144)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(rows, nsb, 1)
    dmin = b[:, :, 2:4].copy().view(np.float16).astype(np.float32).reshape(rows, nsb, 1)
    sc, mn = _scale_min_k(b[:, :, 4:16])
    qs = b[:, :, 16:144].reshape(rows, nsb, 4, 32)
    q = np.stack([qs & 0x0F, qs >> 4], axis=3).reshape(rows, K)  # sub-block j = 2c + h
    dd = (d * sc.astype(np.float32)).astype(np.float16).reshape(rows, nsb * 8)
    mm = (-(dmin * mn.astype(np.float32))).astype(np.float16).reshape(rows, nsb * 8)
    return q, dd, mm


def q5k_rows(raw: np.ndarray, K: int):
    rows = raw.shape[0]
    nsb = K // 256
    b = raw.reshape(rows, nsb, 176)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(rows, nsb, 1)
    dmin = b[:, :, 2:4].copy().view(np.float16).astype(np.float32).reshape(rows, nsb, 1)
    sc, mn = _scale_min_k(b[:, :, 4:16])
    qh = b[:, :, 16:48].reshape(rows, nsb, 1, 32)
    qs = b[:, :, 48:176].reshape(rows, nsb, 4, 32)
    ql = np.stack([qs & 0x0F, qs >> 4], axis=3).reshape(rows, nsb, 8, 32)
    hb = (qh >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)) & 1
    q = (ql | (hb << 4)).reshape(rows, K)
    dd = (d * sc.astype(np.float32)).astype(np.float16).reshape(rows, nsb * 8)
    mm = (-(dmin * mn.astype(np.float32))).astype(np.float16).reshape(rows, nsb * 8)
    return q, dd, mm


def q5_1_rows(raw: np.ndarray, K: int):
    rows = raw.shape[0]
    nb = K // 32
    b = raw.reshape(rows, nb, 24)
    d = b[:, :, 0:2].copy().view(np.float16).reshape(rows, nb)
    m = b[:, :, 2:4].copy().view(np.float16).reshape(rows, nb)
    qh = b[:, :, 4:8].copy().view(np.uint32).reshape(rows, nb, 1)
    hb = ((qh >> np.arange(32, dtype=np.uint32).reshape(1, 1, 32)) & 1).astype(np.uint8)
    qs = b[:, :, 8:24]
    ql = np.concatenate([qs & 0x0F, qs >> 4], axis=-1)  # elems 0..15 low nibbles, 16..31 high
    q = (ql | (hb << 4)).reshape(rows, K)
    return q, d.copy(), m.copy()


def q8_0_rows(raw: np.ndarray, K: int):
    rows = raw.shape[0]
    nb = K // 32
    b = raw.reshape(rows, nb, 34)
    d = b[:, :, 0:2].copy().view(np.float16).reshape(rows, nb)
    q = (b[:, :, 2:34].view(np.int8).astype(np.int16) + 128).astype(np.uint8).reshape(rows, K)
    m = (-128.0 * d.astype(np.float32)).astype(np.float16)
    return q, d.copy(), m


def float_rows_bits(raw: np.ndarray, K: int, ggml_type: int, bits: int = 8):
    """Lossy: float rows -> asymmetric `bits`-bit per 32 (min/max round-to-nearest), d and m rounded
    to f16. 8-bit is used for the BF16 indexer projections (max abs err = d/2 ~ 0.2% of the group
    range); 4/5-bit for the MTP experts, which only exist as BF16."""
    w = to_f32(raw, ggml_type).reshape(-1, K)
    rows = w.shape[0]
    g = w.reshape(rows, K // 32, 32)
    lo, hi = g.min(-1), g.max(-1)
    qmax = (1 << bits) - 1
    dd = ((hi - lo) / qmax).astype(np.float16).astype(np.float32)
    dd = np.where(dd == 0, 1e-8, dd)
    lo16 = lo.astype(np.float16).astype(np.float32)
    q = np.clip(np.rint((g - lo16[:, :, None]) / dd[:, :, None]), 0, qmax).astype(np.uint8).reshape(rows, K)
    return q, dd.astype(np.float16), lo16.astype(np.float16)


def float_rows_8bit(raw: np.ndarray, K: int, ggml_type: int):
    return float_rows_bits(raw, K, ggml_type, 8)


CONVERTERS = {GGML_Q4_K: q4k_rows, GGML_Q5_K: q5k_rows, GGML_Q5_1: q5_1_rows, GGML_Q8_0: q8_0_rows}
FLOAT_TYPES = (GGML_BF16, GGML_F16, GGML_F32)
for _t in FLOAT_TYPES:
    CONVERTERS[_t] = (lambda t: (lambda raw, K: float_rows_8bit(raw, K, t)))(_t)


def rows_to_groups(q, d, m, kb: int):
    """Re-express per-32 (d, m) as per-kb groups (kb in {16, 32}); exact."""
    if kb == 32:
        return q, d, m
    if kb == 16:
        return q, np.repeat(d, 2, axis=1), np.repeat(m, 2, axis=1)
    raise ValueError(kb)


def to_f32(raw: np.ndarray, ggml_type: int) -> np.ndarray:
    if ggml_type == GGML_F32:
        return np.asarray(raw, dtype=np.float32)
    if ggml_type == GGML_F16:
        return np.asarray(raw, dtype=np.float16).astype(np.float32)
    if ggml_type == GGML_BF16:
        u = np.asarray(raw).view(np.uint16).astype(np.uint32) << 16
        return u.view(np.float32)
    raise ValueError(f"not a float type: {ggml_type}")

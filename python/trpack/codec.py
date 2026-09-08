"""TQ block codec: 16-row strips, per-row f16 scale+min per KB-wide k block, unsigned q.

Block layout (bytes):
    d[16] f16 (32) | m[16] f16 (32) | q (16*KB/2 for 4/5-bit, 16*KB for 8-bit) | hi-plane (2*KB, 5-bit only)

4/5-bit nibble layout: for chunk j in 0..KB/8, 64 bytes = 16 rows x 4 k as dwords; the low nibble of
byte (r, i) holds k = 4j + i and the high nibble holds k = KB/2 + 4j + i.  The 5-bit high-bit plane
stores, per chunk j, 64 bits for the low-half elements then 64 bits for the high-half elements,
bit index r*4 + i, little-endian bit order.
8-bit layout: chunk j in 0..KB/4, 64 bytes = 16 rows x 4 k (k = 4j + i).
w[r][k] = d[r] * q + m[r]
"""
from __future__ import annotations

import numpy as np

STRIP = 16
HDR_BYTES = 64


def block_bytes(bits: int, kb: int) -> int:
    if bits == 8:
        return HDR_BYTES + STRIP * kb
    if bits == 4:
        return HDR_BYTES + STRIP * kb // 2
    if bits == 5:
        return HDR_BYTES + STRIP * kb // 2 + 2 * kb
    raise ValueError(bits)


def pad_rows(rows: int) -> int:
    return (rows + STRIP - 1) // STRIP * STRIP


def matrix_bytes(rows: int, k: int, bits: int, kb: int) -> int:
    if k % kb:
        raise ValueError(f"k={k} not a multiple of kb={kb}")
    return pad_rows(rows) // STRIP * (k // kb) * block_bytes(bits, kb)


def pack_tq(q: np.ndarray, d: np.ndarray, m: np.ndarray, bits: int, kb: int) -> np.ndarray:
    """q: uint8 [rows, K] in [0, 2^bits); d, m: float16 [rows, K//kb]. Returns uint8 bytes."""
    rows, K = q.shape
    assert d.shape == (rows, K // kb) and m.shape == (rows, K // kb), (d.shape, m.shape, rows, K, kb)
    assert q.dtype == np.uint8 and d.dtype == np.float16 and m.dtype == np.float16
    rp = pad_rows(rows)
    if rp != rows:
        q = np.concatenate([q, np.zeros((rp - rows, K), np.uint8)])
        d = np.concatenate([d, np.zeros((rp - rows, K // kb), np.float16)])
        m = np.concatenate([m, np.zeros((rp - rows, K // kb), np.float16)])
    S, B = rp // STRIP, K // kb
    hd = np.ascontiguousarray(d.reshape(S, STRIP, B).transpose(0, 2, 1)).view(np.uint8).reshape(S, B, 32)
    hm = np.ascontiguousarray(m.reshape(S, STRIP, B).transpose(0, 2, 1)).view(np.uint8).reshape(S, B, 32)
    parts = [hd, hm]
    if bits == 8:
        qb = q.reshape(S, STRIP, B, kb // 4, 4).transpose(0, 2, 3, 1, 4).reshape(S, B, STRIP * kb)
        parts.append(np.ascontiguousarray(qb))
    else:
        J = kb // 8
        nib = (q & 0x0F).reshape(S, STRIP, B, 2, J, 4)
        lo = nib[:, :, :, 0].transpose(0, 2, 3, 1, 4)  # [S,B,J,16,4]
        hi = nib[:, :, :, 1].transpose(0, 2, 3, 1, 4)
        qb = (lo | (hi << 4)).reshape(S, B, J * 64)
        parts.append(np.ascontiguousarray(qb))
        if bits == 5:
            hb = ((q >> 4) & 1).reshape(S, STRIP, B, 2, J, 4).transpose(0, 2, 4, 3, 1, 5)  # [S,B,J,2,16,4]
            hp = np.packbits(hb.reshape(S, B, J, 2, 64), axis=-1, bitorder="little")  # [S,B,J,2,8]
            parts.append(hp.reshape(S, B, J * 16))
    out = np.concatenate(parts, axis=-1)
    assert out.shape[-1] == block_bytes(bits, kb), (out.shape, block_bytes(bits, kb))
    return np.ascontiguousarray(out).reshape(-1)


def unpack_tq(buf: np.ndarray, rows: int, k: int, bits: int, kb: int) -> np.ndarray:
    """Reference dequant of a packed matrix -> float32 [rows, k]. Test helper only."""
    rp = pad_rows(rows)
    S, B = rp // STRIP, k // kb
    bb = block_bytes(bits, kb)
    blk = np.frombuffer(buf, np.uint8)[: S * B * bb].reshape(S, B, bb)
    d = blk[:, :, 0:32].copy().view(np.float16).astype(np.float32)  # [S,B,16]
    m = blk[:, :, 32:64].copy().view(np.float16).astype(np.float32)
    if bits == 8:
        q = blk[:, :, 64 : 64 + STRIP * kb].reshape(S, B, kb // 4, STRIP, 4).transpose(0, 3, 1, 2, 4).reshape(S, STRIP, B, kb)
    else:
        J = kb // 8
        qb = blk[:, :, 64 : 64 + J * 64].reshape(S, B, J, STRIP, 4)
        lo = (qb & 0x0F).transpose(0, 3, 1, 2, 4)  # [S,16,B,J,4]
        hi = (qb >> 4).transpose(0, 3, 1, 2, 4)
        q = np.stack([lo, hi], axis=3).reshape(S, STRIP, B, kb)
        if bits == 5:
            hp = blk[:, :, 64 + J * 64 : 64 + J * 64 + J * 16].reshape(S, B, J, 2, 8)
            hb = np.unpackbits(hp, axis=-1, bitorder="little").reshape(S, B, J, 2, STRIP, 4)
            hb = hb.transpose(0, 4, 1, 3, 2, 5).reshape(S, STRIP, B, kb)
            q = q | (hb << 4)
    q = q.astype(np.float32)
    w = q * d.transpose(0, 2, 1)[:, :, :, None] + m.transpose(0, 2, 1)[:, :, :, None]
    return w.reshape(rp, k)[:rows]

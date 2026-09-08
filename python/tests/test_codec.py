import numpy as np
import pytest
from trpack.codec import pack_tq, unpack_tq, matrix_bytes, block_bytes


@pytest.mark.parametrize("bits,kb,rows,k", [(4, 32, 32, 64), (5, 32, 80, 2560), (8, 32, 40, 320), (5, 16, 48, 80), (8, 16, 16, 32), (4, 32, 5, 64)])
def test_roundtrip(bits, kb, rows, k):
    rng = np.random.default_rng(0)
    q = rng.integers(0, 1 << bits, size=(rows, k), dtype=np.uint8)
    d = rng.uniform(0.001, 0.1, size=(rows, k // kb)).astype(np.float16)
    m = rng.uniform(-1, 1, size=(rows, k // kb)).astype(np.float16)
    buf = pack_tq(q, d, m, bits, kb)
    assert buf.nbytes == matrix_bytes(rows, k, bits, kb)
    w = unpack_tq(buf, rows, k, bits, kb)
    ref = q.astype(np.float32) * np.repeat(d.astype(np.float32), kb, axis=1) + np.repeat(m.astype(np.float32), kb, axis=1)
    assert np.array_equal(w, ref)


def test_block_bytes():
    assert block_bytes(4, 32) == 64 + 256
    assert block_bytes(5, 32) == 64 + 256 + 64
    assert block_bytes(8, 32) == 64 + 512
    assert block_bytes(5, 16) == 64 + 128 + 32

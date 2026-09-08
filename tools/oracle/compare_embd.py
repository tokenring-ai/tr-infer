#!/usr/bin/env python3
"""Compare two image-embedding dumps (u32 n_tokens, nx, ny, n_embd; f32 rows): oracle vs engine."""
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(16), np.uint32)
        n, nx, ny, d = (int(x) for x in hdr)
        data = np.frombuffer(f.read(n * d * 4), np.float32).reshape(n, d)
    return n, nx, ny, d, data


def main():
    a, b = sys.argv[1], sys.argv[2]
    na, nxa, nya, da, A = load(a)
    nb, nxb, nyb, db, B = load(b)
    print(f"{a}: {na} tokens grid {nxa}x{nya} dim {da}")
    print(f"{b}: {nb} tokens grid {nxb}x{nyb} dim {db}")
    if (na, nxa, nya, da) != (nb, nxb, nyb, db):
        print("SHAPE MISMATCH")
        sys.exit(1)
    diff = A - B
    an = np.linalg.norm(A, axis=1)
    bn = np.linalg.norm(B, axis=1)
    cos = (A * B).sum(1) / (an * bn + 1e-12)
    rel = np.linalg.norm(diff, axis=1) / (an + 1e-12)
    print(f"max |diff| {np.abs(diff).max():.5f}  rms A {np.sqrt((A**2).mean()):.5f}  rms diff {np.sqrt((diff**2).mean()):.5f}")
    print(f"per-token rel L2: mean {rel.mean():.4f} max {rel.max():.4f}   cosine: min {cos.min():.5f} mean {cos.mean():.5f}")
    worst = int(np.argmax(rel))
    print(f"worst token {worst}: rel {rel[worst]:.4f}  A[:4] {A[worst,:4]}  B[:4] {B[worst,:4]}")


if __name__ == "__main__":
    main()

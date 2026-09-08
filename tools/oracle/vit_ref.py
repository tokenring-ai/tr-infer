#!/usr/bin/env python3
"""numpy f32 reference of the vision encoder (llama.cpp's qwen3vl_merger graph) for one image.

  vit_ref.py IMAGE OUT.bin [--quant] [--stretch] [--mmproj PATH]

--quant: use the pack's 8-bit weights (quantised exactly as trpack does) instead of the F16 source,
which separates weight-quantisation error from everything else when comparing with the engine.
Preprocessing mirrors llama.cpp (smart_resize, PAD_CEIL centre padding, Pillow bicubic) unless
--stretch (transformers' behaviour, the engine's default). Output: the oracle's dump format.
"""
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from trpack.blocks import GGML_F32, float_rows_bits, to_f32  # noqa: E402
from trpack.ggufio import GgufModel  # noqa: E402
from trpack.vision import VisionModel  # noqa: E402


def smart_size(w, h, align=32, min_px=8 * 1024, max_px=4096 * 1024):
    rb = lambda x: max(align, int(round(x / align)) * align)
    cb = lambda x: int(math.ceil(x / align)) * align
    fb = lambda x: max(align, int(math.floor(x / align)) * align)
    wb, hb = rb(w), rb(h)
    if hb * wb > max_px:
        beta = np.float32(math.sqrt(np.float32(h * w) / np.float32(max_px)))
        hb, wb = fb(h / beta), fb(w / beta)
    elif hb * wb < min_px:
        beta = np.float32(math.sqrt(np.float32(min_px) / np.float32(h * w)))
        hb, wb = cb(h * beta), cb(w * beta)
    return wb, hb


def preprocess(path, stretch):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    dw, dh = smart_size(w, h)
    if stretch or (dw, dh) == (w, h):
        out = im.resize((dw, dh), Image.BICUBIC) if (dw, dh) != (w, h) else im
    else:
        scale = min(np.float32(dw) / w, np.float32(dh) / h)
        nw, nh = min(int(math.ceil(w * scale)), dw), min(int(math.ceil(h * scale)), dh)
        inner = im.resize((nw, nh), Image.BICUBIC)
        out = Image.new("RGB", (dw, dh), (0, 0, 0))
        out.paste(inner, ((dw - nw) // 2, (dh - nh) // 2))
    a = (np.asarray(out, dtype=np.float32) / 255.0 - 0.5) / 0.5  # [H, W, 3]
    gw, gh = dw // 16, dh // 16
    patches, ys, xs = [], [], []
    for by in range(0, gh, 2):
        for bx in range(0, gw, 2):
            for dy in range(2):
                for dx in range(2):
                    py, px = by + dy, bx + dx
                    blk = a[py * 16 : (py + 1) * 16, px * 16 : (px + 1) * 16, :]  # [ky, kx, c]
                    patches.append(blk.transpose(2, 0, 1).reshape(-1))  # (c, ky, kx)
                    ys.append(py)
                    xs.append(px)
    return np.stack(patches), np.array(ys), np.array(xs), gw, gh


def pos_rows(table, gw, gh):
    n = int(round(math.sqrt(table.shape[0])))
    t = table.reshape(n, n, -1)

    def coord(i, n_out):
        x = i * (n - 1) / (n_out - 1)
        x0 = min(int(math.floor(x)), n - 1)
        return x0, min(x0 + 1, n - 1), x - x0

    rows = []
    for by in range(0, gh, 2):
        for bx in range(0, gw, 2):
            for dy in range(2):
                for dx in range(2):
                    y0, y1, fy = coord(by + dy, gh)
                    x0, x1, fx = coord(bx + dx, gw)
                    top = t[y0, x0] * (1 - fx) + t[y0, x1] * fx
                    bot = t[y1, x0] * (1 - fx) + t[y1, x1] * fx
                    rows.append(top * (1 - fy) + bot * fy)
    return np.stack(rows).astype(np.float32)


def layernorm(x, w, b, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def dequant8(w):
    """Quantise like trpack (8-bit per 32, f16 scale/min) and dequantise."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    K = w.shape[-1]
    q, d, m = float_rows_bits(w.reshape(-1, K), K, GGML_F32, 8)
    r = q.reshape(-1, K // 32, 32).astype(np.float32) * d.astype(np.float32)[:, :, None] + m.astype(np.float32)[:, :, None]
    return r.reshape(w.shape)


def bf16_round(w):
    u = np.ascontiguousarray(w, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) & 0xFFFF0000).view(np.float32)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    image, out = args[0], args[1]
    quant = "--quant" in flags
    bf16 = "--bf16" in flags
    stretch = "--stretch" in flags
    bf16qk = "--bf16qk" in flags  # round q/k (after rope) and the probabilities to bf16, as the AMX attention does
    mmproj = next((f.split("=", 1)[1] for f in flags if f.startswith("--mmproj=")), "/opt/llm/Qwen3.8-Flash-Next-UD-Q4_K_XL/mmproj-F16.gguf")
    g = GgufModel(Path(mmproj))
    vm = VisionModel(g.tensors, g.kv, 8)
    H, NL, NH, T = vm.hidden, vm.n_layer, vm.n_head, 8
    hd = H // NH

    def W(name):
        a = vm.tensors[name].data
        if a.ndim == 2 and name.endswith(".weight") and "position" not in name:
            if quant:
                return dequant8(a)
            if bf16:
                return bf16_round(a)
        return a

    def unpad_cols(a, per, per_pad):
        return np.concatenate([a[..., per_pad * t : per_pad * t + per] for t in range(T)], -1)

    def unpad_rows(a, per, per_pad):
        return np.concatenate([a[per_pad * t : per_pad * t + per] for t in range(T)], 0)

    patches, ys, xs, gw, gh = preprocess(image, stretch)
    N = patches.shape[0]
    x = patches @ W("v.patch_embd.weight").T + W("v.patch_embd.bias") + pos_rows(W("v.position_embd.weight"), gw, gh)
    half, quarter = hd // 2, hd // 4
    inv = 10000.0 ** (-2.0 * np.arange(quarter) / half)
    theta = np.concatenate([ys[:, None] * inv[None, :], xs[:, None] * inv[None, :]], 1).astype(np.float32)  # [N, half]
    cos, sin = np.cos(theta)[:, None, :], np.sin(theta)[:, None, :]

    def rope(v):  # [N, NH, hd]
        a, b = v[..., :half], v[..., half:]
        return np.concatenate([a * cos - b * sin, a * sin + b * cos], -1)

    ff_per = -(-vm.n_ff // T)
    for il in range(NL):
        p = f"v.blk.{il}."
        hn = layernorm(x, W(p + "ln1.weight"), W(p + "ln1.bias"), vm.eps)
        qkv = hn @ W(p + "attn_qkv.weight").T + W(p + "attn_qkv.bias")
        q, k, v = (qkv[:, i * H : (i + 1) * H].reshape(N, NH, hd) for i in range(3))
        q, k = rope(q), rope(k)
        if bf16qk:
            q, k, v = bf16_round(q / math.sqrt(hd)), bf16_round(k), bf16_round(v)
            s = np.einsum("nhd,mhd->hnm", q, k)
        else:
            s = np.einsum("nhd,mhd->hnm", q, k) / math.sqrt(hd)
        if "--logits" in flags:
            print(f"layer {il}: max |logit| {np.abs(s).max():.1f}, max row range {(s.max(-1) - s.min(-1)).max():.1f}")
        s = s - s.max(-1, keepdims=True)
        pr = np.exp(s)
        pr /= pr.sum(-1, keepdims=True)
        if bf16qk:
            pr = bf16_round(pr)
        o = np.einsum("hnm,mhd->nhd", pr, v).reshape(N, H)
        wo = unpad_cols(W(p + "attn_out.weight"), vm.dl, vm.opad)
        x = x + o @ wo.T + W(p + "attn_out.bias")
        hn = layernorm(x, W(p + "ln2.weight"), W(p + "ln2.bias"), vm.eps)
        up = unpad_rows(W(p + "ffn_up.weight"), ff_per, vm.fp)
        ub = unpad_rows(W(p + "ffn_up.bias"), ff_per, vm.fp)
        dn = unpad_cols(W(p + "ffn_down.weight"), ff_per, vm.fp)
        x = x + gelu(hn @ up.T + ub) @ dn.T + W(p + "ffn_down.bias")
    x = layernorm(x, W("v.post_ln.weight"), W("v.post_ln.bias"), vm.eps)
    m = x.reshape(N // 4, 4 * H)
    e = gelu(m @ W("mm.0.weight").T + W("mm.0.bias")) @ W("mm.2.weight").T + W("mm.2.bias")
    e = e.astype(np.float32)
    with open(out, "wb") as f:
        f.write(np.array([N // 4, gw // 2, gh // 2, e.shape[1]], np.uint32).tobytes())
        f.write(e.tobytes())
    print(f"{image}: {gw}x{gh} patches -> {N // 4} tokens, quant={quant} bf16={bf16} bf16qk={bf16qk} stretch={stretch}, rms {np.sqrt((e**2).mean()):.5f} -> {out}")


if __name__ == "__main__":
    main()

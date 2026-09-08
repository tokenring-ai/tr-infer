"""Recompute the PLE block (layer 1) over all dumped positions from res_in/ple_rows and compare res_out.
usage: check_ple.py PREFIX NPOS"""
import sys
import numpy as np
sys.path.insert(0, "python")
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as T
from trpack.ggufio import GgufModel
P = sys.argv[1]; N = int(sys.argv[2])
m = GgufModel("/opt/llm/Qwen3.8-Flash-Next-UD-Q4_K_XL")
def W(name):
    t = m.tensors[name]; return dequantize(np.asarray(t.data), T(t.ggml_type)).reshape(t.dims).astype(np.float64)
H = 2560; hc = 4; eps = 1e-6
Wk, Wv = W("blk.1.ple_key.weight"), W("blk.1.ple_value.weight")
nk, nq, nc = W("blk.1.ple_norm_key.weight"), W("blk.1.ple_norm_query.weight"), W("blk.1.ple_norm_conv.weight")
cw = W("blk.1.ple_conv1d.weight")  # [10240, 4]
ple = m.tensors["per_layer_token_embd.weight"]; raw = np.asarray(ple.data)  # [rows, 90]
sig = lambda v: 1/(1+np.exp(-v)); silu = lambda v: v/(1+np.exp(-v))
def gnorm(x, w): return np.concatenate([x[s*H:(s+1)*H]/np.sqrt((x[s*H:(s+1)*H]**2).mean()+eps)*w[s*H:(s+1)*H] for s in range(hc)])
hist = np.zeros((9, hc*H))
for p in range(N):
    rows = np.fromfile(f"{P}.ple_rows.p{p}.bin", np.float32).astype(np.int64)
    emb = np.concatenate([dequantize(np.ascontiguousarray(raw[r]), T(ple.ggml_type)).astype(np.float64) for r in rows])
    res = np.fromfile(f"{P}.ple_res_in.p{p}.bin", np.float32).astype(np.float64)
    key = gnorm(Wk @ emb, nk); value = Wv @ emb; query = gnorm(res, nq)
    s = np.array([(key[i*H:(i+1)*H] * query[i*H:(i+1)*H]).sum() for i in range(hc)]) / np.sqrt(H)
    gate = sig(np.sign(s) * np.sqrt(np.maximum(np.abs(s), 1e-6)))
    gated = np.concatenate([value * gate[i] for i in range(hc)])
    normalized = gnorm(gated, nc)
    padded = np.vstack([hist, normalized[None]])  # rows: t-9 .. t
    conv = sum(cw[:, k] * padded[9 - (3 - k) * 3] for k in range(4))
    hist = padded[1:]
    out = res + gated + silu(conv)
    ours = np.fromfile(f"{P}.ple_res_out.p{p}.bin", np.float32)
    err = np.abs(ours - out).sum() / np.abs(out).sum()
    print(f"pos {p:2d}: ple res_out rel err {err:.2e}  conv contribution {np.abs(silu(conv)).sum()/np.abs(out).sum():.3f}")

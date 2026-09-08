"""Recompute one layer (pos 0) from dumped res_in in f64 and compare each stage. usage: check_layer.py LAYER PREFIX"""
import sys
import numpy as np
sys.path.insert(0, "python")
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as T
from trpack.ggufio import GgufModel
L = int(sys.argv[1]); P = sys.argv[2]
m = GgufModel("/opt/llm/Qwen3.8-Flash-Next-UD-Q4_K_XL")
def W(name):
    t = m.tensors[name]; return dequantize(np.asarray(t.data), T(t.ggml_type)).reshape(t.dims).astype(np.float64)
def ld(n): return np.fromfile(f"{P}.{n}.bin", np.float32).astype(np.float64)
eps = 1e-6; H = 2560
silu = lambda v: v/(1+np.exp(-v)); sig = lambda v: 1/(1+np.exp(-v))
softplus = lambda v: np.where(v > 20, v, np.log1p(np.exp(np.minimum(v, 20))))
def rel(a, b): return np.abs(a-b).sum()/max(np.abs(b).sum(), 1e-9)
def hc_mix(res, which):
    p = f"blk.{L}.{which}"
    wn, wd, wu, wi = W(p+"_norm.weight"), W(p+"_down.weight"), W(p+"_up.weight"), W(p+"_inject.weight")
    xn = np.concatenate([res[s]/np.sqrt((res[s]**2).mean()+eps)*wn[s*H:(s+1)*H] for s in range(4)])
    lo = silu((wd @ xn)/4); g = sig(wu @ lo); mixed = (xn*g).reshape(4, H).mean(0); inject = wi @ xn
    return mixed, inject
res = ld("res_in").reshape(4, H)
mixed, inject = hc_mix(res, "hc_attn")
print(f"layer {L}: mixed_attn err {rel(ld('mixed_attn'), mixed):.2e}  inject err {rel(ld('inject_attn'), inject):.2e}")
recurrent = (L + 1) % 4 != 0
x = ld("mixed_attn")
if recurrent:
    p = f"blk.{L}."
    qkv = W(p+"attn_qkv.weight") @ x; z = W(p+"attn_gate.weight") @ x; beta = sig(W(p+"ssm_beta.weight") @ x)
    g = W(p+"ssm_a") * softplus(W(p+"ssm_alpha.weight") @ x + W(p+"ssm_dt.bias"))
    conv_w = W(p+"ssm_conv1d.weight"); c = silu(conv_w[:, 3] * qkv)  # zero history at pos 0
    q = c[:2048].reshape(16,128); k = c[2048:4096].reshape(16,128); v = c[4096:].reshape(48,128)
    q = q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),eps); k = k/np.maximum(np.linalg.norm(k,axis=1,keepdims=True),eps)
    fo = np.zeros((48,128)); nrm = W(p+"ssm_norm.weight")
    for h in range(48):
        kh = h % 16; S = np.outer(k[kh], v[h]*beta[h]); y = (q[kh]/np.sqrt(128)) @ S
        fo[h] = y/np.sqrt((y**2).mean()+eps)*nrm*sig(z[h*128:(h+1)*128])
    out = W(p+"ssm_out.weight") @ fo.reshape(-1)
else:
    p = f"blk.{L}."
    qg = (W(p+"attn_q.weight") @ x).reshape(24,2,256); v = (W(p+"attn_v.weight") @ x).reshape(2,256)
    qn = W(p+"attn_q_norm.weight"); o = np.zeros((24,256))
    for h in range(24):
        o[h] = v[h//12] * sig(qg[h,1])  # single key: softmax = 1
    out = W(p+"attn_output.weight") @ o.reshape(-1)
print(f"layer {L}: mixer_out err {rel(ld('mixer_out'), out):.2e}")
res2 = res + ld("mixer_out")[None] * (2*sig(ld("inject_attn")/4))[:, None]
print(f"layer {L}: res_mid err {rel(ld('res_mid'), res2.reshape(-1)):.2e}")
mixed2, inject2 = hc_mix(ld("res_mid").reshape(4, H), "hc_ffn")
print(f"layer {L}: mixed_ffn err {rel(ld('mixed_ffn'), mixed2):.2e}  inject_ffn err {rel(ld('inject_ffn'), inject2):.2e}")
x = ld("mixed_ffn"); p = f"blk.{L}."
logits = W(p+"ffn_gate_inp.weight") @ x
print(f"layer {L}: router err {rel(ld('rlog'), logits):.2e}")
pr = np.exp(logits-logits.max()); pr /= pr.sum(); top = np.argsort(-pr)[:10]; w = pr[top]/pr[top].sum()
tg, tu, td = (m.tensors[p+n] for n in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"))
def expert(t, e):
    return dequantize(np.ascontiguousarray(np.asarray(t.data)[e]), T(t.ggml_type)).reshape(t.dims[1:]).astype(np.float64)
out = np.zeros(H)
for e, we in zip(top, w):
    out += we * (expert(td, e) @ (silu(expert(tg, e) @ x) * (expert(tu, e) @ x)))
sh = W(p+"ffn_down_shexp.weight") @ (silu(W(p+"ffn_gate_shexp.weight") @ x) * (W(p+"ffn_up_shexp.weight") @ x))
ref = out + sig(W(p+"ffn_gate_inp_shexp.weight") @ x) * sh
print(f"layer {L}: ffn_out err {rel(ld('ffn_out'), ref):.2e}   (down type {td.ggml_type}, gate type {tg.ggml_type})")
res3 = ld("res_mid").reshape(4,H) + ld("ffn_out")[None] * (2*sig(ld("inject_ffn")/4))[:, None]
print(f"layer {L}: res_out err {rel(ld('res_out'), res3.reshape(-1)):.2e}")

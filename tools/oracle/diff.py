#!/usr/bin/env python3
"""Compare tr-infer --dump output against llama-eval-callback output for the same prompt.

usage: diff.py llama-eval.txt gen-dump.log
"""
import re, sys
from collections import defaultdict

def parse_llama(path):
    """name -> dict(sum=float, rows=[ [first3.., last3..] per (i2,i1) row ])"""
    out = {}
    cur = None
    for line in open(path, errors="replace"):
        m = re.match(r"\S+:\s+(.+?) = \((\w+)\)\s+(\w+)\(.*\) = \{(.*)\}", line)
        if m:
            name = m.group(1)
            cur = {"rows": [], "ne": [int(x) for x in m.group(4).split(",")]}
            # llama emits some names twice per layer (hc_mixed, hc_combine, hc_inject): suffix repeats with #2
            if name in out:
                name = name + "#2"
                if name in out:
                    cur = None
                    continue
            out[name] = cur
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.startswith("[") and s.endswith("],"):
            vals = [float(v) for v in re.findall(r"-?\d+\.\d+", s)]
            cur["rows"].append(vals)
        elif s.startswith("sum = "):
            cur["sum"] = float(s[6:])
    return out

def parse_ours(path):
    """(name) -> list over tokens of dict(sum, abs, max, first8)"""
    out = defaultdict(list)
    for line in open(path):
        m = re.match(r"tok(\d+) (\S+)\s+sum\s+(\S+) abs\s+(\S+) max\s+(\S+) n (\d+) first \[(.*)\]", line)
        if not m:
            continue
        first = [float(v) for v in m.group(7).split(",")]
        out[m.group(2)].append({"tok": int(m.group(1)), "sum": float(m.group(3)), "abs": float(m.group(4)), "max": float(m.group(5)), "n": int(m.group(6)), "first": first})
    return out

def main():
    ll = parse_llama(sys.argv[1])
    ours = parse_ours(sys.argv[2])
    n_layer = 48
    pairs = [("inp_embd", "model.input_embed"), ("hc_mixed-0", "hc_mixed-0")]
    for il in range(n_layer):
        pairs.append((f"attn_out-{il}", f"linear_attn_out-{il}" if (il + 1) % 4 else f"attn_output-{il}"))
        pairs.append((f"inject_attn-{il}", f"hc_inject-{il}"))
        pairs.append((f"hc_combine-{il}", f"hc_combine-{il}"))
        pairs.append((f"hc_mixed_ffn-{il}", f"hc_mixed-{il}#2"))
        pairs.append((f"inject_ffn-{il}", f"hc_inject-{il}#2"))
        pairs.append((f"ffn_out-{il}", f"ffn_out-{il}"))
        pairs.append((f"l_last-{il}", f"l_last-{il}"))
    pairs += [("result_norm", "result_norm"), ("result_output", "result_output")]
    print(f"{'ours':16s} {'llama':20s} {'sum ours':>14s} {'sum llama':>14s} {'rel':>8s}   first3 tok0 ours | llama")
    worst = 0.0
    for a, b in pairs:
        if a not in ours or b not in ll:
            print(f"{a:16s} {b:20s}   (missing: ours={a in ours} llama={b in ll})")
            continue
        s_ours = sum(t["sum"] for t in ours[a])
        s_ll = ll[b].get("sum", float("nan"))
        rel = abs(s_ours - s_ll) / max(abs(s_ll), 1e-3)
        worst = max(worst, rel) if a != "result_output" else worst
        f_ours = ours[a][0]["first"][:3]
        rows = ll[b]["rows"]
        f_ll = rows[0][:3] if rows else []
        print(f"{a:16s} {b:20s} {s_ours:14.4f} {s_ll:14.4f} {rel:8.4f}   {[round(x,4) for x in f_ours]} | {f_ll}")
    print(f"worst relative sum diff (excluding logits): {worst:.4f}")

if __name__ == "__main__" and not (len(sys.argv) > 3 and sys.argv[3] == "--per-token"):
    main()


def per_token(llama_path, ours_path, pairs):
    """Print first-3 values per prompt token for selected tensors (ours | llama), with rel err."""
    ll = parse_llama(llama_path)
    ours = parse_ours(ours_path)
    for a, b in pairs:
        if a not in ours or b not in ll:
            print(f"{a}: missing (ours={a in ours}, llama={b in ll})"); continue
        rows = ll[b]["rows"]
        ne = ll[b]["ne"]
        # for [x, hc, T] tensors rows are (tok, stream) pairs -> take stream 0 of each token
        stride = ne[1] if len(ne) > 2 and ne[2] > 1 else 1
        print(f"== {a} vs {b} ne={ne}")
        for t, o in enumerate(ours[a]):
            r = t * stride
            if r >= len(rows): break
            lf = rows[r][:3]; of = o["first"][:3]
            rel = max(abs(x - y) for x, y in zip(of, lf)) / max(max(abs(v) for v in lf), 1e-4)
            print(f"  tok{t}: ours {[round(v,4) for v in of]}  llama {lf}  maxrel {rel:.2f}")

if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[3] == "--per-token":
    per_token(sys.argv[1], sys.argv[2], [
        ("conv_silu-0", "conv_output_silu-0"), ("gdn_y-0", "attn_output-0"), ("final_output-0", "final_output-0"),
        ("attn_out-0", "linear_attn_out-0"), ("hc_combine-0", "hc_combine-0"), ("l_last-0", "l_last-0"),
        ("l_last-1", "l_last-1"), ("attn_out-3", "attn_output-3"), ("l_last-3", "l_last-3"), ("l_last-7", "l_last-7"),
        ("ple_embd-1", "ple_embd"), ("ple_gate-1", "ple_gate-1"), ("ple_gated_value-1", "ple_gated_value-1"), ("ple_conv_out-1", "ple_conv_out-1"),
        ("Qcur_full-3", "Qcur_full-3"), ("Kcur-3", "Kcur-3"), ("Vcur-3", "Vcur-3"), ("attn_pregate-3", "attn_pregate-3"), ("gate_sigmoid-3", "gate_sigmoid-3"),
    ])

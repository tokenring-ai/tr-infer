"""Sharding plan: which slice of every GGUF tensor lands on which tile, and in which codec.

Tile-local conventions (n_tiles = 8, hidden 2560, hc 4 streams):
  GDN: llama.cpp broadcasts the n_k key heads to the n_v value heads periodically (ggml_repeat):
  value head H uses key head H % n_k. Tile t owns k-heads {kh*t .. kh*t+kh-1} and the value heads
  that use them: {kh*t + i + n_k*j : i < kh, j < n_v/n_k} in (j, i) order. Its qkv rows are
  [q heads | k heads | v heads] in that local order (1280 rows).
  Attention tile t owns q heads 3t..3t+2 (rows 1536t..+1536 of attn_q, q|gate interleaved per head);
  tiles 0-3 hold kv head 0, tiles 4-7 kv head 1.
  QSA indexer (plan v4): q_proj rows 64t..+64 and k_proj rows 16t..+16 per tile (8-bit from BF16);
  the runtime all-gathers the 640 outputs, so every tile scores blocks with the full indexer.
  Hyper-connection low-rank: hc_down rows 40t..+40; hc_up rows {s*2560 + 320t..+320, s=0..3}.
  Experts: gate/up rows 80t..+80 (TP on n_ff); down cols 80t..+80 (partial sums), KB=16.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .blocks import BITS, GGML_Q4_K, GGML_Q5_K, GGML_Q5_1, GGML_Q8_0, GGML_IQ4_NL


@dataclass
class Spec:
    name: str
    tile: int | None  # None = shared.bin (replicated)
    kind: str  # "tq" | "f32" | "iq4nl_rows"
    shard: dict  # {"kind": "rows", "ranges": [[start, len], ...]} | {"kind": "cols", "start", "len"} | {"kind": "replicate"} | {"kind":"ple_rows","start","len"}
    bits: int = 0
    kb: int = 0
    count: int = 1  # leading (expert) dimension packed as `count` items of `stride` bytes
    layer: int | None = None
    # filled by layout
    rows: int = 0
    k: int = 0
    stride: int = 0
    nbytes: int = 0
    file: str = ""
    offset: int = 0
    shape: list = field(default_factory=list)  # f32 tensors: shard shape
    row_bytes: int = 0  # iq4nl_rows

    def to_json(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, 0, "", []) or k in ("tile", "offset", "nbytes")}


def rows(start, ln):
    return {"kind": "rows", "ranges": [[start, ln]]}


def rows_ranges(rs):
    return {"kind": "rows", "ranges": [list(r) for r in rs]}


def cols(start, ln):
    return {"kind": "cols", "start": start, "len": ln}


REPL = {"kind": "replicate"}


class Planner:
    def __init__(self, cfg: dict, n_tiles: int, tensor_types: dict[str, int], down_q8_to_q5: bool = False, bits_override: dict[str, int] | None = None):
        a = cfg["architecture"]
        g = lambda k, d=None: cfg.get(f"{a}.{k}", d)
        self.n_tiles = n_tiles
        self.hidden = g("embedding_length")
        self.n_layer = g("block_count")
        self.n_head = g("attention.head_count")
        self.n_head_kv = g("attention.head_count_kv")
        self.head_dim = g("attention.key_length")
        self.n_expert = g("expert_count")
        self.n_ff = g("expert_feed_forward_length")
        self.n_ff_shexp = g("expert_shared_feed_forward_length")
        self.hc = g("hyper_connection.count")
        self.hc_lr = g("hyper_connection.low_rank")
        self.d_state = g("ssm.state_size")
        self.n_k_heads = g("ssm.group_count")
        self.n_v_heads = g("ssm.time_step_rank")
        self.full_interval = g("full_attention_interval", 4)
        self.ple_layers = list(g("ple.layers", []) or [])
        self.idx_heads = g("attention.indexer.head_count", 0)
        self.idx_dim = g("attention.indexer.key_length", 0)
        self.types = tensor_types
        self.down_q8_to_q5 = down_q8_to_q5
        self.bits_override = bits_override or {}  # tensor name -> bits (MTP experts from BF16)
        T = n_tiles
        assert self.n_ff % (16 * T) == 0, "n_ff must split into 16-row strips per tile"
        assert self.n_k_heads % T == 0 and self.n_v_heads % T == 0 and self.n_head % T == 0
        assert self.hc_lr % T == 0 and self.hidden % T == 0
        assert self.n_head_kv in (1, 2) and T % self.n_head_kv == 0

    def is_recurrent(self, il: int) -> bool:
        return (il + 1) % self.full_interval != 0

    # ---- helpers -------------------------------------------------------
    def tq(self, name, tile, shard, layer, count=1, kb=32, bits=None):
        ty = self.types[name]
        if bits is None:
            bits = self.bits_override.get(name, BITS[ty])
        return Spec(name, tile, "tq", shard, bits=bits, kb=kb, count=count, layer=layer)

    def f32(self, name, tile, shard, layer):
        return Spec(name, tile, "f32", shard, layer=layer)

    def expert_specs(self, il: int) -> list[Spec]:
        T = self.n_tiles
        per = self.n_ff // T
        out = []
        for t in range(T):
            out.append(self.tq(f"blk.{il}.ffn_gate_exps.weight", t, rows(per * t, per), il, count=self.n_expert))
            out.append(self.tq(f"blk.{il}.ffn_up_exps.weight", t, rows(per * t, per), il, count=self.n_expert))
            dn = f"blk.{il}.ffn_down_exps.weight"
            bits = self.bits_override.get(dn, BITS[self.types[dn]])
            if self.down_q8_to_q5 and bits == 8:
                bits = 5  # lossy requant knob (applied in the writer)
            out.append(self.tq(dn, t, cols(per * t, per), il, count=self.n_expert, kb=16, bits=bits))
        return out

    def layer_specs(self, il: int, recurrent: bool | None = None) -> list[Spec]:
        """`recurrent` overrides the interval rule (the MTP layer is full attention at index 48)."""
        T = self.n_tiles
        H, hc, lr = self.hidden, self.hc, self.hc_lr
        p = f"blk.{il}."
        out: list[Spec] = []
        if recurrent is None:
            recurrent = self.is_recurrent(il)
        # hyper-connections (2 per layer)
        for which in ("hc_attn", "hc_ffn"):
            out.append(self.f32(p + f"{which}_norm.weight", None, REPL, il))
            out.append(self.f32(p + f"{which}_inject.weight", None, REPL, il))
            for t in range(T):
                # K-split: every tile computes all lr rows over its slice of the hc*H input (reduce on LO)
                out.append(self.tq(p + f"{which}_down.weight", t, cols(hc * H // T * t, hc * H // T), il))
                out.append(self.tq(p + f"{which}_up.weight", t, rows_ranges([(s * H + H // T * t, H // T) for s in range(hc)]), il))
        if recurrent:
            kd, vd = self.d_state, self.d_state
            kh, vh = self.n_k_heads // T, self.n_v_heads // T
            key_dim, val_dim = self.n_k_heads * kd, self.n_v_heads * vd
            rep = self.n_v_heads // self.n_k_heads
            for t in range(T):
                # value heads owned by this tile: for j in 0..rep, the kh heads kh*t + n_k*j .. +kh (contiguous)
                v_ranges = [(2 * key_dim + (kh * t + self.n_k_heads * j) * vd, kh * vd) for j in range(rep)]
                qkv_rows = [(kh * kd * t, kh * kd), (key_dim + kh * kd * t, kh * kd)] + v_ranges
                head_ranges = [((kh * t + self.n_k_heads * j) * vd, kh * vd) for j in range(rep)]
                out.append(self.tq(p + "attn_qkv.weight", t, rows_ranges(qkv_rows), il))
                out.append(self.tq(p + "attn_gate.weight", t, rows_ranges(head_ranges), il))
                out.append(self.tq(p + "ssm_out.weight", t, {"kind": "cols_ranges", "ranges": [list(r) for r in head_ranges]}, il))
                out.append(self.f32(p + "ssm_beta.weight", t, rows_ranges([(kh * t + self.n_k_heads * j, kh) for j in range(rep)]), il))
                out.append(self.f32(p + "ssm_alpha.weight", t, rows_ranges([(kh * t + self.n_k_heads * j, kh) for j in range(rep)]), il))
                # conv weight numpy [channels, K=4]: channel slice in the tile local q|k|v order
                out.append(self.f32(p + "ssm_conv1d.weight", t, rows_ranges(qkv_rows), il))  # numpy [C, K=4]: channels are rows
            for n in ("ssm_dt.bias", "ssm_a", "ssm_norm.weight"):
                out.append(self.f32(p + n, None, REPL, il))
        else:
            hd, nh, nkv = self.head_dim, self.n_head, self.n_head_kv
            qh = nh // T
            for t in range(T):
                out.append(self.tq(p + "attn_q.weight", t, rows(qh * hd * 2 * t, qh * hd * 2), il))
                kv = t * nkv // T  # which kv head this tile's q heads use
                out.append(self.tq(p + "attn_k.weight", t, rows(kv * hd, hd), il))
                out.append(self.tq(p + "attn_v.weight", t, rows(kv * hd, hd), il))
                out.append(self.tq(p + "attn_output.weight", t, cols(qh * hd * t, qh * hd), il))
            out.append(self.f32(p + "attn_q_norm.weight", None, REPL, il))
            out.append(self.f32(p + "attn_k_norm.weight", None, REPL, il))
            if p + "indexer.q_proj.weight" in self.types:
                iq, ik = self.idx_dim * self.idx_heads, self.idx_dim
                assert iq % (16 * T) == 0 and ik % (16 * T) == 0
                for t in range(T):
                    out.append(self.tq(p + "indexer.q_proj.weight", t, rows(iq // T * t, iq // T), il))
                    out.append(self.tq(p + "indexer.k_proj.weight", t, rows(ik // T * t, ik // T), il))
                out.append(self.f32(p + "indexer.q_norm.weight", None, REPL, il))
                out.append(self.f32(p + "indexer.k_norm.weight", None, REPL, il))
        # router (K-split), shared expert (TP like an expert), gates
        for t in range(T):
            out.append(self.f32(p + "ffn_gate_inp.weight", t, cols(H // T * t, H // T), il))  # router is F32 in the GGUF
            per = self.n_ff_shexp // T
            out.append(self.tq(p + "ffn_gate_shexp.weight", t, rows(per * t, per), il))
            out.append(self.tq(p + "ffn_up_shexp.weight", t, rows(per * t, per), il))
            out.append(self.tq(p + "ffn_down_shexp.weight", t, cols(per * t, per), il, kb=16))
        out.append(self.f32(p + "ffn_gate_inp_shexp.weight", None, REPL, il))
        if il in self.ple_layers:
            for t in range(T):
                out.append(self.tq(p + "ple_key.weight", t, rows_ranges([(s * H + H // T * t, H // T) for s in range(hc)]), il))
                out.append(self.tq(p + "ple_value.weight", t, rows(H // T * t, H // T), il))
            for n in ("ple_norm_key.weight", "ple_norm_query.weight", "ple_norm_conv.weight", "ple_conv1d.weight"):
                out.append(self.f32(p + n, None, REPL, il))
        out.extend(self.expert_specs(il))
        return out

    def head_specs(self) -> list[Spec]:
        T = self.n_tiles
        H, hc, lr = self.hidden, self.hc, self.hc_lr
        out = [self.f32("output_hc_norm.weight", None, REPL, None)]
        n_vocab = None
        for t in range(T):
            out.append(self.tq("output_hc_down.weight", t, cols(hc * H // T * t, hc * H // T), None))
            out.append(self.tq("output_hc_up.weight", t, rows_ranges([(s * H + H // T * t, H // T) for s in range(hc)]), None))
        return out

    def mtp_specs(self) -> list[Spec]:
        """MTP-only tensors: the draft head's output mixer (as the head's), the two input
        projections (row-split, 8-bit) and the two pre-fc Gemma norms (replicated)."""
        T = self.n_tiles
        H, hc = self.hidden, self.hc
        out = [self.f32("mtp_hc_norm.weight", None, REPL, None), self.f32("mtp_enorm.weight", None, REPL, None), self.f32("mtp_hnorm.weight", None, REPL, None)]
        for t in range(T):
            out.append(self.tq("mtp_hc_down.weight", t, cols(hc * H // T * t, hc * H // T), None))
            out.append(self.tq("mtp_hc_up.weight", t, rows_ranges([(s * H + H // T * t, H // T) for s in range(hc)]), None))
            out.append(self.tq("mtp_fc_embedding.weight", t, rows(H // T * t, H // T), None))
            out.append(self.tq("mtp_fc_hidden.weight", t, rows(H // T * t, H // T), None))
        return out

    def vocab_specs(self, n_vocab: int) -> list[Spec]:
        T = self.n_tiles
        per = -(-n_vocab // T)
        per = (per + 15) // 16 * 16
        out = []
        for t in range(T):
            start = per * t
            ln = max(0, min(per, n_vocab - start))
            out.append(self.tq("output.weight", t, rows(start, ln), None))
            out.append(self.tq("token_embd.weight", t, rows(start, ln), None))
        return out

    def ple_specs(self, n_rows: int) -> list[Spec]:
        T = self.n_tiles
        per = -(-n_rows // T)
        out = []
        for t in range(T):
            start = per * t
            ln = max(0, min(per, n_rows - start))
            out.append(Spec("per_layer_token_embd.weight", t, "iq4nl_rows", {"kind": "ple_rows", "start": start, "len": ln}, layer=None))
        return out

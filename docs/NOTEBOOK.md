# tr-infer-rs notebook

Host `mdmax`: 2× Xeon Max 9470, 8 HBM tiles × 16 GiB (SNC-4 × 2), 104 P-cores, no DDR.
Model: Unsloth `Qwen3.8-Flash-Next-UD-Q4_K_XL` (arch `qwen4exp`), 48 layers, 512 experts top-10, PLE 26.8 GiB.

## 2026-09-04 — P0..P3 first light

| item | result |
|---|---|
| Rust toolchain | rustup stable 1.98.1; AMX via `asm!` (`tdpbf16ps` tile test passes), AVX-512 via `core::arch` |
| Barrier (104 workers, 8×13) | **1.28 µs** global, 0.46 µs tile-local (`tr-infer barrier-bench`) |
| NUMA placement | 100 % local on every tile (`get_mempolicy` samples + `numa_maps`), THP advised |
| Per-tile read bandwidth | 127–179 GB/s with 13 cores (1.26 TB/s aggregate) |
| Loader | `O_DIRECT` 2 MiB chunks from the tile's own cores; 113.85 GiB in **8.8 s** warm / 17.6 s cold |
| Idle | 104 parked workers cost ~4 % of one core (futex sleep) |
| Pack v2 | `python/trpack`: lossless Q4_K/Q5_K/Q5_1/Q8_0 → TQ{4,5,8}_{16,32}; 113.7 GiB in 476 s (6 procs) |
| Per tile | 14.24 GiB weights (experts ~10.0, PLE 3.35, dense ~0.9) + ~0.2 GiB state → MemFree ≈ 0.1 GiB, VmSwap 0.4 GiB (tight) |
| TQ GEMV (1 core) | TQ4_32 8.5 GB/s, TQ5_32 8.4, TQ8_32 10.0, TQ5_16 7.6 (`kbench`) |
| First generate | `The capital of France is` → ` Paris.` (greedy), **25.2 tok/s** decode, prompt 12.3 tok/s (token-at-a-time) |

Greedy 12 tokens: `Paris.\nThe capital of Paris,\n\nThe Paris` — ids `[11751, 13, 198, 760, 6511, 314, 11751, 11, 198, 198, 760, 11751]`.
llama.cpp CPU reference (with PLE): `Paris / Berlin / Rome / Madrid` — per-layer comparison pending (`tools/oracle/diff.py`).

Baselines on this file (exclusive CPU): ik_llama pp64 87 / tg64 8.86; llama.cpp 24.9 / 4.22; old Python tr-infer 6.92 / 0.35.

Pack: `/opt/llm/tr-infer/flashnext-v2` (hash `e0060c5746103727`). Test subset: `/opt/llm/tr-infer/test-l03` (layers 0,3; 16 experts; no PLE).

Commands:
```bash
export PATH=$HOME/.cargo/bin:$PATH
cargo test --workspace --release
./target/release/tr-infer topology | barrier-bench | numa-smoke --mib 1024
./target/release/tr-infer generate --pack /opt/llm/tr-infer/flashnext-v2 -p "The capital of France is" -n 24 [--dump]
./target/release/tr-infer bench --pack /opt/llm/tr-infer/flashnext-v2 --pp 64 --tg 64
python/.venv/bin/trpack pack --out /opt/llm/tr-infer/flashnext-v2   # ~8 min
```

## 2026-09-04 — correctness: the GDN key-head broadcast

Symptom: first predicted token right (" Paris", " dog"), output degrades with context length (26-token prompt ending "The capital of Spain is" → " Paris").
Method: per-block validation against f64 Python references from the GGUF (attention over 26 positions, MoE, PLE over 9-row history, head, deep layers 20/23/45 stage by stage: all ≤ 1e-4), then llama-eval-callback on CPU **and** CUDA (they agree to 3 digits, so llama is a trustworthy oracle at layer 0).
Root cause: llama.cpp broadcasts the 16 key heads to the 48 value heads with `ggml_repeat` = periodic (value head H ↔ key head H mod 16). The old Python port (and my first Rust version) used HF's `repeat_interleave` rule (H div 3). Heads 0 and 47 coincide under both rules, which is exactly what the first spot checks looked at.
Fix: plan_version 2 — tile t owns key heads {2t,2t+1} and value heads {2t+i+16j}; z rows, ssm_out columns (`cols_ranges`), β/α rows follow; runtime uses local k-head j mod 2. Repack required.
Also added: two-level ("pair") int8 activations (default on; weight traffic unchanged), per-phase profiler (`TR_PROFILE=1`), `--dump`, `TR_DUMP_VEC`/`TR_DUMP_LAYER` dumps, `tools/oracle/{diff,check_layer,check_ple}.py`.

**Result after the fix (2026-09-04):** greedy `The capital of France is` → ` Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The` — token-identical to llama.cpp CPU for all 24 tokens; 26-token prompt → ` Madrid. The capital of Portugal is` (same as llama.cpp CUDA). Per-layer `l_last` first values track llama within a few % through layer 39. Decode 25.6 tok/s untuned; prompt 12.3 tok/s (token-at-a-time).

## 2026-09-04 — P4 decode tuning

| change | tg64 | notes |
|---|---:|---|
| first light (redundant hc prep, main-thread scratch) | 32.4 tok/s (30.8 ms) | barriers A/D ~100 µs each: waiting on cores whose private scratch lived on tile 0 |
| tile-local scratch (allocated by the worker), vectorised pair quant, hc prep split across the 13 cores | **59.2 tok/s (16.9 ms)** | barriers 3–5 µs; per layer: gdn 114 µs, moe 72, hc_a 32×2, hc_b 18×2, attn 81 |
| plan v3: hc `*_down` K-split so all 13 cores stream it; LO becomes a reduce | **64.5 tok/s (15.5 ms)** | gdn.proj 56 µs, moe.gateup 44, moe.down 22, hc_a 17, hc_b 20 |

Bandwidth floor per tile ≈ 500 MB dense + 200 MB experts per token at ~130 GB/s ≈ 6 ms; lm_head 80 MB/tile ≈ 0.6 ms.
| GEMV software prefetch (3 blocks ahead; single-core TQ4 8.5→14.9 GB/s), PLE rows gathered once per tile | **67–69 tok/s (14.5 ms)** | gdn.proj 51 µs ≈ 116 GB/s per tile (near HBM), moe.gateup 38, hc_a 15, hc_b 19 |
| `--ple-mmap` (PLE table file-backed) | 62 tok/s, **RSS 88 GiB** (11 GiB/tile) | fallback when HBM is contended |

## 2026-09-04 — P5 batched prefill

`exec_batch.rs`: M tokens per step (default 64) with the same sharding; tile-shared row-major buffers, row-grouped GEMMs (`gemm_tq`, 8 rows per weight pass), experts grouped by token entries, recurrence/conv/PLE/attention sequential in time inside the batch, single-level int8 activations for prefill. Decode path unchanged (pair int8).

| | pp64 | tg | notes |
|---|---:|---:|---|
| token-at-a-time (before) | 60 tok/s | 67 | |
| batch 64 (VNNI, no AMX yet) | **312–423 tok/s** | 64–67 | 26-token prompt gives the same greedy text as llama.cpp CUDA |
| batch 64, pp512 | **437–448 tok/s** | 5-token prompt via the batch path: same 24 greedy tokens as llama.cpp |

Baselines: ik_llama pp64 87 / tg64 8.86; llama.cpp CUDA RTX PRO 6000 pp512 1676 / tg64 92. AMX-BF16 GEMM deferred (VNNI already 5× the gate).

## 2026-09-04 — P6 long context: QSA indexer, lazy KV, chat

QSA (Qwen sparse attention) as llama.cpp `build_qsa_top_k` / `set_input_qsa`: on every full-attention layer an
indexer (q_proj 2560→4×128, k_proj 2560→128, BF16 in the GGUF, packed 8-bit; plan v4, pack `flashnext-v3`)
scores blocks of 4 tokens: `score[b] = Σ_h relu(q_h · rope(rmsnorm(mean raw k of block b)))`. A query at
position p sees `min(p+1, 2048+3)` tokens: the incomplete tail block always, then whole blocks by score
(ties → lowest block; a partial block takes its first tokens). Below 2052 tokens attention is dense, so
everything measured before is unchanged. Runtime: indexer rows sharded 80/tile inside the attention
projection task list, all-gathered via a new mailbox slot (one extra global barrier "I" per attention
layer), block keys pooled/normed/roped once per completed block (state `ik_pooled [ctx/4][128]` per layer
per tile, 128 B/token), scores split across the tile's cores, selection per head core, attention over
(start,len) ranges (`attend_ranges`, no gather copy). Batched prefill does the same per token
(`exec_batch::attn_batch`).

Oracle (`TR_QSA_DEBUG=1 … --dump` vs `llama-eval-callback`, layer 3): indexer q `[0.0204 0.4368 1.2449]`
vs `[0.0191 0.4416 1.249]`, raw k `[-0.7225 0.6153 -1.132]` vs `[-0.7225 0.6073 -1.1303]`, pooled block-0
key `[-0.2206 0.4349 -0.5343]` vs `[-0.2229 0.4243 -0.5348]`, block-0 score for token 3: 137.24 vs 137.30,
token 4: 128.23 vs 128.81 (int8 indexer + int8 activations; llama runs it in bf16). Greedy 24 tokens still
identical to llama.cpp.

KV cache: `--ctx N` now only reserves address space (`Arena::alloc_slice_uninit`, MAP_NORESERVE); pages fault
in from the tile's own cores as the context grows (24 KiB/token/tile f32 K+V, +128 B indexer keys).
CLI: `--chat` (ChatML user turn + `<|im_start|>assistant\n<think>\n`), `--no-think`, `--prompt-file`,
streamed output, stop at `<|im_end|>` (248046).

**Long-context results (exclusive box, `flashnext-v3`):**

| | tr-infer-rs | llama.cpp CPU (same prompt) |
|---|---:|---:|
| prefill 3153 tokens | 8–15 s (214–355 tok/s; run-to-run swap noise) | 247 s (12.8 tok/s) |
| decode at 3K context | **57 tok/s** (17.4 ms) | 4.5 tok/s |
| `bench --pp 3072 --tg 32` | pp 372–390 tok/s, tg 57.5 | |
| short context (unchanged) | pp512 445, tg64 66–69 | |

Parity above the sparse threshold (3133-token prompt, greedy 24): llama.cpp
`quant in Rust, \`llama-quant.h:141\`), rows sharded by vocab.\n\n**` vs ours
`quant in Rust, \`ggml-quants.c:1046\`), rows sharded by vocab.\n\n\n` — same phrase, a different
hallucinated file reference (int8 indexer + int8 activations over 3K tokens; llama's own top-k breaks
score ties arbitrarily). With the question suffix (3153 tokens) both emit end-of-text as the first token.
Tokenizer identical to llama.cpp on the 3153-token prompt (`tr-infer tokenize` vs `llama-tokenize`).
Dense control: 1889-token prompt (below the threshold in both engines) → all 24 greedy tokens identical
(`NOISE.md, BENCH.md, PLAN.md (this file).\n```\n\n## Pack format v2 (contract between`), so the 3K
difference is confined to the sparse selection.

Decode attention at 3K: one head per core cost 290 µs/layer (52 tok/s); split into (head, token-window)
tasks over all 13 cores with online-softmax merge (`attend_partial` / `merge_partials`) and software
prefetch of the scattered 4-token blocks → 278 µs/layer, 57 tok/s. Short-context attention 72 → 93 µs
(indexer projections + the extra global barrier). Prefill profile at 3K: attention 3.1 ms per
64-token batch per layer (17 %), MoE barrier wait 1.2 ms (24 %, load imbalance across tiles) —
the next prefill target, not QSA-related.

Follow-ups: bf16 KV cache (halves attention traffic and doubles the context that fits), query-blocked
prefill attention (stream KV once per 8 queries), MoE batch load balance, AMX-BF16 GEMM.

## 2026-09-05 — 16-bit KV cache

`KvElem` trait in `tr-kernels/attn.rs` (f32, `F16`, `Bf16`): rows load with `vcvtph2ps` / shift, store
with `vcvtps2ph` (RNE) / RNE bit trick; queries and softmax stay f32. `--kv f16|bf16|f32`, default **f16**
(what llama.cpp keeps in its cache). Per token per tile: 12 KiB (was 24) + 128 B indexer keys →
262144-token context = 3 GiB/tile.

| | f32 KV | f16 KV | bf16 KV |
|---|---:|---:|---:|
| tg64 short (A/B back to back) | 67.9–69.7 | 67.5–68.8 | |
| tg32 at 3K context | 57.5 | **59–61** | 62 |
| pp3072 | 372–390 | 375–384 | 376 |
| `The capital of France is` greedy 24 | identical | identical | |
| 26-token prompt | ` Madrid. The capital of Portugal is` | same | |
| 1889-token prompt vs llama.cpp | 24/24 tokens | diverges at token 6 (`ARCHITECTURE.md` vs `PLAN.md`, both hallucinated) | diverges at token 6 |

Per-layer oracle diff (`--dump`, 5 tokens) with f16: `attn_out-{3,7,11,23,31}` sums within 0.1 % of the f32
run and at the same distance from llama.cpp, so the long-prompt flip is a near-tie under 16-bit rounding,
not a kernel error. bf16 is kept as an option; f16 is the default because it matches the oracle's cache.

## 2026-09-05 — query-blocked prefill attention

`attend_block` (`tr-kernels/attn.rs`): a task is (head, block of 8 queries). It builds the union of the
8 queries' visible rows (dense causal prefix or the QSA selection) with an 8-bit membership mask per row,
streams every K row once computing 8 dots (8 zmm accumulators), does a masked softmax per query with a
vectorised `exp512` (`elem::exp_sub_inplace`, also used by the decode kernel now), then streams every V
row once into 8 L1-resident output accumulators. Sums stay unnormalised until the end so results are
bit-identical to the per-query kernel (the 1889-token llama.cpp match is preserved with `--kv f32`;
5- and 26-token greedy outputs unchanged).

| 3K prefill (`--pp 3072`, f16 KV) | per-query kernel | query-blocked | + vector exp |
|---|---:|---:|---:|
| `b.attn` per 64-token batch per layer (worker 0) | 3.10 ms | 1.85 ms | **1.67 ms** (attend 0.87) |
| pp3072 | 372–390 tok/s | 388–413 | **401–411** |
| pp512 (short) | 373–445 | | 370–440 (unchanged) |

Remaining attention time is the 8-way FMA / L1 store loops (prefetch distance 6 vs 16 makes no
difference); the 3 heads sharing one kv head still stream K/V separately (3× the minimum traffic).
Attention is now ~11 % of 3K prefill; the MoE barrier-G imbalance (~24 %) is the next prefill target.

## 2026-09-05 — prefill MoE: the "imbalance" that wasn't, and what was left

Per-worker phase timers (`TR_PROFILE=1`, `pw_report`): in clean runs barrier G waits **44 µs** (1 %),
cores within a tile agree to ~15 µs and tiles spread 870–1015 µs per MoE layer-batch. The 1.2 ms
(24 %) wait seen earlier came from runs with swap activity after llama.cpp's page-cache churn
(`VmSwap > 0` in those lines), not from the work split. MoE sub-phases at batch 64 (worker 0, µs):
gate/up 353 + 137 wait, down 267, combine 127, route 59, act 36, group 19 — gate/up and down are at
the expert-weight bandwidth floor (~69 MB + ~30 MB per tile per layer-batch at ~130 GB/s).

Changes: expert groups gathered into a contiguous per-core buffer so gate/up is one GEMM per expert
(no effect — bandwidth-bound, kept for the cleaner kernel path); largest-first group order so the
round-robin task split is LPT (wait 137 → 70 µs); the per-token combine folded into the down phase
(each core sums its own column range, streams it to the PART rows; one barrier fewer; 127 → ~90 µs);
router selects on logits and exponentiates only the 10 winners (route 59 → 42 µs; same weights to
1e-6, test added — the ulp change flips the same near-tie token the f16 cache flips). Default prefill
batch raised to 256 (`--batch`), which amortises expert reads: MoE 15.9 → 13.1 µs per token-layer.

| | before | after (batch 64) | after (batch 256, default) |
|---|---:|---:|---:|
| MoE per 64-token layer-batch | 1017–1027 µs | 955 µs | 13.1 µs/token-layer |
| pp3072 | 385–400 tok/s | 395 | **435–464** |
| pp512 | 380–443 | | **445–511** |
| tg64 | 66–69 | | 68.4 (unchanged) |

Greedy 5- and 26-token outputs unchanged; the 3133-token continuation differs from the previous run
in one hallucinated number (router rounding). Prefill is now ~20 % MoE, ~20 % GDN, ~13 % attention,
~15 % hyper-connections, the rest exchanges/barriers; the next lever is the GDN chunk kernel.

## 2026-09-05 — GDN prefill: the "chunk kernel" that turned out to be a cache-layout bug

Goal: the GDN block was ~25 % of a 256-token layer-batch (3.5 ms per GDN layer). Sub-phase laps
(`b.gdn.*`, added) said the delta rule was 785 µs + a 444 µs wait, conv 303 µs, the two int8 GEMMs
1.6 ms.

**Why chunking (FLA-style `chunk_gated_delta_rule`) is not the lever here.** Per token and head the
recurrence touches the [128][128] state twice (2·dk·dv MACs); the chunked form needs three
state-sized GEMMs per chunk (w·S, q·S, kᵀ·D → 3·dk·dv MACs per token) plus the C² intra-chunk
terms. In f32 AVX-512 a register-blocked GEMM runs at ~2× the rate of the store-bound recurrence,
so chunking is a wash (≈2050 vs 2048 cycles per token-head); it only pays with BF16 dot products
(≈2×) or AMX tiles (≈5×), at the cost of bf16 state operands. Not worth it once the recurrent kernel
runs at its bound — which it did not:

- **State layout.** The state was [dk][dv] row-major and tasks took 16-column strips, so a task's
  8 KiB working set sat at a 512-byte stride: (i·8 + c) mod 64 hits only 8 of the L1's 64 sets
  (96 lines of capacity for 128 lines) and thrashed to L2. Cores with 4 tasks (32 KiB, 384 lines
  for 512) lost more than cores with 3 — that was the 444 µs "wait". Now chunk-major
  `[head][dv/16][dk][16]`: a task's state is contiguous.
- **One accumulator chain.** kv and y were each a single FMA chain over 128 rows (4-cycle latency
  → ~1024 cycles/token). `gdn_step16` uses 8 chains.
- **Fused sweep.** `gdn_chunk_seq` runs a whole batch for one chunk: the rank-1 update of token t
  and the decay + kv pass of token t+1 share one load/store per state row.

Single core, 4 tasks × 256 tokens (`gdnbench`): 928 → 382 (layout) → 258 (8 chains) →
**145 ns per token-task** (fused), against a ~110 ns port bound. Plus: conv vectorised over 16
channels with the history in registers (`conv_silu_seq16`, 303 → 64 µs), q/k l2-norm and gates
computed once per token instead of once per task (8×), the gated-norm sigmoid vectorised.

| per 256-token GDN layer | before | after |
|---|---:|---:|
| b.gdn.delta (+wait) | 785 (+444) µs | 227 (+14) |
| b.gdn.conv | 303 | 64 |
| b.gdn total | 3520 | 2190 |
| pp512 | 445–511 tok/s | **449–577** |
| pp3072 | 435–464 | **488–516** |
| tg64 / tg32@3K | 68.4 / 62.7 | 68.3 / 62.9 |

Kernel tests compare both new kernels against the f64 reference over 20 steps (state and output);
5-token and 26-token greedy outputs unchanged; the 1889-token near-tie token flips (as every
ulp-level change so far). Decode also uses the chunk-major state (`gdn_step16`), no measurable change.

What is left in GDN is the two int8 GEMMs (qkv+z 1.05 ms, out 0.67 ms = 79 % of the block):
`gemm_tq` at m=8 reaches 132 GMAC/s per core on streamed weights (82 % of VNNI peak) but only
~98 GMAC/s on qkv and ~57 on the K=768 out-proj in the engine. The GEMM kernel is now the lever
for every prefill phase (hc, attention proj, MoE compute), not GDN-specific work.

## 2026-09-05 — AMX-BF16 GEMM for the dense prefill projections

The int8 VNNI `gemm_tq` carried every dense prefill projection (hyper-connection down/up, GDN
qkv/z/out, attention q/k/v/o and indexer, PLE key/value, shared expert) at ~98 GMAC/s per core in
the engine. Its structural ceiling is the per-32-block f32 epilogue (convert, two FMAs per
activation row per block) — AMX-INT8 would not remove it, because int32 tiles cannot accumulate
across blocks with different scales. AMX-BF16 does: unpack the TQ block to bf16 once
(`w = d·q + mn`, rounded once), and the tile accumulates the whole K in f32 with no per-block
epilogue.

Probe (`amxbench`, one core, 2.7 GHz): `tdpbf16ps` alone 1069 GMAC/s; a 2×2-tile loop with
64-byte-aligned operands 1066 GMAC/s from L1 and 1055 GMAC/s streaming 4 KiB/step from a 1 MiB
buffer (unaligned tile rows cost 3×: 346 GMAC/s). Engine-like shape (K=2560, m=256, 7 strips per
core): unpack 30 µs + GEMM 150 µs = 490 GMAC/s (L2-bound: the A rows stream once per strip pair),
against 154 GMAC/s for `gemm_tq` on the same shape.

Kernel (`tr-kernels/src/amx.rs`, `asm!` since the tile intrinsics are unstable):
`init()` requests XTILEDATA (`arch_prctl`) before the pool starts; `unpack_strips` writes strips as
`[k/2 pair-rows][16 n][2]` bf16 (one 64-byte B-tile row per pair-row; low nibbles are k = 4j..,
high nibbles k = KB/2 + 4j.., as in `gemv_impl`); `rows_to_bf16` converts activation rows;
`gemm_bf16` runs 2 A × 2 B tiles into 4 accumulators per (strip pair, 32-row group), storing tiles
straight into `y` with the row stride, and reconfigures the tile rows for the m-tail. Engine:
`Act::{Q8, H}` + `dense_gemm` at every dense site, per-core 1 MiB unpack buffers in the tile arena
(`BatchWs.amx_buf`), `TR_AMX=0` falls back to VNNI. Decode is untouched (bandwidth-bound GEMV).

Two bugs on the way: (1) the shared-expert down projection has K = n_ff/8 = 80, not a multiple of
the 32-deep tile — `rows_to_bf16` overran each row by 16 elements into the next and the GEMM
silently dropped the last 16 k; it showed up as a NaN in the router at 3K after the batch that
straddles the QSA threshold happened to carry an all-zero token. `use_amx(k)` now keeps K % 32 ≠ 0
on VNNI and the kernels assert. (2) the first version malloc'd the unpack buffer per core; with
every node at 0.1 GiB free after load it landed wherever, and socket-1 tiles ran the GEMMs 3× slower.
Arena placement fixed it. `TR_NAN_CHECK=1` (per-phase finite check on tile 0) stays in.

Precision: bf16 rounding of both operands is closer to exact f32 than the int8-per-32 activations
(numpy experiment on Q4 weights: rel. RMS error 2.5e-3 vs 1.1e-2 with heavy-tailed activations,
2.4e-3 vs 5.4e-3 Gaussian). Greedy 5-token output unchanged (24/24 llama.cpp); the 26-token prompt
now ends ` Madrid. The capital of Portugal is Lisbon` where the int8 path (and llama.cpp, which
also uses Q8 activations) put a newline — a 17.1 vs 15.6 logit call on the AMX path, 17.8 vs 16.6
the other way on VNNI.

Back-to-back, clean (VmSwap 0), two reps each; the box drifted faster over the sequence, so the
pairs are what count:

| | VNNI (`TR_AMX=0`) | AMX-BF16 |
|---|---:|---:|
| pp512 | 420 / 522 tok/s | **503 / 643** |
| pp3072 | 475 / 502 | **501 / 535**, then 578 / 612 |
| tg64 / tg32@3K | 68–69 / 61–63 | 67–70 / 60–64 (unchanged) |

Per 256-token layer at pp512: gdn.proj 1049 → 448 µs, gdn.out 673 → 423, attn.proj 1626 → 406,
b.gdn 2190 → 1200. The whole is only ~20 % faster because the dense GEMMs were ~30 % of a batch:
what remains is MoE expert streaming (2.9 ms/layer, bandwidth floor), the attention layer
(6.6 ms/layer at pp512 — of which a 3 ms wait at barrier I that predates AMX and needs a look),
exchanges/barriers (~0.4 ms × 4 per layer) and the hyper-connection non-GEMM work. Kernel-side, the
2×2 tile loop is L2-bound at ~500 GMAC/s (peak 1070): K-blocking so the A group stays in L1 is the
next kernel step if the GEMMs matter again.

## 2026-09-05 — Prefill: the barrier-I stall, the exchanges, and what the socket link really does

Profile-driven pass over everything around the GEMMs at pp512 (`TR_PROFILE=1`, new sub-laps for
the hyper-connections and the exchanges). In order of what was found:

**Barrier I (3 ms/attention layer)** was the K/V cache: allocated `alloc_slice_uninit` under
`MADV_HUGEPAGE` "to fault lazily as the context grows" — so the first write to a layer's cache
mid-prefill took a 2 MiB zero-fill fault under 0.1 GiB/node free memory, on whichever core got
there first, and every other tile waited at the next global barrier. Pre-faulting at load
(`alloc_slice`) took the attention layer from 6.6 to 2.0 ms and rep 0 now matches rep 1.

**Exchanges** (`gather_rows` / `reduce_rows` / `combine_batch`, 0.4 ms × 4 per layer): element-wise
loops with a `slot_ro` per element, and every tile summed all 8 partial rows (21 MB of remote
reads per tile per reduce). Now: row copies; the reduce is a reduce-scatter + all-gather with the
residual combine fused into the gather (no `red` buffer); `part` is stored column-blocked
(`part_block(b)`, 8 × [m][320]) so a tile's slice is one contiguous run per source. Then the
measurements that mattered, with timing-only variants of the reduce (`TR_XPERIMENT`, removed):

| variant | cross-socket bytes per direction | time |
|---|---:|---:|
| flat reduce-scatter, each tile reads its 1280-B slice of every remote row | 5.2 MB unique | 175 µs |
| same, two-phase (contiguous copy then local sum) | 5.2 MB | 225 µs |
| socket-hierarchical: sum inside the socket, swap 654 KB/tile across, gather same-socket | 2.6 MB | 23 + 88 + 83 µs |
| swap step reading own tile / same-socket neighbour / other socket | — | 16 / 32 / 90 µs |
| `mixed` all-gather (each remote line read by all 4 tiles of a socket) | 1.3 MB | 46 µs |

Everything fits one model: **the socket link carries ≈35 GB/s per direction, and same-socket tiles
that read the same remote line share one fetch through the socket's caches** (the old full reduce:
10.4 MB → 300 µs; the flat scatter with unique slices: 5.2 MB → 150 µs; the swap: 2.6 MB → 75 µs;
the gather: 1.3 MB → 37 µs — all at ~35 GB/s). Not the store type (NT vs plain: same), not the
loop shape (memcpy vs AVX: same), not RFO on remotely shared lines (private-destination variant:
same), not prefetch. The hierarchical all-reduce moves the minimum an f32 all-reduce can (each
socket must receive the other's full partial once); halving it again means bf16/f16 partials,
which is a fidelity call not taken. Streaming stores for everything written into a mailbox slot
(`put_row`, `dense_gemm_stream` via a per-core staging buffer) stay: they are worth 20–45 µs per
GEMM/reduce on the same-socket side.

**Hyper-connections** (0.7 ms per call, ×2 per layer): `hc_b` had a scalar `sigmoid` over its
gate columns (107 µs), a scalar mixing loop and 512 `dot`s of 320 per token for the router (206 µs
together); now `sigmoid_inplace`, vector mixing, and a 4×4 register-blocked f32 GEMM
(`smallgemm::gemm_f32_nt`) for the router — kept f32 because routing near-ties are the one place
bf16 inputs would visibly change outputs. `dot` and `sumsq` got 4 accumulator chains (the
10 240-wide injection dots were FMA-latency bound). `route_topk` is one insertion pass instead of
k sweeps with a `vec![false; 512]` per token.

**Attention**: query-block tasks are handed out in zigzag order (block i with block nqb−1−i) so the
causal cost is balanced: `attn.wait` 446 → 84 µs at pp512.

**Tile 3** is consistently 20–30 % slower in every compute phase, and the per-tile timers (now
with the arg-max core) point at core 12 = cpu 51: the NIC's receive queues (`i40e-eno2np1-TxRx-*`)
have their IRQ affinity on node 3's cores and >50 M interrupts have landed on cpu 51. That is a
system setting, not the engine; it costs ~300 µs per layer at barrier C. Fix outside:
`echo 143-155 > /proc/irq/<n>/smp_affinity_list` for those IRQs (the SMT siblings), or move them
off node 3 altogether.

Things tried and dropped: `clflushopt` of tile-stored PART rows (no effect); expert-major MoE
down projection (one 180 KB stream per expert instead of each core's strip range of every
expert: down 754 → 1000 µs, combine 296 → 207, net loss); software prefetch in the scatter.

Back-to-back with the AMX commit's binary (3 reps / 2 reps, VmSwap 0):

| | before (`3e82137`) | now |
|---|---:|---:|
| pp512 | 683–691 tok/s | **873–885** |
| pp3072 | 576–601 | **780** |
| tg32 / tg32@3K | 73 / 67 | 73–74 / 67 (unchanged) |

Per 256-token batch at pp512 (~290 ms): MoE 2.6 ms × 48 = 45 %, hyper-connections 0.5 ms × 96 =
17 %, exchanges+barriers ~15 %, GDN 0.85 ms × 36 = 10 %, attention 1.5 ms × 12 = 6 %. MoE
gate/up streams expert weights at ~93 GB/s per tile against a measured 127–179 GB/s ceiling
(compute at m≈5 rows per expert is interleaved with the streaming, not overlapped); the
hyper-connection injection re-reads `xn` from HBM per stream and the router GEMM is at ~half of
FMA peak. Output: identical to the previous binary on the short prompts (24/24 and 64/64 tokens);
the 3133-token prompt diverges at its 4th token on a three-way tie within 0.2 logits (summation
order of the all-reduce and the 4-chain dot changed).

## 2026-09-05 — MoE: where the time really is, and an AMX-INT8 kernel that did not pay

The notebook had MoE down as "bandwidth floor". Two measurements say otherwise: batch 512 leaves
the per-token gate/up cost unchanged (1.23 ms per 256 tokens, 2.34 per 512 — a bandwidth-bound
phase would stay at 1.23), and kbench on the exact shape (an expert's 80 × 2560 gate matrix,
weights streamed from a 235 MB pool) gives the single-core VNNI kernel 10.7 µs at m = 5 and
12.3 µs at m = 8 = 134 GMAC/s, which is the `vpdpbusd` issue limit. The engine gets 15.5 µs per
expert matrix with 13 cores sharing the tile's HBM. So the routed-expert GEMMs are compute-bound
in the int8 kernel at 5–10 rows, ~25 % above the byte floor at batch 256 and far above it at 512.

**AMX-INT8 (`amx_i8.rs`)**: one `tdpbsud` per (strip, 32-k block) with the packed nibbles expanded
straight into the u8 B-tile rows (the chunk layout already is `[16 n][4 k]`), int32 tile stored,
then the same per-block f32 epilogue as `gemv_impl`. Bit-exact with `gemm_tq` for all six codecs,
row maps and accumulate. First version (one chain per block) 18 µs per expert matrix at every m;
four blocks in flight (tmm0/1/6/7) 12.3 µs at m = 5, 13.9 at m = 8, 24.0 at m = 16 — still behind
VNNI (10.7 / 12.2 / 23.6). The per-block scale epilogue the TQ format forces is the same in both
kernels and is now the dominant cost; the tile loads/stores per block outweigh the multiply
savings. Kept in the tree (tested, unused) as a reference for a block-64 format.

What did pay: the down projection accumulates each expert's rows straight into the per-token sums
(`gemm_tq_rows`, a row map with `accumulate`), so the 28 MB per-entry scratch and the combine pass
are gone (combine 293 → 15 µs; down 768 → 910 because the accumulate is a read-modify-write of
scattered token rows out of L2 — software prefetch of the rows did not help); MoE 2.61 → 2.52 ms.
Also: the hyper-connection injection as one 4×4-blocked f32 GEMM (`xn` read once per 4 rows,
84 → 59 µs) and attention query blocks claimed from a tile-shared atomic counter (`attn.wait`
453 → 40 µs at 3K). pp512 885 → 913–932 tok/s, pp3072 780 (unchanged; attention-bound there).

## 2026-09-05 — Run-to-run variance: page faults on full nodes, and what is left

The same binary measured 700 and 930 tok/s at pp512 an hour apart. Fault counters on the bench
line (`faults minor/major` from `/proc/self/stat`) show one cause: **22 000 major faults during
the first batch and ~1 500 per batch after** — glibc served the per-batch large allocations
(logits vectors and the like, above its 128 KB mmap threshold) with fresh mmaps, every batch
page-faulted them in on nodes with 0.1 GiB free, the pressure swapped the process's own pages
out (`VmSwap` 0.1–0.4 GiB) and they came back as major faults. `mallopt(M_MMAP_THRESHOLD, 32 MiB)`
+ `M_TRIM_THRESHOLD` max (`procinfo::pin_heap`, default, `TR_PIN_HEAP=0` to disable) makes the
fault count constant and rep 0 land within 4 % of the later reps (868 / 903 / 906 / 906 tok/s,
VmSwap 0). `M_TOP_PAD` of 64 MiB was tried in the same experiment and dropped decode to 9 tok/s —
not that. A second slow state remains that is not ours: socket-1 tiles run the MoE 45 % slower
(3.5 vs 2.4 ms) with no faults at all — nodes 4–7 carry 250–640 MB of page cache and the direct
reclaim / compaction counters climb during runs (`pgscan_direct` +4.7 M, `compact_stall` +38 K
in one bench). Not something the engine can fix: the fixes are outside (drop the page cache
before a run, or leave the nodes headroom).

Also measured on the way: the worker cores run at ~2.4 GHz all-core during a batch (3.5 GHz for the
single-core probes), and reading activation rows that other cores just wrote — every dense GEMM
does that for its A operand — moves at ~7 GB/s per core (a private copy of 655 KB took 93 µs and
the GEMM afterwards still took 47 µs vs 22 µs isolated at the higher clock). So the engine's AMX
GEMMs sit at ~2× the probe's time for reasons that are the clock and the on-tile fabric, not
the kernel's K-blocking: that lever is worth ~2 % and was not taken.

State at the end of the day (clean runs, `3e82137` → now): pp512 683–691 → **903–932 tok/s**,
pp3072 576–601 → **780–809**, decode 73 / 67 unchanged; output identical on the short prompts.
Per 256-token batch: MoE 44 % (VNNI at its issue limit, ~25 % above the byte floor), hyper-
connections 17 %, exchanges + barriers 18 % (socket link at its floor; barrier C is tile 3's IRQ
core), GDN 11 %, attention 6 %.

## 2026-09-05 — OpenAI-compatible server

`tr-infer serve`: `crates/tr-infer/src/server/{http,chat,engine,mod}.rs`. Decisions:

- **No async stack.** The engine runs one request at a time, so the server is std `TcpListener`,
  one thread per connection, and a channel into the engine thread that owns the `Model`.
  `http.rs` is ~150 lines of HTTP/1.1: keep-alive, Content-Length and chunked bodies,
  `Expect: 100-continue` (curl sends it for larger bodies and otherwise waits 1 s), chunked
  responses for SSE, CORS headers. Dependencies added: serde/serde_json in tr-infer, clap `env`.
- **Chat template transcribed from the GGUF** (`tokenizer.chat_template`, saved to the scratch
  dir): merged leading system/developer messages, the reasoning-effort instruction line for
  xhigh/low (medium has none), assistant history as `<think>\n{reasoning}\n</think>\n\n{content}`
  (the template's `preserve_thinking` default), generation prompt `<think>\n` or the closed
  empty block. Verified string-equal against jinja2 rendering of the original for three cases
  (unit test `template_matches_jinja` carries the reference strings). Tools/images/tool roles
  are rejected with a 400 rather than rendered wrongly.
- **Reasoning split by token id** (`</think>` = 248069) in the engine, not by text: the prompt
  ends inside `<think>` when thinking is on, so output is reasoning until that id. Whitespace the
  model emits right after `</think>` is dropped; trailing whitespace of the reasoning is held back
  and never sent. Stop ids: `<|im_end|>` (eos) and `<|endoftext|>`.
- **Incremental detokenising** per section: decode the tokens since the last UTF-8-complete
  point, hold back trailing U+FFFD. Byte-level BPE decoding is context-free per token, so a
  window that ends clean can be dropped. Stop strings search the unsent tail plus enough
  context for a straddling match, and the longest suffix that is a proper prefix of a stop
  string is held back.
- **Prefix reuse.** The engine keeps the token list its state covers plus the last logits; a
  prompt that extends it is prefilled from the tail only, anything else resets. Multi-turn
  chats hit it when the client returns `reasoning_content` (111 of 134 tokens reused in the
  test); clients that drop the reasoning miss it, since the template renders an empty think
  block then. Rollback is impossible with the recurrent GDN state, so no partial-prefix reuse.
- **Cancellation.** The connection thread peeks the socket every 250 ms while waiting for
  events and on any write error; the cancel flag is checked per token and per prefill chunk.
  A curl killed 2 s into an xhigh essay: next request answered in 0.2 s.

Tested live on port 8089 with curl and Python `http.client`: models/auth/health, non-stream,
stream with `include_usage`, stop strings, content arrays, 100-continue, keep-alive (two
requests, one connection), error bodies. Decode through the server 74–85 tok/s, same as the CLI.
Not done: `/v1/completions`, logprobs, multiple requests in flight.

### Tools (same day)

`server/tools.rs`. The template dumps each tool with Jinja's `tojson`; transformers overrides that
filter with `json.dumps(ensure_ascii=False)` defaults (`", "` / `": "` separators, insertion
order, no HTML escaping), so `py_json` reproduces that and serde_json now runs with
`preserve_order` (the manifest identity hash sorts keys itself, unaffected). Assistant history
with `tool_calls`: OpenAI clients send `arguments` as a JSON string, which the template refuses;
the server parses it into the mapping first. Tool-role messages get the `<|im_start|>user` /
`<tool_response>` grouping of consecutive tool messages. The whole rendering is checked
byte-equal against jinja2 on the original template (`tests/tools_prompt.txt`).

Output side: the model writes `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>…`.
`ToolStream` splits the content stream: text before the first `<tool_call>` is streamed as
`content` (holding back a partial tag and the whitespace before it), each complete block becomes
one `tool_calls` delta with the full arguments. Parameter values are typed by the tool's schema
(string stays text even if it looks like JSON, anything else is parsed as JSON, untyped decides
by the value), the vLLM qwen3-coder parser's rule. Unparseable blocks fall back to text.

Live: weather call with `reasoning_effort: low` → `tool_calls` with typed arguments (1.7 s);
tool result round-trip answered with 487 of 523 prompt tokens reused; two parallel calls in one
streamed reply; `tool_choice: none` and a no-tool question answer in text.

### Per-phase sampling (same day)

Sampling settings for the `<think>` block separate from the answer: `GenParams { sampling,
think: Option<Sampling> }`; the engine starts with `think` when the prompt is inside an open
think block and rewrites the sampler's temp/top_k/top_p at the `</think>` id (one RNG stream, so
a seed still reproduces a whole reply). Request: `reasoning_temperature/top_p/top_k` flat or
under `reasoning`; server: `--think-temp/--think-top-p/--think-top-k`. Verified: temp 1 thinking
with temp 0 answer gives different reasoning and answers per seed, identical replay per seed;
thinking off ignores the settings. The log line prints both settings.

## 2026-09-05 — Speculative decoding with the model's MTP head

**Where the draft head comes from.** The GGUF has no MTP tensors (llama.cpp's converter sets
`no_mtp = True` for qwen4exp). The AutoRound checkpoint on the box keeps them unquantised:
`/opt/llm/Qwen3.8-Flash-Next-W4A16-AutoRound/model_extra_tensors.safetensors`, 5.2 GB BF16, 1565
`mtp.*` tensors: one full-attention decoder layer (q/k/v/o with the q-gate, q/k norms, QSA indexer
`index_qk_proj`, two hyper-connection mixers, router, 512 experts, shared expert) plus
`fc_embedding`, `fc_hidden` [2560×2560], `pre_fc_norm_embedding` [2560], `pre_fc_norm_hidden`
[10240] and its own `hyper_connection_mixer` (config `mtp = {hybrid, layer_types: [full_attention],
num_hidden_layers: 1}`, `mtp_use_dedicated_embeddings: false`). The forward, transcribed from sglang's
`qwen4_exp_mtp.py`: `e = fc_embedding(gemma_rms(embed(tok)))`, `x[s] = fc_hidden(gemma_rms_10240(h))[s]
+ e` (one RMS over all four streams), the layer, the mixer, the shared LM head; the row for position p
takes the residual after p-1 and token p and predicts p+1. `trpack mtp` (`python/trpack/mtp.py`,
`safetensors.py`: 8-byte length + JSON header + memmap, no torch) maps the names onto the GGUF ones the
engine knows (`blk.48.*`; +1 on every Gemma norm as the converter does; `index_qk_proj` split 512/128
rows), quantises experts round-to-nearest to 4/5 bit and dense to 8 bit (`float_rows_bits`), and
writes an overlay pack (`overlay_of` = base hash) of 228 MiB per tile in 51 s. The base pack and its
plan version are untouched.

**Rollback.** KV rows, the QSA block-key ring and the draft layer's own cache are position-indexed
and just get overwritten; the GDN conv/SSM states and the PLE history are destructive. `Model::verify`
runs the batch with the recurrence row by row through the decode kernel (`gdn_step16`), slot i =
f(slot i-1, row i), the live state only read; conv history slots are built from `[live; raw rows]`;
`commit(j)` swaps slot j with the live buffers. `crates/tr-model/tests/verify.rs` (subset pack, ignored
test) checks that `verify` + `commit(j)` + `step(probe)` equals the sequential reference: 0.0000
max-abs difference for j ≥ 1 with `TR_HIPREC=0` (identical arithmetic), 0.03–0.05 on a scale of 6
otherwise (the decode head quantises its activation in pairs, the batched head in one level); the
wrong prefixes differ by 0.7–1.9. Found on the way: the subset pack's expert fold (`% 16`) produced
duplicate experts per token, which the batch path's per-expert grouping cannot hold; ids are now kept
distinct (`fold_ids`), test packs only.

**The batch path at small M** (`bench --tg-batch M`, decode 13.3–14.1 ms/token):

| M | AMX path (before) | after: VNNI below 32 rows, inject/router split across cores |
|---|---:|---:|
| 2 | 23.3 ms | 17.2 ms |
| 4 | 27.3 ms | 19.5 ms |
| 8 | 33.9 ms | 24.8 ms |
| 16 | 44.9 ms | 43.3 ms (VNNI) |

At M=4 the profile had `b.hc_a.inj` 12 µs and `b.hc_b.mix` 29 µs per call because the inject and
router GEMMs ran on one core per row (decode splits them across cores) — 4 ms per step; and the AMX
GEMMs paid the bf16 unpack of every strip for 4 rows (`b.gdn.proj` 93 vs 50 µs). What is left at M=4
is mostly the MoE: 166 µs per layer against 53 in decode, near the byte floor of 4 tokens' experts
(3.7 MB per tile per token per layer). So verify(k+1) costs about 1 + 0.45·k of a decode step; pp512
is unchanged (884–933 tok/s, AMX from 32 rows, `TR_AMX_MIN_M`).

**Draft head accuracy** (`spec-probe`, greedy main model vs draft): "The capital of France is …":
first draft 92 %, chained second 82 %, third 78 %; a three-paragraph prose answer: 73 / 64 / 44 %.
A draft step costs 1.8 ms (the shared 636 MB LM head is ~0.7 ms of it; the layer, the fused input
projections and five global barriers the rest).

**End to end** (`generate`, 48–200 tokens, k=3): temperature 0 outputs are identical to plain
decoding for the arithmetic prompt with thinking (56 tokens) and diverge once in the prose and the
capitals prompts — at a near-tie of the batched vs the one-token numerics (capitals, token 28:
`TR_LOGIT_DEBUG` shows 77916 at 17.64 against 271 at 17.26, a 2 % margin, below the paths'
difference). Same seed
twice at temperature 0.7 gives the same output. Acceptance 117/249 on prose (2.4 tokens per round),
39/46 on the arithmetic thinking (3.4 per round), 20/58 at temperature 0.7 on a limerick.

**Throughput.** With the pack fully resident the runs with the overlay show `VmSwap` 0.26–0.42 GiB
and 14–37 K major faults per run and end at 52–72 tok/s against 75 for plain decoding: every node
sits at 15.0–15.2 GiB anonymous memory with 50–150 MiB free (base engine 14.5 GiB arena + the
overlay, the k+1 checkpoint slots and the draft cache), and the kernel reclaims by swapping hot pages.
With `--ple-mmap` (PLE table file-backed, 3.4 GiB per tile freed) the same server, same prompts, temperature
0, 250 tokens, page cache warm (second request onwards):

| | prose (hash tables) | prose (short story) |
|---|---:|---:|
| plain | 73.8 / 73.9 tok/s | 74.0 |
| `--spec-k 3` | 83.2 / 83.6 (147/306 drafts accepted) | 81.8 (144/313) |
| `--spec-k 2` | 90.3 / 90.4 (138/224) | 84.2 (128/242) |

So +12 % at k=3 and +17–22 % at k=2 on prose (k=2's verify is 17.2 ms and its drafts cost less;
the third draft is accepted rarely enough not to pay for itself). The arithmetic says this is close to
the ceiling of the current costs: 2.4 tokens per round for 19.5 + 3·1.8 + ~2 ms (verify, drafts, the
post-commit draft extend) against 13.3 ms per token. The levers, none taken yet: a cheaper draft
step (the head dominates; a restricted draft vocabulary keeps speculative sampling exact), folding
the extend into the verify pool run, and freeing per-node memory so the resident configuration works
(`vm.swappiness`, the environmental page-cache reclaim already in the notes).

## 2026-09-05 — Speculative decoding: two pool runs per round

The previous entry's arithmetic: a round at k=3 was verify 19.5 ms + 3 drafts × 1.8 ms + a
post-commit draft extend ~2 ms + host work, for 2.4–2.6 tokens. The draft step's 1.8 ms was only
~1.1 ms of pool work (head 0.65, layer 0.40, projections 0.07): the rest was leaving the pool —
gathering 1 MB of logits to the host, the sampler over 248 K logits, and the wake-up of the next
run — three times per round, plus a fourth run for the extend.

**Draft chain in the pool** (`Model::mtp_draft`, `draft_pick`). The k draft rows now run in one pool
run and every draft is sampled inside it: after the LM-head GEMV each core keeps the top 64
candidates of its slice of the tile's logits, core 0 merges them into a small `cand` mailbox slot,
and after the global barrier every tile's core 0 builds the same distribution from the 8×64
candidates (`Sampler::dist_from`, ties towards the lower id, identical arithmetic on identical
data — no exchange of the result needed) and draws it with a uniform the host pre-drew for that
draft. Worker 0 records (token, distribution) for the accept step. For top-k ≤ 64 the draft
distribution is exactly the host sampler's; beyond that it is the same distribution truncated to
512 candidates, which speculative sampling accepts as any other q. The pick costs 68 µs; a draft
step is now 1.2 ms of pool time and nothing else.

**Extend folded into verify.** The draft rows the extend used to run after `commit` — position P+i
with the main model's residual row i-1 and token d_i — depend on nothing the accept step decides,
so the verify run computes them for all k drafts right after the batched head (0.4 ms for one
draft-layer pass), keeping the main residual rows in `BatchWs::res_keep`; `commit(j)` just marks row
j as the next chain's carry (`mtp_carry_row`), rows above j are overwritten by that chain. The round
is two pool runs: draft chain, verify.

**Host sampler.** `Sampler::dist` cost 0.8 ms per call at temperature 0.7 (top-k 20) and 16 ms with
top-k 0 (`select_nth` through an index vector over the whole vocabulary), 2k+1 calls per round. It
now drops candidates more than 24·temperature below the maximum first (relative probability below
4e-11): 0.18 / 0.19 ms; greedy 0.19 → 0.06 ms (vectorised max, then the first index).

**Results** (`p9.log`/`p11.log`). Temperature 0 outputs: identical to the previous speculative runs
for the capitals and the arithmetic-with-thinking prompts (and to plain decoding, up to the near-tie
already recorded); the prose prompt diverges from the previous speculative run at token 187, top
two logits 21.515 vs 21.431 (`TR_LOGIT_DEBUG` now prints the accepted draft rows too) — the drafts
differ (the chain's first row was a batched extend row before, a single row now), so the verify
batches differ, and that is another near-tie of the batched numerics. Same seed twice at
temperature 0.7: identical, and identical to the previous version's output. `tests/verify.rs`
unchanged (0.0000 for j ≥ 1). Server, `--ple-mmap`, warm, 250 tokens, temperature 0:

| | prose (hash tables) | prose (short story) |
|---|---:|---:|
| plain | 75.1 / 75.2 tok/s | 75.1 |
| `--spec-k 3` | 93.7 / 93.8 (155/283 accepted; was 83.2–83.6) | 81.0 (139/331; was 81.8) |
| `--spec-k 2` | 92.7 / 92.7 (138/224; was 90.3) | 86.3 (128/242; was 84.2) |

Per round: 29.1 → 27.8 ms on the hash-table prose (96 rounds), 28.6 → 27.6 on the story — about
1.2–1.5 ms per round, less than the ~3 ms the arithmetic promised. The verify run grew by the folded
draft pass (0.4 ms) and the residual copy, and what the profile can show of the rest is inflated by
the PLE page faults (`bench --ple-mmap` never gets warm: the table is read at random n-gram
positions, 13 K faults in the third rep; `b.ple` 300 µs per layer where the warm server spends a
few). The k=3 gain on the hash-table prose comes mostly from the drafts: 155 of 283 accepted
against 147 of 306 before — the single-row first draft is slightly better than the batched one was.

What is left per round at k=3, ~27.8 ms: verify ≈ 20 ms (MoE 131 µs per layer at M=4 against 53 in
decode — the byte floor of four tokens' experts; hyper-connections 43 µs per layer against 33),
three drafts 3.6 ms (LM head 0.65 each: a 4-bit copy of `output.weight` for the draft would save
~0.25 ms per draft, 45 MB per tile, ~2.5 % — not taken), the folded draft pass 0.4 ms, host work
(4 MB of logits, 7 sampler calls) ≈ 1 ms. The next real lever is the batch path's per-row cost at
small M, not the draft head.


## 2026-09-05 — Images: the vision encoder, tensor-parallel over the tiles

**Goal.** Send images to `/v1/chat/completions`. The model ships its vision encoder as
`mmproj-F16.gguf` (llama.cpp projector `qwen3vl_merger`: SigLIP-style ViT, hidden 1152, 27 layers,
16 heads × 72, GELU MLP 4304, LayerNorm+bias, 16×16 patches, 48×48 learned positions, 2-D rope,
2×2 merger 4608→4608→2560), 450 M parameters, 862 MiB in f16.

**What the reference does** (llama.cpp `tools/mtmd`, `models/qwen3vl.cpp`, `mtmd-image.cpp`,
`ggml-cpu/ops.cpp`; HF config `Qwen4ExpForConditionalGeneration`): patches in 2×2-block order
((y,x),(y,x+1),(y+1,x),(y+1,x+1)) with channel-major vectors; the two temporal conv kernels applied to
the same frame and summed; positions bilinear ALIGN_CORNERS from 48×48; rope pairs (j, j+36), j<18
rotate by the patch row with freq 1e4^(-2j/36), j≥18 by the column with the same 18 freqs; full
non-causal attention; smart_resize to multiples of 32 within 8..4096 tokens (llama.cpp's default;
HF's default is 64..16384), Pillow bicubic (a=-0.5, fixed point), and — llama.cpp only — `PAD_CEIL`
letterboxing instead of transformers' stretch. Language side: `<|vision_start|>` + n×`<|image_pad|>`
(248056) + `<|vision_end|>`; qwen4exp uses interleaved M-RoPE (`IMROPE`, sections [11, 11, 10] over
the 32 rotated pairs, pair j takes component j % 3); image token i of an nx×ny grid at position p is
(p, p+i/nx, p+i%nx), the next text token is at p+max(nx, ny); KV cells stay sequential; the PLE hashes
image rows as `ple.image_token_id` = 248056; the QSA pooled block keys are roped at the cell index in
every section ("exact for text, approximate for images"), the indexer queries with the M-RoPE triple.

**Design.** Overlay pack `trpack vision` (108 MiB/tile + 16 MiB shared, 4 s to write). Tensor-parallel
like the main model: tile t owns heads 2t, 2t+1 (qkv rows [q|k|v] 432, attention over all patches
locally, `attn_out` K-slice 144 padded to 160), MLP hidden split 544/tile (4304 padded to 4352),
merger hidden 576/tile; the residual `x[N][1152]` replicated on every tile so a layer is two
all-reduces (chunks of `--batch` rows through the batch path's `part`/`rsum`/`rfull` rows and
`allreduce_rows`), LayerNorms and biases recomputed per tile. GEMMs AMX-BF16 straight on stored bf16
strips (new manifest kind `bf16_strips`); attention f32 `gemm_f32_nt` (q/k padded to 80, V^T per
head, 32-query tasks from a counter). Workspace ≈250 MiB/tile at the 4096-token cap. Host side:
`image.rs` (smart_size, Pillow bicubic port, patches, position resampling, `mrope_positions`),
`Model::encode_image`, `Model::step_batch_with(toks, pos3, rope_after, image rows)`,
`RopeTable::apply3`, `Model::rope_delta` (rope position = cell + delta for text after images; decode,
verify and the MTP rows use it, the MTP rows of image cells take the image embedding).

**Oracle.** `tools/oracle/mtmd_embd.cpp` links the user's llama.cpp build (vocab-only text model +
libmtmd) and dumps the projected embeddings of an image; `tools/oracle/vit_ref.py` is a numpy f32
reference (f16 weights, or the pack's quantisation, or bf16-rounded weights); `tr-infer vision-embed`
dumps ours; `compare_embd.py` reports per-token relative L2 and cosine.

**8-bit weights are not good enough here.** First version packed everything 8-bit TQ like the indexer.
Synthetic images matched the oracle at 2–3 % relative per token, but a 320×320 photo gave 10 % mean
with outliers (cos 0.42). numpy isolated it: f32 reference vs oracle 1.2 %; the *same* reference with
the pack's 8-bit weights 7.5 % mean, max 117 %, cos min 0.58; with bf16-rounded weights 0.44 %. So the
encoder went bf16 (2 bytes × 450 M = 108 MiB/tile, the strip layout the AMX kernel reads directly;
needs AMX). After that: g64 1.0 %, 224² 1.1 %, 320² 2.0 %, 448² 2.3 % vs the oracle; engine vs the
f32 reference 1.0 % (320²), 3.1 % (640×480). At 640×480 the oracle itself is noisy: llama.cpp CPU vs
its own CUDA backend disagree by 7 % mean (cos min 0.62) on that photo — f16 activations, a few tokens
are numerically unstable — and both of my implementations sit at 7–9 % from either. Also learned:
JPEG decoders differ (stb_image vs libjpeg vs zune-jpeg) enough to show in the embeddings, so the
comparisons use PNGs; and llama.cpp's PAD_CEIL letterbox changes the output a lot (stretch vs pad:
42 % mean) — the engine stretches by default (transformers' behaviour), `TR_IMAGE_PAD_CEIL=1` pads.

**Timings** (test pack, exclusive box; encode only): 96×96 (9 tokens) 6.5 ms, 224² (49) 22 ms,
320² (100) 43 ms, 448² (196) 89 ms, 640×480 (300) 143 ms, 1024² (1024) 0.79 s, 2048×1536 (3072)
5.3 s. Profile at 1024²: attention 50 % (14.7 ms/layer = 0.7 TFLOPS/tile in f32), oproj/up/down
chunks 40 % (the all-reduces and the 34-strip up GEMM at ~1 TFLOPS/tile); at 3 MP attention is 78 %.
An AMX attention kernel (bf16 QK^T and PV per head) is the lever for large images (done below); the
GEMM side would gain from larger row chunks per all-reduce.

**End to end.** `generate --chat --no-think --image test-1.jpeg` (llama.cpp's test picture, the 1969
"MEN WALK ON MOON" front page): "This image shows the front page of The New York Times from July 21,
1969, with the headline "MEN WALK ON MOON" and subheadings about astronauts landing on the moon's
plain, collecting rocks, and planting a flag. …" — 300 image tokens, encode 0.14 s, prompt 321 tokens
in 1.13 s, 37 tok/s decode with a cold PLE page cache.

**Against llama-mtmd-cli** (same PNG, greedy, 80 tokens, `-ngl 0 --no-mmproj-offload`; llama.cpp
encoded the image in 4.7 s on CPU and prefilled at ~1.6 tok/s): "This image shows the front page of
The New York Times from July 21, 1969, with the headline "MEN WALK ON MOON" and subheadings about
astronauts landing on the moon, collecting rocks, and planting a flag. The newspaper features a small
photo of the lunar surface …" — word for word ours up to "on the moon" vs "on the moon's plain"
(token ~26), then a paraphrase of the same content. The encoders differ by a few percent per token
and llama.cpp letterboxes while we stretch by default, so token identity was never expected; the
description is the same.

**Server** (`--vision --ple-mmap --ctx 8192`, `srv_img_test.py`): one image (321 prompt tokens, encode
0.16 s, prefill 1.14 s, 37 tok/s decode with a cold PLE cache); the follow-up turn with the same image
reused 400 tokens of state and hit the embedding cache (0 encoded, prefill 0.33 s); two images in one
message with text between them, streamed ("The second image is a synthetic pattern …"); 400s for an
image in a system message, a non-data/http URL and undecodable bytes. `usage.prompt_tokens` counts
image tokens; the log line reports `N images (E encoded, S s)`.

**Not done / levers.** AMX attention for the encoder (large images; done below); http(s) image fetching is
synchronous on the connection thread (30 s timeout); video/audio parts are rejected; `--image-max-tokens`
above 4096 works but the workspace grows with it (~60 MiB per 1024 tokens per tile).

## 2026-09-05 — AMX attention for the vision encoder

Attention was the f32 half of the encoder (50 % at 1 MP, 78 % at 3 MP: `gemm_f32_nt` at 0.7 TFLOPS per
tile). Now it is AMX-BF16 like the GEMMs, flash-style per (head, 32-query block) task:

- **Operands.** Q rows bf16 (pre-scaled by 1/√72), K packed as AMX B strips (16 keys × lanes: pair-row
  p holds (k[key][2p], k[key][2p+1]) for 16 keys — the same `[k/2][16][2]` layout the weight strips
  use), V packed per key block as strips of 16 head dims × the block's keys (72 → 80 lanes, 5 strips).
  The rope pass writes them straight from the roped qkv rows (36 u32 stores per key for K, 80 u16 for
  V); the padded keys up to a multiple of 32 are zeroed once per image.
- **Task.** For each block of `TR_VKB` keys: S = Q·K^T (`amx::gemm_bf16_kept`, f32 scores 32 × kb per
  core), online softmax per row (running max and sum; `elem::vmax` + `elem::exp_sub_bf16`, which writes
  the probabilities as bf16 rows directly and returns their f32 sum), rescale the running output by
  exp(m_old − m_new), then O += P·V with the accumulate variant of the same kernel (tile loads of C
  instead of `tilezero`). `gemm_bf16_kept` keeps the tile configuration across calls (`TileRows`), so
  a task loads it once; `release_tiles` at the end of the phase. Everything of a task lives in L1/L2
  (192 KiB of scores/probabilities at kb = 1024); block sizes 256/512/1024 measured within 3 %, larger
  slightly ahead, 128 costs 5–10 % (per-block overhead), default 1024.
- **Precision — the part that mattered.** Plain bf16 q/k matched the f32 path to 0.5–1.7 % on most
  images but broke t320 (5.6 % mean, one token 105 % off, cosine 0.59) and got *worse* with smaller key
  blocks; a unit test of the kernel against a scalar reference on the same bf16 operands passed at
  1e-3, and an in-engine f32 cross-check per task placed every deviation in layer 12, whose attention
  logits reach 72 with row ranges of 106 (`vit_ref.py --logits`; layers 1, 5, 10 sit at 35–43). bf16
  q and k each carry 2^-9 relative error, i.e. ±0.3 on a logit of 70, and that layer's softmax is
  peaked enough that single patches flip. llama.cpp keeps f16 here (2^-11 each). Fix: the classic
  split — q = q_hi + q_lo, k = k_hi + k_lo (each part bf16), and one AMX pass over the concatenated
  lanes [q_hi | q_lo | q_hi] · [k_hi | k_hi | k_lo] (3 × 72 = 216 → 224) accumulates q_hi·k_hi +
  q_lo·k_hi + q_hi·k_lo in f32 (~16 mantissa bits; the lo·lo term is 2^-18). The QK^T GEMM costs
  224/96 of the plain one, the softmax and P·V are unchanged, and P/V stay plain bf16 (probabilities
  and values are benign: the unit test at logits ~N(0, 70) against an f32-q/k reference stays at
  1.1e-3 abs). The engine then matches the old f32 attention to 0.6–1.3 % mean on the small images and
  is at least as close to the oracles as before.
- **Softmax cost.** With the exp skipped, attention dropped 3.7 → 2.2 ms per layer at 1 MP and
  31.7 → 19.8 ms at 3 MP: the exp/max/convert pass was 38 % of the attention time. `exp512_fast`
  (2^f by a degree-4 polynomial after a single-multiply range reduction, 9 instructions vs 15; 3e-6
  relative, below the bf16 rounding of the outputs and self-consistent with the row sums) took 10 %
  off the attention time. The remaining softmax pass is memory-bound in L2 at the larger blocks.

**Timings** (test pack, exclusive box, encode only; before → after): 96² 6.5 → 5.6 ms, 224² 22 →
20 ms, 320² 43 → 43 ms, 448² 89 → 80 ms, 640×480 143 → 124 ms, 1024² 0.79 → 0.46 s, 2048×1536
5.3 → 1.9 s. Profile: attention 3.3 ms/layer (19 %) at 1 MP and 28.7 ms/layer (41 %) at 3 MP (was
14.7 and 76 ms: 4.5× and 2.6×; 3.2 TMAC/s per tile at 3 MP over the 224-lane QK and the 80-lane PV
including the softmax); the chunked oproj/up/down with their two all-reduces per 128-row chunk are
now 64 % / 47 %.

**Accuracy** (per-token relative L2, mean / max). New vs the old f32-attention engine: g64 0.6 % /
0.9 %, 224² 0.8 / 2.5, 320² 1.0 / 6.9, 448² 1.3 / 11, 640×480 2.8 / 60, 1024² 2.0 / 78 (single
tokens of the synthetic test pictures are sensitive, both engines are within 2–3 % of the f32
reference: numpy f32 vs new 2.3 % / 0.91, vs old 2.8 % / 2.6 at 1024²). Versus the llama.cpp oracle
the new engine is as close as or closer than the old one on every image (320² 1.8 vs 2.0 %, 448² 2.3
vs 2.3, 640×480 7.7 vs 8.8). At ≥ 1 MP llama.cpp's CPU encoder itself is far from the f32 reference
(15.5 % mean at 1024², cosine down to 0.19; 24 % from both engines at 3 MP) — its f16 attention path,
not ours; the oracle took 4 min (1 MP) and 12 min (3 MP) on 29 cores for those dumps.

**Unit test** `vision::tests::amx_attention_matches_reference`: packs random q/k/v through the same
layouts (`AttnGeom::kline/vidx`), runs `attn_task` for five (n, block size, heads) combinations
incl. partial last blocks, single-block images and 32-key blocks, and compares with a scalar f64
softmax attention using f32 q/k and bf16-rounded v (max 1.1e-3 abs at logits ~N(0, 70)).

**Server check** (full model, `--ctx 16384 --ple-mmap --mtp … --spec-k 2 --vision …`): the moon-landing
front page is described as before (encode 0.12 s, 321 prompt tokens in 1.38 s, 39 tok/s), the follow-up
turn reuses 399 tokens with the embedding cache hit, two images streamed. 55 workspace tests pass.

## 2026-09-05 — Cumulative-mass expert routing (`--moe-mass`)

Optional replacement for the fixed top-10 routing: per token and MoE layer, experts are taken in
descending router order until their cumulative mass reaches a target, between `--moe-min` and
`--moe-max` (≤ 32). `router::route_mass` (insertion-sorted top-`max`, one softmax denominator, the
walk, then the same renormalisation as `route_topk`; `min = max = k` is bit-identical to top-k).
The MoE workspaces (`TileWs`, `CoreScratch`, `BatchWs`) are sized for 32 experts per token
(≈3 MB more per tile); the batch path's grouping by expert now takes a variable number of entries
per token (`n_sel[mi]`, `n_ent = Σ`, the unused `tok_ent` map dropped). Worker 0 counts (token,
layer) rows and experts (`Model::moe_stats`), printed by `generate`, `bench` and the server log
line; `--moe-profile` (bench, generate) accumulates the mean cumulative mass at every k.

**The router is flat.** Mean cumulative softmax mass of the best k experts over all 512, on a chat
answer (37 prompt + 147 output tokens, 48 layers): k=1 0.039, 2 0.065, 3 0.086, 5 0.119, 10 0.173,
16 0.218, 32 0.302 (random tokens: 0.032 / 0.145 / 0.269). The 10 experts the model runs carry a
sixth of the router's probability; an absolute target of 0.95 would never stop before the cap
(measured: 10.00 experts at 0.95 / 0.9 / 0.8, `--moe-basis all`). Hence the default basis `top`:
the target is measured against the renormalised weights of the `--moe-max` best experts, the
numbers the model multiplies with (`--moe-basis all` keeps the absolute semantics; useful targets
there are 0.05–0.15: 0.15 → 8.2 experts).

**Experts and speed** (greedy, exclusive box). Chat answer, `generate --chat --no-think -n 200`:

| policy | experts/token | tok/s | output |
|---|---|---|---|
| off | 10 | 74 (bench; the run itself was a cold 61) | reference paragraph |
| `--moe-mass 0.95` | 9.67 | 76 | same first two sentences, then a paraphrase |
| `--moe-mass 0.9` | 8.70 | 72 | different wording, same content |
| `--moe-mass 0.8` | 7.13 | 80 | different wording, same content |
| `--moe-mass 0.7` | 5.80 | 79 | different wording, same content (adds "5–10x the bandwidth") |
| `--moe-mass 0.9 --moe-max 16` | 13.33 | 66 | same content |
| `--moe-mass 0.15 --moe-basis all` | 8.20 | 77 | same content, longer |

Bench after a 2048-token random prompt (rep 2): tg64 14.5 ms/tok at 10 experts, 14.1 at 7.2
(0.8), 13.8 at 5.9 (0.7), 13.5 at 3.8 (0.5); at ctx 256 the fixed counts give 13.5 ms (10),
14.9 (16), 18.1 (32) — ≈0.2 ms per expert and token, i.e. the routed experts are ~15 % of a decode
step (1.2 of ~6 GB of weights per token), so the policy buys at most ~7 % decode throughput at four
experts. Prefill (pp2048) is within its ±10 % noise at every setting (854 / 776 / 960 / 864 tok/s
for 10 / 7.4 / 6.1 / 3.9 experts): the batched path streams each expert's weights once per group
of tokens, so fewer entries per token save GEMM rows, not weight traffic. More than 10 experts
costs the same per expert (16: −9 %, 32: −25 % decode).

**Quality** was only eyeballed on the one prompt above (all seven answers are correct, fluent
paragraphs; the greedy paths diverge from the reference already at 0.95, as any change of the
routing does). No perplexity or task evaluation yet: treat any target below 1 as a speed/quality
trade-off outside the model's training regime and compare on your own prompts before serving.

**Tests** `router::mass_tests` (target reached, floor/cap, flat router hits the cap, `top` vs `all`,
`min = max = k` equals `route_topk`, `mass_profile` vs a sorted softmax) and the ignored engine test
`moe_mass` on the subset pack (min = max = 10 bit-identical to top-k on the batch and decode paths;
variable counts — 5.3 / 6.8 / 4.1 / 7.4 / 9.6 experts — with batch vs decode logits within the
verify tolerance and equal argmax; top-k restored after `set_moe(None)`).

## 2026-09-05 — Layered configuration (YAML + environment + flags)

The command line had grown to ~50 flags with the same eight repeated on every subcommand, and the
server command that actually runs on this box is long enough that it lives in shell history rather
than anywhere reviewable. `crates/tr-infer/src/config.rs` now merges four sources with the `config`
crate (0.15, `yaml` + `json` features only): built-in defaults < YAML file(s) < `TR_*` environment <
flags. Full reference in [CONFIG.md](CONFIG.md); annotated schema in `docs/tr-infer.example.yaml`.

**Shape.** `AppConfig` is a tree of `Option` leaves (53 keys in 16 sections), so "not set anywhere"
stays distinguishable from "set to the default". Defaults are applied per command when the merged
tree is turned into the concrete `RuntimeCfg` / `SamplingCfg` / `MoeCfg` / … structs — which is what
lets two commands keep different defaults for the same key (`runtime.ctx` 4096 but 8192 for `serve`,
`sampling.temp` 0.0 for `generate` but 0.6 for `serve`) while a single `runtime.ctx:` in a file still
means "for whatever I run". No default moved: `serve --print-config` and `config --for CMD` print the
resolved values, which is how the old flag defaults were checked one by one.

**Flags.** All overridable flags became `Option<T>` (defaults live in `config.rs` only, and each
flag's help names the key it writes), and the eight repeated groups are `#[command(flatten)]`
`Args` structs — `PackArgs`, `RuntimeArgs`, `OverlayArgs`, `MoeArgs`, `SampleArgs`, `ThinkArgs`,
`ImageArgs` — each of which knows how to write itself into the override tree. Booleans are
`Option<bool>` with `num_args = 0..=1, require_equals, default_missing_value = "true"`: bare
`--ple-mmap` is true, `--ple-mmap=false` turns off what a file set, and an absent flag stays absent
instead of overriding the file with `false`. `require_equals` also stops `--keep` from eating the
positional after it in `load-bench DIR`.

**Details that mattered.**
- `Environment::with_prefix("TR").separator("__")` defaults the *prefix* separator to the key
  separator, so it wanted `TR__RUNTIME__CTX`. `prefix_separator("_")` gives `TR_RUNTIME__CTX`, and
  `TR_PIN_HEAP` keeps mapping to the flat key `pin_heap`.
- Unknown keys are an error for **files** (with the nearest known key by edit distance) but ignored
  for the environment: the engine's own debug knobs share the prefix (`TR_AMX`, `TR_DUMP_VEC`,
  `TR_PROFILE`, `TR_VKB`, `TR_TEST_PACK`, …) and would otherwise have to be renamed. The known-key
  set is derived from `serde_json::to_value(AppConfig::default())`, so it cannot drift from the
  schema; `config --keys` prints it.
- serde_json has only f64, so `0.95f32` serialises as `0.949999988079071`. CLI overrides go through
  the shortest decimal form (`Ov::set_f32`) and the YAML printer prints any f64 that survives a
  round trip through f32 as that f32 (`config::float`), which leaves genuine f64 like
  `load_bench.gib_per_node: 0.1` alone.
- `TR_PIN_HEAP` is still read *before* `Cli::parse()` — the heap policy has to be set before the
  first large allocation — and `runtime.pin_heap` is applied afterwards only if it differs.
- `--print-config` and `config` redact `server.api_key`; `--api-key` keeps its clap `env =
  "TR_API_KEY"`, so that variable still beats a file and loses to the flag, unchanged.

**Behaviour is unchanged** with no config file present: same defaults, same flags (all of the old
ones still parse), same output. New: `-c/--config`, `--no-config`, `--no-env`, `--print-config`,
`--pin-heap`, the `config` subcommand, and the shared groups now give a few commands flags they did
not have (`spec-probe --ple-mmap`, `bench --vision`, …) which they simply pass to the loader.

**Tests** `config::tests` (six): source precedence per key, environment scalar/list/`TR_CONFIG`
parsing, unknown-key rejection with a suggestion, override nesting and f32 precision, the per-command
defaults, and a YAML round trip (what `--print-config` prints, `--config` reads back to an equal
tree). `cargo test --workspace`: 63 passed.

## 2026-09-05 — Persistent prefix cache (rows chunks + snapshots, filesystem LRU)

| item | result |
|---|---|
| 4004-token prompt, cold vs restored | prefill 5.3–5.8 s → 0.09–0.18 s total (restore 40–109 ms: 17 rows chunks + snapshot, 213 MiB, `O_DIRECT` reads) |
| 440-token prompt (system + user), after another request | 0.9–2.05 s → 0.14–0.18 s (restore 74–112 ms at the end of the user span, 7-token tail prefilled) |
| edited last user turn | resumes at the system boundary: 398 of 435 tokens restored in 91 ms |
| conversation one turn on | in-memory slot when the previous request was the same conversation, else the end-of-request snapshot |
| output with / without cache, across restarts, index rebuilt, MTP on, thinking on | byte-identical at temperature 0; `drafts 19/21` and `39/49 accepted` identical after a restore |
| snapshot write (119 MB, `O_DIRECT` + fsync, btrfs on the NVMe root) | 110–250 ms on the writer thread; rows chunks 1–5 ms |
| export on the engine thread (memcpy on the pool) | 3–35 ms for ~125 MiB; 60 ms for the 19 chunks of the 4 K prompt |
| LRU at 0.5 GiB | store stayed at 0.45 GiB over 8 requests; the evicted prompt missed, the kept ones hit |
| `cargo test --workspace` | 90 passed (12 new in `tr-cache`, 1 in `tr-sys`, 1 span test), 4 ignored; `tests/snapshot.rs` bit-exact on the subset pack |

**Shape.** This model's state is not only K/V: the 36 Gated-DeltaNet layers hold a destructive
[Dk][Dv] recurrence per tile (112 MiB across the box) plus the PLE history, while K/V rows and the
QSA block keys are position-indexed (~26 KiB per token unique; K roped at write time, so valid only
at their absolute position). So the cache stores two kinds of objects: *rows chunks* (up to 256
tokens, never crossing a role boundary, tagged system / user / reasoning / tool / assistant) and
*snapshots* (~115 MiB) at message boundaries and request ends; a restore point is a snapshot whose
rows chain back to position 0 without a gap. Both are byte images of tile memory copied on the pool
(`crates/tr-model/src/snapshot.rs`), so a restore is a memcpy and the continuation is bit-identical.
Objects are keyed by a SHA-256 chain over the engine identity (pack hash, `--kv`, tiles, overlay
hashes, packed layers, layout version), the token ids and the image hashes, and each rows chunk
carries its tokens for verification, so a collision cannot restore foreign state. The store is a
`Backend` trait (blob put/get/delete/list + advisory index) with a filesystem implementation
(`<key>.rows` / `<key>.snap`, JSON header padded to 4 KiB, body written `O_DIRECT` through
`DirectFile::create/write_at`, tmp + rename; index rebuilt from headers when missing) and a memory
one for tests; the LRU index lives above it (`crates/tr-cache`).

**Details that mattered.**
- Role tags come from byte offsets of the rendered template (`render_prompt_spans` records where
  each turn starts, `Tok::encode_with_offsets` maps tokens back); every marker is one added token,
  so span edges are exact. `<|im_start|>assistant\n<think>` … `</think>` is one `reasoning` span,
  the rest of the turn `assistant`; tool responses keep their `<|im_start|>user` wrapper as `tool`;
  generated tokens are `reasoning` until id 248069.
- Snapshot cost: 2 per request in practice (the user turn's end during prefill, the request end),
  because spans under 16 tokens — the 7-token generation tail, `<|im_end|>\n` — get none
  (`cache.snapshot_min_tokens`); prefill batches are clamped to those span ends. Each snapshot also
  carries the logits at its position, so a prompt that ends exactly there resumes without a step.
- The writer thread must not hold the index lock during a 250 ms write: the backend sits behind its
  own mutex (`Cache::backend()`), `insert` only indexes and returns what to evict.
- `ik_raw` (the QSA block in progress) and the MTP carry — which lives in `BatchWs::res_keep`
  after a speculative commit, not in `TileWs::mtp_carry` — go into the snapshot; forgetting either
  would restore at a 4-token boundary only, or draft from a stale residual.
- Page cache: writes and reads are `O_DIRECT` with 4 KiB-aligned staging buffers (`AlignedBuf`),
  reused through a bounded pool so a snapshot export does not page-fault 30 K fresh pages each time.

**Limits.** `generate` does not participate (its inline template differs from the server's);
a client that re-sends the assistant turn re-tokenised differently misses the end-of-request
snapshot and resumes at the previous boundary; no partial reuse inside a chunk; one store per
pack (foreign roots are dropped on open); only the filesystem backend exists — a network backend is
the trait plus an implementation.

```bash
tr-infer serve … --cache-dir /opt/llm/tr-infer/kvcache [--cache-max-gib 32 --cache-snapshots message|request]
tr-infer cache --cache-dir /opt/llm/tr-infer/kvcache stats|clear
TR_CACHE_DEBUG=1  # per-object write timings
cargo test --release -p tr-model --test snapshot -- --ignored --nocapture
```

### Per-role policies (same day)

`cache.roles.<role>.{persist, priority, ttl, snapshot}` (`--cache-role ROLE=K:V,…`, repeatable;
`TR_CACHE__ROLES__…`). `persist: false` skips the role's chunks *and everything after them* (the
chain breaks at the first unstored chunk; the end-of-request snapshot is not written either, since
nothing could reach it), so `reasoning=persist:false` keeps think blocks off disk at the price of
resuming at the last boundary before the first one. `priority` turns the LRU into tiers — the
eviction order is `(priority, last use)` ascending, so every priority-1 object goes before any
priority-5 one — with snapshots taking the role of the span ending at them. `ttl` is idle expiry,
checked on every insert and lookup (names dropped inside a lookup are deleted by the caller after
the index lock is released). `snapshot: false` drops the span-end snapshot for a role. Checked on
the box with `reasoning=persist:false`: a thinking request stores system + user chunks and the
user-span snapshot only (2 rows chunks, 2.1 MiB staged, no end-of-request snapshot), its repeat
restores at the end of the user turn (84 of 89 tokens) and produces the identical answer, and the
follow-up turn (whose history carries the think block) stores nothing new since everything up to the
think block is already there. `cache stats` prints each role's policy when it is not the default.

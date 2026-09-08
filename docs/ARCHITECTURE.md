# Architecture

How the engine is built, what the pack contains, how a token flows through the tiles, and how to
measure and debug it. The chronological reasoning behind each decision is in [NOTEBOOK.md](NOTEBOOK.md).

## The machine and the model

Two Xeon Max 9470 sockets in SNC-4 mode: **8 NUMA tiles**, each 16 GiB HBM and 13 physical P-cores
(SMT siblings are not used), 125 GiB in total and **no DDR**. Measured: 127–179 GB/s read bandwidth per
tile (1.26 TB/s aggregate), 1.3 µs global barrier across 104 workers, about 35 GB/s per direction over
the socket link, 2.4 GHz all-core clock under load. Tiles 0–3 are socket 0, tiles 4–7 socket 1.
A tile whose memory overflows swaps even when the box has room elsewhere, so every allocation is placed
deliberately.

Qwen3.8-Flash-Next (`qwen4exp`): hidden 2560, vocabulary 248 320, 48 layers of which 36 are Gated
DeltaNet (GDN) and 12 are full attention (every 4th). Each layer has 4-stream hyper-connections (hc=4,
low rank 320) around the token mixer and around the MoE. MoE: 512 experts with n_ff 640, 10 used per
token plus a shared expert (optionally a cumulative-router-mass count between `--moe-min` and
`--moe-max`, `router::route_mass`; the MoE workspaces hold up to 32 experts per token, and the batch
path groups a variable number of (token, expert) entries per token). Attention: 24 query / 2 KV heads of dimension 256, NEOX rope on 64 dims,
and the QSA indexer (4 heads × 128) that makes attention sparse above 2051 tokens. Layer 1 adds a
per-layer n-gram embedding (PLE) table of 26.8 GiB (IQ4_NL). GDN: 16 key heads, 48 value heads, head
dimension 128, conv kernel 4; value head H uses key head H mod 16 (the periodic broadcast llama.cpp
uses, not `repeat_interleave`).

## Crates

| crate | role |
|---|---|
| `tr-format` | the pack v2 contract: `Manifest` schema, TQ block codec geometry, reference dequantisation, SHA-256 of the identity |
| `tr-sys` | `Topology` from sysfs; `Arena` per tile (anonymous mmap with `MPOL_BIND`, THP advised, 2 MiB bump allocation); `Pool` of pinned workers with hierarchical sense-reversing barriers (`barrier()` global, `node_barrier()` per tile), workers spin during a run and futex-sleep between runs; `O_DIRECT` loader; AMX enablement via `arch_prctl`; `/proc` readings and heap pinning |
| `tr-kernels` | every compute kernel, each with a scalar or f64 reference in `reference.rs` used by the tests; targets Sapphire Rapids (AVX-512 F/BW/VL/VNNI/BF16, AMX) |
| `tr-cache` | the persistent prefix cache: role tags and chunking, prefix chain hashes, the object format (JSON header + body, 4 KiB aligned), the `Backend` trait with the filesystem (`O_DIRECT`) and in-memory implementations, and the LRU index with the restore-point search; knows nothing about the model |
| `tr-model` | the graph: per-tile weight tables, mailboxes, the decode program (`exec.rs`), batched prefill and speculative verify (`exec_batch.rs`), the MTP draft head (`exec_mtp.rs`) and the speculative round (`spec.rs`), the vision encoder (`vision.rs`) and image preprocessing (`image.rs`), per-tile state with checkpoints, sampler, tokenizer wrapper |
| `tr-infer` | CLI, layered configuration (`config.rs`: built-in defaults < YAML file < `TR_*` environment < flags, via the `config` crate; see [CONFIG.md](CONFIG.md)) and the HTTP server ([SERVER.md](SERVER.md)) |
| `python/trpack` | GGUF to pack converter (numpy + gguf-py), with tests against real GGUF blocks; `trpack mtp` packs the draft head from the BF16 safetensors checkpoint as an overlay; `trpack vision` packs the mmproj GGUF (vision encoder) as an overlay |

## Pack format (v2)

A pack is a directory: `manifest.json`, `node0.bin` … `node7.bin` (14.2 GiB each, one per tile),
`shared.bin` (replicated small tensors), `tokenizer.json`. The packer decides the placement plan
(plan v4 for `flashnext-v3`): every large matrix is split across the 8 tiles (tensor parallel), small
vectors are replicated, and each tile's file holds exactly the bytes that tile will own, in load order,
so the loader streams each file into its tile's arena with `O_DIRECT` from that tile's own cores
(113.8 GiB in about 10 s warm, 18 s cold, 100 % NUMA-local placement).

Weights use the **TQ codec**, a lossless re-layout of the GGUF block quants (Q4_K, Q5_K, Q5_1, Q8_0)
into the form the VNNI kernels stream: 16-row strips, per-row f16 scale and minimum per k-block of
32 (16 for the MoE down projection), unsigned 4/5/8-bit values, `w = d*q + m`. Q8_0's signed values are
offset by 128 with the offset folded into `m`, so one kernel handles every width. Only the `d*sc`
products are rounded to f16 (relative error at most 2^-11); everything else is exact. The block layout
is documented at the top of `crates/tr-format/src/codec.rs` and `python/trpack/codec.py`. The QSA
indexer projections (BF16 in the GGUF) are packed 8-bit; the PLE table stays IQ4_NL as rows; norms
and small vectors are f32.

The manifest lists every tensor with its tile, kind (`tq`, `f32`, `iq4nl_rows`, `bf16_strips` —
bf16 weights pre-laid out as AMX B-tile strips, used by the vision encoder), codec, shape, byte range
and file. `identity` is hashed (canonical JSON, first 8 bytes of SHA-256) into the pack
hash; the shard SHA-256 of the source GGUF is recorded.

Per tile at rest: about 10.0 GiB experts, 3.35 GiB PLE, 0.9 GiB dense, 0.2 GiB state and mailboxes,
leaving about 0.1 GiB free. `--ple-mmap` maps the PLE rows from the file instead (page cache) and
saves 3.4 GiB per tile.

## Execution model

The pool runs one thread per physical core, pinned, grouped by tile (`WorkerCtx { node, local, global }`).
A forward pass is one `pool.run(closure)`: every worker executes the same program (SPMD) on its tile's
slice of the model. Inside a tile, work is split across the 13 cores by 16-row weight strips, by rows of
the batch, or by tasks claimed from an atomic counter; results that other tiles need are written to
**mailboxes**.

A mailbox is a region in each tile's own memory with fixed slots (`MailboxLayout`: `lo`, `mixed`, `part`,
`rlog`, `rsum`, `rfull`, `logits`, `idx`, …). Producers write locally; after a global barrier consumers
read the remote slices they need. Reads of lines other cores just wrote run at about 7 GB/s per core,
and same-socket tiles reading the same remote line share one fetch through the socket's caches, so
the reduction of the token-mixer and MoE partial sums is **socket-hierarchical**: sum inside the socket
(`reduce_scatter`), swap only the unique slices across the socket link, gather from same-socket tiles,
with the residual combine fused into the gather. All mailbox writes go through a per-core stage buffer
and streaming stores.

### One layer in decode

Per token, per layer, seven global barriers (A–E, G, plus I on attention layers):

| step | work on every tile | barrier |
|---|---|---|
| S0/S1 | PLE rows (layer 1); hyper-connection mix part 1: per-stream RMS norm, injection, low-rank down projection into `lo` | A |
| S2 | gather `lo` from all tiles, SiLU, low-rank up projection, this tile's slice of the mixed input into `mixed` | B |
| S3 | gather `mixed`; token mixer on the tile's heads: GDN (projections, conv, gated delta recurrence, gated norm, output projection) or attention (QKV, rope, KV cache append, QSA indexer slice, barrier **I**, block scoring and selection, attention, output projection); partial sums into `part` | C |
| S4 | reduce `part`, combine with the residual streams, hc mix part 1 for the MoE | D |
| S5 | hc part 2, router partial logits into `rlog` | E |
| S6 | gather router logits, softmax top-10, shared expert plus the routed experts this tile holds, partial sums into `part` | G |
| S7 | reduce and combine | next layer |

After the last layer: final mix, RMS norm, the tile's slice of the LM head into `logits`, barrier, and
the main thread gathers the logits. Activations for the int8 GEMVs are quantised per (row, k-block)
with two-level ("pair") int8 in decode and single-level int8 in prefill (as llama.cpp does).

The per-token cost is close to the memory floor: about 500 MB of dense weights and 200 MB of expert
weights per tile per token at about 130 GB/s, about 6 ms, plus the LM head; measured 14 ms per token
including everything else.

### Batched prefill

`step_batch` runs M tokens (default 256) through the same sharding with tile-shared row-major buffers.
Differences from decode: the dense projections use **AMX-BF16** tiles (TQ strips unpacked to bf16 in
tile order, f32 accumulation; `TR_AMX=0` falls back to VNNI); the GDN recurrence runs a chunk-major
fused multi-token kernel; attention is query-blocked with tasks claimed dynamically and zigzag-ordered
for balance; expert GEMMs are grouped by expert over the tokens that route to it, with the down
projection accumulating straight into per-token sums through a row map; the KV cache is pre-faulted at
load so no tile stalls on 2 MiB zero-fill faults mid-batch. Per 256-token batch the time splits roughly:
MoE 44 % (the int8 VNNI kernel at its instruction-issue limit for 5–10 rows per expert), hyper-connections
17 %, exchanges and barriers 18 %, GDN 11 %, attention 6 %.

### State and context

`TileState` per tile: GDN conv history and the [Dk][Dv] f32 recurrent state per value head the tile
owns, the K/V cache for its attention heads (`--kv f16` default, `bf16`, `f32`), the pooled QSA block
keys, and the PLE n-gram history. `--ctx N` sizes the cache; f16 costs about 12 KiB per token per tile.
Above 2051 tokens the QSA indexer selects the 2048 best-scoring 4-token blocks (plus the tail) for each
query; below that attention is dense and bit-for-bit what it was before QSA existed.

The engine has no general rollback: `Model::reset()` clears everything, `step`/`step_batch` append. The
server's prefix reuse works by continuing from the current state when a new prompt extends it.
Speculative decoding needs to undo rejected drafts: `Model::verify` runs the batch with the recurrent
state (GDN conv history and [Dk][Dv] states, PLE history) checkpointed after every row into `k+1`
per-tile slots (`TileState::ckpt`, 14.4 MiB each) — the recurrence runs row by row with the decode
kernel so slot i is exactly the state after row i, the live state is only read — and `Model::commit(j)`
swaps slot j with the live buffers (pointer swap, no copy). K/V rows, the QSA block keys and the draft
layer's own cache are position-indexed and simply get overwritten by the next tokens at those positions.

### Persistent prefix cache

`Model::export_rows(a, b)` / `import_rows` copy the position-indexed state of `[a, b)` as one byte
image (per attention layer incl. the draft layer: K then V rows of every K/V head from the first tile
that holds it, then the QSA block keys of the blocks completed inside the range; ~26 KiB per token
unique), and `export_snapshot` / `import_snapshot` the recurrent state at `n_past` (per tile the
GDN conv history and [Dk][Dv] states of every recurrent layer and the draft carry, once the PLE
history and every attention layer's raw indexer keys of the block in progress, then `n_past`,
`prev_tokens`, `rope_delta` and a caller vector — the server's last logits; ~115 MiB). Both run on
the pool with every tile copying its own slice, so a restore is a memcpy and the continued run is
bit-identical (`crates/tr-model/tests/snapshot.rs`). `Model::cache_identity` names what changes these
bytes (pack hash, K/V type, tile count, overlay hashes, packed layers, layout version) and becomes
the root of every prefix key.

The server side (`crates/tr-infer/src/server/prefix_cache.rs`, crate `tr-cache`) tags every prompt
token with its role from byte offsets of the rendered template (`chat::render_prompt_spans` +
`Tok::encode_with_offsets`; generated tokens are `reasoning` until `</think>`, then `assistant`),
chunks the sequence at role boundaries and every `chunk_len` tokens, keys each chunk by the chain
hash of everything before its end, and stores rows chunks plus snapshots (at span ends of ≥ 16 tokens
during prefill, and at request end). A request looks for the deepest snapshot of its prompt whose
rows chain is complete, restores it, and prefills the rest; the writer thread persists new objects
from staging buffers and the LRU index evicts by bytes. See [SERVER.md](SERVER.md#prefix-cache).

### Speculative decoding (MTP)

The checkpoint ships a multi-token-prediction head: one full-attention decoder layer (with its own
hyper-connection mixers, QSA indexer and 512-expert MoE) plus `fc_embedding`/`fc_hidden`, two Gemma
norms and an output mixer, sharing `token_embd` and `output.weight` with the main model. It is not in
the GGUF; `trpack mtp` reads it from `model_extra_tensors.safetensors` of the AutoRound checkpoint
(BF16), maps the HF names onto the GGUF names the engine already knows (`blk.48.*`, norms +1 as the
GGUF converter does, `index_qk_proj` split into indexer q/k), quantises experts round-to-nearest to
4-bit (gate/up) / 5-bit (down) and everything dense to 8-bit, and writes a separate overlay pack that
references the base pack's hash; `--mtp DIR` loads it into the same tile arenas (`overlay_of` check).

The draft row for position p takes the main model's 4-stream residual after position p-1 (`hc_hidden`,
kept per tile as the *carry* after every committed step) and token p: `x[s] = fc_hidden(rms_all(h))[s]
+ fc_embedding(rms(embed(tok)))`, one RMS over all 10240 elements, then the layer, then the draft mixer
and the shared LM head. Rows go through the batched layer path (`layer_batch`) with the draft layer's
weights and its own K/V cache (`TileState::mtp_attn`); the two projections are row-split across tiles
and exchanged through the `part` mailbox rows. During prefill every chunk also runs its draft rows
(token q with residual q-1, position 0 has none), so the draft cache covers the prompt.

A round (`spec::Speculator::round`), with `t0` sampled for position P, is two pool runs. The draft
run (`Model::mtp_draft`) computes the head's row for (carry, t0) and chains k-1 more rows, sampling
every draft *inside the pool*: after each LM-head GEMV every core keeps the top 64 candidates of its
slice of the tile's logits, core 0 merges them into the tile's `cand` mailbox slot, and after a global
barrier every tile's core 0 builds the same distribution from all tiles' candidates
(`Sampler::dist_from`, ties towards the lower id) and draws with a uniform the host pre-drew — so no
tile ever leaves the pool or ships 1 MB of logits to the host between drafts (the union of per-tile
top-64 contains the global top-64, so for top-k ≤ 64 the draft distribution is the host sampler's;
beyond that it is the same distribution truncated to 512 candidates, still a valid draft). The verify
run (`worker_batch`, `Verify`) returns k+1 logit rows with state checkpoints, keeps the main model's
residual rows (`BatchWs::res_keep`) and, in the same run, refreshes the draft head's K/V for positions
P+1..P+k from them (draft rows above the accept point are overwritten by the next chain). The host
then applies speculative sampling (j accepted drafts, the next pending token), `commit(j)` swaps in
checkpoint j and marks residual row j as the next round's carry (`mtp_carry_row`, consumed by the next
draft run; a decode step or a prefill chunk writes the carry directly). Measured on the full model:
verify at M=4 costs 19.5 ms against 13.3 ms for a decode step (the batch path keeps VNNI below 32
rows and splits the hyper-connection inject/router across cores for small M; it was 27.3 ms with the
AMX path), a draft step 1.8 ms; the draft head's first token matches the greedy main model 92 % of
the time on list-like text and 73 % on prose (chained: 64 %, 44 %).

The host sampler (`Sampler::dist`) drops candidates more than 24·temperature below the maximum
before the top-k selection (relative probability < 4e-11): 0.8 ms → 0.18 ms per call over the
248 K vocabulary, 16 ms → 0.19 ms with top-k 0; a speculative round calls it 2k+1 times.

### Vision encoder (images)

The mmproj that ships with the model (`qwen3vl_merger`: a SigLIP-style ViT, hidden 1152, 27 layers,
16 heads of 72, GELU MLP 4304, LayerNorm with bias, 16×16 patches, a learned 48×48 position table,
2-D rope on q/k, and a 2×2 merger MLP 4608→4608→2560 into the language model's hidden size) is packed
by `trpack vision` as an overlay (`/opt/llm/tr-infer/flashnext-v3-vision`, 108 MiB per tile + 16 MiB
shared) and loaded with `--vision DIR`. Weights are **bf16** in the AMX strip layout, not 8-bit:
measured against a numpy f32 reference on a photo, 8-bit per-32 weights cost 7.5 % relative error on
the output embeddings with outlier tokens above 100 %, bf16 0.4 %; the encoder is 450 M parameters,
so 2 bytes each is cheap. The encoder therefore needs AMX.

`image.rs` preprocesses an RGB8 image the way transformers' Qwen-VL processor does: resize to
multiples of 32 per side keeping the aspect ratio within the token budget (`--image-min/max-tokens`),
a Pillow-style bicubic (the same fixed-point port llama.cpp uses), `(x/255 - 0.5)/0.5`, patches cut in
2×2-block order with channel-major vectors, the position table bilinearly resampled (align corners) to
the patch grid. The default fit stretches to the target size (transformers); `TR_IMAGE_PAD_CEIL=1`
letterboxes like llama.cpp, which the oracle comparisons use.

`vision.rs` runs the encoder tensor-parallel over the tiles in one pool run: tile t owns heads 2t and
2t+1 of every layer (its rows of the fused qkv projection, full attention for those heads over all
patches, the matching K columns of `attn_out`, padded 144→160), a 1/8 slice of the MLP hidden
(544 of 4352, padded) and of the merger hidden (576 of 4608). The residual stream `x[N][1152]` is
replicated on every tile — LayerNorms, bias adds and the patch embedding are recomputed per tile —
so a layer needs exactly two all-reduces (attn_out and MLP down partial sums), done in chunks of
`--batch` rows through the batch path's `part`/`rsum`/`rfull` mailbox rows and its socket-aware
`allreduce_rows`. Dense GEMMs are AMX-BF16 with f32 accumulation (`amx::gemm_bf16` straight on the
stored strips, tasks of 2 strips × 512 rows claimed from per-phase counters). Attention is AMX-BF16 as
well, flash-style per (head, 32-query block) task: K is packed as B strips (16 keys × lanes), V per
key block (strips of 16 head dims × the block's keys), scores for one block of `TR_VKB` (1024) keys at
a time, an online softmax whose probabilities go straight to bf16 rows (`elem::exp_sub_bf16`, a
degree-4 exp), and P·V accumulated in the tiles (`amx::gemm_bf16_kept`, which keeps the tile config
across calls and can accumulate into `y`). The logits need more than bf16 precision — this ViT's
attention logits reach 70 with row ranges over 100, and bf16 q/k cost up to 100 % on single output
tokens — so q and k are split into bf16 hi + lo parts and one pass over lanes [q_hi | q_lo | q_hi] ·
[k_hi | k_hi | k_lo] (216 → 224) accumulates the three cross terms (~16 mantissa bits); P and V stay
bf16. Workspace per tile is sized for `--image-max-tokens` (≈250 MiB at 4096 tokens = 16384 patches).
Timings (exclusive box): 640×480 (300 tokens) 0.12 s, 1024×1024 (1024 tokens) 0.46 s, 2048×1536
(3072 tokens) 1.9 s, where attention is 19 % and 41 % and the chunked oproj/up/down (two all-reduces
per chunk of `--batch` rows) 64 % and 47 %.

Against llama.cpp's mmproj (`tools/oracle/mtmd_embd.cpp`, linked to the user's build; embeddings of the
same image) the engine agrees to 1–2 % relative per token on images up to 448×448 and to 2–3 % against
the f32 numpy reference (`tools/oracle/vit_ref.py`) on 640×480 and 1024², where llama.cpp's own CPU and
GPU backends disagree by 7 % and its CPU encoder is 15 % (1 MP) to 24 % (3 MP) from the f32 reference
(f16 activations; a few tokens of a photo are numerically unstable).

On the language-model side an image is `<|vision_start|>` + n×`<|image_pad|>` + `<|vision_end|>` in
the token stream; the pad rows take the encoder's output as their input embedding instead of
`token_embd` (`ImageSeg` rows in `step_batch_with`), the PLE hashes them as the pad id (as llama.cpp
does), and rope positions follow Qwen-VL's interleaved M-RoPE (`RopeTable::apply3`, sections
[11, 11, 10] of the 32 rotated pairs): image token i of an nx×ny grid at position p gets
(p, p + i/nx, p + i%nx) and the text after it continues at p + max(nx, ny). Cells (KV rows, `n_past`)
stay sequential; `Model::rope_delta` carries the offset so decode, verify and the MTP draft rows
compute the text position as cell + delta. The QSA block keys keep their cell-based rope, as in
llama.cpp. The MTP draft rows of image positions take the image embedding too.

## Kernels

| file | kernel |
|---|---|
| `gemv.rs` | TQ × int8 GEMV/GEMM on VNNI: streams the weight bytes once per call for up to `MAX_M` rows; software prefetch 3 blocks ahead; `gemm_tq_rows` with a row map for the MoE down accumulation |
| `amx.rs` | AMX-BF16 GEMM for prefill (tile config, strip unpack, `tdpbf16ps` loop, f32 epilogue) |
| `amx_i8.rs` | AMX-INT8 TQ GEMM, bit-exact with the VNNI path, kept as a tested negative result (slower: the per-32-block scale epilogue dominates) |
| `gdn.rs` | gated delta recurrence per value head, decode step and chunk-major batch kernel |
| `conv.rs` | depthwise causal conv1d with history state |
| `attn.rs` | causal softmax attention, decode fast path, query blocks, range attention for sparse selections, online-softmax partial merge |
| `qsa.rs` | indexer block scores and the visible-token selection (mirrors llama.cpp `build_qsa_top_k`) |
| `rope.rs`, `elem.rs`, `quant.rs`, `router.rs`, `smallgemm.rs`, `ple.rs`, `iq4nl.rs` | NEOX rope; RMS norm, SiLU, sigmoid, axpy and friends; activation quantisation; softmax top-k with lowest-index ties and the cumulative-mass policy (`route_mass`, `mass_profile`); 4×4 register-blocked f32 GEMM for the router and injection; PLE n-gram hash; IQ4_NL row dequant |

Every kernel has a test against its reference; `cargo test --workspace --release` runs them (43
tests). `kbench` (TQ GEMV bandwidth and the MoE expert shape, VNNI vs AMX-INT8), `amxbench` (tile
throughput and clock), `gdnbench` (state layouts) are single-core probes in `crates/tr-kernels/src/bin`.

## Correctness

The oracle is llama.cpp on CPU (`llama-completion --device none --temp 0`, `llama-eval-callback`).
Greedy `The capital of France is` gives the same 24 tokens (`Paris … Berlin … Rome … Madrid`); a
1889-token prompt gives 24 identical tokens; at 3.1K tokens (sparse attention in both engines) the
phrase agrees and one hallucinated detail differs, which is expected with an int8 indexer where llama's
top-k breaks ties arbitrarily. The tokenizer matches `llama-tokenize` on a 3153-token prompt.
`tools/oracle/` holds the per-layer comparison scripts (`diff.py`, `check_layer.py`, `check_ple.py`,
`dump_tq_fixture.py`) that work from `--dump` and `TR_DUMP_VEC` output.

The server's chat template is a transcription of the GGUF's `tokenizer.chat_template`, verified
string-equal against jinja2 rendering of the original for plain, system, multi-turn with reasoning and
tool-calling conversations (`crates/tr-infer/tests/tools_prompt.txt` is the tools reference).

## Profiling and debugging

| environment variable | effect |
|---|---|
| `TR_PROFILE=1` | per-phase laps for worker (0,0) and per-tile min/mean/max of the batch phases with the slowest core, printed by `bench` |
| `TR_AMX=0` | VNNI instead of AMX for the prefill projections |
| `TR_AMX_MIN_M` | batch rows from which the AMX path is used (default 32; VNNI's 8-rows-per-pass wins below, measured equal at 16) |
| `TR_PIN_HEAP` | heap policy; default `mt` (no mmap for large allocations, no trim), `0` disables. Without it every batch page-faulted on full tiles and the process swapped at random |
| `TR_NAN_CHECK=1` | NaN checks after the batched reductions |
| `TR_LOGIT_DEBUG=1` | top-3 logits per generated token on stderr |
| `TR_QSA_OFF=1`, `TR_QSA_DEBUG=1` | dense attention at every length; score blocks even for dense queries (oracle checks) |
| `TR_DUMP_LAYER=n`, `TR_DUMP_VEC=path` | write layer-n activations and named vectors for the oracle scripts |
| `TR_HIPREC=0` | single-level int8 activations in decode instead of the two-level ("pair") quantisation (default on; weight traffic is the same) |

`bench` prints per rep: tok/s, RSS, `VmSwap` (must be 0), `THP` (anonymous huge pages), and
`faults minor/major` (must be constant across reps). A number from a rep with swap or growing faults is
not a measurement.

## Environment on this box (not fixable in the engine)

- NIC receive-queue interrupts (`i40e-eno2np1-TxRx-*`, IRQs 626 and 677) are pinned to node 3's cores;
  cpu 51 (tile 3's last worker) has taken over 50 million of them. Tile 3 runs 20–30 % slower in every
  phase and every batch waits for it. Fix: `echo 143-155 > /proc/irq/626/smp_affinity_list` (and 677).
- Socket-1 nodes carry 250–640 MB of page cache; direct reclaim and compaction during a run slow tiles
  4–7 by up to 45 % at random with no faults in the process.
- The memlock limit is 8 MiB, so neither the arenas nor the binary can be locked in memory.
- llama.cpp on the same GGUF and this engine each need about 114 GiB; running both swaps one out.

### Sequence-aware continuous execution

`sequence.rs` separates resident sequence state from the single shared tensor-parallel executor.
A generation-counted `SequenceId` owns per-tile recurrent/PLE histories, positions and recent tokens,
raw QSA block keys, and a logical page table. Shared tile-local pools store KV and completed QSA
blocks in 64-token pages. Physical backing uses raw pointers with lifetime tied to owned NUMA arenas;
only disjoint owned pages may be written. Forks share committed pages and copy recurrent state on
the pool. Append preparation performs copy-on-write before any layer executes.

`Model::forward_batch` validates a list of sequence/token segments, preflights aggregate page demand,
and packs their rows into the existing batch workspace. Dense projections and expert grouping run
across the entire batch. GDN and PLE retain separate histories at segment boundaries; attention query
blocks belong to one sequence and translate selected logical rows to physical rows before invoking
the existing attention kernel. All workers follow the same barriers. Continuous rows use a fixed tile-reduction order to avoid worker-assignment-dependent rounding. Only requested final-segment
residuals are compacted for the output head. Per-segment expert statistics avoid cross-request attribution.

Snapshot adapters temporarily select one sequence outside execution. Row import/export translates
logical positions to physical pages, preserving the `state-v1` serialized format and cache identity.
The server scheduler owns admission, chunked prefill, choice forking, sampling, cancellation and
reclamation. No separate model or worker pool is created per request.

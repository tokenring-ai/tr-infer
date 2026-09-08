# tr-infer (Rust)

CPU-only, NUMA-first inference engine for **Qwen3.8-Flash-Next** (GGUF architecture `qwen4exp`,
Unsloth `UD-Q4_K_XL`) on a dual **Xeon Max 9470** with HBM only: 8 NUMA tiles × 16 GiB × 13 P-cores.
Every kernel is written from scratch in Rust (AVX-512 VNNI int8, AMX-BF16, `asm!` for tiles); the only
Python is the offline packer that converts the GGUF into a per-tile pack. It ships an OpenAI-compatible
server with streaming, reasoning, function calling, per-phase sampling and image input (the model's
vision encoder, tensor-parallel over the tiles).

| | decode (tg64) | pp512 | pp3072 | decode at 3K ctx |
|---|---:|---:|---:|---:|
| **tr-infer-rs** | **67–73 tok/s** | **903–932 tok/s** | **780–809 tok/s** | **57–67 tok/s** |
| ik_llama.cpp (same GGUF) | 8.9 | | | |
| llama.cpp CPU | 4.2 | | 12.8 | 4.5 |

Greedy output is token-identical to llama.cpp through 1889-token prompts (details in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#correctness)).

## Documents

| file | what |
|---|---|
| [docs/SERVER.md](docs/SERVER.md) | `tr-infer serve`: endpoints, request fields, thinking control, tools, streaming format, operations |
| [docs/CONFIG.md](docs/CONFIG.md) | configuration: YAML file, `TR_*` environment, flags, and the key reference ([example file](docs/tr-infer.example.yaml)) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | crates, pack format, execution model (tiles, barriers, mailboxes), kernels, state, profiling, environment notes |
| [docs/NOTEBOOK.md](docs/NOTEBOOK.md) | chronological engineering log with every measurement and negative result |
| `~/xeon_max/tr-infer/README.md` | lab notes for this box (packs on disk, baselines, constraints) |

## Quick start

```bash
export PATH=$HOME/.cargo/bin:$PATH
cargo build --release && cargo test --workspace --release

# one-time: GGUF -> pack (about 8 minutes with 6 processes; destination must be a real filesystem, not /tmp)
python/.venv/bin/trpack pack --gguf /opt/llm/Qwen3.8-Flash-Next-UD-Q4_K_XL --out /opt/llm/tr-infer/flashnext-v3
# optional: the MTP draft head for speculative decoding (1 minute, from the BF16 safetensors checkpoint)
python/.venv/bin/trpack mtp --base /opt/llm/tr-infer/flashnext-v3 --out /opt/llm/tr-infer/flashnext-v3-mtp
python/.venv/bin/trpack vision --out /opt/llm/tr-infer/flashnext-v3-vision   # vision encoder from the mmproj GGUF (4 s)

P=/opt/llm/tr-infer/flashnext-v3
./target/release/tr-infer generate --pack $P -p "The capital of France is" -n 24
./target/release/tr-infer generate --pack $P --chat -p "Explain HBM in two sentences." -n 256 --no-think
./target/release/tr-infer bench --pack $P --pp 512 --tg 64
./target/release/tr-infer serve --pack $P --host 0.0.0.0 --port 8089 --api-key SECRET
./target/release/tr-infer generate --pack $P --mtp $P-mtp --spec-k 3 --chat -p "..." -n 256   # speculative decoding
./target/release/tr-infer generate --pack $P --vision $P-vision --chat --no-think --image photo.jpg -p "Describe this image." -n 100
./target/release/tr-infer serve --pack $P --host 0.0.0.0 --port 8089 --api-key SECRET --ple-mmap --vision $P-vision   # image parts in chat messages
./target/release/tr-infer bench --pack $P --moe-profile                      # cumulative router mass per k; then --moe-mass 0.9 [--moe-min 1 --moe-max 10]
```

Every flag is also a configuration key, so a settled setup can live in a file instead
([docs/CONFIG.md](docs/CONFIG.md), [example](docs/tr-infer.example.yaml)):

```bash
cp docs/tr-infer.example.yaml /etc/tr-infer/config.yaml   # or ./tr-infer.yaml, or --config FILE
./target/release/tr-infer config --for serve              # what `serve` will use, defaults filled in
./target/release/tr-infer serve                           # ... and run it
TR_RUNTIME__CTX=8192 ./target/release/tr-infer bench      # environment overrides the file, flags override both
```

The model loads in about 10 s (114 GiB with `O_DIRECT` from each tile's own cores) and uses
115 GiB resident, 14.5 GiB per tile. **Only one engine fits on the box**: check
`pgrep -af 'tr-infer serve'` before starting another, and never run llama.cpp on the same GGUF at the
same time. `--ple-mmap` keeps the 27 GiB per-layer-embedding table file-backed (88 GiB resident,
decode 62 tok/s) when HBM is needed for something else.

## Commands

| command | purpose |
|---|---|
| `generate` | prompt in, text out; `--chat/--no-think`, `--prompt-file`, `--temp/--top-k/--top-p/--seed`, `--ctx`, `--kv f16\|bf16\|f32`, `--batch`, `--dump` (oracle statistics) |
| `bench` | pp/tg throughput on random tokens, `--reps`; prints RSS, swap, THP and page-fault counts per rep (swap must be 0 and faults constant for a trustworthy number); `--tg-batch M` times the batched path at M rows, `--mtp` times speculative decoding |
| `serve` | OpenAI-compatible API, see [docs/SERVER.md](docs/SERVER.md); `--mtp DIR --spec-k 3` for speculative decoding, `--vision DIR` for images, `--cache-dir DIR` for the persistent prefix cache, `--continuous --kv-cache-mib N` for concurrent text requests and `n` choices |
| `cache` | `stats` / `clear` of a prefix cache store (`--cache-dir DIR`), no model load |
| `vision-embed` | encode one image with the vision overlay and dump the embeddings (oracle comparison with `tools/oracle/mtmd_embd.cpp` / `vit_ref.py`) |
| `spec-probe` | draft-head accuracy probe: greedy main model vs first/second/third draft |
| `tokenize` | token ids of a prompt with the pack's tokenizer, no model load |
| `topology`, `numa-smoke`, `load-bench`, `barrier-bench` | box diagnostics: NUMA layout, placement and bandwidth per tile, `O_DIRECT` load rate, barrier latency |
| `kbench`, `amxbench`, `gdnbench` (tr-kernels bins) | single-core kernel probes |

| `config` | print the merged configuration, `--for CMD` the values a command will use, `--keys` the schema |

`tr-infer <command> --help` lists every flag and the configuration key it writes; `--print-config`
shows what a command would run with, without running it.

## Repository layout

```
crates/tr-format   pack v2 contract: manifest schema, TQ block codec, reference dequant
crates/tr-sys      topology, per-tile arenas (MPOL_BIND + THP), pinned SPMD pool and barriers, O_DIRECT loader, AMX enablement
crates/tr-kernels  kernels: TQ GEMV/GEMM (VNNI), AMX-BF16 GEMM, GDN recurrence, conv, attention, QSA indexer, rope, router, PLE hash, quantisation
crates/tr-cache    persistent prefix cache: role-tagged chunks, prefix chain hashes, object format, Backend trait (fs, memory), LRU index
crates/tr-model    the qwen4exp graph: weights per tile, mailboxes, decode program, batched prefill, state, sampler, tokenizer
crates/tr-infer    CLI, layered configuration (config.rs) and the HTTP server (crates/tr-infer/src/server/)
python/trpack      offline packer (numpy + gguf-py) incl. `trpack mtp` (draft-head overlay), `trpack vision` (encoder overlay), tests
tools/oracle       llama.cpp comparison scripts (per-layer diffs, PLE check)
docs/              this documentation and the notebook
```

## Constraints on this box

From the lab notes; they still apply. Do not take or kill port `:11434`; do not overwrite
`runQwen3.8-27B.sh`; do not pop the sglang AMX stash or reuse its venvs; do not export
`OMP_NUM_THREADS`/`OMP_PROC_BIND` in a parent shell; never stage GGUFs, packs or the prefix cache on
`/tmp` (tmpfs, no `O_DIRECT`; the cache goes under `/opt/llm/tr-infer/`); never `swapoff` or `drop_caches`; benchmarks need the box to themselves. A 16 GiB tile
that overflows is a swap event even when other tiles have room.

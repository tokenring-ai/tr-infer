# Configuration

Every setting `tr-infer` has can come from four places. Later ones win:

1. **built-in defaults** — the values the flags used to carry, listed in the table below and in
   `crates/tr-infer/src/config.rs` (`defaults`);
2. **YAML file(s)** — `--config FILE` (repeatable, later files win), else `$TR_CONFIG` (a
   `:`-separated list), else the search path
   `/etc/tr-infer/config.yaml`, `$XDG_CONFIG_HOME/tr-infer/config.yaml`, `./tr-infer.yaml`
   (each optional; all that exist are merged in that order);
3. **environment** — `TR_<SECTION>__<KEY>`, e.g. `TR_RUNTIME__CTX=60000`, `TR_MOE__MASS=0.8`,
   `TR_SERVER__PORT=8089`. One underscore after `TR`, **two** between path segments;
4. **command line** — every flag names the key it writes in its `--help` text.

Merging is per key, not per file: a file that only sets `runtime.ctx` leaves everything else alone.
`--no-config` skips step 2, `--no-env` skips step 3.

```bash
tr-infer config                  # the merged configuration, as YAML
tr-infer config --for serve      # ... plus the values `serve` will actually use
tr-infer config --keys           # every settable key, one per line
tr-infer serve --print-config …  # the same, for the command you were about to run, then exit
```

`--print-config` and `config` redact `server.api_key`.

## Files

[`docs/tr-infer.example.yaml`](tr-infer.example.yaml) is a fully annotated file. A working
server config is short:

```yaml
pack: /opt/llm/tr-infer/flashnext-v3
runtime:
  ctx: 60000
  ple_mmap: true
overlays:
  mtp: /opt/llm/tr-infer/flashnext-v3-mtp
spec:
  k: 2
moe:
  mass: 0.8
server:
  host: 0.0.0.0
  port: 8089
  api_key: SECRET
```

```bash
tr-infer serve -c /etc/tr-infer/config.yaml
```

An unknown key in a **file** is an error, with the nearest known key as a hint — a typo in a config
file is otherwise invisible:

```
Error: /etc/tr-infer/config.yaml: unknown configuration key(s)
  runtime.cxt  (did you mean runtime.ctx?)
```

Unknown `TR_*` variables are ignored instead, because the engine has its own debug knobs under the
same prefix (`TR_AMX`, `TR_DUMP_VEC`, `TR_PROFILE`, `TR_VKB`, …; see
[ARCHITECTURE.md](ARCHITECTURE.md)).

## Booleans on the command line

Flags that are booleans take an optional value, so a `true` in a file stays overridable:

```bash
tr-infer bench --ple-mmap          # true
tr-infer bench --ple-mmap=false    # false, even if the file says true
```

Omitting the flag means "not given" and leaves the file/environment value alone. The value must be
attached with `=`.

## Keys

Two defaults depend on the command, so that one entry in a file still means "for whatever I run":

| key | default | except |
|---|---|---|
| `runtime.ctx` | 4096 | 8192 for `serve`, 1024 for `vision-embed` |
| `sampling.temp` | 0.0 (`generate`) | 0.6 for `serve` |

### Shared

| key | flag | default | meaning |
|---|---|---|---|
| `pack` | `--pack` | — | pack directory produced by `trpack`; required by every command that loads a model |
| `runtime.ctx` | `--ctx` | 4096 / 8192 | context window, prompt + completion; the KV cache is reserved for it |
| `runtime.batch` | `--batch` | 256 | prompt tokens per batched step; 1 disables the batch path |
| `runtime.kv` | `--kv` | `f16` | K/V cache element type: `f16`, `bf16`, `f32` |
| `runtime.cores_per_node` | `--cores-per-node` | all physical cores | cores used per NUMA tile |
| `runtime.ple_mmap` | `--ple-mmap` | `false` | keep the PLE table file-backed instead of resident in HBM (−3.4 GiB/tile) |
| `runtime.pin_heap` | `--pin-heap` | `mt` | glibc heap policy: `t` no trim, `m` no mmap, `0` off. `TR_PIN_HEAP` is read before anything else allocates and is the reliable way to set it |
| `overlays.mtp` | `--mtp` | off | MTP draft-head overlay pack: speculative decoding |
| `overlays.vision` | `--vision` | off | vision encoder overlay pack: image input |
| `spec.k` | `--spec-k` | 3 | draft tokens per round with an MTP overlay (1–7) |
| `moe.mass` | `--moe-mass` | off | cumulative-mass expert routing target |
| `moe.min`, `moe.max` | `--moe-min`, `--moe-max` | 1, `expert_used_count` | expert count bounds (max ≤ 32) |
| `moe.basis` | `--moe-basis` | `top` | what the mass is measured against: `top` or `all` |
| `moe.profile` | `--moe-profile` | `false` | report the mean cumulative router mass of the best k experts |
| `sampling.temp`, `.top_k`, `.top_p` | `--temp`, `--top-k`, `--top-p` | 0.0 / 0.6, 20, 0.95 | sampling |
| `sampling.seed` | `--seed` | 42 | `generate` only; the server seeds each request from the clock unless the request sets `seed` |
| `sampling.think_temp`, `.think_top_k`, `.think_top_p` | `--think-*` | unset | `serve` only: sampling inside the `<think>` block (unset = same as the answer) |
| `image.min_tokens`, `.max_tokens` | `--image-min-tokens`, `--image-max-tokens` | 8, 4096 | image size limits in tokens of 32×32 pixels |

### `serve`

| key | flag | default |
|---|---|---|
| `server.host`, `server.port` | `--host`, `--port` | `127.0.0.1`, 8080 |
| `server.api_key` | `--api-key`, env `TR_API_KEY` | none (required unless `server.no_auth`) |
| `server.no_auth` | `--no-auth` | `false` |
| `server.model_name` | `--model-name` | the pack directory name |
| `server.reasoning` | `--reasoning` | `xhigh` |
| `cache.dir` | `--cache-dir` | unset = no prefix cache; a directory on a real disk (never tmpfs), see [SERVER.md](SERVER.md#prefix-cache) |
| `cache.backend` | `--cache-backend` | `fs` (the only one) |
| `cache.chunk_len` | `--cache-chunk-len` | 256 tokens per rows chunk at most; chunks also end at every role boundary |
| `cache.max_gib` | `--cache-max-gib` | 32 (LRU byte budget of the store) |
| `cache.snapshots` | `--cache-snapshots` | `message`: a recurrent-state snapshot at the end of every prompt span of at least `snapshot_min_tokens` and at the end of each request; `request`: request ends only |
| `cache.snapshot_min_tokens` | `--cache-snapshot-min-tokens` | 16 |
| `cache.staging_mib` | `--cache-staging-mib` | 1024 MiB of host memory for objects waiting to be written |
| `cache.roles.<role>.persist` | `--cache-role ROLE=persist:BOOL` | `true`; `false` stores no chunks of that role — and nothing after them, since the chain breaks there |
| `cache.roles.<role>.priority` | `--cache-role ROLE=priority:N` | 5; eviction order 0..=9, lower priorities are evicted first (LRU within a priority) |
| `cache.roles.<role>.ttl` | `--cache-role ROLE=ttl:SECONDS` | none; objects of the role idle for longer are dropped, budget or not (0 = none) |
| `cache.roles.<role>.snapshot` | `--cache-role ROLE=snapshot:BOOL` | `true`; whether a prompt span of the role ends in a snapshot |

`<role>` is `system`, `user`, `reasoning`, `tool` or `assistant`; a snapshot takes the role of the
span that ends at it. `--cache-role` is repeatable and takes several keys at once:
`--cache-role reasoning=persist:false --cache-role tool=ttl:3600,priority:1`; in the environment,
`TR_CACHE__ROLES__REASONING__PERSIST=false`.

`TR_API_KEY` is still read directly, so it keeps beating a key set in a file and loses to
`--api-key`, as before. `TR_SERVER__API_KEY` works too.

### `generate`

| key | flag | default |
|---|---|---|
| `generate.prompt` | `-p`, `--prompt` | `The capital of France is` |
| `generate.prompt_file` | `--prompt-file` | unset |
| `generate.n_predict` | `-n`, `--n-predict` | 32 |
| `generate.chat` | `--chat` | `false` |
| `generate.no_think` | `--no-think` | `false` |
| `generate.dump` | `--dump` | `false` |
| `generate.images` | `--image` (repeatable) | none |

### `bench` and the diagnostics

| key | flag | default |
|---|---|---|
| `bench.pp`, `bench.tg`, `bench.reps` | `--pp`, `--tg`, `--reps` | 64, 64, 2 |
| `bench.tg_batch` | `--tg-batch` | unset |
| `spec_probe.prompt`, `.chat`, `.n_predict` | `-p`, `--chat`, `-n` | prompt as `generate`, `false`, 64 |
| `tokenize.prompt`, `.prompt_file` | `-p`, `--prompt-file` | empty |
| `vision_embed.image`, `.out`, `.reps` | `--image`, `--out`, `--reps` | required, required, 1 |
| `numa_smoke.mib` | `--mib` | 1024 |
| `load_bench.dir`, `.gib_per_node`, `.keep` | positional, `--gib-per-node`, `--keep` | required, 2.0, `false` |
| `barrier_bench.rounds` | `--rounds` | 100000 |

## Environment examples

```bash
export TR_PACK=/opt/llm/tr-infer/flashnext-v3
export TR_RUNTIME__CTX=60000
export TR_RUNTIME__PLE_MMAP=true
export TR_OVERLAYS__MTP=/opt/llm/tr-infer/flashnext-v3-mtp
export TR_SPEC__K=2
export TR_MOE__MASS=0.8
export TR_SERVER__PORT=8089
export TR_API_KEY=SECRET
export TR_CACHE__DIR=/opt/llm/tr-infer/kvcache
tr-infer serve
```

Lists use commas: `TR_GENERATE__IMAGES=a.png,b.jpg`.

### Continuous batching

The following server keys also accept flags of the same spelling with hyphens, or environment
variables such as `TR_SERVER__CONTINUOUS=true`:

```yaml
server:
  continuous: true       # default false; text only, without vision/MTP
  max_sequences: 16     # default 16; each n choice consumes one
  kv_cache_mib: 1024     # required when continuous=true; shared KV/QSA MiB per tile
  prefill_chunk: 32     # default 32; prompt rows per iteration during decode
  max_queue: 128        # default 128; waiting requests
runtime:
  batch: 256           # must exceed max_sequences
```

The KV pool budget excludes recurrent state and workspaces. See [SERVER.md](SERVER.md#continuous-batching-and-multiple-choices)
for admission, multi-choice sampling, memory accounting, and streaming behavior.

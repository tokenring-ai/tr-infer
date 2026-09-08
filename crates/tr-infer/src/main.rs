mod config;
mod server;

use crate::config::{defaults, AppConfig, Ov};
use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;
use tr_sys::topology::Topology;

const ABOUT: &str = "TokenRing Infer (Rust): NUMA-first CPU engine for Qwen3.8-Flash-Next";
const LONG_ABOUT: &str = "TokenRing Infer (Rust): NUMA-first CPU engine for Qwen3.8-Flash-Next.

Every setting can come from a YAML file, the environment or the command line; later wins:
  1. built-in defaults
  2. --config FILE (repeatable), else $TR_CONFIG, else /etc/tr-infer/config.yaml,
     $XDG_CONFIG_HOME/tr-infer/config.yaml, ./tr-infer.yaml
  3. TR_<SECTION>__<KEY> environment variables (TR_RUNTIME__CTX=60000, TR_MOE__MASS=0.8)
  4. flags

`tr-infer config` prints the merged configuration, `config --for serve` the values a command
will actually use. Each flag's help names the config key it writes.";

#[derive(Parser)]
#[command(name = "tr-infer", about = ABOUT, long_about = LONG_ABOUT)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
    #[command(flatten)]
    global: GlobalArgs,
}

/// Where configuration comes from. Valid before or after the subcommand.
#[derive(Args, Clone, Default)]
#[command(next_help_heading = "Configuration sources")]
struct GlobalArgs {
    /// YAML configuration file; repeatable, later files win. Replaces the default search path
    #[arg(long = "config", short = 'c', global = true, value_name = "FILE")]
    config: Vec<PathBuf>,
    /// Do not read any configuration file (not the search path, not $TR_CONFIG)
    #[arg(long, global = true)]
    no_config: bool,
    /// Do not read TR_* environment variables as configuration
    #[arg(long, global = true)]
    no_env: bool,
    /// Print the merged configuration and exit without doing any work
    #[arg(long, global = true)]
    print_config: bool,
}

/// Pack directory.
#[derive(Args, Clone, Default)]
struct PackArgs {
    /// Pack directory produced by trpack  [config: pack]
    #[arg(long, value_name = "DIR")]
    pack: Option<PathBuf>,
}
impl PackArgs {
    fn apply(&self, o: &mut Ov) {
        o.set("pack", &self.pack);
    }
}

/// Model-load settings shared by every command that loads a model  [config section: runtime].
#[derive(Args, Clone, Default)]
struct RuntimeArgs {
    /// Context window, prompt + completion  [config: runtime.ctx; default 4096, serve 8192]
    #[arg(long, value_name = "N")]
    ctx: Option<usize>,
    /// Prompt batch size in tokens per step, 1 = token at a time  [config: runtime.batch; default 256]
    #[arg(long, value_name = "N")]
    batch: Option<usize>,
    /// K/V cache element type: f16 (as llama.cpp), bf16, f32  [config: runtime.kv; default f16]
    #[arg(long, value_name = "TYPE")]
    kv: Option<String>,
    /// Cores used per NUMA tile  [config: runtime.cores_per_node; default: every physical core]
    #[arg(long, value_name = "N")]
    cores_per_node: Option<usize>,
    /// Keep the PLE table file-backed (page cache) instead of resident in HBM, -3.4 GiB/tile
    /// [config: runtime.ple_mmap]
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL")]
    ple_mmap: Option<bool>,
    /// glibc heap policy: t = no trim, m = no mmap, 0 = off  [config: runtime.pin_heap; default mt]
    #[arg(long, value_name = "MODE")]
    pin_heap: Option<String>,
}
impl RuntimeArgs {
    fn apply(&self, o: &mut Ov) {
        o.set("runtime.ctx", &self.ctx);
        o.set("runtime.batch", &self.batch);
        o.set("runtime.kv", &self.kv);
        o.set("runtime.cores_per_node", &self.cores_per_node);
        o.set("runtime.ple_mmap", &self.ple_mmap);
        o.set("runtime.pin_heap", &self.pin_heap);
    }
}

/// Overlay packs and speculative decoding  [config sections: overlays, spec].
#[derive(Args, Clone, Default)]
struct OverlayArgs {
    /// MTP overlay pack (trpack mtp): speculative decoding with --spec-k drafts per round
    /// [config: overlays.mtp]
    #[arg(long, value_name = "DIR")]
    mtp: Option<PathBuf>,
    /// Draft tokens per round, 1..7  [config: spec.k; default 3]
    #[arg(long, value_name = "N")]
    spec_k: Option<usize>,
    /// Vision encoder overlay pack (trpack vision): image input  [config: overlays.vision]
    #[arg(long, value_name = "DIR")]
    vision: Option<PathBuf>,
}
impl OverlayArgs {
    fn apply(&self, o: &mut Ov) {
        o.set("overlays.mtp", &self.mtp);
        o.set("overlays.vision", &self.vision);
        o.set("spec.k", &self.spec_k);
    }
}

/// Cumulative-mass expert routing (off = the model's fixed top-k)  [config section: moe].
#[derive(Args, Clone, Default)]
struct MoeArgs {
    /// Take experts in router order until their softmax mass reaches this target, e.g. 0.95
    /// [config: moe.mass]
    #[arg(long, value_name = "M")]
    moe_mass: Option<f32>,
    /// Fewest experts per token with --moe-mass  [config: moe.min; default 1]
    #[arg(long, value_name = "N")]
    moe_min: Option<usize>,
    /// Most experts per token with --moe-mass  [config: moe.max; default: expert_used_count, limit 32]
    #[arg(long, value_name = "N")]
    moe_max: Option<usize>,
    /// What the target mass is measured against: "top" = the --moe-max best experts (their
    /// renormalised weights), "all" = the softmax over all experts (this model's top-10 hold
    /// only ~15 % of it)  [config: moe.basis; default top]
    #[arg(long, value_name = "BASIS")]
    moe_basis: Option<String>,
    /// Report the mean cumulative router mass of the best k experts (k = 1..32)  [config: moe.profile]
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL")]
    moe_profile: Option<bool>,
}
impl MoeArgs {
    fn apply(&self, o: &mut Ov) {
        o.set_f32("moe.mass", &self.moe_mass);
        o.set("moe.min", &self.moe_min);
        o.set("moe.max", &self.moe_max);
        o.set("moe.basis", &self.moe_basis);
        o.set("moe.profile", &self.moe_profile);
    }
}

/// Sampling  [config section: sampling].
#[derive(Args, Clone, Default)]
struct SampleArgs {
    /// Sampling temperature  [config: sampling.temp; default 0 for generate, 0.6 for serve]
    #[arg(long, value_name = "T")]
    temp: Option<f32>,
    /// Top-k  [config: sampling.top_k; default 20]
    #[arg(long, value_name = "N")]
    top_k: Option<usize>,
    /// Top-p  [config: sampling.top_p; default 0.95]
    #[arg(long, value_name = "P")]
    top_p: Option<f32>,
}
impl SampleArgs {
    fn apply(&self, o: &mut Ov) {
        o.set_f32("sampling.temp", &self.temp);
        o.set("sampling.top_k", &self.top_k);
        o.set_f32("sampling.top_p", &self.top_p);
    }
}

/// Sampling inside the `<think>` block when the request does not set reasoning_temperature etc.
/// (unset: same as the answer). E.g. --think-temp 1.0 to explore while thinking.
#[derive(Args, Clone, Default)]
struct ThinkArgs {
    /// [config: sampling.think_temp]
    #[arg(long, value_name = "T")]
    think_temp: Option<f32>,
    /// [config: sampling.think_top_p]
    #[arg(long, value_name = "P")]
    think_top_p: Option<f32>,
    /// [config: sampling.think_top_k]
    #[arg(long, value_name = "N")]
    think_top_k: Option<usize>,
}
impl ThinkArgs {
    fn apply(&self, o: &mut Ov) {
        o.set_f32("sampling.think_temp", &self.think_temp);
        o.set_f32("sampling.think_top_p", &self.think_top_p);
        o.set("sampling.think_top_k", &self.think_top_k);
    }
}

/// Image size limits in tokens (an image is resized into min..max tokens of 32x32 pixels)
/// [config section: image].
#[derive(Args, Clone, Default)]
struct ImageArgs {
    /// [config: image.min_tokens; default 8]
    #[arg(long, value_name = "N")]
    image_min_tokens: Option<usize>,
    /// [config: image.max_tokens; default 4096]
    #[arg(long, value_name = "N")]
    image_max_tokens: Option<usize>,
}
impl ImageArgs {
    fn apply(&self, o: &mut Ov) {
        o.set("image.min_tokens", &self.image_min_tokens);
        o.set("image.max_tokens", &self.image_max_tokens);
    }
}

/// Persistent prefix cache of the server.
#[derive(Args, Clone, Default)]
#[command(next_help_heading = "Prefix cache")]
struct CacheArgs {
    /// Store directory on a real disk (never tmpfs); unset = no cache  [config: cache.dir]
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<PathBuf>,
    /// Backend: fs  [config: cache.backend; default fs]
    #[arg(long, value_name = "NAME")]
    cache_backend: Option<String>,
    /// Longest rows chunk in tokens; chunks also end at role boundaries  [config: cache.chunk_len; default 256]
    #[arg(long, value_name = "N")]
    cache_chunk_len: Option<usize>,
    /// LRU byte budget in GiB  [config: cache.max_gib; default 32]
    #[arg(long, value_name = "GIB")]
    cache_max_gib: Option<f64>,
    /// Recurrent-state snapshots: message (every prompt span end + request end) or request
    /// [config: cache.snapshots; default message]
    #[arg(long, value_name = "POLICY")]
    cache_snapshots: Option<String>,
    /// Spans shorter than this get no snapshot  [config: cache.snapshot_min_tokens; default 16]
    #[arg(long, value_name = "N")]
    cache_snapshot_min_tokens: Option<usize>,
    /// Host memory for objects waiting to be written, MiB  [config: cache.staging_mib; default 1024]
    #[arg(long, value_name = "MIB")]
    cache_staging_mib: Option<usize>,
    /// Per-role policy, repeatable: ROLE=KEY:VALUE[,KEY:VALUE] with role system|user|reasoning|tool|assistant
    /// and keys persist (bool), priority (0-9), ttl (seconds, 0 = none), snapshot (bool); e.g.
    /// --cache-role reasoning=persist:false --cache-role tool=ttl:3600,priority:1  [config: cache.roles.ROLE.KEY]
    #[arg(long = "cache-role", value_name = "ROLE=K:V,..", value_parser = parse_cache_role)]
    cache_role: Vec<CacheRoleArg>,
}

/// One `--cache-role` value, parsed and typed.
#[derive(Clone, Debug)]
struct CacheRoleArg {
    role: String,
    pairs: Vec<(String, serde_json::Value)>,
}

fn parse_cache_role(s: &str) -> Result<CacheRoleArg, String> {
    let (role, rest) = s.split_once('=').ok_or("expected ROLE=KEY:VALUE[,KEY:VALUE]")?;
    let role = role.trim();
    if tr_cache::Role::parse(role).is_none() {
        return Err(format!("unknown role {role:?}: system, user, reasoning, tool or assistant"));
    }
    let mut pairs = Vec::new();
    for kv in rest.split(',').filter(|x| !x.trim().is_empty()) {
        let (k, v) = kv.split_once(':').ok_or_else(|| format!("{kv:?}: expected KEY:VALUE"))?;
        let (k, v) = (k.trim(), v.trim());
        let val = match k {
            "persist" | "snapshot" => serde_json::Value::Bool(v.parse::<bool>().map_err(|_| format!("{k} must be true or false"))?),
            "priority" => serde_json::Value::from(v.parse::<u8>().ok().filter(|&p| p <= tr_cache::PRIORITY_MAX).ok_or_else(|| format!("priority must be 0..={}", tr_cache::PRIORITY_MAX))?),
            "ttl" => serde_json::Value::from(v.parse::<u64>().map_err(|_| "ttl must be a number of seconds".to_string())?),
            other => return Err(format!("unknown key {other:?}: persist, priority, ttl or snapshot")),
        };
        pairs.push((k.to_string(), val));
    }
    if pairs.is_empty() {
        return Err("no KEY:VALUE given".into());
    }
    Ok(CacheRoleArg { role: role.to_string(), pairs })
}

impl CacheArgs {
    fn apply(&self, o: &mut Ov) {
        for r in &self.cache_role {
            for (k, v) in &r.pairs {
                o.set(&format!("cache.roles.{}.{k}", r.role), &Some(v.clone()));
            }
        }
        o.set("cache.dir", &self.cache_dir);
        o.set("cache.backend", &self.cache_backend);
        o.set("cache.chunk_len", &self.cache_chunk_len);
        o.set("cache.max_gib", &self.cache_max_gib);
        o.set("cache.snapshots", &self.cache_snapshots);
        o.set("cache.snapshot_min_tokens", &self.cache_snapshot_min_tokens);
        o.set("cache.staging_mib", &self.cache_staging_mib);
    }
}

#[derive(Subcommand)]
enum CacheOp {
    /// Object counts, bytes and ages of the store
    Stats,
    /// Delete every object in the store
    Clear,
}

#[derive(Subcommand)]
enum Cmd {
    /// Inspect or empty the server's prefix cache store (`--cache-dir` or cache.dir)
    Cache {
        #[command(flatten)]
        cache: CacheArgs,
        #[command(subcommand)]
        op: CacheOp,
    },
    /// Print the merged configuration (all sources) as YAML
    Config {
        /// Also print the settings this command will actually use, defaults filled in:
        /// generate, bench, serve, spec-probe, vision-embed, tokenize
        #[arg(long = "for", value_name = "COMMAND")]
        for_cmd: Option<String>,
        /// List every settable key instead, one per line (the schema itself)
        #[arg(long)]
        keys: bool,
    },
    /// Print NUMA topology
    Topology,
    /// Encode an image with the vision overlay and dump the embeddings (oracle file format)
    VisionEmbed {
        #[command(flatten)]
        pack: PackArgs,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        overlays: OverlayArgs,
        #[command(flatten)]
        images: ImageArgs,
        /// Image file to encode  [config: vision_embed.image]
        #[arg(long, value_name = "FILE")]
        image: Option<PathBuf>,
        /// Output file  [config: vision_embed.out]
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,
        /// Encode passes (timing)  [config: vision_embed.reps; default 1]
        #[arg(long, value_name = "N")]
        reps: Option<usize>,
    },
    /// Allocate memory on every tile, fill it from the tile's own cores, verify placement and measure bandwidth
    NumaSmoke {
        /// [config: numa_smoke.mib; default 1024]
        #[arg(long, value_name = "MIB")]
        mib: Option<usize>,
    },
    /// Write synthetic per-node blobs under DIR (real filesystem, not /tmp) and time loading them
    /// into node-bound memory with O_DIRECT from each node's own cores
    LoadBench {
        /// [config: load_bench.dir]
        dir: Option<PathBuf>,
        /// [config: load_bench.gib_per_node; default 2.0]
        #[arg(long, value_name = "GIB")]
        gib_per_node: Option<f64>,
        /// Skip writing if the files already exist with the right size  [config: load_bench.keep]
        #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL")]
        keep: Option<bool>,
    },
    /// Generate text from a pack
    Generate {
        #[command(flatten)]
        pack: PackArgs,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        overlays: OverlayArgs,
        #[command(flatten)]
        sampling: SampleArgs,
        #[command(flatten)]
        moe: MoeArgs,
        #[command(flatten)]
        images: ImageArgs,
        /// [config: generate.prompt; default "The capital of France is"]
        #[arg(short, long)]
        prompt: Option<String>,
        /// Read the prompt from a file instead of -p  [config: generate.prompt_file]
        #[arg(long, value_name = "FILE")]
        prompt_file: Option<PathBuf>,
        /// [config: generate.n_predict; default 32]
        #[arg(short = 'n', long, value_name = "N")]
        n_predict: Option<usize>,
        /// [config: sampling.seed; default 42]
        #[arg(long, value_name = "N")]
        seed: Option<u64>,
        /// Print per-layer activation statistics for the prompt tokens (oracle comparison)
        /// [config: generate.dump]
        #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL")]
        dump: Option<bool>,
        /// Wrap the prompt in the model's ChatML template (user turn + assistant generation prompt)
        /// [config: generate.chat]
        #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL")]
        chat: Option<bool>,
        /// With --chat: disable thinking (emits an empty <think> block)  [config: generate.no_think]
        #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL")]
        no_think: Option<bool>,
        /// Image file(s) placed before the prompt text in the user turn (needs --chat and --vision)
        /// [config: generate.images]
        #[arg(long, value_name = "FILE")]
        image: Vec<PathBuf>,
    },
    /// Benchmark prompt processing and generation (pp / tg tokens), exclusive box
    Bench {
        #[command(flatten)]
        pack: PackArgs,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        overlays: OverlayArgs,
        #[command(flatten)]
        moe: MoeArgs,
        /// Prompt tokens per rep  [config: bench.pp; default 64]
        #[arg(long, value_name = "N")]
        pp: Option<usize>,
        /// Generated tokens per rep  [config: bench.tg; default 64]
        #[arg(long, value_name = "N")]
        tg: Option<usize>,
        /// [config: bench.reps; default 2]
        #[arg(long, value_name = "N")]
        reps: Option<usize>,
        /// Decode M identical rows per step through the batched path (speculative-verify cost probe)
        /// [config: bench.tg_batch]
        #[arg(long, value_name = "M")]
        tg_batch: Option<usize>,
    },
    /// Serve an OpenAI-compatible API (/v1/chat/completions, /v1/models) from a pack
    Serve {
        #[command(flatten)]
        pack: PackArgs,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        overlays: OverlayArgs,
        #[command(flatten)]
        sampling: SampleArgs,
        #[command(flatten)]
        think: ThinkArgs,
        #[command(flatten)]
        moe: MoeArgs,
        #[command(flatten)]
        images: ImageArgs,
        /// Text-only continuous batching [config: server.continuous; default false]
        #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true")]
        continuous: Option<bool>,
        /// Maximum active choices [config: server.max_sequences; default 16]
        #[arg(long)]
        max_sequences: Option<usize>,
        /// KV pool MiB per tile, required with --continuous [config: server.kv_cache_mib]
        #[arg(long)]
        kv_cache_mib: Option<usize>,
        /// Prompt rows per iteration during decode [config: server.prefill_chunk; default 32]
        #[arg(long)]
        prefill_chunk: Option<usize>,
        /// Maximum waiting requests [config: server.max_queue; default 128]
        #[arg(long)]
        max_queue: Option<usize>,
        /// [config: server.host; default 127.0.0.1]
        #[arg(long)]
        host: Option<String>,
        /// [config: server.port; default 8080]
        #[arg(long)]
        port: Option<u16>,
        /// Bearer token clients must send; required unless --no-auth  [config: server.api_key]
        #[arg(long, env = "TR_API_KEY")]
        api_key: Option<String>,
        /// Serve without an API key (only sensible on a private interface)  [config: server.no_auth]
        #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL")]
        no_auth: Option<bool>,
        /// Model id reported by /v1/models  [config: server.model_name; default: the pack directory name]
        #[arg(long)]
        model_name: Option<String>,
        /// Default reasoning effort when the request does not set one: xhigh (template default),
        /// medium, low, none  [config: server.reasoning; default xhigh]
        #[arg(long)]
        reasoning: Option<String>,
        #[command(flatten)]
        cache: CacheArgs,
    },
    /// MTP draft-head probe: greedy decode with the main model and report how often the draft
    /// head's prediction (first draft and chained second draft) matches it
    SpecProbe {
        #[command(flatten)]
        pack: PackArgs,
        #[command(flatten)]
        runtime: RuntimeArgs,
        #[command(flatten)]
        overlays: OverlayArgs,
        /// [config: spec_probe.prompt; default "The capital of France is"]
        #[arg(short, long)]
        prompt: Option<String>,
        /// [config: spec_probe.chat]
        #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "true", value_name = "BOOL")]
        chat: Option<bool>,
        /// [config: spec_probe.n_predict; default 64]
        #[arg(short = 'n', long, value_name = "N")]
        n_predict: Option<usize>,
    },
    /// Print the token ids of a prompt (pack tokenizer only, no model load)
    Tokenize {
        #[command(flatten)]
        pack: PackArgs,
        /// [config: tokenize.prompt]
        #[arg(short, long)]
        prompt: Option<String>,
        /// [config: tokenize.prompt_file]
        #[arg(long, value_name = "FILE")]
        prompt_file: Option<PathBuf>,
    },
    /// Measure barrier round-trip latency across all workers
    BarrierBench {
        /// [config: barrier_bench.rounds; default 100000]
        #[arg(long, value_name = "N")]
        rounds: Option<usize>,
        /// [config: runtime.cores_per_node]
        #[arg(long, value_name = "N")]
        cores_per_node: Option<usize>,
    },
}

impl Cmd {
    /// The name `config --for` uses, for commands that have settings of their own.
    fn name(&self) -> Option<&'static str> {
        match self {
            Cmd::Generate { .. } => Some("generate"),
            Cmd::Bench { .. } => Some("bench"),
            Cmd::Serve { .. } => Some("serve"),
            Cmd::SpecProbe { .. } => Some("spec-probe"),
            Cmd::VisionEmbed { .. } => Some("vision-embed"),
            Cmd::Tokenize { .. } => Some("tokenize"),
            Cmd::Cache { .. } => Some("cache"),
            _ => None,
        }
    }

    /// The flags the user actually gave, as the highest-priority configuration source.
    fn overrides(&self) -> Ov {
        let mut o = Ov::default();
        match self {
            Cmd::Config { .. } | Cmd::Topology => {}
            Cmd::Cache { cache, .. } => cache.apply(&mut o),
            Cmd::VisionEmbed { pack, runtime, overlays, images, image, out, reps } => {
                pack.apply(&mut o);
                runtime.apply(&mut o);
                overlays.apply(&mut o);
                images.apply(&mut o);
                o.set("vision_embed.image", image);
                o.set("vision_embed.out", out);
                o.set("vision_embed.reps", reps);
            }
            Cmd::NumaSmoke { mib } => o.set("numa_smoke.mib", mib),
            Cmd::LoadBench { dir, gib_per_node, keep } => {
                o.set("load_bench.dir", dir);
                o.set("load_bench.gib_per_node", gib_per_node);
                o.set("load_bench.keep", keep);
            }
            Cmd::Generate { pack, runtime, overlays, sampling, moe, images, prompt, prompt_file, n_predict, seed, dump, chat, no_think, image } => {
                pack.apply(&mut o);
                runtime.apply(&mut o);
                overlays.apply(&mut o);
                sampling.apply(&mut o);
                moe.apply(&mut o);
                images.apply(&mut o);
                o.set("generate.prompt", prompt);
                o.set("generate.prompt_file", prompt_file);
                o.set("generate.n_predict", n_predict);
                o.set("sampling.seed", seed);
                o.set("generate.dump", dump);
                o.set("generate.chat", chat);
                o.set("generate.no_think", no_think);
                o.set_list("generate.images", image);
            }
            Cmd::Bench { pack, runtime, overlays, moe, pp, tg, reps, tg_batch } => {
                pack.apply(&mut o);
                runtime.apply(&mut o);
                overlays.apply(&mut o);
                moe.apply(&mut o);
                o.set("bench.pp", pp);
                o.set("bench.tg", tg);
                o.set("bench.reps", reps);
                o.set("bench.tg_batch", tg_batch);
            }
            Cmd::Serve { pack, runtime, overlays, sampling, think, moe, images, cache, host, port, api_key, no_auth, model_name, reasoning, continuous, max_sequences, kv_cache_mib, prefill_chunk, max_queue } => {
                pack.apply(&mut o);
                runtime.apply(&mut o);
                overlays.apply(&mut o);
                sampling.apply(&mut o);
                think.apply(&mut o);
                moe.apply(&mut o);
                images.apply(&mut o);
                cache.apply(&mut o);
                o.set("server.continuous", continuous);
                o.set("server.max_sequences", max_sequences);
                o.set("server.kv_cache_mib", kv_cache_mib);
                o.set("server.prefill_chunk", prefill_chunk);
                o.set("server.max_queue", max_queue);
                o.set("server.host", host);
                o.set("server.port", port);
                o.set("server.api_key", api_key);
                o.set("server.no_auth", no_auth);
                o.set("server.model_name", model_name);
                o.set("server.reasoning", reasoning);
            }
            Cmd::SpecProbe { pack, runtime, overlays, prompt, chat, n_predict } => {
                pack.apply(&mut o);
                runtime.apply(&mut o);
                overlays.apply(&mut o);
                o.set("spec_probe.prompt", prompt);
                o.set("spec_probe.chat", chat);
                o.set("spec_probe.n_predict", n_predict);
            }
            Cmd::Tokenize { pack, prompt, prompt_file } => {
                pack.apply(&mut o);
                o.set("tokenize.prompt", prompt);
                o.set("tokenize.prompt_file", prompt_file);
            }
            Cmd::BarrierBench { rounds, cores_per_node } => {
                o.set("barrier_bench.rounds", rounds);
                o.set("runtime.cores_per_node", cores_per_node);
            }
        }
        o
    }
}

fn main() -> Result<()> {
    // Heap policy: large allocations (logits, per-batch vectors) must not come from fresh mmaps and
    // the heap must not be trimmed, or every batch page-faults on nodes with no free memory and the
    // process starts swapping (22 K major faults in the first batch, 30-50 % slower runs at random).
    // TR_PIN_HEAP: letters t (no trim), m (no mmap for large allocations); default "mt", "0" = off.
    // Read before anything else allocates; `runtime.pin_heap` can still change it below.
    let early = std::env::var("TR_PIN_HEAP").unwrap_or_else(|_| defaults::PIN_HEAP.to_string());
    if early != "0" {
        tr_sys::procinfo::pin_heap(&early);
    }
    let cli = Cli::parse();
    let g = &cli.global;
    let (cfg, prov) = config::load(&g.config, g.no_config, g.no_env, cli.cmd.overrides().into_value())?;
    let want = cfg.pin_heap();
    if want != early && want != "0" {
        tr_sys::procinfo::pin_heap(&want);
    }
    if g.print_config {
        print_config(&cfg, &prov, cli.cmd.name())?;
        return Ok(());
    }
    if !prov.files.is_empty() && !matches!(cli.cmd, Cmd::Config { .. }) {
        eprintln!("{}", prov.summary());
    }
    match &cli.cmd {
        Cmd::Config { for_cmd, keys } => {
            if *keys {
                for k in config::known_keys() {
                    println!("{k}");
                }
            } else {
                print_config(&cfg, &prov, for_cmd.as_deref())?;
            }
        }
        Cmd::Topology => {
            let t = Topology::discover()?;
            print!("{}", t.summary());
            println!("nodes {}  physical cores {}  AMX {}", t.n_nodes(), t.n_physical(), tr_sys::amx::enable_amx());
        }
        Cmd::VisionEmbed { .. } => cmd_vision_embed(&cfg)?,
        Cmd::NumaSmoke { .. } => numa_smoke(cfg.numa_smoke.mib.unwrap_or(defaults::NUMA_SMOKE_MIB))?,
        Cmd::LoadBench { .. } => {
            let dir = cfg.load_bench.dir.clone().ok_or_else(|| anyhow::anyhow!("no directory: pass it as the argument or set load_bench.dir"))?;
            load_bench(&dir, cfg.load_bench.gib_per_node.unwrap_or(defaults::LOAD_BENCH_GIB), cfg.load_bench.keep.unwrap_or(false))?
        }
        Cmd::BarrierBench { .. } => barrier_bench(cfg.barrier_bench.rounds.unwrap_or(defaults::BARRIER_ROUNDS), cfg.runtime.cores_per_node)?,
        Cmd::Tokenize { .. } => cmd_tokenize(&cfg)?,
        Cmd::SpecProbe { .. } => cmd_spec_probe(&cfg)?,
        Cmd::Bench { .. } => cmd_bench(&cfg)?,
        Cmd::Serve { .. } => cmd_serve(&cfg)?,
        Cmd::Cache { op, .. } => cmd_cache(&cfg, op)?,
        Cmd::Generate { .. } => cmd_generate(&cfg)?,
    }
    Ok(())
}

/// Replace the API key with a placeholder: `--print-config` output ends up in logs and issues.
fn redact(mut v: serde_json::Value) -> serde_json::Value {
    if let Some(k) = v.pointer_mut("/server/api_key") {
        if !k.is_null() {
            *k = serde_json::Value::String("<set, redacted>".into());
        }
    }
    v
}

/// `--print-config` and `tr-infer config`: the merged tree, plus the resolved view of one command.
fn print_config(cfg: &AppConfig, prov: &config::Provenance, for_cmd: Option<&str>) -> Result<()> {
    println!("# {}", prov.summary());
    println!("# merged configuration (only what a source actually set)");
    print!("{}", config::to_yaml(&redact(serde_json::to_value(cfg)?), true));
    let Some(name) = for_cmd else { return Ok(()) };
    let v = match name {
        "generate" => serde_json::json!({"runtime": cfg.runtime(defaults::CTX), "sampling": cfg.sampling(defaults::TEMP), "moe": cfg.moe(), "generate": cfg.generate()}),
        "bench" => serde_json::json!({"runtime": cfg.runtime(defaults::CTX), "moe": cfg.moe(), "bench": cfg.bench()}),
        "serve" => serde_json::json!({"runtime": cfg.runtime(defaults::CTX_SERVE), "sampling": cfg.sampling(defaults::TEMP_SERVE), "moe": cfg.moe(), "server": cfg.server(), "cache": cfg.cache()}),
        "cache" => serde_json::json!({"cache": cfg.cache()}),
        "spec-probe" | "spec_probe" => serde_json::json!({"runtime": cfg.runtime(defaults::CTX), "spec_probe": {"prompt": cfg.spec_probe.prompt.clone().unwrap_or_else(|| defaults::PROMPT.into()), "chat": cfg.spec_probe.chat.unwrap_or(false), "n_predict": cfg.spec_probe.n_predict.unwrap_or(defaults::N_PREDICT_PROBE)}}),
        "vision-embed" | "vision_embed" => serde_json::json!({"runtime": cfg.runtime(defaults::CTX_VISION_EMBED), "vision_embed": {"image": cfg.vision_embed.image, "out": cfg.vision_embed.out, "reps": cfg.vision_embed.reps.unwrap_or(defaults::REPS)}}),
        "tokenize" => serde_json::json!({"pack": cfg.pack, "tokenize": {"prompt": cfg.tokenize.prompt.clone().unwrap_or_default(), "prompt_file": cfg.tokenize.prompt_file}}),
        other => anyhow::bail!("--for must name a command that has settings: generate, bench, serve, cache, spec-probe, vision-embed, tokenize (got {other})"),
    };
    println!("\n# effective settings for `{name}` (defaults filled in)");
    print!("{}", config::to_yaml(&redact(v), false));
    Ok(())
}

fn cmd_tokenize(cfg: &AppConfig) -> Result<()> {
    let pack = cfg.runtime(defaults::CTX);
    let pack = pack.pack()?;
    let text = match &cfg.tokenize.prompt_file {
        Some(p) => std::fs::read_to_string(p)?,
        None => cfg.tokenize.prompt.clone().unwrap_or_default(),
    };
    let m = tr_format::Manifest::load(pack)?;
    let tok = tr_model::tokenizer::Tok::load(&pack.join(&m.tokenizer))?;
    let ids = tok.encode(&text, false)?;
    for id in &ids {
        println!("{id}");
    }
    eprintln!("{} tokens", ids.len());
    Ok(())
}

fn cmd_vision_embed(cfg: &AppConfig) -> Result<()> {
    let rt = cfg.runtime(defaults::CTX_VISION_EMBED);
    let vision = rt.vision.clone().ok_or_else(|| anyhow::anyhow!("vision-embed needs --vision (config: overlays.vision)"))?;
    let image = cfg.vision_embed.image.clone().ok_or_else(|| anyhow::anyhow!("vision-embed needs --image (config: vision_embed.image)"))?;
    let out = cfg.vision_embed.out.clone().ok_or_else(|| anyhow::anyhow!("vision-embed needs --out (config: vision_embed.out)"))?;
    vision_embed(&rt, &vision, &image, &out, cfg.vision_embed.reps.unwrap_or(defaults::REPS))
}

fn cmd_spec_probe(cfg: &AppConfig) -> Result<()> {
    let rt = cfg.runtime(defaults::CTX);
    let mtp = rt.mtp.clone().ok_or_else(|| anyhow::anyhow!("spec-probe needs --mtp (config: overlays.mtp)"))?;
    spec_probe(&rt, &mtp, cfg.spec_probe.prompt.as_deref().unwrap_or(defaults::PROMPT), cfg.spec_probe.chat.unwrap_or(false), cfg.spec_probe.n_predict.unwrap_or(defaults::N_PREDICT_PROBE))
}

fn cmd_bench(cfg: &AppConfig) -> Result<()> {
    bench(&cfg.runtime(defaults::CTX), &cfg.bench(), &cfg.moe())
}

fn cmd_serve(cfg: &AppConfig) -> Result<()> {
    let rt = cfg.runtime(defaults::CTX_SERVE);
    let sv = cfg.server();
    let sm = cfg.sampling(defaults::TEMP_SERVE);
    anyhow::ensure!(sv.api_key.is_some() || sv.no_auth, "set --api-key (or TR_API_KEY, or server.api_key), or pass --no-auth");
    let reasoning = server::chat::Thinking::parse(&sv.reasoning).ok_or_else(|| anyhow::anyhow!("reasoning must be xhigh, medium, low or none (got {})", sv.reasoning))?;
    let cache = cache_opts(cfg)?;
    server::serve(server::ServeOpts {
        continuous: sv.continuous,
        max_sequences: sv.max_sequences,
        kv_cache_mib: sv.kv_cache_mib,
        prefill_chunk: sv.prefill_chunk,
        max_queue: sv.max_queue,
        pack: rt.pack()?.to_path_buf(),
        host: sv.host,
        port: sv.port,
        api_key: sv.api_key,
        model_name: sv.model_name,
        ctx: rt.ctx,
        batch: rt.batch,
        cores_per_node: rt.cores_per_node,
        ple_mmap: rt.ple_mmap,
        kv: rt.kv()?,
        temp: sm.temp,
        top_p: sm.top_p,
        top_k: sm.top_k,
        think_temp: sm.think_temp,
        think_top_p: sm.think_top_p,
        think_top_k: sm.think_top_k,
        reasoning,
        mtp: rt.mtp.clone(),
        spec_k: rt.spec_k,
        vision: rt.vision.clone(),
        image_min_tokens: rt.image_min_tokens,
        image_max_tokens: rt.image_max_tokens,
        moe: cfg.moe().kernel()?,
        cache,
    })
}

/// The prefix cache settings, None when no directory is configured.
fn cache_opts(cfg: &AppConfig) -> Result<Option<server::prefix_cache::CacheOpts>> {
    let c = cfg.cache();
    c.validate()?;
    let Some(dir) = c.dir else { return Ok(None) };
    let snapshots = server::prefix_cache::SnapshotPolicy::parse(&c.snapshots).ok_or_else(|| anyhow::anyhow!("cache.snapshots must be message or request (got {})", c.snapshots))?;
    anyhow::ensure!(c.max_gib > 0.0, "cache.max_gib must be positive");
    anyhow::ensure!(c.chunk_len > 0, "cache.chunk_len must be positive");
    Ok(Some(server::prefix_cache::CacheOpts {
        dir,
        backend: c.backend,
        chunk_len: c.chunk_len,
        max_bytes: (c.max_gib * (1u64 << 30) as f64) as u64,
        snapshots,
        snapshot_min_tokens: c.snapshot_min_tokens,
        staging_bytes: (c.staging_mib as u64) << 20,
        roles: c.roles,
    }))
}

/// `tr-infer cache stats|clear`: works on the object headers alone (no model needed).
fn cmd_cache(cfg: &AppConfig, op: &CacheOp) -> Result<()> {
    use tr_cache::{Backend, FsBackend, Kind};
    let c = cfg.cache();
    let dir = c.dir.ok_or_else(|| anyhow::anyhow!("no store: pass --cache-dir DIR or set cache.dir"))?;
    anyhow::ensure!(c.backend == "fs", "only the fs backend exists (got {})", c.backend);
    let mut be = FsBackend::open(&dir)?;
    let metas = be.list()?;
    match op {
        CacheOp::Stats => {
            let mut by_root: std::collections::BTreeMap<String, (usize, usize, u64, u64, u64)> = Default::default();
            for m in &metas {
                let e = by_root.entry(m.root.hex()).or_insert((0, 0, 0, u64::MAX, 0));
                match m.kind {
                    Kind::Rows => e.0 += 1,
                    Kind::Snapshot => e.1 += 1,
                }
                e.2 += m.footprint();
                e.3 = e.3.min(m.last_used);
                e.4 = e.4.max(m.last_used);
            }
            let total: u64 = metas.iter().map(|m| m.footprint()).sum();
            println!("{}: {} objects, {:.2} GiB (budget {:.1} GiB)", dir.display(), metas.len(), total as f64 / (1u64 << 30) as f64, c.max_gib);
            let age = |t: u64| -> String {
                let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
                let d = now.saturating_sub(t);
                if d < 3600 { format!("{}m", d / 60) } else if d < 86400 { format!("{}h", d / 3600) } else { format!("{}d", d / 86400) }
            };
            for (root, (rows, snaps, bytes, oldest, newest)) in &by_root {
                println!("  root {root}: {rows} rows chunks, {snaps} snapshots, {:.2} GiB, last used {} .. {} ago", *bytes as f64 / (1u64 << 30) as f64, age(*newest), age(*oldest));
            }
            let mut roles: std::collections::BTreeMap<&str, (usize, usize, usize)> = Default::default();
            for m in &metas {
                let e = roles.entry(m.role.map(|r| r.name()).unwrap_or("(no role)")).or_default();
                match m.kind {
                    Kind::Rows => {
                        e.0 += 1;
                        e.1 += m.end - m.start;
                    }
                    Kind::Snapshot => e.2 += 1,
                }
            }
            for (r, (n, toks, snaps)) in &roles {
                let p = c.roles.get(tr_cache::Role::parse(r).unwrap_or(tr_cache::Role::User));
                let pol = if tr_cache::Role::parse(r).is_some() && *p != tr_cache::RolePolicy::default() { format!("  [persist {}, priority {}, ttl {}, snapshots {}]", p.persist, p.priority, p.ttl.map(|t| t.to_string()).unwrap_or_else(|| "none".into()), p.snapshot) } else { String::new() };
                println!("  {r}: {n} chunks, {toks} tokens, {snaps} snapshots{pol}");
            }
        }
        CacheOp::Clear => {
            for m in &metas {
                be.delete(&m.name())?;
            }
            be.put_index(b"{}")?;
            println!("{}: {} objects deleted", dir.display(), metas.len());
        }
    }
    Ok(())
}

fn cmd_generate(cfg: &AppConfig) -> Result<()> {
    let rt = cfg.runtime(defaults::CTX);
    let g = cfg.generate();
    let mut text = match &g.prompt_file {
        Some(p) => std::fs::read_to_string(p)?,
        None => g.prompt.clone(),
    };
    anyhow::ensure!(g.images.is_empty() || (g.chat && rt.vision.is_some()), "images need chat mode and a vision overlay (--chat --vision)");
    if g.chat {
        let gen = if g.no_think { "<think>\n\n</think>\n\n" } else { "<think>\n" };
        let placeholders = server::chat::IMAGE_PLACEHOLDER.repeat(g.images.len());
        text = format!("<|im_start|>user\n{placeholders}{}<|im_end|>\n<|im_start|>assistant\n{gen}", text.trim_end());
    }
    generate(&rt, &g, &cfg.sampling(defaults::TEMP), &cfg.moe(), &text)
}

/// Decode an image file (any format the `image` crate reads) to RGB8.
fn load_rgb(path: &std::path::Path) -> Result<(Vec<u8>, usize, usize)> {
    let img = image::open(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    Ok((img.into_raw(), w, h))
}

fn vision_embed(rt: &config::RuntimeCfg, vision: &std::path::Path, image: &std::path::Path, out: &std::path::Path, reps: usize) -> Result<()> {
    use std::io::Write;
    let t0 = Instant::now();
    let mut model = tr_model::exec::Model::load_with(rt.pack()?, rt.ctx, rt.cores_per_node, &tr_model::weights::LoadOptions { ple_mmap: rt.ple_mmap, batch_max: rt.batch, kv: rt.kv()?, vision: Some(vision.to_path_buf()), image_min_tokens: rt.image_min_tokens, image_max_tokens: rt.image_max_tokens, ..Default::default() })?;
    eprintln!("loaded in {:.1} s: {}", t0.elapsed().as_secs_f64(), model.load_summary());
    let (rgb, w, h) = load_rgb(image)?;
    let vc = model.vision.as_ref().unwrap().cfg.clone();
    let t1 = Instant::now();
    let pt = tr_model::image::prepare(&rgb, w, h, &vc.prep_params());
    let (tw, th) = pt.tokens(vc.merge);
    eprintln!("image {w}x{h} -> {}x{} ({}x{} patches, {} tokens) prep {:.1} ms", pt.width, pt.height, pt.gw, pt.gh, tw * th, t1.elapsed().as_secs_f64() * 1e3);
    let mut embd = Vec::new();
    for r in 0..reps.max(1) {
        let t2 = Instant::now();
        embd = model.encode_image(&pt);
        eprintln!("encode rep {r}: {:.1} ms", t2.elapsed().as_secs_f64() * 1e3);
    }
    if model.profile.enabled {
        eprintln!("{}", model.profile.report());
    }
    let mut f = std::fs::File::create(out)?;
    for v in [tw * th, tw, th, vc.proj] {
        f.write_all(&(v as u32).to_le_bytes())?;
    }
    let bytes: Vec<u8> = embd.iter().flat_map(|v| v.to_le_bytes()).collect();
    f.write_all(&bytes)?;
    let rms = (embd.iter().map(|v| v * v).sum::<f32>() / embd.len() as f32).sqrt();
    eprintln!("wrote {} ({} tokens, rms {rms:.5}, first {:?})", out.display(), tw * th, &embd[..4]);
    Ok(())
}

fn barrier_bench(rounds: usize, cpn: Option<usize>) -> Result<()> {
    let topo = Topology::discover()?;
    let mut pool = tr_sys::pool::Pool::new(&topo, cpn)?;
    println!("workers {} ({} nodes x {} cores)", pool.n_workers(), pool.n_nodes(), pool.cores_per_node());
    // warm-up
    pool.run(|ctx| for _ in 0..1000 { ctx.barrier(); });
    let t0 = Instant::now();
    pool.run(|ctx| for _ in 0..rounds { ctx.barrier(); });
    let dt = t0.elapsed();
    println!("global barrier: {:.2} us/round over {rounds} rounds", dt.as_secs_f64() * 1e6 / rounds as f64);
    let t0 = Instant::now();
    pool.run(|ctx| for _ in 0..rounds { ctx.node_barrier(); });
    let dt = t0.elapsed();
    println!("node barrier:   {:.2} us/round", dt.as_secs_f64() * 1e6 / rounds as f64);
    // idle check: workers must futex-sleep, not spin, between runs
    let c0 = process_cpu_secs();
    std::thread::sleep(std::time::Duration::from_secs(2));
    let c1 = process_cpu_secs();
    println!("idle: {:.1}% CPU over 2 s with {} workers parked", (c1 - c0) / 2.0 * 100.0, pool.n_workers());
    Ok(())
}

fn numa_smoke(mib: usize) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    use tr_sys::numa::{sample_placement, Arena};
    let topo = Topology::discover()?;
    let mut pool = tr_sys::pool::Pool::new(&topo, None)?;
    let bytes = mib << 20;
    let mut arenas: Vec<Arena> = Vec::new();
    let mut bufs: Vec<(usize, usize)> = Vec::new();
    for n in &topo.nodes {
        let mut a = Arena::new(n.id, bytes)?;
        let p = a.alloc(bytes, 2 << 20)?;
        bufs.push((p as usize, bytes));
        arenas.push(a);
    }
    let cpn = pool.cores_per_node();
    // fill from owning cores
    let t0 = Instant::now();
    pool.run(|ctx| {
        let (p, len) = bufs[ctx.node];
        let r = ctx.range_local(len / 8);
        let s = unsafe { std::slice::from_raw_parts_mut(p as *mut u64, len / 8) };
        for (i, v) in s[r.clone()].iter_mut().enumerate() {
            *v = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
        }
    });
    let fill = t0.elapsed();
    println!("fill {} MiB/tile on {} tiles: {:.2} s ({:.1} GB/s aggregate)", mib, topo.n_nodes(), fill.as_secs_f64(), (bytes * topo.n_nodes()) as f64 / fill.as_secs_f64() / 1e9);
    for (n, (p, len)) in bufs.iter().enumerate() {
        let placed = sample_placement(*p as *const u8, *len, 64)?;
        let maps = tr_sys::numa::numa_maps_pages(*p as *const u8, *len)?;
        println!("tile {n}: sampled pages {:?}  numa_maps pages {:?}", placed, maps);
    }
    // per-tile read bandwidth: each tile's cores stream their own buffer, repeated
    let reps = 5;
    let sink = AtomicU64::new(0);
    let per_node_ns: Vec<AtomicU64> = (0..topo.n_nodes()).map(|_| AtomicU64::new(0)).collect();
    pool.run(|ctx| {
        let (p, len) = bufs[ctx.node];
        let s = unsafe { std::slice::from_raw_parts(p as *const u64, len / 8) };
        let r = ctx.range_local(len / 8);
        ctx.barrier();
        let t0 = Instant::now();
        let mut acc = 0u64;
        for _ in 0..reps {
            acc = acc.wrapping_add(sum_u64(std::hint::black_box(&s[r.clone()])));
        }
        ctx.node_barrier();
        if ctx.local == 0 {
            per_node_ns[ctx.node].store(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        sink.fetch_add(acc, Ordering::Relaxed);
    });
    let mut agg = 0.0;
    for n in 0..topo.n_nodes() {
        let ns = per_node_ns[n].load(Ordering::Relaxed) as f64;
        let gbs = (bytes * reps) as f64 / ns;
        agg += gbs;
        println!("tile {n}: read {:.1} GB/s with {cpn} cores", gbs);
    }
    println!("aggregate {:.1} GB/s  (checksum {})", agg, sink.load(Ordering::Relaxed) % 1000);
    println!("{}", tr_sys::procinfo::mem_summary());
    Ok(())
}

#[inline(never)]
fn sum_u64(s: &[u64]) -> u64 {
    // 8 independent accumulators so the loop is bandwidth-bound, not latency-bound.
    let mut acc = [0u64; 8];
    let chunks = s.chunks_exact(8);
    let rem = chunks.remainder();
    for c in chunks {
        for i in 0..8 {
            acc[i] = acc[i].wrapping_add(c[i]);
        }
    }
    let mut t = rem.iter().fold(0u64, |a, &b| a.wrapping_add(b));
    for a in acc {
        t = t.wrapping_add(a);
    }
    t
}

fn load_bench(dir: &std::path::Path, gib: f64, keep: bool) -> Result<()> {
    use std::io::Write;
    use std::sync::Arc;
    use tr_sys::loader::{parallel_load, DirectFile, Section};
    use tr_sys::numa::Arena;
    if dir.starts_with("/tmp") {
        anyhow::bail!("refusing to stage under /tmp (tmpfs, no O_DIRECT)");
    }
    std::fs::create_dir_all(dir)?;
    let topo = Topology::discover()?;
    let bytes = ((gib * (1u64 << 30) as f64) as usize / (2 << 20)) * (2 << 20);
    let paths: Vec<_> = (0..topo.n_nodes()).map(|n| dir.join(format!("node{n}.bin"))).collect();
    let t0 = Instant::now();
    for (n, p) in paths.iter().enumerate() {
        if keep && std::fs::metadata(p).map(|m| m.len() as usize == bytes).unwrap_or(false) {
            continue;
        }
        let mut f = std::fs::File::create(p)?;
        let chunk: Vec<u8> = (0..(8 << 20)).map(|i| ((i * 7 + n * 13) % 251) as u8).collect();
        let mut left = bytes;
        while left > 0 {
            let l = left.min(chunk.len());
            f.write_all(&chunk[..l])?;
            left -= l;
        }
        f.sync_all()?;
    }
    println!("wrote {} x {:.2} GiB in {:.1} s", paths.len(), gib, t0.elapsed().as_secs_f64());
    // drop page cache for these files so the read is a real NVMe read
    for p in &paths {
        let f = std::fs::File::open(p)?;
        unsafe {
            use std::os::unix::io::AsRawFd;
            libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
    let mut pool = tr_sys::pool::Pool::new(&topo, None)?;
    let mut arenas = Vec::new();
    let mut sections = Vec::new();
    for (n, p) in paths.iter().enumerate() {
        let mut a = Arena::new(n, bytes)?;
        let dst = a.alloc(bytes, 2 << 20)?;
        sections.push(Section { node: n, file: Arc::new(DirectFile::open(p)?), offset: 0, dst, len: bytes });
        arenas.push(a);
    }
    let t0 = Instant::now();
    let total = parallel_load(&mut pool, &sections)?;
    let dt = t0.elapsed().as_secs_f64();
    println!("loaded {:.2} GiB in {:.1} s = {:.2} GB/s", total as f64 / (1u64 << 30) as f64, dt, total as f64 / dt / 1e9);
    // verify content + placement of one page per 256 MiB
    for (n, s) in sections.iter().enumerate() {
        let sl = unsafe { std::slice::from_raw_parts(s.dst, s.len) };
        let mut off = 0;
        while off < s.len {
            let want = ((off % (8 << 20)) * 7 + n * 13) % 251;
            anyhow::ensure!(sl[off] as usize == want, "node {n} byte {off}: {} != {want}", sl[off]);
            off += 256 << 20;
        }
        let placed = tr_sys::numa::sample_placement(s.dst, s.len, 32)?;
        anyhow::ensure!(placed.len() == 1 && placed.contains_key(&n), "node {n} placement {placed:?}");
    }
    println!("content + placement OK; {}", tr_sys::procinfo::mem_summary());
    Ok(())
}

fn process_cpu_secs() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after = s.rsplit(')').next().unwrap_or("");
    let f: Vec<&str> = after.split_whitespace().collect();
    // fields after the comm: state(0) ... utime is index 11, stime 12
    let ticks: f64 = f.get(11).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) + f.get(12).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    ticks / unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64
}

/// `prompt` is the final text (the chat template has already been applied when `g.chat`).
fn generate(rt: &config::RuntimeCfg, g: &config::GenerateCfg, sm: &config::SamplingCfg, moe_cfg: &config::MoeCfg, prompt: &str) -> Result<()> {
    use tr_model::exec::{Dump, Model};
    use tr_model::sampler::Sampler;
    use tr_model::tokenizer::Tok;
    let (pack, ctx, cpn, ple_mmap, batch) = (rt.pack()?, rt.ctx, rt.cores_per_node, rt.ple_mmap, rt.batch);
    let (kv, mtp, vision, image_max_tokens) = (rt.kv()?, rt.mtp.clone(), rt.vision.clone(), rt.image_max_tokens);
    let (n_predict, dump, images) = (g.n_predict, g.dump, g.images.as_slice());
    let stream = g.chat; // the chat template is streamed as it decodes
    let (temp, top_k, top_p, seed) = (sm.temp, sm.top_k, sm.top_p, sm.seed);
    let (moe, moe_profile) = (moe_cfg.kernel()?, moe_cfg.profile);
    let t0 = Instant::now();
    let spec_k = if mtp.is_some() && !dump { rt.spec_k.clamp(1, 7) } else { 0 };
    let mut model = Model::load_with(pack, ctx, cpn, &tr_model::weights::LoadOptions { ple_mmap, batch_max: if dump { 1 } else { batch.max(spec_k + 1).max(1) }, kv, mtp: if dump { None } else { mtp }, spec_k, vision, image_max_tokens, ..Default::default() })?;
    model.set_moe(moe.resolve(model.cfg.n_expert_used, model.cfg.n_expert).map_err(anyhow::Error::msg)?)?;
    let moe0 = model.moe_stats.snapshot();
    if moe_profile {
        model.moe_stats.profile_start();
    }
    let mut spec = if spec_k > 0 { Some(tr_model::spec::Speculator::new(spec_k)) } else { None };
    eprintln!("{} (total {:.1} s)", model.load_summary(), t0.elapsed().as_secs_f64());
    let tok = Tok::load(&pack.join(&model.manifest.tokenizer))?;
    let mut ids = tok.encode(prompt, false)?;
    // images: preprocess, expand the pad tokens, encode
    let mut places = Vec::new();
    let mut embds: Vec<Vec<f32>> = Vec::new();
    if !images.is_empty() {
        let vc = model.vision.as_ref().ok_or_else(|| anyhow::anyhow!("--image needs --vision"))?.cfg.clone();
        let pad = tok.token_id("<|image_pad|>").ok_or_else(|| anyhow::anyhow!("no <|image_pad|> token"))?;
        let mut prepared = Vec::new();
        for path in images {
            let (rgb, w, h) = load_rgb(path)?;
            let pt = tr_model::image::prepare(&rgb, w, h, &vc.prep_params());
            let (nx, ny) = pt.tokens(vc.merge);
            eprintln!("image {}: {w}x{h} -> {}x{} px, {nx}x{ny} = {} tokens", path.display(), pt.width, pt.height, nx * ny);
            let hash = server::images::patches_hash(&pt);
            prepared.push(server::images::Prepared { patches: std::sync::Arc::new(pt), hash, nx, ny });
        }
        let (expanded, pl) = server::images::expand_pads(&ids, pad, &prepared).map_err(|e| anyhow::anyhow!(e))?;
        ids = expanded;
        places = pl;
        let te = Instant::now();
        for p in &prepared {
            embds.push(model.encode_image(&p.patches));
        }
        eprintln!("encoded {} images in {:.2} s", prepared.len(), te.elapsed().as_secs_f64());
    }
    anyhow::ensure!(ids.len() + n_predict <= ctx, "prompt ({}) + n_predict ({n_predict}) exceeds --ctx {ctx}", ids.len());
    if ids.len() <= 64 {
        eprintln!("prompt tokens ({}): {:?}", ids.len(), ids);
    } else {
        eprintln!("prompt tokens ({}): {:?} ... {:?}", ids.len(), &ids[..16], &ids[ids.len() - 8..]);
    }
    let mut sampler = Sampler::new(temp, top_k, top_p, seed);
    model.reset();
    let dumper = if dump { Some(Dump::new()) } else { None };
    let t1 = Instant::now();
    let mut logits = Vec::new();
    if dump {
        for (i, &id) in ids.iter().enumerate() {
            logits = model.step(id, dumper.as_ref());
            if let Some(d) = &dumper {
                let entries = std::mem::take(&mut *d.entries.lock().unwrap());
                for (name, st) in entries {
                    println!("tok{i} {name:22} sum {:14.6} abs {:14.6} max {:10.6} n {} first {:?}", st[0], st[1], st[2], st[3] as usize, &st[4..]);
                }
            }
        }
    } else if images.is_empty() {
        let bm = model.batch_max();
        for chunk in ids.chunks(bm) {
            logits = model.step_batch(chunk);
        }
    } else {
        let (pos3, after) = tr_model::image::mrope_positions(ids.len(), &places);
        let hh = model.cfg.hidden;
        let bm = model.batch_max();
        let mut a = 0;
        while a < ids.len() {
            let b = (a + bm).min(ids.len());
            let mut segs = Vec::new();
            for (pl, e) in places.iter().zip(&embds) {
                let (s0, s1) = (pl.row.max(a), (pl.row + pl.nx * pl.ny).min(b));
                if s0 < s1 {
                    segs.push(tr_model::exec_batch::ImageSeg { row: s0 - a, n: s1 - s0, embd: &e[(s0 - pl.row) * hh..(s1 - pl.row) * hh] });
                }
            }
            logits = model.step_batch_with(&ids[a..b], Some(&pos3[a..b]), Some(after[b - 1]), &segs);
            a = b;
        }
    }
    let prompt_s = t1.elapsed().as_secs_f64();
    let mut out_ids = Vec::new();
    let mut printed = 0usize; // chars of the decoded output already streamed
    let t2 = Instant::now();
    let mut show = |out_ids: &[u32], printed: &mut usize| -> Result<()> {
        if stream {
            // print the stable prefix (a token may end mid-UTF-8-sequence, shown as U+FFFD)
            let text = tok.decode(out_ids)?;
            let stable = text.trim_end_matches('\u{FFFD}');
            if stable.len() > *printed {
                use std::io::Write;
                print!("{}", &stable[*printed..]);
                std::io::stdout().flush()?;
                *printed = stable.len();
            }
        }
        Ok(())
    };
    let stop_at = [model.cfg.eos_id, server::engine::THINK_CLOSE];
    let mut next = sampler.sample(&logits);
    while out_ids.len() < n_predict {
        if std::env::var("TR_LOGIT_DEBUG").is_ok() {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            idx.select_nth_unstable_by(2, |&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            let mut top: Vec<usize> = idx[..3].to_vec();
            top.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            eprintln!("top3: {}", top.iter().map(|&i| format!("{i}:{:.3}", logits[i])).collect::<Vec<_>>().join(" "));
        }
        out_ids.push(next);
        if next == model.cfg.eos_id {
            break;
        }
        show(&out_ids, &mut printed)?;
        if out_ids.len() >= n_predict {
            break;
        }
        match spec.as_mut() {
            Some(sp) => {
                let r = sp.round(&mut model, next, n_predict - out_ids.len(), &mut sampler, &stop_at);
                let mut done = false;
                for &d in &r.accepted {
                    out_ids.push(d);
                    if d == model.cfg.eos_id {
                        done = true;
                        break;
                    }
                }
                show(&out_ids, &mut printed)?;
                logits = r.next_logits;
                next = r.next;
                if done {
                    break;
                }
            }
            None => {
                logits = model.step(next, None);
                next = sampler.sample(&logits);
            }
        }
    }
    let gen_s = t2.elapsed().as_secs_f64();
    let text = tok.decode(&out_ids)?;
    if stream {
        println!("{}", &text[printed.min(text.len())..]);
    } else {
        println!("{text}");
    }
    eprintln!("ids: {:?}", out_ids);
    let spec_desc = spec.as_ref().map(|s| format!("; speculative k={}: {} rounds, {}/{} drafts accepted", s.k, s.stats.rounds, s.stats.accepted, s.stats.drafted)).unwrap_or_default();
    let moe_desc = model.moe_policy().map(|p| format!("; moe mass {} of {:?} [{}..{}]: {:.2} experts/token", p.mass, p.basis, p.min, p.max, model.moe_stats.mean_since(moe0))).unwrap_or_default();
    let cum = model.moe_stats.profile_report();
    if !cum.is_empty() {
        eprintln!("cumulative router mass of the best k experts (mean over tokens and MoE layers):\n{}", cum.iter().enumerate().map(|(j, m)| format!("k={} {:.3}", j + 1, m)).collect::<Vec<_>>().join("  "));
    }
    eprintln!("prompt {} tok in {:.2} s ({:.2} tok/s); generated {} tok in {:.2} s ({:.2} tok/s){spec_desc}{moe_desc}; {}", ids.len(), prompt_s, ids.len() as f64 / prompt_s, out_ids.len(), gen_s, out_ids.len() as f64 / gen_s, tr_sys::procinfo::mem_summary());
    Ok(())
}

/// Greedy decode with the main model; at every position ask the draft head for the next token
/// (from the carry) and, chained, the one after; count hits. Exercises prefill hook, carry,
/// mtp_extend, mtp_step and the draft K/V without the speculative loop itself.
fn spec_probe(rt: &config::RuntimeCfg, mtp: &std::path::Path, prompt: &str, chat: bool, n: usize) -> Result<()> {
    use tr_model::exec::Model;
    use tr_model::exec_mtp::MtpIn;
    use tr_model::sampler::Sampler;
    use tr_model::tokenizer::Tok;
    let pack = rt.pack()?;
    let spec_k = rt.spec_k;
    let t0 = Instant::now();
    let mut model = Model::load_with(pack, rt.ctx, rt.cores_per_node, &tr_model::weights::LoadOptions { ple_mmap: rt.ple_mmap, batch_max: rt.batch.max(spec_k + 1), kv: rt.kv()?, mtp: Some(mtp.to_path_buf()), spec_k, ..Default::default() })?;
    eprintln!("{} (total {:.1} s)", model.load_summary(), t0.elapsed().as_secs_f64());
    let tok = Tok::load(&pack.join(&model.manifest.tokenizer))?;
    let text = if chat { format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n") } else { prompt.to_string() };
    let ids = tok.encode(&text, true)?;
    model.reset();
    let bm = model.batch_max();
    let mut logits = Vec::new();
    for chunk in ids.chunks(bm) {
        logits = model.step_batch(chunk);
    }
    let mut t_cur = Sampler::greedy(&logits);
    let (mut hit1, mut hit2, mut hit3, mut tried) = (0usize, 0usize, 0usize, 0usize);
    let mut out = Vec::new();
    let mut td = 0f64;
    for _ in 0..n {
        // draft from the committed state: row for position n_past (token t_cur, hidden = carry)
        let ts = Instant::now();
        let l1 = model.mtp_extend(&[t_cur], model.n_past, &[MtpIn::Carry], None);
        let d1 = Sampler::greedy(&l1);
        let l2 = model.mtp_step(d1, model.n_past + 1);
        let d2 = Sampler::greedy(&l2);
        let l3 = model.mtp_step(d2, model.n_past + 2);
        let d3 = Sampler::greedy(&l3);
        td += ts.elapsed().as_secs_f64();
        // truth: main model, two steps ahead (the second step is undone by re-running below)
        logits = model.step(t_cur, None);
        let t1 = Sampler::greedy(&logits);
        out.push(t_cur);
        tried += 1;
        hit1 += usize::from(d1 == t1);
        if d1 == t1 {
            // chained draft is only meaningful when the first one was right; peek two more steps
            let saved = model.n_past;
            let l = model.step(t1, None);
            let t2 = Sampler::greedy(&l);
            hit2 += usize::from(d2 == t2);
            if d2 == t2 {
                let l = model.step(t2, None);
                hit3 += usize::from(d3 == Sampler::greedy(&l));
            }
            // rewind: replay the sequence up to `saved` (no state rollback in this probe)
            model.reset();
            for chunk in ids.chunks(bm) {
                model.step_batch(chunk);
            }
            for &x in &out {
                logits = model.step(x, None);
            }
            assert_eq!(model.n_past, saved);
        }
        t_cur = t1;
        if t_cur == model.cfg.eos_id {
            break;
        }
    }
    println!("{}", tok.decode(&out)?);
    println!("draft hits: first {hit1}/{tried} ({:.1} %), second {hit2}/{hit1}, third {hit3}/{hit2}; draft time {:.2} ms per 3 drafts", 100.0 * hit1 as f64 / tried.max(1) as f64, 1000.0 * td / tried.max(1) as f64);
    Ok(())
}

fn bench(rt: &config::RuntimeCfg, b: &config::BenchCfg, moe_cfg: &config::MoeCfg) -> Result<()> {
    use tr_model::exec::Model;
    use tr_model::sampler::Sampler;
    let (pack, ctx, cpn, ple_mmap, batch) = (rt.pack()?, rt.ctx, rt.cores_per_node, rt.ple_mmap, rt.batch);
    let (kv, mtp) = (rt.kv()?, rt.mtp.clone());
    let (pp, tg, reps, tg_batch) = (b.pp, b.tg, b.reps, b.tg_batch);
    let (moe, moe_profile) = (moe_cfg.kernel()?, moe_cfg.profile);
    let t0 = Instant::now();
    let spec_k = if mtp.is_some() { rt.spec_k.clamp(1, 7) } else { 0 };
    let mut model = Model::load_with(pack, ctx, cpn, &tr_model::weights::LoadOptions { ple_mmap, batch_max: batch.max(spec_k + 1), kv, mtp, spec_k, ..Default::default() })?;
    eprintln!("{} (total {:.1} s)", model.load_summary(), t0.elapsed().as_secs_f64());
    model.set_moe(moe.resolve(model.cfg.n_expert_used, model.cfg.n_expert).map_err(anyhow::Error::msg)?)?;
    let moe_desc = |m: &Model, pp0: (u64, u64), tg0: (u64, u64)| {
        let Some(p) = m.moe_policy() else { return String::new() };
        let (r, e) = (tg0.0 - pp0.0, tg0.1 - pp0.1);
        let pp_mean = if r > 0 { e as f64 / r as f64 } else { 0.0 };
        format!("   moe mass {} of {:?} [{}..{}]: {:.2} experts/token pp, {:.2} tg", p.mass, p.basis, p.min, p.max, pp_mean, m.moe_stats.mean_since(tg0))
    };
    let mut rng = 12345u64;
    for r in 0..reps {
        if r + 1 == reps {
            model.profile.clear(); // the report covers the last rep only (earlier reps warm the page cache)
            if moe_profile {
                model.moe_stats.profile_start();
            }
        }
        model.reset();
        let moe_pp = model.moe_stats.snapshot();
        let t1 = Instant::now();
        let mut logits = Vec::new();
        let mut prompt = Vec::with_capacity(pp);
        for _ in 0..pp {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            prompt.push(((rng >> 33) % 200000) as u32);
        }
        let bm = model.batch_max();
        for chunk in prompt.chunks(bm) {
            logits = model.step_batch(chunk);
        }
        let pp_s = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let moe_tg = model.moe_stats.snapshot();
        let mut next = Sampler::greedy(&logits);
        match tg_batch {
            None if spec_k > 0 => {
                // speculative greedy decode on the random prompt's continuation
                let mut sp = tr_model::spec::Speculator::new(spec_k);
                let mut sampler = Sampler::new(0.0, 1, 1.0, 0);
                let mut n = 0usize;
                let stop_at = [model.cfg.eos_id];
                while n < tg {
                    let r = sp.round(&mut model, next, tg - n, &mut sampler, &stop_at);
                    n += 1 + r.accepted.len();
                    next = r.next;
                }
                let tg_s = t2.elapsed().as_secs_f64();
                println!("rep {r}: pp{pp} {:.2} tok/s ({:.2} s)   tg{n} speculative k={spec_k}: {:.2} tok/s ({:.1} ms/tok), {} rounds, {}/{} drafts accepted   {}{}", pp as f64 / pp_s, pp_s, n as f64 / tg_s, tg_s * 1000.0 / n as f64, sp.stats.rounds, sp.stats.accepted, sp.stats.drafted, tr_sys::procinfo::mem_summary(), moe_desc(&model, moe_pp, moe_tg));
                continue;
            }
            None => {
                for _ in 0..tg {
                    logits = model.step(next, None);
                    next = Sampler::greedy(&logits);
                }
            }
            Some(m) => {
                let steps = tg.div_ceil(m);
                for _ in 0..steps {
                    logits = model.step_batch(&vec![next; m]);
                    next = Sampler::greedy(&logits);
                }
                let tg_s = t2.elapsed().as_secs_f64();
                println!("rep {r}: pp{pp} {:.2} tok/s ({:.2} s)   batch-decode M={m}: {steps} steps, {:.2} ms/step ({:.2} tok/s if all rows count)   {}{}", pp as f64 / pp_s, pp_s, tg_s * 1000.0 / steps as f64, (steps * m) as f64 / tg_s, tr_sys::procinfo::mem_summary(), moe_desc(&model, moe_pp, moe_tg));
                continue;
            }
        }
        let tg_s = t2.elapsed().as_secs_f64();
        println!("rep {r}: pp{pp} {:.2} tok/s ({:.2} s)   tg{tg} {:.2} tok/s ({:.1} ms/tok)   {}{}", pp as f64 / pp_s, pp_s, tg as f64 / tg_s, tg_s * 1000.0 / tg as f64, tr_sys::procinfo::mem_summary(), moe_desc(&model, moe_pp, moe_tg));
    }
    let cum = model.moe_stats.profile_report();
    if !cum.is_empty() {
        println!("cumulative router mass of the best k experts (mean over the last rep's tokens and MoE layers):");
        println!("{}", cum.iter().enumerate().map(|(j, m)| format!("k={} {:.3}", j + 1, m)).collect::<Vec<_>>().join("  "));
    }
    if model.profile.enabled {
        println!("profile (worker 0,0):\n{}", model.profile.report());
        println!("per-worker batch phases:\n{}", model.profile.pw_report());
    }
    Ok(())
}

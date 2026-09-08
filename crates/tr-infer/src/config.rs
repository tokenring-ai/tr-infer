//! Layered configuration.
//!
//! Every setting the binary has can be given in four places; later ones win:
//!
//! 1. built-in defaults (the `defaults` module below — the values the flags used to carry),
//! 2. YAML file(s): `--config FILE` (repeatable), else `$TR_CONFIG`, else the search path
//!    `/etc/tr-infer/config.yaml`, `$XDG_CONFIG_HOME/tr-infer/config.yaml`, `./tr-infer.yaml`,
//! 3. environment variables `TR_<SECTION>__<KEY>` (double underscore nests, e.g.
//!    `TR_RUNTIME__CTX=60000`, `TR_MOE__MASS=0.8`, `TR_SERVER__PORT=8089`),
//! 4. command-line flags.
//!
//! [`AppConfig`] is the merged *request* tree: every leaf is an `Option`, so "not set anywhere"
//! stays distinguishable from "set to the default value". Commands turn it into the concrete
//! `*Cfg` structs at the bottom of this file, which is where the defaults are applied — a few of
//! them differ per command (`generate --temp` defaults to 0, `serve` to 0.6, and the context
//! window is 4096 for the one-shot commands but 8192 for the server), so a single `runtime.ctx`
//! in the file still means "for whatever I run".
//!
//! Unknown keys in a *file* are an error (a typo in a config file is otherwise invisible);
//! unknown `TR_*` variables are ignored, because the engine has its own debug knobs (`TR_AMX`,
//! `TR_DUMP_VEC`, ...) that share the prefix.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

pub const ENV_PREFIX: &str = "TR";
pub const ENV_SEPARATOR: &str = "__";

/// The values the command line used to hard-code. Referenced from the `*Cfg` builders so the
/// defaults exist exactly once.
pub mod defaults {
    pub const CTX: usize = 4096;
    pub const CTX_SERVE: usize = 8192;
    pub const CTX_VISION_EMBED: usize = 1024;
    pub const BATCH: usize = 256;
    pub const KV: &str = "f16";
    pub const PIN_HEAP: &str = "mt";

    pub const SPEC_K: usize = 3;
    pub const MOE_BASIS: &str = "top";

    pub const TEMP: f32 = 0.0;
    pub const TEMP_SERVE: f32 = 0.6;
    pub const TOP_P: f32 = 0.95;
    pub const TOP_K: usize = 20;
    pub const SEED: u64 = 42;

    pub const IMAGE_MIN_TOKENS: usize = 8;
    pub const IMAGE_MAX_TOKENS: usize = 4096;

    pub const CACHE_BACKEND: &str = "fs";
    pub const CACHE_CHUNK_LEN: usize = 256;
    pub const CACHE_MAX_GIB: f64 = 32.0;
    pub const CACHE_SNAPSHOTS: &str = "message";
    pub const CACHE_SNAPSHOT_MIN_TOKENS: usize = 16;
    pub const CACHE_STAGING_MIB: usize = 1024;

    pub const HOST: &str = "127.0.0.1";
    pub const PORT: u16 = 8080;
    pub const REASONING: &str = "xhigh";

    pub const PROMPT: &str = "The capital of France is";
    pub const N_PREDICT: usize = 32;
    pub const N_PREDICT_PROBE: usize = 64;

    pub const BENCH_PP: usize = 64;
    pub const BENCH_TG: usize = 64;
    pub const BENCH_REPS: usize = 2;

    pub const NUMA_SMOKE_MIB: usize = 1024;
    pub const LOAD_BENCH_GIB: f64 = 2.0;
    pub const BARRIER_ROUNDS: usize = 100_000;
    pub const REPS: usize = 1;
}

macro_rules! section {
    ($(#[$m:meta])* $name:ident { $($(#[$fm:meta])* $f:ident : $t:ty),* $(,)? }) => {
        $(#[$m])*
        #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
        #[serde(default)]
        pub struct $name {
            $($(#[$fm])* pub $f: Option<$t>,)*
        }
    };
}

section!(
    /// Everything that shapes a model load and is shared by every command that loads one.
    Runtime {
        /// Context window (prompt + completion) in tokens.
        ctx: usize,
        /// Prompt batch size (tokens per batched step); 1 disables the batch path.
        batch: usize,
        /// K/V cache element type: `f16`, `bf16` or `f32`.
        kv: String,
        /// Cores used per NUMA tile (unset = all physical cores of the tile).
        cores_per_node: usize,
        /// Keep the PLE n-gram table file-backed instead of resident in HBM (-3.4 GiB/tile).
        ple_mmap: bool,
        /// glibc heap policy, letters `t` (no trim) and `m` (no mmap); `0` disables.
        /// Also readable as `TR_PIN_HEAP`, which is applied before anything else allocates.
        pin_heap: String,
    }
);

section!(
    /// Optional overlay packs loaded into the same tile arenas as the base pack.
    Overlays {
        /// MTP draft head (`trpack mtp`): enables speculative decoding.
        mtp: PathBuf,
        /// Vision encoder (`trpack vision`): enables image input.
        vision: PathBuf,
    }
);

section!(
    /// Speculative decoding.
    Spec {
        /// Draft tokens per round (1..7).
        k: usize,
    }
);

section!(
    /// Cumulative-mass expert routing (`mass` unset = the model's fixed top-k).
    Moe {
        /// Take experts in router order until their softmax mass reaches this target.
        mass: f32,
        /// Fewest experts per token.
        min: usize,
        /// Most experts per token (limit 32).
        max: usize,
        /// What the mass is measured against: `top` (the best `max` experts, renormalised) or
        /// `all` (the softmax over all experts).
        basis: String,
        /// Report the mean cumulative router mass of the best k experts.
        profile: bool,
    }
);

section!(
    /// Sampling defaults. `think_*` apply inside the `<think>` block (server only).
    Sampling {
        temp: f32,
        top_k: usize,
        top_p: f32,
        seed: u64,
        think_temp: f32,
        think_top_k: usize,
        think_top_p: f32,
    }
);

section!(
    /// Image input limits, in merged tokens of 32x32 pixels.
    Image {
        min_tokens: usize,
        max_tokens: usize,
    }
);

section!(
    /// `tr-infer serve`.
    Server {
        /// Opt in to text-only continuous batching.
        continuous: bool,
        /// Maximum simultaneously resident generation choices.
        max_sequences: usize,
        /// Shared KV pool budget in MiB per NUMA tile (required in continuous mode).
        kv_cache_mib: usize,
        /// Prompt rows per iteration while generation is active.
        prefill_chunk: usize,
        /// Maximum waiting requests.
        max_queue: usize,
        host: String,
        port: u16,
        /// Bearer token clients must send; also readable as `TR_API_KEY`.
        api_key: String,
        /// Serve without an API key (only sensible on a private interface).
        no_auth: bool,
        /// Model id reported by `/v1/models` (unset = the pack directory name).
        model_name: String,
        /// Default reasoning effort: `xhigh`, `medium`, `low` or `none`.
        reasoning: String,
    }
);

section!(
    /// What the prefix cache does with chunks of one role (`cache.roles.<role>`).
    CacheRole {
        /// Store chunks of this role at all. Nothing after an unstored chunk is restorable.
        persist: bool,
        /// Eviction order 0..=9: lower priorities are evicted first (LRU within a priority).
        priority: u8,
        /// Idle expiry in seconds: dropped when unused for longer, budget or not.
        ttl: u64,
        /// Take a snapshot where a prompt span of this role ends.
        snapshot: bool,
    }
);

/// The per-role policies of the prefix cache, one table per role.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheRoles {
    #[serde(deserialize_with = "or_default")]
    pub system: CacheRole,
    #[serde(deserialize_with = "or_default")]
    pub user: CacheRole,
    #[serde(deserialize_with = "or_default")]
    pub reasoning: CacheRole,
    #[serde(deserialize_with = "or_default")]
    pub tool: CacheRole,
    #[serde(deserialize_with = "or_default")]
    pub assistant: CacheRole,
}

impl CacheRoles {
    pub fn get(&self, role: tr_cache::Role) -> &CacheRole {
        match role {
            tr_cache::Role::System => &self.system,
            tr_cache::Role::User => &self.user,
            tr_cache::Role::Reasoning => &self.reasoning,
            tr_cache::Role::Tool => &self.tool,
            tr_cache::Role::Assistant => &self.assistant,
        }
    }
}

/// Persistent prefix cache of `tr-infer serve` (off unless `dir` is set).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Cache {
    /// Store directory (on a real disk, never tmpfs). Unset = no cache.
    pub dir: Option<PathBuf>,
    /// Backend name; only `fs` exists.
    pub backend: Option<String>,
    /// Longest rows chunk in tokens (chunks also end at every role boundary).
    pub chunk_len: Option<usize>,
    /// LRU byte budget of the store, in GiB.
    pub max_gib: Option<f64>,
    /// Where recurrent-state snapshots are taken: `message` (every span end of the prompt
    /// plus the end of each request) or `request` (the end of each request only).
    pub snapshots: Option<String>,
    /// With `snapshots: message`, only spans of at least this many tokens end in a snapshot.
    pub snapshot_min_tokens: Option<usize>,
    /// Host memory for objects waiting to be written, in MiB.
    pub staging_mib: Option<usize>,
    /// Per-role policies: `roles.reasoning.persist: false`, `roles.tool.ttl: 3600`, ...
    #[serde(deserialize_with = "or_default")]
    pub roles: CacheRoles,
}

section!(
    /// `tr-infer generate`.
    Generate {
        prompt: String,
        /// Read the prompt from a file instead of `prompt`.
        prompt_file: PathBuf,
        n_predict: usize,
        /// Wrap the prompt in the model's ChatML template.
        chat: bool,
        /// With `chat`: disable thinking (emits an empty `<think>` block).
        no_think: bool,
        /// Print per-layer activation statistics for the prompt tokens (oracle comparison).
        dump: bool,
        /// Image files placed before the prompt text in the user turn.
        images: Vec<PathBuf>,
    }
);

section!(
    /// `tr-infer bench`.
    Bench {
        pp: usize,
        tg: usize,
        reps: usize,
        /// Decode M identical rows per step through the batched path.
        tg_batch: usize,
    }
);

section!(
    /// `tr-infer spec-probe`.
    SpecProbe {
        prompt: String,
        chat: bool,
        n_predict: usize,
    }
);

section!(
    /// `tr-infer tokenize`.
    Tokenize {
        prompt: String,
        prompt_file: PathBuf,
    }
);

section!(
    /// `tr-infer vision-embed`.
    VisionEmbed {
        image: PathBuf,
        out: PathBuf,
        reps: usize,
    }
);

section!(
    /// `tr-infer numa-smoke`.
    NumaSmoke { mib: usize }
);

section!(
    /// `tr-infer load-bench`.
    LoadBench {
        dir: PathBuf,
        gib_per_node: f64,
        /// Skip writing if the files already exist with the right size.
        keep: bool,
    }
);

section!(
    /// `tr-infer barrier-bench`.
    BarrierBench { rounds: usize }
);

/// A YAML section written with nothing under it (`runtime:`) parses as null, which is not a typo
/// and must mean "this section sets nothing".
fn or_default<'de, D, T>(d: D) -> std::result::Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(d)?.unwrap_or_default())
}

/// The merged configuration request. Every leaf is optional; see the module docs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    /// Pack directory (`trpack`), required by every command that loads a model.
    pub pack: Option<PathBuf>,
    #[serde(deserialize_with = "or_default")]
    pub runtime: Runtime,
    #[serde(deserialize_with = "or_default")]
    pub overlays: Overlays,
    #[serde(deserialize_with = "or_default")]
    pub spec: Spec,
    #[serde(deserialize_with = "or_default")]
    pub moe: Moe,
    #[serde(deserialize_with = "or_default")]
    pub sampling: Sampling,
    #[serde(deserialize_with = "or_default")]
    pub image: Image,
    #[serde(deserialize_with = "or_default")]
    pub server: Server,
    #[serde(deserialize_with = "or_default")]
    pub cache: Cache,
    #[serde(deserialize_with = "or_default")]
    pub generate: Generate,
    #[serde(deserialize_with = "or_default")]
    pub bench: Bench,
    #[serde(deserialize_with = "or_default")]
    pub spec_probe: SpecProbe,
    #[serde(deserialize_with = "or_default")]
    pub tokenize: Tokenize,
    #[serde(deserialize_with = "or_default")]
    pub vision_embed: VisionEmbed,
    #[serde(deserialize_with = "or_default")]
    pub numa_smoke: NumaSmoke,
    #[serde(deserialize_with = "or_default")]
    pub load_bench: LoadBench,
    #[serde(deserialize_with = "or_default")]
    pub barrier_bench: BarrierBench,
}

// ---------------------------------------------------------------------------------------------
// Command-line overrides
// ---------------------------------------------------------------------------------------------

/// Nested JSON built from the flags the user actually passed, used as the last (winning) source.
/// Flags that were not given must not appear at all, or they would override the file.
#[derive(Default, Debug)]
pub struct Ov(Map<String, Value>);

impl Ov {
    pub fn set<T: Serialize>(&mut self, path: &str, v: &Option<T>) {
        if let Some(x) = v {
            self.put(path, serde_json::to_value(x).expect("override serialises"));
        }
    }
    /// f32 through its shortest decimal form: `json!(0.95f32)` would widen to 0.949999988079071.
    pub fn set_f32(&mut self, path: &str, v: &Option<f32>) {
        if let Some(x) = v.filter(|x| x.is_finite()) {
            self.put(path, serde_json::from_str(&format!("{x}")).expect("finite f32 is JSON"));
        }
    }
    /// A repeatable flag: an empty list means "not given".
    pub fn set_list<T: Serialize>(&mut self, path: &str, v: &[T]) {
        if !v.is_empty() {
            self.put(path, serde_json::to_value(v).expect("override serialises"));
        }
    }
    fn put(&mut self, path: &str, v: Value) {
        let mut cur = &mut self.0;
        let mut it = path.split('.').peekable();
        while let Some(k) = it.next() {
            if it.peek().is_none() {
                cur.insert(k.to_string(), v);
                return;
            }
            let e = cur.entry(k.to_string()).or_insert_with(|| Value::Object(Map::new()));
            if !e.is_object() {
                *e = Value::Object(Map::new());
            }
            cur = e.as_object_mut().expect("just made a table");
        }
    }
    pub fn into_value(self) -> Value {
        Value::Object(self.0)
    }
}

// ---------------------------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------------------------

/// Where the merged configuration came from, for `--print-config` and the server banner.
#[derive(Debug, Clone, Default)]
pub struct Provenance {
    pub files: Vec<PathBuf>,
    pub env: bool,
}

impl Provenance {
    pub fn summary(&self) -> String {
        let files = if self.files.is_empty() { "(none)".to_string() } else { self.files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ") };
        format!("config files: {files}; {}{ENV_PREFIX}_* environment", if self.env { "" } else { "no " })
    }
}

/// Config files to read when none were named on the command line, lowest priority first.
pub fn search_path() -> Vec<PathBuf> {
    let mut v = vec![PathBuf::from("/etc/tr-infer/config.yaml")];
    let home = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")));
    if let Some(h) = home {
        v.push(h.join("tr-infer").join("config.yaml"));
    }
    v.push(PathBuf::from("tr-infer.yaml"));
    v
}

/// Merge defaults, files, environment and command line into an [`AppConfig`].
///
/// `files`: explicit `--config` paths (each must exist). Empty = the search path, or `$TR_CONFIG`
/// if that is set. `no_config` skips files entirely, `no_env` skips the environment.
pub fn load(files: &[PathBuf], no_config: bool, no_env: bool, cli: Value) -> Result<(AppConfig, Provenance)> {
    let env_map: Option<std::collections::HashMap<String, String>> = None;
    load_from(files, no_config, no_env, cli, env_map)
}

/// [`load`] with the environment injectable, so tests do not race on the process environment.
pub fn load_from(files: &[PathBuf], no_config: bool, no_env: bool, cli: Value, env: Option<std::collections::HashMap<String, String>>) -> Result<(AppConfig, Provenance)> {
    let mut prov = Provenance { files: Vec::new(), env: !no_env };
    let mut b = config::Config::builder();
    let from_env = match &env {
        Some(m) => m.get("TR_CONFIG").cloned(),
        None => std::env::var("TR_CONFIG").ok(),
    };
    let explicit: Vec<PathBuf> = if !files.is_empty() {
        files.to_vec()
    } else {
        match from_env {
            Some(p) if !p.is_empty() => p.split(':').map(PathBuf::from).collect(),
            _ => Vec::new(),
        }
    };
    let required = !explicit.is_empty();
    let list = if no_config {
        Vec::new()
    } else if required {
        explicit
    } else {
        search_path()
    };
    for p in list {
        if !p.is_file() {
            if required {
                bail!("config file {} does not exist", p.display());
            }
            continue;
        }
        check_keys(&p)?;
        b = b.add_source(config::File::from(p.clone()).format(config::FileFormat::Yaml).required(true));
        prov.files.push(p);
    }
    if !no_env {
        let mut e = config::Environment::with_prefix(ENV_PREFIX).prefix_separator("_").separator(ENV_SEPARATOR).try_parsing(true).list_separator(",").with_list_parse_key("generate.images").ignore_empty(true);
        if let Some(m) = env {
            e = e.source(Some(m.into_iter().collect()));
        }
        b = b.add_source(e);
    }
    let cli = serde_json::to_string(&cli)?;
    b = b.add_source(config::File::from_str(&cli, config::FileFormat::Json));
    let merged = b.build().context("merging configuration sources")?;
    let cfg: AppConfig = merged.try_deserialize().context("reading the merged configuration")?;
    Ok((cfg, prov))
}

/// Reject keys a config file has that the schema does not, with the nearest known key as a hint.
/// Only files are checked: `TR_*` also carries the engine's debug knobs.
fn check_keys(path: &Path) -> Result<()> {
    use config::Source;
    let src = config::File::from(path.to_path_buf()).format(config::FileFormat::Yaml).required(true);
    let map = src.collect().with_context(|| format!("reading {}", path.display()))?;
    let mut got = Vec::new();
    leaf_keys(&map, "", &mut got);
    let known = known_keys();
    let bad: Vec<&String> = got.iter().filter(|k| !known.iter().any(|n| n == *k)).collect();
    if bad.is_empty() {
        return Ok(());
    }
    let mut msg = format!("{}: unknown configuration key(s)", path.display());
    for k in bad {
        match known.iter().min_by_key(|n| edit_distance(n, k)).filter(|n| edit_distance(n, k) <= 3 + k.len() / 4) {
            Some(n) => msg.push_str(&format!("\n  {k}  (did you mean {n}?)")),
            None => msg.push_str(&format!("\n  {k}")),
        }
    }
    bail!(msg)
}

fn leaf_keys(map: &config::Map<String, config::Value>, prefix: &str, out: &mut Vec<String>) {
    for (k, v) in map {
        let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
        match &v.kind {
            // `runtime:` with nothing under it, or `runtime: {}`, sets no key and is not a typo.
            config::ValueKind::Table(t) => {
                if !t.is_empty() {
                    leaf_keys(t, &path, out);
                }
            }
            config::ValueKind::Nil => {}
            _ => out.push(path),
        }
    }
}

/// Every settable key path, derived from the schema itself (all leaves serialise as null).
pub fn known_keys() -> Vec<String> {
    let mut out = Vec::new();
    json_leaves(&serde_json::to_value(AppConfig::default()).expect("schema serialises"), "", &mut out);
    out
}

fn json_leaves(v: &Value, prefix: &str, out: &mut Vec<String>) {
    match v {
        Value::Object(m) => {
            for (k, val) in m {
                let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                json_leaves(val, &path, out);
            }
        }
        _ => out.push(prefix.to_string()),
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + usize::from(a[i - 1] != b[j - 1]));
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

// ---------------------------------------------------------------------------------------------
// YAML output (for `--print-config` and `tr-infer config`)
// ---------------------------------------------------------------------------------------------

/// Render a JSON value as YAML. `skip_null` drops unset leaves (and sections that become empty),
/// which is what makes the merged view readable.
pub fn to_yaml(v: &Value, skip_null: bool) -> String {
    let mut s = String::new();
    emit(v, 0, skip_null, &mut s);
    if s.is_empty() {
        s.push_str("{}\n");
    }
    s
}

fn is_empty(v: &Value, skip_null: bool) -> bool {
    match v {
        Value::Null => skip_null,
        Value::Object(m) => m.values().all(|x| is_empty(x, skip_null)),
        _ => false,
    }
}

fn emit(v: &Value, indent: usize, skip_null: bool, out: &mut String) {
    let pad = "  ".repeat(indent);
    match v {
        Value::Object(m) => {
            for (k, val) in m {
                if is_empty(val, skip_null) {
                    continue;
                }
                match val {
                    Value::Object(_) => {
                        out.push_str(&format!("{pad}{k}:\n"));
                        emit(val, indent + 1, skip_null, out);
                    }
                    Value::Array(a) if !a.is_empty() => {
                        out.push_str(&format!("{pad}{k}:\n"));
                        for item in a {
                            out.push_str(&format!("{pad}  - {}\n", scalar(item)));
                        }
                    }
                    Value::Array(_) => out.push_str(&format!("{pad}{k}: []\n")),
                    _ => out.push_str(&format!("{pad}{k}: {}\n", scalar(val))),
                }
            }
        }
        _ => out.push_str(&format!("{pad}{}\n", scalar(v))),
    }
}

/// f32 arrives here widened to f64 (serde_json has only f64), so 0.8f32 would print as
/// 0.800000011920929. Anything that survives a round trip through f32 is printed as that f32's
/// shortest form; a genuine f64 like 0.1 does not survive and keeps its own shortest form.
fn float(v: f64) -> String {
    let s = if (v as f32) as f64 == v { format!("{}", v as f32) } else { format!("{v}") };
    if s.contains(['.', 'e', 'E', 'n', 'i']) {
        s
    } else {
        format!("{s}.0")
    }
}

fn scalar(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Number(n) if n.as_i64().is_none() && n.as_u64().is_none() => n.as_f64().map(float).unwrap_or_else(|| n.to_string()),
        Value::String(s) => {
            let plain = !s.is_empty() && s.bytes().all(|c| c.is_ascii_alphanumeric() || b"_./+-@".contains(&c)) && !matches!(s.as_str(), "true" | "false" | "null" | "yes" | "no" | "on" | "off" | "~") && s.parse::<f64>().is_err();
            if plain {
                s.clone()
            } else {
                format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n"))
            }
        }
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// Resolved settings: the merged request plus the defaults, per command
// ---------------------------------------------------------------------------------------------

/// Everything a model load needs. `ctx_default` differs per command, see the module docs.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeCfg {
    pub pack: Option<PathBuf>,
    pub ctx: usize,
    pub batch: usize,
    pub kv: String,
    pub cores_per_node: Option<usize>,
    pub ple_mmap: bool,
    pub mtp: Option<PathBuf>,
    pub vision: Option<PathBuf>,
    pub spec_k: usize,
    pub image_min_tokens: usize,
    pub image_max_tokens: usize,
}

impl RuntimeCfg {
    /// The pack directory, or an error naming every way it can be set.
    pub fn pack(&self) -> Result<&Path> {
        match &self.pack {
            Some(p) => Ok(p.as_path()),
            None => bail!("no pack: pass --pack DIR, set `pack:` in a config file, or export TR_PACK"),
        }
    }
    pub fn kv(&self) -> Result<tr_model::state::KvType> {
        tr_model::state::KvType::parse(&self.kv).ok_or_else(|| anyhow::anyhow!("kv must be f32, f16 or bf16 (got {})", self.kv))
    }
}

/// Sampling with the defaults applied; `temp_default` differs between `generate` and `serve`.
#[derive(Debug, Clone, Serialize)]
pub struct SamplingCfg {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
    pub think_temp: Option<f32>,
    pub think_top_k: Option<usize>,
    pub think_top_p: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MoeCfg {
    pub mass: Option<f32>,
    pub min: Option<usize>,
    pub max: Option<usize>,
    pub basis: String,
    pub profile: bool,
}

impl MoeCfg {
    pub fn kernel(&self) -> Result<tr_kernels::router::MoeArgs> {
        let basis = match self.basis.as_str() {
            "top" => tr_kernels::router::MassBasis::Top,
            "all" => tr_kernels::router::MassBasis::All,
            other => bail!("moe basis must be top or all, not {other}"),
        };
        Ok(tr_kernels::router::MoeArgs { mass: self.mass, min: self.min, max: self.max, basis })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ServerCfg {
    pub continuous: bool,
    pub max_sequences: usize,
    pub kv_cache_mib: Option<usize>,
    pub prefill_chunk: usize,
    pub max_queue: usize,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub no_auth: bool,
    pub model_name: Option<String>,
    pub reasoning: String,
}

/// The prefix cache with defaults applied; `dir` unset means no cache.
#[derive(Debug, Clone, Serialize)]
pub struct CacheCfg {
    pub dir: Option<PathBuf>,
    pub backend: String,
    pub chunk_len: usize,
    pub max_gib: f64,
    pub snapshots: String,
    pub snapshot_min_tokens: usize,
    pub staging_mib: usize,
    pub roles: tr_cache::Policies,
}

impl CacheCfg {
    /// Validate the per-role policies (priority range) — the rest is checked where it is used.
    pub fn validate(&self) -> Result<()> {
        for r in tr_cache::Role::ALL {
            let p = self.roles.get(r);
            if p.priority > tr_cache::PRIORITY_MAX {
                bail!("cache.roles.{}.priority must be 0..={} (got {})", r.name(), tr_cache::PRIORITY_MAX, p.priority);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GenerateCfg {
    pub prompt: String,
    pub prompt_file: Option<PathBuf>,
    pub n_predict: usize,
    pub chat: bool,
    pub no_think: bool,
    pub dump: bool,
    pub images: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchCfg {
    pub pp: usize,
    pub tg: usize,
    pub reps: usize,
    pub tg_batch: Option<usize>,
}

impl AppConfig {
    pub fn runtime(&self, ctx_default: usize) -> RuntimeCfg {
        RuntimeCfg {
            pack: self.pack.clone(),
            ctx: self.runtime.ctx.unwrap_or(ctx_default),
            batch: self.runtime.batch.unwrap_or(defaults::BATCH),
            kv: self.runtime.kv.clone().unwrap_or_else(|| defaults::KV.into()),
            cores_per_node: self.runtime.cores_per_node,
            ple_mmap: self.runtime.ple_mmap.unwrap_or(false),
            mtp: self.overlays.mtp.clone(),
            vision: self.overlays.vision.clone(),
            spec_k: self.spec.k.unwrap_or(defaults::SPEC_K),
            image_min_tokens: self.image.min_tokens.unwrap_or(defaults::IMAGE_MIN_TOKENS),
            image_max_tokens: self.image.max_tokens.unwrap_or(defaults::IMAGE_MAX_TOKENS),
        }
    }
    pub fn sampling(&self, temp_default: f32) -> SamplingCfg {
        SamplingCfg {
            temp: self.sampling.temp.unwrap_or(temp_default),
            top_k: self.sampling.top_k.unwrap_or(defaults::TOP_K),
            top_p: self.sampling.top_p.unwrap_or(defaults::TOP_P),
            seed: self.sampling.seed.unwrap_or(defaults::SEED),
            think_temp: self.sampling.think_temp,
            think_top_k: self.sampling.think_top_k,
            think_top_p: self.sampling.think_top_p,
        }
    }
    pub fn moe(&self) -> MoeCfg {
        MoeCfg {
            mass: self.moe.mass,
            min: self.moe.min,
            max: self.moe.max,
            basis: self.moe.basis.clone().unwrap_or_else(|| defaults::MOE_BASIS.into()),
            profile: self.moe.profile.unwrap_or(false),
        }
    }
    pub fn server(&self) -> ServerCfg {
        ServerCfg {
            continuous: self.server.continuous.unwrap_or(false),
            max_sequences: self.server.max_sequences.unwrap_or(16),
            kv_cache_mib: self.server.kv_cache_mib,
            prefill_chunk: self.server.prefill_chunk.unwrap_or(32),
            max_queue: self.server.max_queue.unwrap_or(128),
            host: self.server.host.clone().unwrap_or_else(|| defaults::HOST.into()),
            port: self.server.port.unwrap_or(defaults::PORT),
            api_key: self.server.api_key.clone().filter(|k| !k.is_empty()),
            no_auth: self.server.no_auth.unwrap_or(false),
            model_name: self.server.model_name.clone().filter(|k| !k.is_empty()),
            reasoning: self.server.reasoning.clone().unwrap_or_else(|| defaults::REASONING.into()),
        }
    }
    pub fn cache(&self) -> CacheCfg {
        CacheCfg {
            dir: self.cache.dir.clone().filter(|d| !d.as_os_str().is_empty()),
            backend: self.cache.backend.clone().unwrap_or_else(|| defaults::CACHE_BACKEND.into()),
            chunk_len: self.cache.chunk_len.unwrap_or(defaults::CACHE_CHUNK_LEN),
            max_gib: self.cache.max_gib.unwrap_or(defaults::CACHE_MAX_GIB),
            snapshots: self.cache.snapshots.clone().unwrap_or_else(|| defaults::CACHE_SNAPSHOTS.into()),
            snapshot_min_tokens: self.cache.snapshot_min_tokens.unwrap_or(defaults::CACHE_SNAPSHOT_MIN_TOKENS),
            staging_mib: self.cache.staging_mib.unwrap_or(defaults::CACHE_STAGING_MIB),
            roles: {
                let mut p = tr_cache::Policies::default();
                for r in tr_cache::Role::ALL {
                    let c = self.cache.roles.get(r);
                    let d = tr_cache::RolePolicy::default();
                    *p.get_mut(r) = tr_cache::RolePolicy { persist: c.persist.unwrap_or(d.persist), priority: c.priority.unwrap_or(d.priority), ttl: c.ttl.filter(|&t| t > 0), snapshot: c.snapshot.unwrap_or(d.snapshot) };
                }
                p
            },
        }
    }
    pub fn generate(&self) -> GenerateCfg {
        GenerateCfg {
            prompt: self.generate.prompt.clone().unwrap_or_else(|| defaults::PROMPT.into()),
            prompt_file: self.generate.prompt_file.clone(),
            n_predict: self.generate.n_predict.unwrap_or(defaults::N_PREDICT),
            chat: self.generate.chat.unwrap_or(false),
            no_think: self.generate.no_think.unwrap_or(false),
            dump: self.generate.dump.unwrap_or(false),
            images: self.generate.images.clone().unwrap_or_default(),
        }
    }
    pub fn bench(&self) -> BenchCfg {
        BenchCfg {
            pp: self.bench.pp.unwrap_or(defaults::BENCH_PP),
            tg: self.bench.tg.unwrap_or(defaults::BENCH_TG),
            reps: self.bench.reps.unwrap_or(defaults::BENCH_REPS),
            tg_batch: self.bench.tg_batch,
        }
    }
    /// The heap policy, which `TR_PIN_HEAP` also sets (see `main`).
    pub fn pin_heap(&self) -> String {
        self.runtime.pin_heap.clone().unwrap_or_else(|| defaults::PIN_HEAP.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> Option<HashMap<String, String>> {
        Some(pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
    }

    struct TmpYaml(PathBuf);
    impl TmpYaml {
        fn new(tag: &str, body: &str) -> Self {
            let p = std::env::temp_dir().join(format!("tr-infer-cfg-{tag}-{}.yaml", std::process::id()));
            std::fs::write(&p, body).unwrap();
            TmpYaml(p)
        }
    }
    impl Drop for TmpYaml {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn schema_keys_cover_the_sections() {
        let k = known_keys();
        for want in ["pack", "runtime.ctx", "runtime.ple_mmap", "overlays.mtp", "spec.k", "moe.mass", "moe.basis", "sampling.temp", "sampling.think_top_k", "image.max_tokens", "server.api_key", "server.reasoning", "cache.dir", "cache.snapshots", "cache.roles.reasoning.persist", "cache.roles.tool.ttl", "generate.images", "bench.tg_batch", "load_bench.gib_per_node"] {
            assert!(k.iter().any(|x| x == want), "missing {want} in {k:?}");
        }
        // sections are tables, not leaves
        assert!(!k.iter().any(|x| x == "runtime"));
    }

    #[test]
    fn file_then_env_then_cli() {
        let f = TmpYaml::new("prec", "pack: /packs/base\nruntime:\n  ctx: 1000\n  batch: 64\n  kv: bf16\nmoe:\n  mass: 0.8\nserver:\n  port: 1111\n  host: 0.0.0.0\n");
        // file only
        let (c, p) = load_from(&[f.0.clone()], false, true, Value::Object(Map::new()), None).unwrap();
        assert_eq!(c.pack.as_deref(), Some(Path::new("/packs/base")));
        assert_eq!(c.runtime.ctx, Some(1000));
        assert_eq!(c.server.port, Some(1111));
        assert_eq!(p.files, vec![f.0.clone()]);
        // env beats the file, and only for the keys it names
        let e = env(&[("TR_RUNTIME__CTX", "2000"), ("TR_SERVER__PORT", "2222"), ("TR_DUMP_VEC", "head_res")]);
        let (c, _) = load_from(&[f.0.clone()], false, false, Value::Object(Map::new()), e.clone()).unwrap();
        assert_eq!(c.runtime.ctx, Some(2000));
        assert_eq!(c.runtime.batch, Some(64), "untouched keys survive");
        assert_eq!(c.server.port, Some(2222));
        assert_eq!(c.server.host.as_deref(), Some("0.0.0.0"));
        // the command line beats both
        let mut ov = Ov::default();
        ov.set("runtime.ctx", &Some(3000usize));
        ov.set_f32("moe.mass", &Some(0.95));
        let (c, _) = load_from(&[f.0.clone()], false, false, ov.into_value(), e).unwrap();
        assert_eq!(c.runtime.ctx, Some(3000));
        assert_eq!(c.server.port, Some(2222));
        assert_eq!(c.moe.mass, Some(0.95));
        // --no-config drops the file, --no-env drops the environment
        let (c, p) = load_from(&[f.0.clone()], true, true, Value::Object(Map::new()), None).unwrap();
        assert_eq!(c, AppConfig::default());
        assert!(p.files.is_empty());
    }

    #[test]
    fn env_parses_scalars_lists_and_the_config_path() {
        let e = env(&[("TR_RUNTIME__PLE_MMAP", "true"), ("TR_SAMPLING__TEMP", "0.6"), ("TR_SPEC__K", "2"), ("TR_GENERATE__IMAGES", "a.png,b.jpg"), ("TR_PACK", "/packs/x")]);
        let (c, _) = load_from(&[], true, false, Value::Object(Map::new()), e).unwrap();
        assert_eq!(c.runtime.ple_mmap, Some(true));
        assert_eq!(c.sampling.temp, Some(0.6));
        assert_eq!(c.spec.k, Some(2));
        assert_eq!(c.generate.images, Some(vec![PathBuf::from("a.png"), PathBuf::from("b.jpg")]));
        assert_eq!(c.pack.as_deref(), Some(Path::new("/packs/x")));
        // TR_CONFIG names the file when --config does not
        let f = TmpYaml::new("trconfig", "runtime:\n  ctx: 77\n");
        let e = env(&[("TR_CONFIG", f.0.to_str().unwrap())]);
        let (c, p) = load_from(&[], false, false, Value::Object(Map::new()), e).unwrap();
        assert_eq!(c.runtime.ctx, Some(77));
        assert_eq!(p.files, vec![f.0.clone()]);
    }

    #[test]
    fn unknown_file_keys_are_rejected_with_a_hint() {
        let f = TmpYaml::new("typo", "runtime:\n  cxt: 4096\n");
        let e = load_from(&[f.0.clone()], false, true, Value::Object(Map::new()), None).unwrap_err().to_string();
        assert!(e.contains("runtime.cxt"), "{e}");
        assert!(e.contains("runtime.ctx"), "{e}");
        let f = TmpYaml::new("typo2", "moe:\n  masss: 0.9\nnonsense: 1\n");
        let e = load_from(&[f.0.clone()], false, true, Value::Object(Map::new()), None).unwrap_err().to_string();
        assert!(e.contains("moe.masss") && e.contains("moe.mass"), "{e}");
        assert!(e.contains("nonsense"), "{e}");
        // a missing --config file is an error, a missing searched file is not
        assert!(load_from(&[PathBuf::from("/nope/x.yaml")], false, true, Value::Object(Map::new()), None).is_err());
        // an empty or absent section is neither a typo nor a parse error
        let f = TmpYaml::new("empty", "runtime:\nmoe: {}\nserver:\n  port: 1\n");
        let (c, _) = load_from(&[f.0.clone()], false, true, Value::Object(Map::new()), None).unwrap();
        assert_eq!(c.server.port, Some(1));
        assert_eq!(c.runtime, Runtime::default());
    }

    #[test]
    fn overrides_nest_and_keep_f32_precision() {
        let mut o = Ov::default();
        o.set("runtime.ctx", &Some(8usize));
        o.set("runtime.kv", &Some("bf16".to_string()));
        o.set::<usize>("runtime.batch", &None);
        o.set_f32("sampling.temp", &Some(0.95));
        o.set_f32("sampling.top_p", &None);
        o.set_list("generate.images", &[PathBuf::from("a.png")]);
        o.set_list::<PathBuf>("overlays.mtp", &[]);
        let v = o.into_value();
        assert_eq!(v["runtime"]["ctx"], serde_json::json!(8));
        assert_eq!(v["runtime"]["kv"], serde_json::json!("bf16"));
        assert!(v["runtime"].get("batch").is_none());
        assert_eq!(v["sampling"]["temp"].to_string(), "0.95", "f32 must not widen to 0.949999988");
        assert!(v["sampling"].get("top_p").is_none());
        assert_eq!(v["generate"]["images"], serde_json::json!(["a.png"]));
        assert!(v.get("overlays").is_none());
        // and it survives the merge as an f32
        let (c, _) = load_from(&[], true, true, v, None).unwrap();
        assert_eq!(c.sampling.temp, Some(0.95f32));
    }

    #[test]
    fn cache_roles_nest_in_files_env_and_overrides() {
        let f = TmpYaml::new("roles", "cache:\n  dir: /kv\n  roles:\n    reasoning:\n      persist: false\n    tool: { ttl: 3600, priority: 1 }\n    user:\n");
        let e = env(&[("TR_CACHE__ROLES__SYSTEM__PRIORITY", "9")]);
        let mut ov = Ov::default();
        ov.set("cache.roles.assistant.snapshot", &Some(false));
        let (c, _) = load_from(&[f.0.clone()], false, false, ov.into_value(), e).unwrap();
        let p = c.cache().roles;
        assert!(!p.reasoning.persist);
        assert_eq!((p.tool.ttl, p.tool.priority), (Some(3600), 1));
        assert_eq!(p.system.priority, 9);
        assert!(!p.assistant.snapshot);
        assert_eq!(p.user, tr_cache::RolePolicy::default());
        // an unknown role key in a file is caught like any other typo
        let g = TmpYaml::new("roles2", "cache:\n  roles:\n    reasoning:\n      persits: false\n");
        let err = load_from(&[g.0.clone()], false, true, Value::Object(Map::new()), None).unwrap_err().to_string();
        assert!(err.contains("cache.roles.reasoning.persits") && err.contains("persist"), "{err}");
    }

    #[test]
    fn defaults_are_applied_per_command() {
        let c = AppConfig::default();
        assert_eq!(c.runtime(defaults::CTX).ctx, 4096);
        assert_eq!(c.runtime(defaults::CTX_SERVE).ctx, 8192);
        assert_eq!(c.sampling(defaults::TEMP).temp, 0.0);
        assert_eq!(c.sampling(defaults::TEMP_SERVE).temp, 0.6);
        assert_eq!(c.runtime(defaults::CTX).batch, 256);
        assert_eq!(c.runtime(defaults::CTX).kv, "f16");
        assert_eq!(c.runtime(defaults::CTX).spec_k, 3);
        assert_eq!(c.moe().basis, "top");
        assert_eq!(c.server().port, 8080);
        assert_eq!(c.generate().n_predict, 32);
        assert!(c.cache().dir.is_none(), "no cache unless a directory is set");
        assert_eq!(c.cache().chunk_len, 256);
        assert_eq!(c.cache().snapshots, "message");
        assert_eq!(c.cache().max_gib, 32.0);
        assert_eq!(c.cache().roles, tr_cache::Policies::default());
        // per-role policies resolve with defaults per field; ttl 0 means none
        let mut c = AppConfig::default();
        c.cache.roles.reasoning.persist = Some(false);
        c.cache.roles.tool.ttl = Some(3600);
        c.cache.roles.tool.priority = Some(1);
        c.cache.roles.user.ttl = Some(0);
        let p = c.cache().roles;
        assert!(!p.reasoning.persist && p.reasoning.priority == 5 && p.reasoning.snapshot);
        assert_eq!(p.tool.ttl, Some(3600));
        assert_eq!(p.tool.priority, 1);
        assert_eq!(p.user.ttl, None);
        assert!(c.cache().validate().is_ok());
        c.cache.roles.system.priority = Some(12);
        assert!(c.cache().validate().is_err());
        assert!(c.runtime(defaults::CTX).pack().is_err());
        // one runtime.ctx in the file still means "for whatever I run"
        let mut c = AppConfig::default();
        c.runtime.ctx = Some(60000);
        assert_eq!(c.runtime(defaults::CTX).ctx, 60000);
        assert_eq!(c.runtime(defaults::CTX_SERVE).ctx, 60000);
        assert!(c.runtime(defaults::CTX).kv().is_ok());
        c.runtime.kv = Some("q8".into());
        assert!(c.runtime(defaults::CTX).kv().is_err());
        c.moe.basis = Some("sideways".into());
        assert!(c.moe().kernel().is_err());
    }

    #[test]
    fn yaml_output_is_readable_and_reloadable() {
        let f = TmpYaml::new("yaml", "pack: /packs/base\nruntime:\n  ctx: 1000\n  ple_mmap: true\nserver:\n  host: 0.0.0.0\n  api_key: \"s3cr:et\"\ngenerate:\n  images:\n    - a.png\n");
        let (c, _) = load_from(&[f.0.clone()], false, true, Value::Object(Map::new()), None).unwrap();
        let y = to_yaml(&serde_json::to_value(&c).unwrap(), true);
        assert!(y.contains("pack: /packs/base"), "{y}");
        assert!(y.contains("  ctx: 1000"), "{y}");
        assert!(y.contains("  ple_mmap: true"), "{y}");
        assert!(y.contains("\"s3cr:et\""), "{y}");
        assert!(y.contains("    - a.png"), "{y}");
        assert!(!y.contains("null"), "unset leaves are dropped:\n{y}");
        // f32 must read back as it was written, not as its f64 widening
        let mut c3 = c.clone();
        c3.moe.mass = Some(0.8);
        c3.sampling.top_p = Some(0.95);
        c3.load_bench.gib_per_node = Some(0.1);
        let y3 = to_yaml(&serde_json::to_value(&c3).unwrap(), true);
        assert!(y3.contains("mass: 0.8") && !y3.contains("0.800000011920929"), "{y3}");
        assert!(y3.contains("top_p: 0.95"), "{y3}");
        assert!(y3.contains("gib_per_node: 0.1"), "f64 keeps its own precision:\n{y3}");
        // what it prints, it can read back
        let g = TmpYaml::new("yaml2", &y);
        let (c2, _) = load_from(&[g.0.clone()], false, true, Value::Object(Map::new()), None).unwrap();
        assert_eq!(c, c2);
        // the resolved view keeps the nulls, so "unset" stays visible
        let r = to_yaml(&serde_json::to_value(c.runtime(defaults::CTX)).unwrap(), false);
        assert!(r.contains("cores_per_node: null"), "{r}");
    }
}

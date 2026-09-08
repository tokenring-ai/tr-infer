//! OpenAI-compatible HTTP server: `POST /v1/chat/completions` (streaming and not, with
//! `reasoning_content` split out of the `<think>` block), `GET /v1/models`, `GET /health`.
//! Bearer API key on everything under `/v1`. One engine thread; connections queue on a channel.

pub mod chat;
pub mod engine;
pub mod http;
pub mod images;
pub mod prefix_cache;
pub mod tools;

use anyhow::Result;
use chat::{ChatRequest, Thinking};
use engine::{Engine, Event, GenParams, Job, JobImage, Sampling};
use serde_json::{json, Value};
use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tr_model::tokenizer::Tok;

pub struct ServeOpts {
    pub continuous: bool,
    pub max_sequences: usize,
    pub kv_cache_mib: Option<usize>,
    pub prefill_chunk: usize,
    pub max_queue: usize,
    pub pack: std::path::PathBuf,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub model_name: Option<String>,
    pub ctx: usize,
    pub batch: usize,
    pub cores_per_node: Option<usize>,
    pub ple_mmap: bool,
    pub kv: tr_model::state::KvType,
    pub temp: f32,
    pub top_p: f32,
    pub top_k: usize,
    /// Sampling inside the `<think>` block when the request does not say (None = as the answer)
    pub think_temp: Option<f32>,
    pub think_top_p: Option<f32>,
    pub think_top_k: Option<usize>,
    pub reasoning: Thinking,
    /// MTP overlay pack (speculative decoding) and draft tokens per round.
    pub mtp: Option<std::path::PathBuf>,
    pub spec_k: usize,
    /// Vision encoder overlay pack (image input) and image size limits in tokens.
    pub vision: Option<std::path::PathBuf>,
    pub image_min_tokens: usize,
    pub image_max_tokens: usize,
    /// Cumulative-mass expert routing (mass unset = the model's fixed top-k).
    pub moe: tr_kernels::router::MoeArgs,
    /// Persistent prefix cache (None = off).
    pub cache: Option<prefix_cache::CacheOpts>,
}

/// Image preprocessing settings of the loaded vision encoder and the pad token it expands.
struct VisionInfo {
    prep: tr_model::image::PrepParams,
    pad: u32,
}

struct Shared {
    jobs: Mutex<SyncSender<Job>>,
    queued: Arc<AtomicUsize>,
    max_queue: usize,
    max_choices: usize,
    page_capacity: usize,
    stats: Arc<engine::SchedulerStats>,
    tok: Arc<Tok>,
    api_key: Option<String>,
    model_id: String,
    ctx: usize,
    temp: f32,
    top_p: f32,
    top_k: usize,
    think_temp: Option<f32>,
    think_top_p: Option<f32>,
    think_top_k: Option<usize>,
    reasoning: Thinking,
    counter: AtomicU64,
    started: u64,
    vision: Option<VisionInfo>,
}

pub fn serve(o: ServeOpts) -> Result<()> {
    use tr_model::exec::Model;
    let t0 = std::time::Instant::now();
    anyhow::ensure!(o.max_queue > 0, "max queue must be positive");
    if o.continuous {
        anyhow::ensure!(o.mtp.is_none() && o.vision.is_none(), "continuous batching requires text-only execution without MTP or vision");
        anyhow::ensure!(o.max_sequences > 0 && o.max_sequences < o.batch, "max sequences must be positive and smaller than batch size");
        anyhow::ensure!(o.prefill_chunk > 0, "prefill chunk must be positive");
        anyhow::ensure!(o.kv_cache_mib.is_some_and(|n| n > 0), "--continuous requires a positive --kv-cache-mib budget per tile");
    }
    let spec_k = if o.mtp.is_some() { o.spec_k } else { 0 };
    anyhow::ensure!(o.mtp.is_none() || (1..=7).contains(&o.spec_k), "--spec-k must be 1..7 with --mtp");
    let mut model = Model::load_with(&o.pack, o.ctx, o.cores_per_node, &tr_model::weights::LoadOptions { ple_mmap: o.ple_mmap, batch_max: o.batch.max(spec_k + 1).max(1), kv: o.kv, mtp: o.mtp.clone(), spec_k, vision: o.vision.clone(), image_min_tokens: o.image_min_tokens, image_max_tokens: o.image_max_tokens, moe: None })?;
    model.set_moe(o.moe.resolve(model.cfg.n_expert_used, model.cfg.n_expert).map_err(anyhow::Error::msg)?)?;
    if let Some(p) = model.moe_policy() {
        eprintln!("expert routing: cumulative mass {} of {:?} with {}..{} experts per token (model default: top-{})", p.mass, p.basis, p.min, p.max, model.cfg.n_expert_used);
    }
    eprintln!("{} (total {:.1} s)", model.load_summary(), t0.elapsed().as_secs_f64());
    if o.continuous { model.enable_sequences(o.max_sequences, o.kv_cache_mib.unwrap())?; }
    let page_capacity = model.page_capacity();
    let tok = Arc::new(Tok::load(&o.pack.join(&model.manifest.tokenizer))?);
    let prefix = match &o.cache {
        Some(c) => {
            let pc = prefix_cache::PrefixCache::open(c, &model.cache_identity())?;
            if o.continuous { pc.warm_staging(model.snapshot_bytes(model.cfg.n_vocab), model.rows_bytes(0, c.chunk_len.min(o.ctx))); }
            eprintln!("{}", pc.describe());
            Some(pc)
        }
        None => None,
    };
    let vision = match model.vision.as_ref() {
        Some(v) => {
            let pad = tok.token_id("<|image_pad|>").ok_or_else(|| anyhow::anyhow!("tokenizer has no <|image_pad|> token"))?;
            eprintln!("vision encoder loaded: {} layers, images of {}..{} tokens", v.cfg.n_layer, o.image_min_tokens, o.image_max_tokens);
            Some(VisionInfo { prep: v.cfg.prep_params(), pad })
        }
        None => None,
    };
    let model_id = o.model_name.clone().unwrap_or_else(|| o.pack.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "tr-infer".into()));
    let (jtx, jrx): (SyncSender<Job>, Receiver<Job>) = mpsc::sync_channel(o.max_queue);
    let listener = TcpListener::bind((o.host.as_str(), o.port))?;
    let stats = Arc::new(engine::SchedulerStats::default());
    let shared = Arc::new(Shared {
        stats: stats.clone(),
        jobs: Mutex::new(jtx),
        queued: Arc::new(AtomicUsize::new(0)),
        max_queue: o.max_queue,
        max_choices: if o.continuous { o.max_sequences } else { 1 },
        page_capacity,
        tok: tok.clone(),
        api_key: o.api_key,
        model_id,
        ctx: o.ctx,
        temp: o.temp,
        top_p: o.top_p,
        top_k: o.top_k,
        think_temp: o.think_temp,
        think_top_p: o.think_top_p,
        think_top_k: o.think_top_k,
        reasoning: o.reasoning,
        counter: AtomicU64::new(0),
        started: now(),
        vision,
    });
    eprintln!(
        "listening on http://{}:{}/v1  model id {:?}  ctx {}  auth {}  default reasoning {:?}",
        o.host,
        o.port,
        shared.model_id,
        o.ctx,
        if shared.api_key.is_some() { "bearer key" } else { "none" },
        o.reasoning
    );
    std::thread::Builder::new().name("http-accept".into()).spawn(move || {
        for s in listener.incoming() {
            match s {
                Ok(s) => {
                    let sh = shared.clone();
                    let _ = std::thread::Builder::new().name("http-conn".into()).spawn(move || handle_conn(s, sh));
                }
                Err(e) => eprintln!("accept: {e}"),
            }
        }
    })?;
    let mut engine = Engine::new(model, tok, spec_k, prefix);
    engine.stats = stats;
    if o.continuous { engine.run_continuous(jrx, o.prefill_chunk); } else { engine.run(jrx); }
    Ok(())
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn error_json(status: u16, msg: &str) -> (u16, Vec<u8>) {
    let ty = match status {
        401 => "authentication_error",
        404 => "not_found_error",
        400 => "invalid_request_error",
        _ => "server_error",
    };
    (status, json!({"error": {"message": msg, "type": ty, "param": null, "code": null}}).to_string().into_bytes())
}

fn handle_conn(mut s: TcpStream, sh: Arc<Shared>) {
    let _ = s.set_nodelay(true);
    let _ = s.set_read_timeout(Some(Duration::from_secs(60)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(60)));
    let mut r = BufReader::new(match s.try_clone() {
        Ok(c) => c,
        Err(_) => return,
    });
    loop {
        let req = match http::read_request(&mut r) {
            Ok(Some(q)) => q,
            Ok(None) => return,
            Err(e) => {
                if e.kind() == std::io::ErrorKind::InvalidData {
                    let (st, body) = error_json(400, &e.to_string());
                    let _ = http::write_response(&mut s, st, "application/json", &body, false);
                }
                return;
            }
        };
        let keep = req.keep_alive;
        let done = route(&mut s, &req, &sh);
        if done.is_err() || !keep {
            return;
        }
    }
}

fn authorized(req: &http::Request, sh: &Shared) -> bool {
    match &sh.api_key {
        None => true,
        Some(k) => req.header("authorization").map(|v| v.strip_prefix("Bearer ").map(|t| t.trim() == k).unwrap_or(false)).unwrap_or(false),
    }
}

fn route(s: &mut TcpStream, req: &http::Request, sh: &Arc<Shared>) -> std::io::Result<()> {
    let keep = req.keep_alive;
    let path = req.path.trim_end_matches('/');
    if req.method == "OPTIONS" {
        return http::write_response(s, 204, "text/plain", b"", keep);
    }
    if path == "/health" || path == "" {
        let body = json!({"status": "ok", "model": sh.model_id, "uptime_s": now() - sh.started,
            "scheduler": {"queued_requests": sh.queued.load(Ordering::Relaxed), "active_sequences": sh.stats.active.load(Ordering::Relaxed),
                "idle_pages": sh.stats.idle_pages.load(Ordering::Relaxed), "kv_total_pages": sh.page_capacity, "kv_free_pages": sh.stats.free_pages.load(Ordering::Relaxed),
                "iterations": sh.stats.iterations.load(Ordering::Relaxed), "decode_rows": sh.stats.decode_rows.load(Ordering::Relaxed),
                "prefill_rows": sh.stats.prefill_rows.load(Ordering::Relaxed), "max_batch_sequences": sh.stats.max_batch_sequences.load(Ordering::Relaxed)}}).to_string();
        return http::write_response(s, 200, "application/json", body.as_bytes(), keep);
    }
    if !authorized(req, sh) {
        let (st, body) = error_json(401, "invalid or missing API key (Authorization: Bearer ...)");
        return http::write_response(s, st, "application/json", &body, keep);
    }
    let model_obj = json!({"id": sh.model_id, "object": "model", "created": sh.started, "owned_by": "tr-infer"});
    match (req.method.as_str(), path) {
        ("GET", "/v1/models") => {
            let body = json!({"object": "list", "data": [model_obj]}).to_string();
            http::write_response(s, 200, "application/json", body.as_bytes(), keep)
        }
        ("GET", p) if p.starts_with("/v1/models/") => {
            let id = &p["/v1/models/".len()..];
            if id == sh.model_id {
                http::write_response(s, 200, "application/json", model_obj.to_string().as_bytes(), keep)
            } else {
                let (st, body) = error_json(404, &format!("model {id:?} not found"));
                http::write_response(s, st, "application/json", &body, keep)
            }
        }
        ("POST", "/v1/chat/completions") => chat_completions(s, req, sh),
        _ => {
            let (st, body) = error_json(404, &format!("no route for {} {}", req.method, req.path));
            http::write_response(s, st, "application/json", &body, keep)
        }
    }
}

struct Prepared {
    ids: Vec<u32>,
    spans: Vec<chat::Span>,
    images: Vec<JobImage>,
    params: GenParams,
    stream: bool,
    include_usage: bool,
    specs: Vec<tools::ToolSpec>,
}

fn prepare(req: &ChatRequest, sh: &Shared) -> Result<Prepared, String> {
    if !(1..=sh.max_choices).contains(&req.n.unwrap_or(1)) {
        return Err(format!("n must be 1..{} (enable continuous batching for multiple choices)", sh.max_choices));
    }
    let choice = req.tool_choice.as_ref().map(|c| c.as_str().unwrap_or("named").to_string()).unwrap_or_else(|| "auto".into());
    let tools: Vec<Value> = match choice.as_str() {
        "none" => Vec::new(),
        "auto" => req.tools.clone().unwrap_or_default(),
        _ => return Err("tool_choice must be \"auto\" or \"none\" (the model decides; forcing a call is not supported)".into()),
    };
    let mut specs = Vec::new();
    for t in &tools {
        specs.push(tools::ToolSpec::from_value(t).ok_or("each tool needs function.name")?);
    }
    let mode = chat::resolve_thinking(req, sh.reasoning)?;
    let rendered = chat::render_prompt_spans(&req.messages, mode, &tools)?;
    let (prompt, thinking_open, srcs) = (rendered.text, rendered.thinking_open, rendered.images);
    let (ids, offsets) = sh.tok.encode_with_offsets(&prompt, false).map_err(|e| e.to_string())?;
    let mut spans = chat::token_spans(&rendered.segs, &offsets);
    let mut job_images = Vec::new();
    let ids = if srcs.is_empty() {
        ids
    } else {
        let vi = sh.vision.as_ref().ok_or("image input needs the vision encoder (start the server with --vision)")?;
        let mut prepared = Vec::with_capacity(srcs.len());
        for (i, src) in srcs.iter().enumerate() {
            prepared.push(images::prepare(src, &vi.prep).map_err(|e| format!("image {}: {e}", i + 1))?);
        }
        let sizes: Vec<usize> = prepared.iter().map(|p| p.nx * p.ny).collect();
        spans = chat::expand_spans(&spans, &ids, vi.pad, &sizes);
        let (ids, places) = images::expand_pads(&ids, vi.pad, &prepared)?;
        for (pl, pr) in places.into_iter().zip(prepared) {
            job_images.push(JobImage { place: pl, patches: pr.patches, hash: pr.hash });
        }
        ids
    };
    if ids.len() >= sh.ctx {
        return Err(format!("prompt is {} tokens, the context window is {} tokens", ids.len(), sh.ctx));
    }
    let room = sh.ctx - ids.len();
    let max_tokens = req.max_completion_tokens.or(req.max_tokens).unwrap_or(room).min(room).max(1);
    if sh.page_capacity > 0 && engine::continuous_pages(ids.len(), max_tokens, req.n.unwrap_or(1)).map_or(true, |n| n > sh.page_capacity) {
        return Err("request exceeds the KV pool capacity; reduce n, prompt length, or max_tokens".into());
    }
    let check = |temp: f32, top_p: f32| -> Result<(), String> {
        if !(0.0..=2.0).contains(&temp) {
            return Err("temperature must be between 0 and 2".into());
        }
        if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
            return Err("top_p must be in (0, 1]".into());
        }
        Ok(())
    };
    let sampling = Sampling { temp: req.temperature.unwrap_or(sh.temp), top_k: req.top_k.unwrap_or(sh.top_k), top_p: req.top_p.unwrap_or(sh.top_p) };
    check(sampling.temp, sampling.top_p)?;
    // thinking phase: request reasoning_* (flat or nested) > server --think-* > the answer's settings
    let rp = req.reasoning.as_ref();
    let t_temp = req.reasoning_temperature.or(rp.and_then(|r| r.temperature)).or(sh.think_temp);
    let t_top_p = req.reasoning_top_p.or(rp.and_then(|r| r.top_p)).or(sh.think_top_p);
    let t_top_k = req.reasoning_top_k.or(rp.and_then(|r| r.top_k)).or(sh.think_top_k);
    let think = if mode != Thinking::Off && (t_temp.is_some() || t_top_p.is_some() || t_top_k.is_some()) {
        let t = Sampling { temp: t_temp.unwrap_or(sampling.temp), top_k: t_top_k.unwrap_or(sampling.top_k), top_p: t_top_p.unwrap_or(sampling.top_p) };
        check(t.temp, t.top_p).map_err(|e| format!("reasoning {e}"))?;
        Some(t)
    } else {
        None
    };
    let seed = match req.seed {
        Some(v) => v as u64,
        None => SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0) ^ sh.counter.load(Ordering::Relaxed).rotate_left(32),
    };
    let stop: Vec<String> = req.stop.as_ref().map(|s| match s {
        chat::Stop::One(x) => vec![x.clone()],
        chat::Stop::Many(v) => v.clone(),
    }).unwrap_or_default().into_iter().filter(|s| !s.is_empty()).collect();
    if stop.len() > 16 {
        return Err("at most 16 stop sequences".into());
    }
    Ok(Prepared {
        ids,
        spans,
        images: job_images,
        params: GenParams { max_tokens, sampling, think, seed, stop, thinking_open },
        stream: req.stream,
        include_usage: req.stream_options.as_ref().map(|o| o.include_usage).unwrap_or(false),
        specs,
    })
}

fn chat_completions(s: &mut TcpStream, req: &http::Request, sh: &Arc<Shared>) -> std::io::Result<()> {
    let keep = req.keep_alive;
    let creq: ChatRequest = match serde_json::from_slice(&req.body) {
        Ok(r) => r,
        Err(e) => { let (st, body) = error_json(400, &format!("invalid request body: {e}")); return http::write_response(s, st, "application/json", &body, keep); }
    };
    let p = match prepare(&creq, sh) {
        Ok(p) => p,
        Err(msg) => { let (st, body) = error_json(400, &msg); return http::write_response(s, st, "application/json", &body, keep); }
    };
    if sh.queued.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| (n < sh.max_queue).then_some(n + 1)).is_err() {
        let (st, body) = error_json(503, "request queue is full");
        return http::write_response(s, st, "application/json", &body, keep);
    }
    let counter = sh.counter.fetch_add(1, Ordering::Relaxed);
    let id = format!("chatcmpl-{:x}{:04x}", sh.started, counter);
    let created = now();
    let count = creq.n.unwrap_or(1);
    // A stalled client must not retain an unbounded stream or block model execution.
    let (tx, rx) = mpsc::sync_channel::<(usize, Event)>(256);
    let cancel = Arc::new(AtomicBool::new(false));
    struct CancelOnDrop(Arc<AtomicBool>);
    impl Drop for CancelOnDrop { fn drop(&mut self) { self.0.store(true, Ordering::Relaxed); } }
    let _cancel_on_drop = CancelOnDrop(cancel.clone());
    let n_prompt = p.ids.len();
    let stream = p.stream;
    let include_usage = p.include_usage;
    let mut choices: Vec<_> = (0..count).map(|_| HttpChoice {
        parser: tools::ToolStream::new(p.specs.clone()), content: String::new(), reasoning: String::new(), calls: Vec::new(), finish: None,
    }).collect();
    let job = Job { n: count, queued: sh.queued.clone(), id: id.clone(), ids: p.ids, spans: p.spans, params: p.params,
        tx: engine::EventTx { inner: tx, index: 0, cancel: cancel.clone() }, cancel: cancel.clone(), images: p.images };
    if sh.jobs.lock().map(|j| j.try_send(job).is_err()).unwrap_or(true) {
        sh.queued.fetch_sub(1, Ordering::Relaxed);
        let (st, body) = error_json(503, "engine unavailable or request queue full");
        return http::write_response(s, st, "application/json", &body, keep);
    }
    let mut started = false;
    let mut done = 0usize;
    let mut failure = None;
    while done < count {
        let (index, event) = match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(e) => e,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if http::peer_closed(s) { return Ok(()); }
                if cancel.load(Ordering::Relaxed) { failure = Some("request cancelled because its output buffer filled".to_string()); break; }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => { failure = Some("engine closed the response before completion".to_string()); break; }
        };
        if let Event::Error(e) = event { failure = Some(e); break; }
        if index >= count { failure = Some("invalid choice index from engine".to_string()); break; }
        if stream && !started {
            http::begin_event_stream(s, keep)?;
            started = true;
            for i in 0..count {
                http::write_chunk(s, completion_chunk(&id, created, &sh.model_id, i, json!({"role": "assistant", "content": ""}), None).as_bytes())?;
            }
        }
        let c = &mut choices[index];
        let chunk = |delta: Value, reason: Option<&str>| completion_chunk(&id, created, &sh.model_id, index, delta, reason);
        // Include the choice in tool-call IDs as well as the outer choices index.
        let choice_id = format!("{id}_{index}");
        match event {
            Event::Prefilled { prompt_tokens, reused, restored, restore_s, prompt_s, images, image_s } => {
                eprintln!("{id}: {prompt_tokens} prompt tokens, {reused} reused ({restored} restored in {restore_s:.3}s), prefill {prompt_s:.3}s, {images} images in {image_s:.3}s, {count} choices");
            }
            Event::Reasoning(t) => {
                if stream { http::write_chunk(s, chunk(json!({"reasoning_content": t}), None).as_bytes())?; }
                else { c.reasoning.push_str(&t); }
            }
            Event::Content(t) => {
                for o in c.parser.push(&t) { emit(s, o, stream, &mut c.content, &mut c.calls, &chunk, &choice_id)?; }
            }
            Event::Done(f) => {
                if c.finish.is_some() { continue; }
                for o in c.parser.finish() { emit(s, o, stream, &mut c.content, &mut c.calls, &chunk, &choice_id)?; }
                let reason = choice_finish(&c.calls, &f);
                if stream { http::write_chunk(s, chunk(json!({}), Some(reason)).as_bytes())?; }
                eprintln!("{id}[{index}]: {} completion tokens, {} reasoning, {:.3}s, finish {}, drafts {}/{}, experts {:.2}",
                    f.completion_tokens, f.reasoning_tokens, f.gen_s, f.reason, f.accepted, f.drafted, f.experts);
                c.finish = Some(f); done += 1;
            }
            Event::Error(_) => unreachable!(),
        }
    }
    if let Some(message) = failure {
        let (status, body) = error_json(500, &message);
        if !started { return http::write_response(s, status, "application/json", &body, keep); }
        http::write_chunk(s, format!("data: {}\n\n", String::from_utf8_lossy(&body)).as_bytes())?;
        http::write_chunk(s, b"data: [DONE]\n\n")?;
        return http::end_stream(s);
    }
    let completion: usize = choices.iter().filter_map(|c| c.finish.as_ref()).map(|f| f.completion_tokens).sum();
    let reasoning: usize = choices.iter().filter_map(|c| c.finish.as_ref()).map(|f| f.reasoning_tokens).sum();
    let usage = json!({"prompt_tokens": n_prompt, "completion_tokens": completion, "total_tokens": n_prompt + completion,
        "completion_tokens_details": {"reasoning_tokens": reasoning}});
    if stream {
        if !started { http::begin_event_stream(s, keep)?; }
        if include_usage {
            let obj = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": sh.model_id, "choices": [], "usage": usage});
            http::write_chunk(s, format!("data: {obj}\n\n").as_bytes())?;
        }
        http::write_chunk(s, b"data: [DONE]\n\n")?;
        http::end_stream(s)
    } else {
        let choices: Vec<_> = choices.into_iter().enumerate().map(|(index, c)| {
            let reason = choice_finish(&c.calls, c.finish.as_ref().unwrap());
            let mut message = json!({"role": "assistant", "content": if c.content.is_empty() && !c.calls.is_empty() { Value::Null } else { Value::String(c.content) }});
            if !c.calls.is_empty() { message["tool_calls"] = Value::Array(c.calls); }
            if !c.reasoning.is_empty() { message["reasoning_content"] = Value::String(c.reasoning); }
            json!({"index": index, "message": message, "logprobs": null, "finish_reason": reason})
        }).collect();
        let body = json!({"id": id, "object": "chat.completion", "created": created, "model": sh.model_id, "choices": choices, "usage": usage});
        http::write_response(s, 200, "application/json", body.to_string().as_bytes(), keep)
    }
}

struct HttpChoice {
    parser: tools::ToolStream,
    content: String,
    reasoning: String,
    calls: Vec<Value>,
    finish: Option<engine::Finish>,
}
fn choice_finish(calls: &[Value], f: &engine::Finish) -> &'static str {
    if !calls.is_empty() { "tool_calls" } else if f.reason == "length" { "length" } else { "stop" }
}
fn completion_chunk(id: &str, created: u64, model: &str, index: usize, delta: Value, reason: Option<&str>) -> String {
    let obj = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": index, "delta": delta, "logprobs": null, "finish_reason": reason}]});
    format!("data: {obj}\n\n")
}

/// Forward one parsed piece of the content stream: text as a `content` delta (or accumulated),
/// a tool call as a `tool_calls` delta with the complete arguments (or accumulated).
fn emit(s: &mut TcpStream, o: tools::Out, stream: bool, content: &mut String, calls: &mut Vec<Value>, chunk: &dyn Fn(Value, Option<&str>) -> String, id: &str) -> std::io::Result<()> {
    match o {
        tools::Out::Text(t) => {
            if stream {
                http::write_chunk(s, chunk(json!({"content": t}), None).as_bytes())
            } else {
                content.push_str(&t);
                Ok(())
            }
        }
        tools::Out::Call(c) => {
            let idx = calls.len();
            let call = json!({"id": format!("call_{}_{idx}", &id["chatcmpl-".len()..]), "type": "function",
                "function": {"name": c.name, "arguments": Value::Object(c.arguments).to_string()}});
            let mut delta_call = call.clone();
            delta_call["index"] = json!(idx);
            calls.push(call);
            if stream {
                http::write_chunk(s, chunk(json!({"tool_calls": [delta_call]}), None).as_bytes())
            } else {
                Ok(())
            }
        }
    }
}

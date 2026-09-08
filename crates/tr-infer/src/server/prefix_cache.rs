//! The persistent prefix cache as the engine sees it: restore the deepest cached prefix of a
//! prompt before prefill, snapshot at message boundaries during prefill, and after a request
//! hand the new rows chunks and the final snapshot to a writer thread.
//!
//! Only the copies out of / into tile memory run on the engine thread (`Model::export_*`,
//! `import_*`); the backend writes run on the writer thread from staging buffers, so a request
//! never waits for the disk except when the staging budget is exhausted by a large first-seen
//! prompt.
use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;
use tr_cache::{chunk_spans, prefix_keys, root_key, AlignedBuf, Backend, Cache, FsBackend, Kind, ObjectMeta, Policies, PrefixKey, Role, Span};
use tr_model::exec::Model;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotPolicy {
    /// At the end of every prompt span of at least `snapshot_min_tokens`, and at the end of each request.
    Message,
    /// At the end of each request only.
    Request,
}
impl SnapshotPolicy {
    pub fn parse(s: &str) -> Option<SnapshotPolicy> {
        match s {
            "message" => Some(SnapshotPolicy::Message),
            "request" => Some(SnapshotPolicy::Request),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CacheOpts {
    pub dir: PathBuf,
    pub backend: String,
    pub chunk_len: usize,
    pub max_bytes: u64,
    pub snapshots: SnapshotPolicy,
    pub snapshot_min_tokens: usize,
    pub staging_bytes: u64,
    /// Per-role policies (persist, eviction priority, idle ttl, snapshots).
    pub roles: Policies,
}

/// Host buffers for objects in flight, bounded in bytes and recycled.
struct Staging {
    budget: u64,
    inner: Mutex<StagingInner>,
    cv: Condvar,
}
struct StagingInner {
    /// Bytes of every buffer that exists (leased or free).
    allocated: u64,
    free: Vec<AlignedBuf>,
}

impl Staging {
    fn new(budget: u64) -> Staging {
        Staging { budget: budget.max(1), inner: Mutex::new(StagingInner { allocated: 0, free: Vec::new() }), cv: Condvar::new() }
    }
    /// A buffer of `len` bytes; None when the budget is exhausted and `block` is false.
    fn take(&self, len: usize, block: bool) -> Option<AlignedBuf> {
        let mut g = if block { self.inner.lock().unwrap() } else { self.inner.try_lock().ok()? };
        loop {
            // reuse: the smallest free buffer that fits (and is not absurdly larger)
            if let Some(i) = (0..g.free.len()).filter(|&i| g.free[i].capacity() >= len && g.free[i].capacity() <= 2 * len.max(1 << 20)).min_by_key(|&i| g.free[i].capacity()) {
                let mut b = g.free.swap_remove(i);
                b.set_len(len);
                return Some(b);
            }
            let need = AlignedBuf::new(0).capacity().max(len.div_ceil(4096) * 4096) as u64;
            // make room by dropping free buffers
            while g.allocated + need > self.budget && !g.free.is_empty() {
                let b = g.free.pop().unwrap();
                g.allocated -= b.capacity() as u64;
            }
            if g.allocated + need <= self.budget {
                g.allocated += need;
                drop(g); // zeroing a large new buffer must not hold the staging mutex
                return Some(AlignedBuf::new(len));
            }
            if !block {
                return None;
            }
            g = self.cv.wait(g).unwrap();
        }
    }
    fn give(&self, buf: AlignedBuf) {
        let mut g = self.inner.lock().unwrap();
        g.free.push(buf);
        self.cv.notify_all();
    }
}

struct WriteJob {
    meta: ObjectMeta,
    body: AlignedBuf,
    flush: bool,
}

/// What a restore did.
#[derive(Clone, Debug, Default)]
pub struct Restored {
    pub pos: usize,
    pub rows: usize,
    pub bytes: u64,
    pub seconds: f64,
    /// The logits stored with the snapshot (the prompt ends exactly at the restore point).
    pub last_logits: Option<Vec<f32>>,
}

#[derive(Clone, Debug, Default)]
pub struct Saved {
    pub rows: usize,
    pub snapshots: usize,
    pub bytes: u64,
    pub seconds: f64,
    /// Snapshots skipped because no staging buffer was free.
    pub skipped: usize,
}

type ReadTask = Box<dyn FnOnce() + Send>;

pub struct PrefixCache {
    reader_tx: SyncSender<ReadTask>,
    reader: Option<std::thread::JoinHandle<()>>,
    cache: Arc<Mutex<Cache>>,
    root: PrefixKey,
    chunk_len: usize,
    policy: SnapshotPolicy,
    min_tokens: usize,
    roles: Policies,
    staging: Arc<Staging>,
    tx: SyncSender<WriteJob>,
    writer: Option<std::thread::JoinHandle<()>>,
    describe: String,
}

impl PrefixCache {
    /// Open the store for an engine with `identity` (`Model::cache_identity`).
    pub fn open(o: &CacheOpts, identity: &[String]) -> Result<PrefixCache> {
        let backend: Box<dyn Backend> = match o.backend.as_str() {
            "fs" => Box::new(FsBackend::open(&o.dir).with_context(|| format!("prefix cache at {}", o.dir.display()))?),
            other => bail!("unknown cache backend {other:?} (only `fs` exists)"),
        };
        let describe = backend.describe();
        let mat: Vec<&str> = identity.iter().map(|s| s.as_str()).collect();
        let root = root_key(&mat);
        let cache = Cache::open_with(backend, o.max_bytes, root, o.roles.clone())?;
        let cache = Arc::new(Mutex::new(cache));
        let staging = Arc::new(Staging::new(o.staging_bytes));
        let (reader_tx, reader_rx) = sync_channel::<ReadTask>(64);
        let reader = std::thread::Builder::new().name("cache-reader".into()).spawn(move || { for read in reader_rx { read(); } })?;
        let (tx, rx) = sync_channel::<WriteJob>(64);
        let writer = {
            let cache = cache.clone();
            let staging = staging.clone();
            std::thread::Builder::new().name("cache-writer".into()).spawn(move || writer_loop(rx, cache, staging))?
        };
        Ok(PrefixCache { reader_tx, reader: Some(reader), cache, root, chunk_len: o.chunk_len.max(1), policy: o.snapshots, min_tokens: o.snapshot_min_tokens, roles: o.roles.clone(), staging, tx, writer: Some(writer), describe })
    }

    pub fn describe(&self) -> String {
        let c = self.cache.lock().unwrap();
        let r = c.open_report();
        let s = c.stats();
        format!(
            "prefix cache at {}: {} objects ({} rows, {} snapshots), {:.2} GiB of {:.1} GiB{}{}; chunks of {} tokens, snapshots {}; roles: {}",
            self.describe,
            s.objects,
            s.rows,
            s.snapshots,
            s.bytes as f64 / (1u64 << 30) as f64,
            s.budget as f64 / (1u64 << 30) as f64,
            if r.index_loaded { "" } else { " (index rebuilt from the object headers)" },
            if r.dropped_foreign > 0 { format!(", {} objects of another model dropped", r.dropped_foreign) } else { String::new() },
            self.chunk_len,
            match self.policy {
                SnapshotPolicy::Message => format!("at message boundaries (spans of >= {} tokens) and request ends", self.min_tokens),
                SnapshotPolicy::Request => "at request ends".into(),
            },
            self.roles.describe()
        )
    }

    pub fn roles(&self) -> &Policies {
        &self.roles
    }

    /// The first position from which nothing is stored: the start of the first span whose role
    /// is not persisted (chunks and snapshots beyond it would be unreachable).
    fn persist_cut(&self, spans: &[Span]) -> usize {
        spans.iter().filter(|s| !self.roles.get(s.role).persist).map(|s| s.start).min().unwrap_or(usize::MAX)
    }

    pub fn stats(&self) -> tr_cache::Stats {
        self.cache.lock().unwrap().stats()
    }

    /// Prefix keys of a prompt (`len + 1` entries).
    pub fn keys(&self, ids: &[u32], images: &[(usize, u64, usize)]) -> Vec<PrefixKey> {
        prefix_keys(&self.root, ids, images)
    }
    /// Continue the chain over generated tokens.
    pub fn extend_keys(&self, keys: &mut Vec<PrefixKey>, toks: &[u32]) {
        let last = *keys.last().expect("keys");
        let more = prefix_keys(&last, toks, &[]);
        keys.extend_from_slice(&more[1..]);
    }

    /// Restore the deepest cached prefix of the prompt into the model (which must be reset).
    /// On any error the model is reset again and None is returned.
    pub fn restore(&self, model: &mut Model, ids: &[u32], keys: &[PrefixKey]) -> Option<Restored> {
        let t0 = Instant::now();
        let (plan, expired) = {
            let mut c = self.cache.lock().unwrap();
            let plan = c.best_restore(keys);
            let plan = match plan {
                Some(p) if p.pos == ids.len() && p.snapshot.bytes as usize <= model.snapshot_bytes(0) => {
                    // the whole prompt, but no logits stored with it: the next-best point instead
                    c.best_restore(&keys[..ids.len()])
                }
                p => p,
            };
            (plan, c.take_pending_delete())
        };
        if !expired.is_empty() {
            let be = self.cache.lock().unwrap().backend();
            let mut be = be.lock().unwrap();
            for n in &expired {
                let _ = be.delete(n);
            }
        }
        let plan = plan?;
        match self.restore_plan(model, ids, &plan) {
            Ok(mut r) => {
                r.seconds = t0.elapsed().as_secs_f64();
                Some(r)
            }
            Err(e) => {
                eprintln!("prefix cache: restore at {} failed: {e:#}", plan.pos);
                model.reset();
                None
            }
        }
    }

    fn restore_plan(&self, model: &mut Model, ids: &[u32], plan: &tr_cache::RestorePlan) -> Result<Restored> {
        let mut out = Restored { pos: plan.pos, ..Default::default() };
        for r in &plan.rows {
            if r.tokens[..] != ids[r.start..r.end] {
                bail!("rows {}..{} hold other tokens (hash collision?)", r.start, r.end);
            }
            let want = model.rows_bytes(r.start, r.end);
            if want as u64 != r.bytes {
                bail!("rows {}..{} are {} bytes, this engine expects {want}", r.start, r.end, r.bytes);
            }
            let mut buf = self.staging.take(want, true).expect("blocking take");
            if let Err(e) = self.read(r.key, Kind::Rows, &mut buf[..]) {
                self.staging.give(buf);
                return Err(e);
            }
            model.import_rows(r.start, r.end, &buf[..]);
            out.rows += 1;
            out.bytes += r.bytes;
            self.staging.give(buf);
        }
        let s = &plan.snapshot;
        let mut buf = self.staging.take(s.bytes as usize, true).expect("blocking take");
        if let Err(e) = self.read(s.key, Kind::Snapshot, &mut buf[..]) {
            self.staging.give(buf);
            return Err(e);
        }
        let host = model.import_snapshot(&buf[..]);
        self.staging.give(buf);
        let host = host?;
        if host.n_past != plan.pos {
            bail!("snapshot is at position {} but was indexed at {}", host.n_past, plan.pos);
        }
        out.bytes += s.bytes;
        if !host.extra.is_empty() {
            out.last_logits = Some(host.extra);
        }
        Ok(out)
    }

    /// Read one object through the backend without holding the index lock during the I/O;
    /// an unreadable object is dropped from the index and deleted.
    fn read(&self, key: PrefixKey, kind: Kind, body: &mut [u8]) -> Result<()> {
        let be = self.cache.lock().unwrap().backend();
        let name = tr_cache::object::object_name(&key, kind);
        let r = be.lock().unwrap().get(&name, body);
        match r {
            Ok(_) => {
                self.cache.lock().unwrap().touch_object(key, kind);
                Ok(())
            }
            Err(e) => {
                if let Some(n) = self.cache.lock().unwrap().forget(key, kind) {
                    let _ = be.lock().unwrap().delete(&n);
                }
                Err(e)
            }
        }
    }

    /// Prompt positions where prefill should stop and snapshot: span ends in `(from, len]` of
    /// spans long enough whose role takes snapshots, under the `message` policy, and not beyond
    /// the first span that is not persisted (nothing after it is restorable).
    pub fn snapshot_points(&self, spans: &[Span], from: usize, len: usize) -> Vec<(usize, Role)> {
        if self.policy != SnapshotPolicy::Message {
            return Vec::new();
        }
        let cut = self.persist_cut(spans);
        let mut v: Vec<(usize, Role)> = spans.iter().filter(|s| s.len() >= self.min_tokens && s.end > from && s.end <= len && s.end <= cut && self.roles.get(s.role).snapshot).map(|s| (s.end, s.role)).collect();
        v.sort_unstable_by_key(|x| x.0);
        v.dedup_by_key(|x| x.0);
        v
    }

    /// Snapshot the live state at `pos` (key `key`, the end of a `role` span) with `logits`
    /// unless it is stored already. Never blocks: without a free staging buffer the snapshot is
    /// skipped (returns false).
    pub fn snapshot_now(&self, model: &mut Model, key: PrefixKey, pos: usize, role: Role, logits: &[f32]) -> Option<bool> {
        if self.cache.lock().unwrap().has(key, Kind::Snapshot) {
            return None;
        }
        let n = model.snapshot_bytes(logits.len());
        let Some(mut buf) = self.staging.take(n, false) else { return Some(false) };
        model.export_snapshot(&mut buf[..], logits);
        let meta = ObjectMeta::snapshot(self.root, key, pos, Some(role), n as u64);
        self.send(WriteJob { meta, body: buf, flush: false });
        Some(true)
    }

    /// End of a request: store every rows chunk of `spans` (over the model's committed
    /// positions) that is missing and whose role is persisted, then the snapshot at `n_past`
    /// with `last_logits` — unless a chunk before it was not persisted (it would be unreachable).
    pub fn save(&self, model: &mut Model, toks: &[u32], keys: &[PrefixKey], spans: &[Span], last_logits: &[f32]) -> Saved {
        let t0 = Instant::now();
        let n = model.n_past;
        debug_assert_eq!(toks.len(), n);
        debug_assert_eq!(keys.len(), n + 1);
        let mut rep = Saved::default();
        let cut = self.persist_cut(spans);
        let chunks: Vec<Span> = chunk_spans(spans, 0, self.chunk_len).into_iter().filter(|c| c.end <= n && c.end <= cut).collect();
        let last = chunks.len().saturating_sub(1);
        let end_role = spans.iter().filter(|s| s.start < n).last().map(|s| s.role);
        for (i, ch) in chunks.iter().enumerate() {
            if self.cache.lock().unwrap().has(keys[ch.end], Kind::Rows) {
                continue;
            }
            let bytes = model.rows_bytes(ch.start, ch.end);
            let mut buf = self.staging.take(bytes, true).expect("blocking take");
            model.export_rows(ch.start, ch.end, &mut buf[..]);
            let meta = ObjectMeta::rows(self.root, keys[ch.end], keys[ch.start], ch.start, ch.end, ch.role, toks[ch.start..ch.end].to_vec(), bytes as u64);
            rep.rows += 1;
            rep.bytes += bytes as u64;
            self.send(WriteJob { meta, body: buf, flush: i == last });
        }
        if n > 0 && n <= cut && !self.cache.lock().unwrap().has(keys[n], Kind::Snapshot) {
            let bytes = model.snapshot_bytes(last_logits.len());
            let mut buf = self.staging.take(bytes, true).expect("blocking take");
            model.export_snapshot(&mut buf[..], last_logits);
            let meta = ObjectMeta::snapshot(self.root, keys[n], n, end_role, bytes as u64);
            rep.snapshots += 1;
            rep.bytes += bytes as u64;
            self.send(WriteJob { meta, body: buf, flush: true });
        } else if rep.rows > 0 {
            // the last rows job already flushes
        }
        rep.seconds = t0.elapsed().as_secs_f64();
        rep
    }

    fn send(&self, job: WriteJob) {
        if let Err(e) = self.tx.send(job) {
            eprintln!("prefix cache: writer gone: {e}");
        }
    }
}

impl Drop for PrefixCache {
    fn drop(&mut self) {
        // Readers are cancelled by dropping their request receivers before cache shutdown.
        let (reader_tx, _) = sync_channel(1);
        drop(std::mem::replace(&mut self.reader_tx, reader_tx));
        if let Some(r) = self.reader.take() { let _ = r.join(); }
        // close the channel so the writer drains and exits, then wait for it
        let (tx, _rx) = sync_channel::<WriteJob>(1);
        drop(std::mem::replace(&mut self.tx, tx));
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
    }
}

fn writer_loop(rx: Receiver<WriteJob>, cache: Arc<Mutex<Cache>>, staging: Arc<Staging>) {
    let be = cache.lock().unwrap().backend();
    for job in rx {
        let t0 = Instant::now();
        let name = job.meta.name();
        let bytes = job.meta.bytes;
        // the write holds only the backend lock; the index lock is taken for the bookkeeping
        let res = (|| -> Result<()> {
            cache.lock().unwrap().check(&job.meta)?;
            be.lock().unwrap().put(&job.meta, &job.body[..]).with_context(|| format!("store {name}"))?;
            let evicted = cache.lock().unwrap().insert(job.meta)?;
            if !evicted.is_empty() {
                let mut b = be.lock().unwrap();
                for n in &evicted {
                    b.delete(n)?;
                }
            }
            Ok(())
        })();
        if job.flush {
            if let Err(e) = cache.lock().unwrap().flush() {
                eprintln!("prefix cache: index write failed: {e:#}");
            }
        }
        staging.give(job.body);
        match res {
            Ok(()) => {
                if std::env::var("TR_CACHE_DEBUG").is_ok() {
                    eprintln!("prefix cache: wrote {name} ({bytes} bytes, {:.1} ms)", t0.elapsed().as_secs_f64() * 1e3);
                }
            }
            Err(e) => eprintln!("prefix cache: write of {name} failed: {e:#}"),
        }
    }
    let _ = cache.lock().map(|mut c| c.flush());
}

/// A staged restore object returns its budget automatically when consumed or cancelled.
pub struct RestoreObject {
    pub kind: Kind,
    pub start: usize,
    pub end: usize,
    body: Option<AlignedBuf>,
    staging: Arc<Staging>,
}
impl RestoreObject { pub fn bytes(&self) -> &[u8] { &self.body.as_ref().unwrap()[..] } }
impl Drop for RestoreObject {
    fn drop(&mut self) { if let Some(b) = self.body.take() { self.staging.give(b); } }
}

impl PrefixCache {
    /// Preallocate the common staging sizes before accepting requests.
    pub fn warm_staging(&self, snapshot_bytes: usize, row_bytes: usize) {
        let mut buffers = Vec::new();
        for bytes in [snapshot_bytes, row_bytes, snapshot_bytes, row_bytes] {
            if let Some(buffer) = self.staging.take(bytes, false) { buffers.push(buffer); }
        }
        for buffer in buffers { self.staging.give(buffer); }
    }

    /// Disk I/O and index lookup happen off the executor. A bounded channel plus staging
    /// leases limits in-flight data; dropping the receiver cancels the restore.
    pub fn restore_async(&self, ids: Vec<u32>, keys: Vec<PrefixKey>, empty_snapshot_bytes: usize) -> Receiver<Result<RestoreObject>> {
        let (tx, rx) = sync_channel(1);
        let cache = self.cache.clone();
        let staging = self.staging.clone();
        let task: ReadTask = Box::new(move || {
            let run = || -> Result<()> {
                let (plan, expired) = {
                    let mut c = cache.lock().unwrap();
                    let p = c.best_restore(&keys);
                    let plan = match p {
                        Some(p) if p.pos == ids.len() && p.snapshot.bytes as usize <= empty_snapshot_bytes => c.best_restore(&keys[..ids.len()]),
                        p => p,
                    };
                    (plan, c.take_pending_delete())
                };
                let backend = cache.lock().unwrap().backend();
                for name in expired { let _ = backend.lock().unwrap().delete(&name); }
                let Some(plan) = plan else { return Ok(()) };
                for object in plan.rows.iter().chain(std::iter::once(&plan.snapshot)) {
                    if object.kind == Kind::Rows {
                        anyhow::ensure!(object.start <= object.end && object.end <= ids.len() && object.tokens == ids[object.start..object.end], "cached rows do not match prompt");
                    }
                    let mut body = staging.take(object.bytes as usize, false).ok_or_else(|| anyhow::anyhow!("restore staging budget unavailable"))?;
                    let result = backend.lock().unwrap().get(&object.name(), &mut body[..]);
                    if let Err(e) = result {
                        staging.give(body);
                        let forgotten = cache.lock().unwrap().forget(object.key, object.kind);
                        if let Some(name) = forgotten { let _ = backend.lock().unwrap().delete(&name); }
                        return Err(e);
                    }
                    cache.lock().unwrap().touch_object(object.key, object.kind);
                    let staged = RestoreObject { kind: object.kind, start: object.start, end: object.end, body: Some(body), staging: staging.clone() };
                    if tx.send(Ok(staged)).is_err() { return Ok(()); }
                }
                Ok(())
            };
            if let Err(e) = run() { let _ = tx.send(Err(e)); }
        });
        // A full reader queue is a cache miss, never executor backpressure.
        let _ = self.reader_tx.try_send(task);
        rx
    }

    /// Best-effort save for continuous execution: never waits for staging or writer capacity.
    /// The index is acquired with try_lock because a background flush can hold it during I/O.
    pub fn save_ready(&self, model: &mut Model, toks: &[u32], keys: &[PrefixKey], spans: &[Span], logits: &[f32]) {
        let n = model.n_past;
        let cut = self.persist_cut(spans);
        for ch in chunk_spans(spans, 0, self.chunk_len).into_iter().filter(|c| c.end <= n && c.end <= cut) {
            let exists = match self.cache.try_lock() { Ok(c) => c.has(keys[ch.end], Kind::Rows), Err(_) => return };
            if exists { continue; }
            let bytes = model.rows_bytes(ch.start, ch.end);
            let Some(mut body) = self.staging.take(bytes, false) else { return };
            model.export_rows(ch.start, ch.end, &mut body[..]);
            let meta = ObjectMeta::rows(self.root, keys[ch.end], keys[ch.start], ch.start, ch.end, ch.role, toks[ch.start..ch.end].to_vec(), bytes as u64);
            if let Err(e) = self.tx.try_send(WriteJob { meta, body, flush: true }) { self.staging.give(match e { std::sync::mpsc::TrySendError::Full(j) | std::sync::mpsc::TrySendError::Disconnected(j) => j.body }); return; }
        }
        if n == 0 || n > cut { return; }
        let exists = match self.cache.try_lock() { Ok(c) => c.has(keys[n], Kind::Snapshot), Err(_) => return };
        if exists { return; }
        let bytes = model.snapshot_bytes(logits.len());
        let Some(mut body) = self.staging.take(bytes, false) else { return };
        model.export_snapshot(&mut body[..], logits);
        let role = spans.iter().filter(|s| s.start < n).last().map(|s| s.role);
        let meta = ObjectMeta::snapshot(self.root, keys[n], n, role, bytes as u64);
        if let Err(e) = self.tx.try_send(WriteJob { meta, body, flush: true }) { self.staging.give(match e { std::sync::mpsc::TrySendError::Full(j) | std::sync::mpsc::TrySendError::Disconnected(j) => j.body }); }
    }
}

#[cfg(test)]
mod continuous_tests {
    use super::*;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[test]
    fn async_restore_cancellation_returns_staging_and_missing_rows_fall_back() {
        let dir = std::env::temp_dir().join(format!("tr-async-cache-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
        let cache = PrefixCache::open(&CacheOpts {
            dir: dir.clone(), backend: "fs".into(), chunk_len: 64, max_bytes: 1 << 20,
            snapshots: SnapshotPolicy::Request, snapshot_min_tokens: 1, staging_bytes: 16384,
            roles: Policies::default(),
        }, &["async-test".into()]).unwrap();
        let ids = vec![10, 20];
        let keys = cache.keys(&ids, &[]);
        let rows = ObjectMeta::rows(cache.root, keys[2], keys[0], 0, 2, Role::User, ids.clone(), 4);
        let snap = ObjectMeta::snapshot(cache.root, keys[2], 2, Some(Role::User), 4);
        let backend = cache.cache.lock().unwrap().backend();
        for meta in [&rows, &snap] {
            backend.lock().unwrap().put(meta, &[1, 2, 3, 4]).unwrap();
            cache.cache.lock().unwrap().insert(meta.clone()).unwrap();
        }
        // Cancel with a one-object channel possibly full. The next restore proves that the
        // reader did not stay blocked and all abandoned staging leases were returned.
        drop(cache.restore_async(ids.clone(), keys.clone(), 0));
        let reader = cache.restore_async(ids.clone(), keys.clone(), 0);
        for kind in [Kind::Rows, Kind::Snapshot] {
            let object = reader.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
            assert_eq!(object.kind, kind);
            assert_eq!(object.bytes(), &[1, 2, 3, 4]);
        }
        drop(reader);
        backend.lock().unwrap().delete(&rows.name()).unwrap();
        let reader = cache.restore_async(ids, keys, 0);
        assert!(reader.recv_timeout(Duration::from_secs(5)).unwrap().is_err());
        drop(reader);
        drop(cache);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn nonblocking_staging_does_not_wait_for_budget_or_mutex() {
        let staging = Staging::new(4096);
        let b = staging.take(4096, false).unwrap();
        assert!(staging.take(1, false).is_none());
        staging.give(b);
        let guard = staging.inner.lock().unwrap();
        assert!(staging.take(1, false).is_none());
        drop(guard);
        assert!(staging.take(1, false).is_some());
    }
}

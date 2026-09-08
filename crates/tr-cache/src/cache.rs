//! The policy layer over a backend: an in-memory index of every object, LRU order, a byte
//! budget, and the restore-point search.
use crate::backend::Backend;
use crate::key::PrefixKey;
use crate::object::{now, object_name, Kind, ObjectMeta, FORMAT};
use crate::policy::{Policies, PRIORITY_MAX};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

/// The backend behind its own lock, so a slow write never blocks index lookups.
pub type SharedBackend = Arc<Mutex<Box<dyn Backend>>>;

struct Entry {
    meta: ObjectMeta,
    /// LRU position: higher = used more recently.
    seq: u64,
    /// Eviction tier from the role policy (lower goes first).
    tier: u8,
}

/// A resumable prefix: the snapshot at `pos` and the rows chunks covering `0..pos`, in order.
#[derive(Clone, Debug)]
pub struct RestorePlan {
    pub pos: usize,
    pub snapshot: ObjectMeta,
    pub rows: Vec<ObjectMeta>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Stats {
    pub objects: usize,
    pub rows: usize,
    pub snapshots: usize,
    pub bytes: u64,
    pub budget: u64,
    pub oldest: Option<u64>,
    pub newest: Option<u64>,
    pub hits: u64,
    pub misses: u64,
    pub evicted: u64,
    pub expired: u64,
}

/// What `open` found.
#[derive(Clone, Debug, Default)]
pub struct OpenReport {
    pub objects: usize,
    pub bytes: u64,
    /// Objects of another engine identity, deleted (one store holds one root).
    pub dropped_foreign: usize,
    pub index_loaded: bool,
}

#[derive(Serialize, Deserialize)]
struct IndexFile {
    format: u32,
    root: PrefixKey,
    /// `(name, last_used, seq)`.
    objects: Vec<(String, u64, u64)>,
}

pub struct Cache {
    backend: SharedBackend,
    root: PrefixKey,
    index: HashMap<(PrefixKey, Kind), Entry>,
    /// Eviction order: `(tier, seq)` ascending = lowest priority, least recently used first.
    lru: BTreeMap<(u8, u64), (PrefixKey, Kind)>,
    policies: Policies,
    seq: u64,
    used: u64,
    budget: u64,
    dirty: bool,
    /// Names dropped from the index by expiry inside `best_restore`, awaiting backend deletion.
    pending_delete: Vec<String>,
    report: OpenReport,
    hits: u64,
    misses: u64,
    evicted: u64,
    expired: u64,
}

impl Cache {
    /// `open_with` and the default policies.
    pub fn open(backend: Box<dyn Backend>, budget: u64, root: PrefixKey) -> Result<Cache> {
        Self::open_with(backend, budget, root, Policies::default())
    }

    /// Open a store: list the backend's objects (the authority), drop objects of another root,
    /// and take the LRU order from the stored index when there is one.
    pub fn open_with(backend: Box<dyn Backend>, budget: u64, root: PrefixKey, policies: Policies) -> Result<Cache> {
        let backend: SharedBackend = Arc::new(Mutex::new(backend));
        let mut be = backend.lock().unwrap();
        let stored = be.get_index().unwrap_or(None).and_then(|b| serde_json::from_slice::<IndexFile>(&b).ok()).filter(|i| i.format == FORMAT && i.root == root);
        let order: HashMap<String, (u64, u64)> = stored.as_ref().map(|i| i.objects.iter().map(|(n, l, s)| (n.clone(), (*l, *s))).collect()).unwrap_or_default();
        let mut report = OpenReport { index_loaded: stored.is_some(), ..Default::default() };
        let mut metas = be.list()?;
        let mut foreign = Vec::new();
        metas.retain(|m| {
            if m.root == root {
                true
            } else {
                foreign.push(m.name());
                false
            }
        });
        for n in &foreign {
            be.delete(n)?;
        }
        drop(be);
        report.dropped_foreign = foreign.len();
        // LRU order: stored seq when known, else by last_used; ties by name
        metas.sort_by(|a, b| {
            let ka = order.get(&a.name()).map(|x| x.1).unwrap_or(0);
            let kb = order.get(&b.name()).map(|x| x.1).unwrap_or(0);
            (ka, a.last_used, a.name()).cmp(&(kb, b.last_used, b.name()))
        });
        let mut c = Cache { backend, root, index: HashMap::new(), lru: BTreeMap::new(), policies, seq: 0, used: 0, budget, dirty: false, pending_delete: Vec::new(), report: OpenReport::default(), hits: 0, misses: 0, evicted: 0, expired: 0 };
        for mut m in metas {
            if let Some((l, _)) = order.get(&m.name()) {
                m.last_used = *l;
            }
            c.insert_entry(m);
        }
        report.objects = c.index.len();
        report.bytes = c.used;
        c.report = report;
        c.dirty = !c.report.index_loaded;
        let mut gone = c.expire_idle();
        gone.extend(c.evict_to_budget());
        c.delete_all(&gone)?;
        Ok(c)
    }

    pub fn policies(&self) -> &Policies {
        &self.policies
    }

    /// The backend, for I/O outside the index lock (see `insert` / `forget`).
    pub fn backend(&self) -> SharedBackend {
        self.backend.clone()
    }
    fn delete_all(&self, names: &[String]) -> Result<()> {
        if names.is_empty() {
            return Ok(());
        }
        let mut be = self.backend.lock().unwrap();
        for n in names {
            be.delete(n)?;
        }
        Ok(())
    }

    pub fn root(&self) -> PrefixKey {
        self.root
    }
    pub fn budget(&self) -> u64 {
        self.budget
    }
    pub fn open_report(&self) -> &OpenReport {
        &self.report
    }
    pub fn describe(&self) -> String {
        self.backend.lock().unwrap().describe()
    }

    fn tier_of(&self, meta: &ObjectMeta) -> u8 {
        self.policies.of(meta.role).priority.min(PRIORITY_MAX)
    }

    fn insert_entry(&mut self, meta: ObjectMeta) {
        self.seq += 1;
        let id = (meta.key, meta.kind);
        let tier = self.tier_of(&meta);
        self.used += meta.footprint();
        if let Some(old) = self.index.insert(id, Entry { meta, seq: self.seq, tier }) {
            self.used -= old.meta.footprint();
            self.lru.remove(&(old.tier, old.seq));
        }
        self.lru.insert((tier, self.seq), id);
    }

    fn touch(&mut self, id: (PrefixKey, Kind)) {
        if let Some(e) = self.index.get_mut(&id) {
            self.seq += 1;
            self.lru.remove(&(e.tier, e.seq));
            e.seq = self.seq;
            e.meta.last_used = now();
            self.lru.insert((e.tier, self.seq), id);
            self.dirty = true;
        }
    }

    /// Drop objects idle for longer than their role's `ttl` from the index; their names.
    pub fn expire_idle(&mut self) -> Vec<String> {
        if !self.policies.any_ttl() {
            return Vec::new();
        }
        let t = now();
        let dead: Vec<(PrefixKey, Kind)> = self.index.values().filter(|e| self.policies.of(e.meta.role).ttl.map(|ttl| t.saturating_sub(e.meta.last_used) > ttl).unwrap_or(false)).map(|e| (e.meta.key, e.meta.kind)).collect();
        let mut out = Vec::new();
        for id in dead {
            if let Some(m) = self.remove(id) {
                out.push(m.name());
                self.expired += 1;
            }
        }
        out
    }

    pub fn has(&self, key: PrefixKey, kind: Kind) -> bool {
        self.index.contains_key(&(key, kind))
    }

    /// The deepest restore point of a prompt whose prefix keys are `keys` (`keys[p]` = hash of
    /// positions `0..p`): the largest `p >= 1` with a snapshot whose rows chunks chain back to
    /// position 0 through objects that are all present. Touches every object of the plan.
    /// Expired objects are dropped from the index first (the caller deletes them: `take_expired`).
    pub fn best_restore(&mut self, keys: &[PrefixKey]) -> Option<RestorePlan> {
        let gone = self.expire_idle();
        self.pending_delete.extend(gone);
        for p in (1..keys.len()).rev() {
            let Some(snap) = self.index.get(&(keys[p], Kind::Snapshot)) else { continue };
            if snap.meta.end != p {
                continue;
            }
            let snapshot = snap.meta.clone();
            let mut rows = Vec::new();
            let mut cur = p;
            let mut ok = true;
            while cur > 0 {
                match self.index.get(&(keys[cur], Kind::Rows)) {
                    Some(e) if e.meta.end == cur && e.meta.start < cur && keys[e.meta.start] == e.meta.parent => {
                        rows.push(e.meta.clone());
                        cur = e.meta.start;
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                rows.reverse();
                for r in &rows {
                    self.touch((r.key, Kind::Rows));
                }
                self.touch((snapshot.key, Kind::Snapshot));
                self.hits += 1;
                return Some(RestorePlan { pos: p, snapshot, rows });
            }
        }
        self.misses += 1;
        None
    }

    /// Store an object (a no-op that touches it when the key is already present): the
    /// backend write, then `insert`. Callers that must not hold the index lock during the
    /// write (the server's writer thread) do the two steps themselves.
    pub fn put(&mut self, meta: ObjectMeta, body: &[u8]) -> Result<()> {
        let id = (meta.key, meta.kind);
        if self.index.contains_key(&id) {
            self.touch(id);
            return Ok(());
        }
        self.check(&meta)?;
        self.backend.lock().unwrap().put(&meta, body).with_context(|| format!("store {}", meta.name()))?;
        let evicted = self.insert(meta)?;
        self.delete_all(&evicted)
    }

    /// Validate an object for this store before writing it.
    pub fn check(&self, meta: &ObjectMeta) -> Result<()> {
        anyhow::ensure!(meta.root == self.root, "object root {} is not this store's {}", meta.root, self.root);
        meta.validate()
    }

    /// Index an object the backend already holds; returns the names evicted (or expired) to
    /// stay within the budget (already gone from the index; the caller deletes them from the
    /// backend).
    pub fn insert(&mut self, meta: ObjectMeta) -> Result<Vec<String>> {
        let id = (meta.key, meta.kind);
        if self.index.contains_key(&id) {
            self.touch(id);
            return Ok(std::mem::take(&mut self.pending_delete));
        }
        self.check(&meta)?;
        self.insert_entry(meta);
        self.dirty = true;
        let mut gone = std::mem::take(&mut self.pending_delete);
        gone.extend(self.expire_idle());
        gone.extend(self.evict_to_budget());
        Ok(gone)
    }

    /// Names dropped from the index by `best_restore`'s expiry that still need deleting.
    pub fn take_pending_delete(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_delete)
    }

    /// Read an object's body (`body.len()` must be its `bytes`); touches it. An unreadable
    /// object is forgotten and deleted.
    pub fn get(&mut self, key: PrefixKey, kind: Kind, body: &mut [u8]) -> Result<ObjectMeta> {
        let name = object_name(&key, kind);
        anyhow::ensure!(self.index.contains_key(&(key, kind)), "no object {name} in the index");
        let r = self.backend.lock().unwrap().get(&name, body);
        match r {
            Ok(m) => {
                self.touch((key, kind));
                Ok(m)
            }
            Err(e) => {
                if let Some(n) = self.forget(key, kind) {
                    let _ = self.backend.lock().unwrap().delete(&n);
                }
                Err(e)
            }
        }
    }

    /// Drop an object from the index (the caller deletes it from the backend); its name.
    pub fn forget(&mut self, key: PrefixKey, kind: Kind) -> Option<String> {
        self.remove((key, kind)).map(|m| m.name())
    }

    /// Mark an object as just used.
    pub fn touch_object(&mut self, key: PrefixKey, kind: Kind) {
        self.touch((key, kind));
    }

    pub fn meta(&self, key: PrefixKey, kind: Kind) -> Option<&ObjectMeta> {
        self.index.get(&(key, kind)).map(|e| &e.meta)
    }

    fn remove(&mut self, id: (PrefixKey, Kind)) -> Option<ObjectMeta> {
        let e = self.index.remove(&id)?;
        self.lru.remove(&(e.tier, e.seq));
        self.used -= e.meta.footprint();
        self.dirty = true;
        Some(e.meta)
    }

    /// Drop least recently used objects from the index until the store fits the budget;
    /// returns their names for deletion.
    fn evict_to_budget(&mut self) -> Vec<String> {
        let mut out = Vec::new();
        while self.used > self.budget {
            let Some((&(_, seq), &id)) = self.lru.iter().next() else { break };
            debug_assert!(self.index.get(&id).map(|e| e.seq == seq).unwrap_or(false));
            if let Some(m) = self.remove(id) {
                out.push(m.name());
                self.evicted += 1;
            }
        }
        out
    }

    /// Persist the index (LRU order and hit times) if anything changed.
    pub fn flush(&mut self) -> Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let mut objects: Vec<(String, u64, u64)> = self.index.values().map(|e| (e.meta.name(), e.meta.last_used, e.seq)).collect();
        objects.sort_by_key(|o| o.2);
        let f = IndexFile { format: FORMAT, root: self.root, objects };
        self.backend.lock().unwrap().put_index(&serde_json::to_vec(&f)?)?;
        self.dirty = false;
        Ok(())
    }

    pub fn stats(&self) -> Stats {
        let mut s = Stats { objects: self.index.len(), bytes: self.used, budget: self.budget, hits: self.hits, misses: self.misses, evicted: self.evicted, expired: self.expired, ..Default::default() };
        for e in self.index.values() {
            match e.meta.kind {
                Kind::Rows => s.rows += 1,
                Kind::Snapshot => s.snapshots += 1,
            }
            s.oldest = Some(s.oldest.map_or(e.meta.last_used, |o| o.min(e.meta.last_used)));
            s.newest = Some(s.newest.map_or(e.meta.last_used, |o| o.max(e.meta.last_used)));
        }
        s
    }

    /// Delete every object.
    pub fn clear(&mut self) -> Result<()> {
        let ids: Vec<_> = self.index.keys().copied().collect();
        let names: Vec<String> = ids.into_iter().filter_map(|id| self.remove(id)).map(|m| m.name()).collect();
        self.delete_all(&names)?;
        self.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MemBackend;
    use crate::key::{prefix_keys, root_key};
    use crate::role::{chunk_spans, Role, Span};

    fn root() -> PrefixKey {
        root_key(&["pack", "f16", "8", "-", "-", "1"])
    }

    /// Store rows chunks for `spans` of `toks` (chunk_len 4) and a snapshot at every span end.
    fn save(c: &mut Cache, toks: &[u32], spans: &[Span], snapshots_at: &[usize]) {
        let keys = prefix_keys(&c.root(), toks, &[]);
        for ch in chunk_spans(spans, 0, 4) {
            let body = vec![ch.role as u8; 100 * ch.len()];
            let m = ObjectMeta::rows(c.root(), keys[ch.end], keys[ch.start], ch.start, ch.end, ch.role, toks[ch.start..ch.end].to_vec(), body.len() as u64);
            c.put(m, &body).unwrap();
        }
        for &p in snapshots_at {
            let body = vec![0xAAu8; 5000];
            let role = spans.iter().find(|s| s.end == p).map(|s| s.role);
            c.put(ObjectMeta::snapshot(c.root(), keys[p], p, role, body.len() as u64), &body).unwrap();
        }
    }

    #[test]
    fn restore_finds_the_deepest_complete_chain() {
        let mut c = Cache::open(Box::new(MemBackend::new()), 1 << 30, root()).unwrap();
        let toks: Vec<u32> = (100..110).collect();
        let spans = [Span::new(0, 6, Role::System), Span::new(6, 10, Role::User)];
        save(&mut c, &toks, &spans, &[6, 10]);
        assert_eq!(c.stats().rows, 3, "6 = 4 + 2, then 4");
        assert_eq!(c.stats().snapshots, 2);
        // the same prompt: the end snapshot
        let keys = prefix_keys(&root(), &toks, &[]);
        let plan = c.best_restore(&keys).unwrap();
        assert_eq!(plan.pos, 10);
        assert_eq!(plan.rows.iter().map(|r| (r.start, r.end)).collect::<Vec<_>>(), vec![(0, 4), (4, 6), (6, 10)]);
        assert_eq!(plan.rows[2].tokens, toks[6..10]);
        // a prompt that diverges inside the user turn: the system boundary
        let mut t2 = toks.clone();
        t2[8] = 999;
        t2.push(1);
        let k2 = prefix_keys(&root(), &t2, &[]);
        let plan = c.best_restore(&k2).unwrap();
        assert_eq!(plan.pos, 6);
        assert_eq!(plan.rows.len(), 2);
        // a prompt shorter than the first snapshot: nothing
        assert!(c.best_restore(&keys[..5]).is_none());
        // a prompt that is exactly the prefix up to a snapshot
        assert_eq!(c.best_restore(&keys[..7]).unwrap().pos, 6);
        // a different root sees nothing
        let other = prefix_keys(&root_key(&["other"]), &toks, &[]);
        assert!(c.best_restore(&other).is_none());
        assert_eq!(c.stats().hits, 3);
        assert_eq!(c.stats().misses, 2);
    }

    #[test]
    fn broken_chain_falls_back_and_is_repaired_by_the_next_save() {
        let mut c = Cache::open(Box::new(MemBackend::new()), 1 << 30, root()).unwrap();
        let toks: Vec<u32> = (0..10).collect();
        let spans = [Span::new(0, 6, Role::System), Span::new(6, 10, Role::User)];
        save(&mut c, &toks, &spans, &[6, 10]);
        let keys = prefix_keys(&root(), &toks, &[]);
        // lose the chunk [4,6): the snapshot at 6 and the one at 10 are both unreachable
        c.remove((keys[6], Kind::Rows));
        assert!(c.best_restore(&keys).is_none());
        // the next save re-creates exactly the missing chunk (dedupe by key)
        let before = c.stats().objects;
        save(&mut c, &toks, &spans, &[6, 10]);
        assert_eq!(c.stats().objects, before + 1);
        assert_eq!(c.best_restore(&keys).unwrap().pos, 10);
    }

    #[test]
    fn lru_evicts_by_bytes_across_kinds_and_touch_protects() {
        // rows chunk footprint = 4096 header + 4096 body; snapshot 4096 + 8192
        let mut c = Cache::open(Box::new(MemBackend::new()), 70_000, root()).unwrap();
        let toks: Vec<u32> = (0..12).collect();
        let spans = [Span::new(0, 12, Role::User)];
        save(&mut c, &toks, &spans, &[4, 8, 12]);
        assert_eq!(c.stats().bytes, 3 * 8192 + 3 * 12288);
        assert_eq!(c.stats().objects, 6);
        let keys = prefix_keys(&root(), &toks, &[]);
        // a hit touches its chain: rows 0..4, 4..8 and the snapshot at 8
        assert_eq!(c.best_restore(&keys[..9]).unwrap().pos, 8);
        // a big object pushes the store over budget: the untouched objects go first, oldest first
        let big = vec![1u8; 20_000];
        c.put(ObjectMeta::snapshot(root(), root_key(&["big"]), 99, None, big.len() as u64), &big).unwrap();
        assert!(c.stats().bytes <= 70_000);
        assert!(!c.has(keys[12], Kind::Rows), "rows 8..12 was the least recently used");
        assert!(!c.has(keys[4], Kind::Snapshot), "then the snapshot at 4");
        assert!(c.has(keys[12], Kind::Snapshot));
        assert!(c.has(keys[4], Kind::Rows) && c.has(keys[8], Kind::Rows) && c.has(keys[8], Kind::Snapshot));
        assert_eq!(c.stats().evicted, 2);
        assert_eq!(c.best_restore(&keys).unwrap().pos, 8, "12 lost its chain, 8 is intact");
        // a store opened over budget trims itself
        let c2 = Cache::open(Box::new(MemBackend::new()), 0, root()).unwrap();
        assert_eq!(c2.stats().objects, 0);
    }

    #[test]
    fn index_persists_lru_order_and_rebuilds_from_headers() {
        let mut be = MemBackend::new();
        let toks: Vec<u32> = (0..8).collect();
        let spans = [Span::new(0, 8, Role::Tool)];
        {
            let mut c = Cache::open(Box::new(std::mem::take(&mut be)), 1 << 30, root()).unwrap();
            save(&mut c, &toks, &spans, &[8]);
            let keys = prefix_keys(&root(), &toks, &[]);
            c.touch((keys[4], Kind::Rows)); // the oldest object becomes the newest
            c.flush().unwrap();
            // take the backend back out through a raw pointer-free route: rebuild a new backend
            let mut fresh = MemBackend::new();
            let be0 = c.backend();
            let mut be0 = be0.lock().unwrap();
            for m in be0.list().unwrap() {
                let mut b = vec![0u8; m.bytes as usize];
                be0.get(&m.name(), &mut b).unwrap();
                fresh.put(&m, &b).unwrap();
            }
            fresh.put_index(&be0.get_index().unwrap().unwrap()).unwrap();
            be = fresh;
        }
        let keys = prefix_keys(&root(), &toks, &[]);
        // with the index: the touched chunk survives a tight budget, the snapshot is the first out
        let mut c = Cache::open(Box::new(be), 1 << 30, root()).unwrap();
        assert!(c.open_report().index_loaded);
        assert_eq!(c.open_report().objects, 3);
        let mut order: Vec<_> = c.lru.values().copied().collect();
        assert_eq!(order.pop(), Some((keys[4], Kind::Rows)), "touched last");
        // without the index (corrupt): rebuilt from headers, objects intact
        let be1 = c.backend();
        let mut be1 = be1.lock().unwrap();
        be1.put_index(b"not json").unwrap();
        let mut fresh = MemBackend::new();
        for m in be1.list().unwrap() {
            let mut b = vec![0u8; m.bytes as usize];
            be1.get(&m.name(), &mut b).unwrap();
            fresh.put(&m, &b).unwrap();
        }
        fresh.put_index(b"not json").unwrap();
        let c2 = Cache::open(Box::new(fresh), 1 << 30, root()).unwrap();
        assert!(!c2.open_report().index_loaded);
        assert_eq!(c2.stats().objects, 3);
        // a foreign root's objects are dropped on open
        let mut fb = MemBackend::new();
        let m = ObjectMeta::snapshot(root_key(&["foreign"]), root_key(&["x"]), 1, None, 4);
        fb.put(&m, &[0; 4]).unwrap();
        let ok = ObjectMeta::snapshot(root(), root_key(&["y"]), 1, None, 4);
        fb.put(&ok, &[0; 4]).unwrap();
        let c3 = Cache::open(Box::new(fb), 1 << 30, root()).unwrap();
        assert_eq!(c3.open_report().dropped_foreign, 1);
        assert_eq!(c3.stats().objects, 1);
    }

    #[test]
    fn priority_tiers_evict_low_roles_first_and_ttl_expires_idle_objects() {
        use crate::policy::{Policies, RolePolicy};
        let mut pol = Policies::default();
        pol.reasoning = RolePolicy { priority: 1, ..Default::default() };
        pol.system = RolePolicy { priority: 9, ..Default::default() };
        // budget: 3 rows chunks (8192 each) + 2 snapshots (12288) = 49152; one more chunk overflows
        let mut c = Cache::open_with(Box::new(MemBackend::new()), 49_152, root(), pol.clone()).unwrap();
        let toks: Vec<u32> = (0..12).collect();
        let spans = [Span::new(0, 4, Role::System), Span::new(4, 8, Role::Reasoning), Span::new(8, 12, Role::User)];
        let keys = prefix_keys(&root(), &toks, &[]);
        save(&mut c, &toks, &spans, &[4, 12]);
        assert_eq!(c.stats().objects, 5);
        // the user chunk is the most recent, the system chunk the oldest, yet the reasoning
        // chunk (priority 1) goes first when something has to
        let extra = vec![0u8; 100];
        let m = ObjectMeta::rows(root(), root_key(&["x"]), root_key(&["p"]), 100, 101, Role::Assistant, vec![7], extra.len() as u64);
        c.put(m, &extra).unwrap();
        assert!(!c.has(keys[8], Kind::Rows), "reasoning chunk evicted first");
        assert!(c.has(keys[4], Kind::Rows) && c.has(keys[12], Kind::Rows));
        // the snapshot at 4 carries the system role (tier 9); the one at 12 the user role (tier 5):
        // the next overflow takes the user-tier objects (oldest first: rows 8..12 then snap 12)
        let m = ObjectMeta::rows(root(), root_key(&["y"]), root_key(&["p"]), 200, 201, Role::Assistant, vec![7], 8000);
        c.put(m, &vec![0u8; 8000]).unwrap();
        assert!(c.has(keys[4], Kind::Rows) && c.has(keys[4], Kind::Snapshot), "system tier survives");
        assert!(!c.has(keys[12], Kind::Rows));
        // ttl: objects idle longer than the role's ttl are dropped on the next insert / lookup
        let mut pol2 = Policies::default();
        pol2.tool = RolePolicy { ttl: Some(10), ..Default::default() };
        let mut c2 = Cache::open_with(Box::new(MemBackend::new()), 1 << 30, root(), pol2).unwrap();
        let spans2 = [Span::new(0, 4, Role::Tool), Span::new(4, 8, Role::User)];
        save(&mut c2, &toks[..8], &spans2, &[8]);
        assert_eq!(c2.stats().objects, 3);
        // age the tool chunk
        let k2 = prefix_keys(&root(), &toks[..8], &[]);
        c2.index.get_mut(&(k2[4], Kind::Rows)).unwrap().meta.last_used -= 100;
        assert!(c2.best_restore(&k2).is_none(), "the chain lost its tool chunk");
        assert_eq!(c2.stats().expired, 1);
        assert_eq!(c2.take_pending_delete().len(), 1);
        assert_eq!(c2.stats().objects, 2);
        // policies are visible and describable
        assert_eq!(c.policies().reasoning.priority, 1);
        assert!(pol.describe().contains("reasoning: priority 1"));
        assert_eq!(Policies::default().describe(), "default for every role");
    }

    #[test]
    fn get_errors_forget_the_object_and_put_errors_leave_the_index_clean() {
        let mut be = MemBackend::new();
        be.fail_next_put = true;
        let mut c = Cache::open(Box::new(be), 1 << 30, root()).unwrap();
        let m = ObjectMeta::snapshot(root(), root_key(&["s"]), 3, None, 4);
        assert!(c.put(m.clone(), &[0; 4]).is_err());
        assert_eq!(c.stats().objects, 0);
        c.put(m.clone(), &[0; 4]).unwrap();
        assert_eq!(c.stats().objects, 1);
        // wrong buffer length -> backend error -> object dropped
        let mut wrong = [0u8; 3];
        assert!(c.get(m.key, Kind::Snapshot, &mut wrong).is_err());
        assert_eq!(c.stats().objects, 0);
        // wrong root is refused
        let bad = ObjectMeta::snapshot(root_key(&["z"]), root_key(&["s2"]), 3, None, 4);
        assert!(c.put(bad, &[0; 4]).is_err());
        c.clear().unwrap();
        assert_eq!(c.stats().bytes, 0);
    }
}

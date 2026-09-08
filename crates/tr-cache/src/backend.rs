//! The storage backend: a blob store keyed by object name. The filesystem backend is the real
//! one; `MemBackend` is for tests. Network backends implement the same trait.
use crate::object::ObjectMeta;
use anyhow::{bail, Result};
use std::collections::BTreeMap;

pub trait Backend: Send {
    /// Store `body` under `meta.name()`, atomically (a reader never sees a partial object).
    fn put(&mut self, meta: &ObjectMeta, body: &[u8]) -> Result<()>;
    /// Read the object `name` into `body` (whose length must equal the stored `bytes`); returns
    /// the stored metadata.
    fn get(&mut self, name: &str, body: &mut [u8]) -> Result<ObjectMeta>;
    /// Remove `name`; not an error when it does not exist.
    fn delete(&mut self, name: &str) -> Result<()>;
    /// The metadata of every object (headers only), in any order.
    fn list(&mut self) -> Result<Vec<ObjectMeta>>;
    /// Store / read the serialized index (informational: LRU order and hit times).
    fn put_index(&mut self, json: &[u8]) -> Result<()>;
    fn get_index(&mut self) -> Result<Option<Vec<u8>>>;
    /// Where the data lives, for log lines.
    fn describe(&self) -> String;
}

/// In-memory backend (tests).
#[derive(Default)]
pub struct MemBackend {
    objects: BTreeMap<String, (ObjectMeta, Vec<u8>)>,
    index: Option<Vec<u8>>,
    /// Test hook: fail the next `put`.
    pub fail_next_put: bool,
}

impl MemBackend {
    pub fn new() -> MemBackend {
        MemBackend::default()
    }
    pub fn len(&self) -> usize {
        self.objects.len()
    }
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
    pub fn names(&self) -> Vec<String> {
        self.objects.keys().cloned().collect()
    }
}

impl Backend for MemBackend {
    fn put(&mut self, meta: &ObjectMeta, body: &[u8]) -> Result<()> {
        if self.fail_next_put {
            self.fail_next_put = false;
            bail!("injected put failure");
        }
        if body.len() as u64 != meta.bytes {
            bail!("body is {} bytes, header says {}", body.len(), meta.bytes);
        }
        self.objects.insert(meta.name(), (meta.clone(), body.to_vec()));
        Ok(())
    }
    fn get(&mut self, name: &str, body: &mut [u8]) -> Result<ObjectMeta> {
        let Some((m, b)) = self.objects.get(name) else { bail!("no object {name}") };
        if b.len() != body.len() {
            bail!("object {name} is {} bytes, buffer {}", b.len(), body.len());
        }
        body.copy_from_slice(b);
        Ok(m.clone())
    }
    fn delete(&mut self, name: &str) -> Result<()> {
        self.objects.remove(name);
        Ok(())
    }
    fn list(&mut self) -> Result<Vec<ObjectMeta>> {
        Ok(self.objects.values().map(|(m, _)| m.clone()).collect())
    }
    fn put_index(&mut self, json: &[u8]) -> Result<()> {
        self.index = Some(json.to_vec());
        Ok(())
    }
    fn get_index(&mut self) -> Result<Option<Vec<u8>>> {
        Ok(self.index.clone())
    }
    fn describe(&self) -> String {
        "memory".into()
    }
}

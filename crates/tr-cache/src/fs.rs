//! Filesystem backend: one file per object in a directory, written to `<name>.tmp` and renamed
//! into place. Bodies go through O_DIRECT (the box's page cache is not to be filled with
//! hundreds of MiB per request beside 16 GiB tiles); headers are small and read buffered.
use crate::backend::Backend;
use crate::object::{decode_header, encode_header, header_json_len, padded, parse_name, ObjectMeta, ALIGN};
use anyhow::{bail, Context, Result};
use std::io::Read;
use std::path::{Path, PathBuf};
use tr_sys::loader::DirectFile;

pub struct FsBackend {
    dir: PathBuf,
}

const INDEX: &str = "index.json";

impl FsBackend {
    /// Open (creating) the directory.
    pub fn open(dir: &Path) -> Result<FsBackend> {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        // leftovers of an interrupted write are never valid
        for e in std::fs::read_dir(dir)? {
            let e = e?;
            if e.file_name().to_string_lossy().ends_with(".tmp") {
                let _ = std::fs::remove_file(e.path());
            }
        }
        Ok(FsBackend { dir: dir.to_path_buf() })
    }
    pub fn dir(&self) -> &Path {
        &self.dir
    }
    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }
    /// Read and decode only the header of `path`.
    fn read_header(path: &Path) -> Result<(ObjectMeta, usize)> {
        let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut first = [0u8; 8];
        f.read_exact(&mut first).with_context(|| format!("read header of {}", path.display()))?;
        let n = header_json_len(&first)?;
        let mut buf = vec![0u8; 8 + n];
        buf[..8].copy_from_slice(&first);
        f.read_exact(&mut buf[8..]).with_context(|| format!("read header of {}", path.display()))?;
        decode_header(&buf)
    }
}

impl Backend for FsBackend {
    fn put(&mut self, meta: &ObjectMeta, body: &[u8]) -> Result<()> {
        if body.len() as u64 != meta.bytes {
            bail!("body is {} bytes, header says {}", body.len(), meta.bytes);
        }
        let name = meta.name();
        let tmp = self.path(&format!("{name}.tmp"));
        let fin = self.path(&name);
        let header = encode_header(meta)?;
        {
            let f = DirectFile::create(&tmp)?;
            f.write_at(0, &header)?;
            f.write_at(header.len() as u64, body)?;
            // pad the file to the alignment so a later O_DIRECT read of the whole body is legal
            let total = header.len() + body.len();
            let pad = padded(total) - total;
            if pad > 0 {
                let z = vec![0u8; pad];
                f.write_at(total as u64, &z)?;
            }
            f.sync()?;
        }
        std::fs::rename(&tmp, &fin).with_context(|| format!("rename {} -> {}", tmp.display(), fin.display()))?;
        Ok(())
    }

    fn get(&mut self, name: &str, body: &mut [u8]) -> Result<ObjectMeta> {
        let path = self.path(name);
        let (meta, off) = Self::read_header(&path)?;
        if meta.bytes != body.len() as u64 {
            bail!("object {name} is {} bytes, buffer {}", meta.bytes, body.len());
        }
        if meta.name() != name {
            bail!("object {name} carries the header of {}", meta.name());
        }
        let f = DirectFile::open(&path)?;
        if (f.len()? as usize) < off + body.len() {
            bail!("object {name} is truncated");
        }
        // an aligned buffer and offset read the whole body with O_DIRECT (see DirectFile::read_at)
        debug_assert_eq!(off % ALIGN, 0);
        f.read_at(off as u64, body)?;
        Ok(meta)
    }

    fn delete(&mut self, name: &str) -> Result<()> {
        match std::fs::remove_file(self.path(name)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e).with_context(|| format!("delete {name}")),
        }
    }

    fn list(&mut self) -> Result<Vec<ObjectMeta>> {
        let mut out = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for e in std::fs::read_dir(&self.dir)? {
            let e = e?;
            let n = e.file_name().to_string_lossy().into_owned();
            if parse_name(&n).is_some() {
                names.push(n);
            }
        }
        names.sort();
        for n in names {
            match Self::read_header(&self.path(&n)) {
                Ok((m, _)) if m.name() == n => out.push(m),
                Ok((m, _)) => eprintln!("prefix cache: {n} carries the header of {} (ignored)", m.name()),
                Err(e) => eprintln!("prefix cache: {n}: {e:#} (ignored)"),
            }
        }
        Ok(out)
    }

    fn put_index(&mut self, json: &[u8]) -> Result<()> {
        let tmp = self.path("index.json.tmp");
        std::fs::write(&tmp, json).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, self.path(INDEX))?;
        Ok(())
    }

    fn get_index(&mut self) -> Result<Option<Vec<u8>>> {
        match std::fs::read(self.path(INDEX)) {
            Ok(b) => Ok(Some(b)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).context("read index.json"),
        }
    }

    fn describe(&self) -> String {
        self.dir.display().to_string()
    }
}

#[cfg(test)]
pub(crate) fn test_dir(tag: &str) -> PathBuf {
    // Not /tmp: tmpfs has no O_DIRECT, and the box rule is no staging on tmpfs.
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../target/test-tmp/tr-cache-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::root_key;
    use crate::role::Role;
    use crate::AlignedBuf;

    #[test]
    fn put_get_list_delete() {
        let dir = test_dir("fs");
        let mut b = FsBackend::open(&dir).unwrap();
        let root = root_key(&["r"]);
        let k1 = root_key(&["k1"]);
        let k2 = root_key(&["k2"]);
        let body: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        let m1 = ObjectMeta::rows(root, k1, root, 0, 3, Role::System, vec![1, 2, 3], body.len() as u64);
        b.put(&m1, &body).unwrap();
        // unaligned source buffer and an aligned one both work; the file is padded to 4 KiB
        let mut ab = AlignedBuf::new(3 * ALIGN);
        for (i, x) in ab.iter_mut().enumerate() {
            *x = (i * 7 % 250) as u8;
        }
        let m2 = ObjectMeta::snapshot(root, k2, 3, None, ab.len() as u64);
        b.put(&m2, &ab).unwrap();
        assert_eq!(std::fs::metadata(dir.join(m1.name())).unwrap().len() % ALIGN as u64, 0);
        let mut got = AlignedBuf::new(body.len());
        let m = b.get(&m1.name(), &mut got).unwrap();
        assert_eq!(m, m1);
        assert_eq!(&got[..], &body[..]);
        let mut got2 = AlignedBuf::new(ab.len());
        b.get(&m2.name(), &mut got2).unwrap();
        assert_eq!(&got2[..], &ab[..]);
        // wrong buffer size is an error, not a short read
        let mut small = vec![0u8; 10];
        assert!(b.get(&m1.name(), &mut small).is_err());
        // a leftover .tmp and a stray file are ignored by list
        std::fs::write(dir.join("zzz.rows.tmp"), b"junk").unwrap();
        std::fs::write(dir.join("notes.txt"), b"junk").unwrap();
        let mut l = b.list().unwrap();
        l.sort_by_key(|m| m.name());
        let mut want = vec![m1.clone(), m2.clone()];
        want.sort_by_key(|m| m.name());
        assert_eq!(l, want);
        // a corrupt object is skipped by list and fails get
        std::fs::write(dir.join(format!("{}.rows", root_key(&["bad"]).hex())), b"\x05\0\0\0\0\0\0\0garbage").unwrap();
        assert_eq!(b.list().unwrap().len(), 2);
        b.delete(&m1.name()).unwrap();
        b.delete(&m1.name()).unwrap();
        assert_eq!(b.list().unwrap(), vec![m2.clone()]);
        assert!(b.get_index().unwrap().is_none());
        b.put_index(b"{\"x\":1}").unwrap();
        assert_eq!(b.get_index().unwrap().unwrap(), b"{\"x\":1}");
        // reopening removes .tmp leftovers
        drop(b);
        let _ = FsBackend::open(&dir).unwrap();
        assert!(!dir.join("zzz.rows.tmp").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

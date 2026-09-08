//! O_DIRECT reads into caller-provided (node-bound) memory. Falls back to buffered pread for
//! the unaligned parts, so any (offset, len, buffer) works; aligned 2 MiB sections are fast.
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::Path;

pub const DIRECT_ALIGN: usize = 4096;

pub struct DirectFile {
    direct: File,
    buffered: File,
}

impl DirectFile {
    pub fn open(path: &Path) -> Result<DirectFile> {
        let direct = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .with_context(|| format!("open O_DIRECT {}", path.display()))?;
        let buffered = File::open(path).with_context(|| format!("open {}", path.display()))?;
        Ok(DirectFile { direct, buffered })
    }

    /// Create (truncate) a file for writing with O_DIRECT; `write_at` mirrors `read_at`.
    pub fn create(path: &Path) -> Result<DirectFile> {
        let direct = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .with_context(|| format!("create O_DIRECT {}", path.display()))?;
        let buffered = std::fs::OpenOptions::new().write(true).open(path).with_context(|| format!("open {}", path.display()))?;
        Ok(DirectFile { direct, buffered })
    }

    pub fn len(&self) -> Result<u64> {
        Ok(self.buffered.metadata()?.len())
    }

    /// Write all of `src` at `offset`. The 4 KiB-aligned middle goes through O_DIRECT (no page
    /// cache left behind), the unaligned head and tail through the buffered handle.
    pub fn write_at(&self, offset: u64, src: &[u8]) -> Result<()> {
        let len = src.len();
        let ptr_align = src.as_ptr() as usize % DIRECT_ALIGN;
        let off_align = offset as usize % DIRECT_ALIGN;
        if ptr_align == off_align {
            let head = if off_align == 0 { 0 } else { (DIRECT_ALIGN - off_align).min(len) };
            let mid_end = len - (len - head) % DIRECT_ALIGN;
            if head > 0 {
                self.buffered.write_all_at(&src[..head], offset)?;
            }
            if mid_end > head {
                write_all_direct(&self.direct, &src[head..mid_end], offset + head as u64)?;
            }
            if mid_end < len {
                self.buffered.write_all_at(&src[mid_end..], offset + mid_end as u64)?;
            }
            Ok(())
        } else {
            self.buffered.write_all_at(src, offset).context("buffered pwrite")
        }
    }

    /// Flush file data and metadata to the device.
    pub fn sync(&self) -> Result<()> {
        self.buffered.sync_all().context("fsync")
    }

    /// Read exactly `dst.len()` bytes at `offset`. Uses O_DIRECT for the 4 KiB-aligned middle.
    pub fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<()> {
        let len = dst.len();
        let ptr_align = dst.as_ptr() as usize % DIRECT_ALIGN;
        let off_align = offset as usize % DIRECT_ALIGN;
        // Direct I/O needs offset, length and buffer address all aligned and identically phased.
        if ptr_align == off_align {
            let head = if off_align == 0 { 0 } else { (DIRECT_ALIGN - off_align).min(len) };
            let mid_end = len - (len - head) % DIRECT_ALIGN;
            if head > 0 {
                self.buffered.read_exact_at(&mut dst[..head], offset)?;
            }
            if mid_end > head {
                read_exact_direct(&self.direct, &mut dst[head..mid_end], offset + head as u64)?;
            }
            if mid_end < len {
                self.buffered.read_exact_at(&mut dst[mid_end..], offset + mid_end as u64)?;
            }
            Ok(())
        } else {
            self.buffered.read_exact_at(dst, offset).context("buffered pread")
        }
    }
}

fn read_exact_direct(f: &File, mut dst: &mut [u8], mut offset: u64) -> Result<()> {
    while !dst.is_empty() {
        // Cap single requests to keep the kernel's bio sizes reasonable.
        let want = dst.len().min(64 << 20);
        let n = f.read_at(&mut dst[..want], offset).context("O_DIRECT pread")?;
        if n == 0 {
            bail!("short O_DIRECT read at offset {offset}");
        }
        dst = &mut dst[n..];
        offset += n as u64;
    }
    Ok(())
}

fn write_all_direct(f: &File, mut src: &[u8], mut offset: u64) -> Result<()> {
    while !src.is_empty() {
        let want = src.len().min(64 << 20);
        let n = f.write_at(&src[..want], offset).context("O_DIRECT pwrite")?;
        if n == 0 {
            bail!("short O_DIRECT write at offset {offset}");
        }
        src = &src[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn direct_write_round_trips() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../target/test-tmp/loader-w-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.bin");
        let data: Vec<u8> = (0..(1 << 20) + 4096 + 777).map(|i| (i * 17 % 253) as u8).collect();
        let mut buf = vec![0u8; data.len() + DIRECT_ALIGN];
        let shift = (DIRECT_ALIGN - buf.as_ptr() as usize % DIRECT_ALIGN) % DIRECT_ALIGN;
        buf[shift..shift + data.len()].copy_from_slice(&data);
        {
            let f = DirectFile::create(&path).unwrap();
            // aligned buffer at aligned offset, then an unaligned tail write appended at an odd offset
            f.write_at(0, &buf[shift..shift + (1 << 20)]).unwrap();
            f.write_at(1 << 20, &buf[shift + (1 << 20)..shift + data.len()]).unwrap();
            f.sync().unwrap();
        }
        assert_eq!(std::fs::read(&path).unwrap(), data);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn direct_read_matches_buffered() {
        // Not /tmp: tmpfs does not support O_DIRECT (and the box rule is no staging on tmpfs).
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../target/test-tmp/loader-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blob.bin");
        let data: Vec<u8> = (0..(3 << 20) + 12345).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::File::create(&path).unwrap().write_all(&data).unwrap();
        let f = DirectFile::open(&path).unwrap();
        // Aligned buffer, various offsets/lengths.
        let mut buf = vec![0u8; (2 << 20) + 8192 + 4096 + 64];
        let base = buf.as_ptr() as usize;
        let shift = (DIRECT_ALIGN - base % DIRECT_ALIGN) % DIRECT_ALIGN;
        for (off, n) in [(0usize, 2 << 20), (4096, 1 << 20), (100, 5000), (8192 + 7, 70000), (1 << 20, (2 << 20) + 8192)] {
            let dst = &mut buf[shift..shift + n];
            f.read_at(off as u64, dst).unwrap();
            assert_eq!(dst, &data[off..off + n], "off {off} n {n}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

/// One contiguous copy from a file into node-bound memory.
pub struct Section {
    pub node: usize,
    pub file: std::sync::Arc<DirectFile>,
    pub offset: u64,
    pub dst: *mut u8,
    pub len: usize,
}
unsafe impl Send for Section {}
unsafe impl Sync for Section {}

/// Load all sections in parallel: each node's workers read that node's sections in 2 MiB
/// chunks (dynamic per-node counter), so the first touch of every page happens on the owning
/// tile. Returns total bytes read.
pub fn parallel_load(pool: &mut crate::pool::Pool, sections: &[Section]) -> Result<u64> {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    const CHUNK: usize = 2 << 20;
    // Per-node chunk lists.
    let n_nodes = pool.n_nodes();
    let mut chunks: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); n_nodes]; // (section, start, len)
    for (si, s) in sections.iter().enumerate() {
        if s.node >= n_nodes {
            bail!("section on node {} but pool has {} nodes", s.node, n_nodes);
        }
        let mut off = 0;
        while off < s.len {
            let l = (s.len - off).min(CHUNK);
            chunks[s.node].push((si, off, l));
            off += l;
        }
    }
    let counters: Vec<AtomicUsize> = (0..n_nodes).map(|_| AtomicUsize::new(0)).collect();
    let total = AtomicU64::new(0);
    let errors = std::sync::Mutex::new(Vec::<String>::new());
    pool.run(|ctx| {
        let list = &chunks[ctx.node];
        loop {
            let i = counters[ctx.node].fetch_add(1, Ordering::Relaxed);
            if i >= list.len() {
                break;
            }
            let (si, start, len) = list[i];
            let s = &sections[si];
            let dst = unsafe { std::slice::from_raw_parts_mut(s.dst.add(start), len) };
            match s.file.read_at(s.offset + start as u64, dst) {
                Ok(()) => {
                    total.fetch_add(len as u64, Ordering::Relaxed);
                }
                Err(e) => errors.lock().unwrap().push(format!("node {} section {si} +{start}: {e}", ctx.node)),
            }
        }
    });
    let errs = errors.into_inner().unwrap();
    if !errs.is_empty() {
        bail!("load errors: {}", errs.join("; "));
    }
    Ok(total.load(Ordering::Relaxed))
}

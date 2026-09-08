//! Per-tile memory arenas: anonymous mmap bound to one NUMA node with MPOL_BIND, THP-advised,
//! bump-allocated in 2 MiB units. Placement can be verified through get_mempolicy / numa_maps.
use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::ptr;

pub const MPOL_BIND: libc::c_int = 2;
pub const MPOL_F_NODE: libc::c_int = 1 << 0;
pub const MPOL_F_ADDR: libc::c_int = 1 << 1;
pub const MPOL_MF_STRICT: libc::c_uint = 1 << 0;
pub const HUGE_2M: usize = 2 << 20;

pub fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) / a * a
}

/// A region of memory bound to one NUMA node.
pub struct Arena {
    base: *mut u8,
    len: usize,
    used: usize,
    pub node: usize,
}
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// Reserve `len` bytes (rounded to 2 MiB) bound to `node`. Pages are not touched here.
    pub fn new(node: usize, len: usize) -> Result<Arena> {
        let len = align_up(len.max(HUGE_2M), HUGE_2M);
        unsafe {
            // Over-reserve by 2 MiB so we can hand back a 2 MiB-aligned base for THP.
            let raw = libc::mmap(
                ptr::null_mut(),
                len + HUGE_2M,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            );
            if raw == libc::MAP_FAILED {
                bail!("mmap({} bytes) failed: {}", len, std::io::Error::last_os_error());
            }
            let base = align_up(raw as usize, HUGE_2M) as *mut u8;
            // Trim the unaligned head/tail so the vma is exactly [base, base+len).
            let head = base as usize - raw as usize;
            if head > 0 {
                libc::munmap(raw, head);
            }
            let tail = HUGE_2M - head;
            if tail > 0 {
                libc::munmap(base.add(len) as *mut _, tail);
            }
            bind_to_node(base, len, node)?;
            if libc::madvise(base as *mut _, len, libc::MADV_HUGEPAGE) != 0 {
                eprintln!("warning: madvise(MADV_HUGEPAGE) failed: {}", std::io::Error::last_os_error());
            }
            Ok(Arena { base, len, used: 0, node })
        }
    }

    pub fn capacity(&self) -> usize {
        self.len
    }
    pub fn used(&self) -> usize {
        self.used
    }
    pub fn base(&self) -> *mut u8 {
        self.base
    }

    /// Bump-allocate `size` bytes with `align` (≤ 2 MiB). Returns a raw pointer; the memory lives
    /// as long as the arena. Not thread-safe; allocate from one thread, fill from many.
    pub fn alloc(&mut self, size: usize, align: usize) -> Result<*mut u8> {
        let off = align_up(self.used, align.max(64));
        if off + size > self.len {
            bail!(
                "arena node {} exhausted: need {} + {} > {}",
                self.node,
                off,
                size,
                self.len
            );
        }
        self.used = off + size;
        Ok(unsafe { self.base.add(off) })
    }

    /// Allocate a zero-initialised slice of `T` (T must be plain data).
    pub fn alloc_slice<T: Copy>(&mut self, n: usize) -> Result<&'static mut [T]> {
        let p = self.alloc(n * std::mem::size_of::<T>(), std::mem::align_of::<T>().max(64))?;
        unsafe {
            ptr::write_bytes(p, 0, n * std::mem::size_of::<T>());
            Ok(std::slice::from_raw_parts_mut(p as *mut T, n))
        }
    }

    /// Allocate a slice without touching it: pages fault in lazily on first write (from the
    /// tile's own workers), so a large reservation costs nothing until used (KV cache growth).
    pub fn alloc_slice_uninit<T: Copy>(&mut self, n: usize) -> Result<&'static mut [T]> {
        let p = self.alloc(n * std::mem::size_of::<T>(), std::mem::align_of::<T>().max(64))?;
        Ok(unsafe { std::slice::from_raw_parts_mut(p as *mut T, n) })
    }

    /// Lock all pages of the arena in RAM. Returns Err if refused (caller decides to warn or fail).
    pub fn mlock(&self) -> Result<()> {
        let r = unsafe { libc::mlock(self.base as *const _, self.len) };
        if r != 0 {
            bail!("mlock({} MiB, node {}): {}", self.len >> 20, self.node, std::io::Error::last_os_error());
        }
        Ok(())
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.base as *mut _, self.len);
        }
    }
}

/// mbind(addr, len, MPOL_BIND, {node}).
pub fn bind_to_node(addr: *mut u8, len: usize, node: usize) -> Result<()> {
    let mut mask = [0u64; 16]; // up to 1024 nodes
    mask[node / 64] |= 1u64 << (node % 64);
    let maxnode = (mask.len() * 64) as libc::c_ulong;
    let r = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            addr as *mut libc::c_void,
            len as libc::c_ulong,
            MPOL_BIND as libc::c_ulong,
            mask.as_ptr(),
            maxnode,
            MPOL_MF_STRICT as libc::c_ulong,
        )
    };
    if r != 0 {
        bail!("mbind(node {node}): {}", std::io::Error::last_os_error());
    }
    Ok(())
}

/// Node id of the page containing `addr` (the page must already be populated).
pub fn node_of_addr(addr: *const u8) -> Result<usize> {
    let mut node: libc::c_int = -1;
    let r = unsafe {
        libc::syscall(
            libc::SYS_get_mempolicy,
            &mut node as *mut libc::c_int,
            ptr::null_mut::<u64>(),
            0 as libc::c_ulong,
            addr as *mut libc::c_void,
            (MPOL_F_NODE | MPOL_F_ADDR) as libc::c_ulong,
        )
    };
    if r != 0 {
        bail!("get_mempolicy: {}", std::io::Error::last_os_error());
    }
    Ok(node as usize)
}

/// Sample `samples` evenly spaced pages of [addr, addr+len) and return node -> count.
pub fn sample_placement(addr: *const u8, len: usize, samples: usize) -> Result<BTreeMap<usize, usize>> {
    let mut out = BTreeMap::new();
    let step = (len / samples.max(1)).max(4096);
    let mut off = 0;
    while off < len {
        let n = node_of_addr(unsafe { addr.add(off) })?;
        *out.entry(n).or_insert(0) += 1;
        off += step;
    }
    Ok(out)
}

/// Per-node resident page counts of every vma overlapping [addr, addr+len), from /proc/self/numa_maps.
pub fn numa_maps_pages(addr: *const u8, len: usize) -> Result<BTreeMap<usize, u64>> {
    let text = std::fs::read_to_string("/proc/self/numa_maps").context("read numa_maps")?;
    let lo = addr as usize;
    let hi = lo + len;
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let Some(a) = it.next() else { continue };
        let Ok(start) = usize::from_str_radix(a, 16) else { continue };
        if start < lo || start >= hi {
            continue;
        }
        for tok in it {
            if let Some(rest) = tok.strip_prefix('N') {
                if let Some((n, c)) = rest.split_once('=') {
                    if let (Ok(n), Ok(c)) = (n.parse::<usize>(), c.parse::<u64>()) {
                        *out.entry(n).or_insert(0) += c;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Raise RLIMIT_MEMLOCK as far as allowed. Returns the resulting soft limit in bytes.
pub fn raise_memlock_limit() -> u64 {
    unsafe {
        let mut rl: libc::rlimit = std::mem::zeroed();
        libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rl);
        let want = libc::rlimit { rlim_cur: libc::RLIM_INFINITY, rlim_max: libc::RLIM_INFINITY };
        if libc::setrlimit(libc::RLIMIT_MEMLOCK, &want) == 0 {
            return u64::MAX;
        }
        let hard = libc::rlimit { rlim_cur: rl.rlim_max, rlim_max: rl.rlim_max };
        libc::setrlimit(libc::RLIMIT_MEMLOCK, &hard);
        libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rl);
        rl.rlim_cur as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::Topology;

    #[test]
    fn arena_alloc_and_placement() {
        let topo = Topology::discover().unwrap();
        let node = topo.nodes.last().unwrap().id;
        let mut a = Arena::new(node, 8 << 20).unwrap();
        let s = a.alloc_slice::<u8>(4 << 20).unwrap();
        s.fill(1);
        let placed = sample_placement(s.as_ptr(), s.len(), 8).unwrap();
        assert_eq!(placed.keys().copied().collect::<Vec<_>>(), vec![node], "placement {placed:?}");
        assert!(a.used() >= 4 << 20);
        assert!(a.alloc(16 << 20, 64).is_err());
    }
}

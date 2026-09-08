//! Pinned SPMD worker pool: one thread per physical core, grouped by NUMA node, with a
//! hierarchical sense-reversing barrier (intra-node lines local to the node, one leader line
//! per node for the cross-node step). Workers spin during a run and futex-sleep when idle.
use crate::affinity::{pin_to_cpu, set_thread_name};
use crate::numa::Arena;
use crate::topology::Topology;
use anyhow::{bail, Result};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[repr(align(128))]
#[derive(Default)]
pub struct Padded<T>(pub T);

pub const MAX_CORES_PER_NODE: usize = 16;

/// Per-node synchronisation block, allocated in that node's own memory.
#[repr(C)]
pub struct NodeSync {
    /// Bumped by the orchestrator to start a run; workers spin / futex-wait on it.
    pub run_gen: Padded<AtomicU64>,
    /// Barrier release word, written by the node leader.
    pub release: Padded<AtomicU64>,
    /// Cross-node barrier flag of this node's leader (read remotely by the other leaders).
    pub leader_flag: Padded<AtomicU64>,
    /// Per-worker arrival words (barrier) and done words (end of run).
    pub arrive: [Padded<AtomicU64>; MAX_CORES_PER_NODE],
    pub done: [Padded<AtomicU64>; MAX_CORES_PER_NODE],
}

struct Shared {
    nodes: Vec<&'static NodeSync>,
    cores_per_node: usize,
    n_nodes: usize,
    /// Pointer to the current run closure (valid for the duration of the run).
    job: AtomicUsize,
    job_vt: AtomicUsize,
}

/// Handed to the run closure on every worker.
pub struct WorkerCtx<'a> {
    pub node: usize,
    pub local: usize,
    pub global: usize,
    pub n_nodes: usize,
    pub cores_per_node: usize,
    shared: &'a Shared,
    gen: std::cell::Cell<u64>,
}

impl<'a> WorkerCtx<'a> {
    pub fn n_workers(&self) -> usize {
        self.n_nodes * self.cores_per_node
    }

    /// Full barrier across all workers of all nodes.
    #[inline]
    pub fn barrier(&self) {
        let g = self.gen.get() + 1;
        self.gen.set(g);
        let ns = self.shared.nodes[self.node];
        if self.local == 0 {
            // wait for the node's own workers
            for i in 1..self.cores_per_node {
                spin_until(&ns.arrive[i].0, g);
            }
            // cross-node: publish own flag, wait for the other leaders
            ns.leader_flag.0.store(g, Ordering::Release);
            for n in 0..self.n_nodes {
                if n != self.node {
                    spin_until(&self.shared.nodes[n].leader_flag.0, g);
                }
            }
            ns.release.0.store(g, Ordering::Release);
        } else {
            ns.arrive[self.local].0.store(g, Ordering::Release);
            spin_until(&ns.release.0, g);
        }
    }

    /// Barrier among the workers of this node only. Uses the same generation counter, so
    /// every worker of the node must call it the same number of times.
    #[inline]
    pub fn node_barrier(&self) {
        let g = self.gen.get() + 1;
        self.gen.set(g);
        let ns = self.shared.nodes[self.node];
        if self.local == 0 {
            for i in 1..self.cores_per_node {
                spin_until(&ns.arrive[i].0, g);
            }
            ns.release.0.store(g, Ordering::Release);
        } else {
            ns.arrive[self.local].0.store(g, Ordering::Release);
            spin_until(&ns.release.0, g);
        }
    }

    /// Split `n` items into a contiguous range for this worker among all workers.
    pub fn range_global(&self, n: usize) -> std::ops::Range<usize> {
        split(n, self.global, self.n_workers())
    }
    /// Split `n` items among the workers of this node.
    pub fn range_local(&self, n: usize) -> std::ops::Range<usize> {
        split(n, self.local, self.cores_per_node)
    }
}

pub fn split(n: usize, i: usize, k: usize) -> std::ops::Range<usize> {
    let base = n / k;
    let rem = n % k;
    let start = i * base + i.min(rem);
    let len = base + usize::from(i < rem);
    start..start + len
}

#[inline(always)]
fn spin_until(a: &AtomicU64, g: u64) {
    while a.load(Ordering::Acquire) < g {
        std::hint::spin_loop();
    }
}

fn futex_wait(word: &AtomicU64, expected: u64, timeout: Duration) {
    // Compare the low 32 bits: futex works on u32. Generations change by 1 each run.
    let ts = libc::timespec { tv_sec: timeout.as_secs() as _, tv_nsec: timeout.subsec_nanos() as _ };
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            word as *const AtomicU64 as *const u32,
            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
            expected as u32,
            &ts as *const libc::timespec,
        );
    }
}

fn futex_wake(word: &AtomicU64) {
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            word as *const AtomicU64 as *const u32,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            i32::MAX,
        );
    }
}

pub struct Pool {
    shared: Arc<Shared>,
    run_gen: u64,
    /// Caller-side ctx generation for the orchestrator (worker (0,0)).
    handles: Vec<std::thread::JoinHandle<()>>,
    pub cpus: Vec<Vec<usize>>,
}

const IDLE_SPIN: Duration = Duration::from_micros(300);

impl Pool {
    /// Build a pool over `cores_per_node` physical cores of every node in `topo` (default: all,
    /// capped by the smallest node). The calling thread becomes worker (0,0) and is pinned.
    pub fn new(topo: &Topology, cores_per_node: Option<usize>) -> Result<Pool> {
        let n_nodes = topo.n_nodes();
        let min_cores = topo.nodes.iter().map(|n| n.physical.len()).min().unwrap_or(0);
        let cpn = cores_per_node.unwrap_or(min_cores).min(min_cores);
        if cpn == 0 || cpn > MAX_CORES_PER_NODE {
            bail!("cores_per_node {cpn} out of range (max {MAX_CORES_PER_NODE})");
        }
        let cpus: Vec<Vec<usize>> = topo.nodes.iter().map(|n| n.physical[..cpn].to_vec()).collect();
        // One sync block per node in that node's memory. Arenas are leaked deliberately.
        let mut nodes: Vec<&'static NodeSync> = Vec::with_capacity(n_nodes);
        for n in &topo.nodes {
            let mut arena = Arena::new(n.id, 2 << 20)?;
            let bytes = arena.alloc_slice::<u8>(std::mem::size_of::<NodeSync>())?;
            let p = bytes.as_mut_ptr() as *mut NodeSync;
            unsafe {
                std::ptr::write_bytes(p as *mut u8, 0, std::mem::size_of::<NodeSync>());
                nodes.push(&*p);
            }
            std::mem::forget(arena);
        }
        let shared = Arc::new(Shared {
            nodes,
            cores_per_node: cpn,
            n_nodes,
            job: AtomicUsize::new(0),
            job_vt: AtomicUsize::new(0),
        });
        pin_to_cpu(cpus[0][0])?;
        set_thread_name("tr-w0.0");
        let mut handles = Vec::new();
        for node in 0..n_nodes {
            for local in 0..cpn {
                if node == 0 && local == 0 {
                    continue;
                }
                let sh = shared.clone();
                let cpu = cpus[node][local];
                handles.push(
                    std::thread::Builder::new()
                        .name(format!("tr-w{node}.{local}"))
                        .stack_size(8 << 20)
                        .spawn(move || worker_main(sh, node, local, cpu))?,
                );
            }
        }
        Ok(Pool { shared, run_gen: 0, handles, cpus })
    }

    pub fn n_nodes(&self) -> usize {
        self.shared.n_nodes
    }
    pub fn cores_per_node(&self) -> usize {
        self.shared.cores_per_node
    }
    pub fn n_workers(&self) -> usize {
        self.shared.n_nodes * self.shared.cores_per_node
    }

    /// Run `f` on every worker (SPMD), including the calling thread as worker (0,0).
    /// Returns when all workers finished.
    pub fn run<F: Fn(&WorkerCtx) + Sync>(&mut self, f: F) {
        let fref: &(dyn Fn(&WorkerCtx) + Sync) = &f;
        let (data, vt): (usize, usize) = unsafe { std::mem::transmute(fref) };
        self.shared.job.store(data, Ordering::Relaxed);
        self.shared.job_vt.store(vt, Ordering::Relaxed);
        self.run_gen += 1;
        let g = self.run_gen;
        for n in 0..self.shared.n_nodes {
            let ns = self.shared.nodes[n];
            ns.run_gen.0.store(g, Ordering::Release);
            futex_wake(&ns.run_gen.0);
        }
        // run our own share
        let ctx = WorkerCtx { node: 0, local: 0, global: 0, n_nodes: self.shared.n_nodes, cores_per_node: self.shared.cores_per_node, shared: &self.shared, gen: std::cell::Cell::new(barrier_gen_base(g)) };
        f(&ctx);
        // wait for everyone
        for n in 0..self.shared.n_nodes {
            let ns = self.shared.nodes[n];
            for i in 0..self.shared.cores_per_node {
                if n == 0 && i == 0 {
                    continue;
                }
                spin_until(&ns.done[i].0, g);
            }
        }
    }
}

/// Barrier generations restart from a run-specific base so runs cannot alias each other's
/// barrier counts (each run may use a different number of barriers).
#[inline]
fn barrier_gen_base(run_gen: u64) -> u64 {
    run_gen << 32
}

fn worker_main(sh: Arc<Shared>, node: usize, local: usize, cpu: usize) {
    if let Err(e) = pin_to_cpu(cpu) {
        eprintln!("worker {node}.{local}: {e}");
    }
    set_thread_name(&format!("tr-w{node}.{local}"));
    let ns = sh.nodes[node];
    let mut seen = 0u64;
    loop {
        // wait for a new run generation
        let mut t0: Option<Instant> = None;
        let g = loop {
            let g = ns.run_gen.0.load(Ordering::Acquire);
            if g > seen {
                break g;
            }
            match t0 {
                None => t0 = Some(Instant::now()),
                Some(t) if t.elapsed() > IDLE_SPIN => {
                    futex_wait(&ns.run_gen.0, seen, Duration::from_millis(50));
                }
                _ => {}
            }
            std::hint::spin_loop();
        };
        seen = g;
        let data = sh.job.load(Ordering::Relaxed);
        let vt = sh.job_vt.load(Ordering::Relaxed);
        let f: &(dyn Fn(&WorkerCtx) + Sync) = unsafe { std::mem::transmute((data, vt)) };
        let ctx = WorkerCtx { node, local, global: node * sh.cores_per_node + local, n_nodes: sh.n_nodes, cores_per_node: sh.cores_per_node, shared: &sh, gen: std::cell::Cell::new(barrier_gen_base(g)) };
        f(&ctx);
        ns.done[local].0.store(g, Ordering::Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        // Workers are detached spinners/sleepers; the process exits with them.
        for h in self.handles.drain(..) {
            drop(h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn split_covers_all() {
        let mut total = 0;
        for i in 0..13 {
            total += split(100, i, 13).len();
        }
        assert_eq!(total, 100);
        assert_eq!(split(5, 0, 8), 0..1);
        assert_eq!(split(5, 7, 8), 5..5);
    }

    #[test]
    fn barrier_stress() {
        let topo = Topology::discover().unwrap();
        let mut pool = Pool::new(&topo, Some(2)).unwrap();
        let rounds = 2000u64;
        let counter = AtomicU64::new(0);
        let nw = pool.n_workers() as u64;
        pool.run(|ctx| {
            for r in 0..rounds {
                counter.fetch_add(1, Ordering::Relaxed);
                ctx.barrier();
                // After the barrier every worker must see the full count for this round.
                let c = counter.load(Ordering::Relaxed);
                assert_eq!(c, (r + 1) * nw, "worker {} round {r}", ctx.global);
                ctx.barrier();
            }
        });
        // second run: node-local barriers plus per-run work
        let hits = AtomicU64::new(0);
        pool.run(|ctx| {
            for _ in 0..100 {
                ctx.node_barrier();
            }
            hits.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(hits.load(Ordering::Relaxed), nw);
    }
}

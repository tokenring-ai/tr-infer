//! NUMA topology from sysfs: nodes, their physical cores (SMT siblings excluded), memory, distances.
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
    /// All logical CPUs of the node (including SMT siblings).
    pub cpus: Vec<usize>,
    /// One logical CPU per physical core (the lowest sibling id).
    pub physical: Vec<usize>,
    pub mem_total_kb: u64,
    pub mem_free_kb: u64,
    pub distances: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct Topology {
    pub nodes: Vec<Node>,
}

/// Parse a Linux cpulist such as `0-12,104-116`.
pub fn parse_cpulist(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in s.trim().split(',') {
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let a: usize = a.trim().parse().unwrap_or(0);
            let b: usize = b.trim().parse().unwrap_or(a);
            out.extend(a..=b);
        } else if let Ok(v) = part.trim().parse() {
            out.push(v);
        }
    }
    out
}

fn read_trim(p: &Path) -> Result<String> {
    Ok(fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?.trim().to_string())
}

fn meminfo_kb(p: &Path, key: &str) -> u64 {
    // lines look like: "Node 0 MemTotal:       16124320 kB"
    fs::read_to_string(p)
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.contains(key)).and_then(|l| {
                l.split_whitespace().rev().nth(1).and_then(|v| v.parse().ok())
            })
        })
        .unwrap_or(0)
}

impl Topology {
    pub fn discover() -> Result<Topology> {
        let root = Path::new("/sys/devices/system/node");
        let mut ids: Vec<usize> = fs::read_dir(root)
            .context("read /sys/devices/system/node")?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let n = e.file_name().into_string().ok()?;
                n.strip_prefix("node")?.parse::<usize>().ok()
            })
            .collect();
        ids.sort_unstable();
        let mut nodes = Vec::with_capacity(ids.len());
        for id in ids {
            let dir = root.join(format!("node{id}"));
            let cpus = parse_cpulist(&read_trim(&dir.join("cpulist"))?);
            let mut physical = Vec::new();
            let mut seen_cores: Vec<usize> = Vec::new();
            for &cpu in &cpus {
                let sib = Path::new("/sys/devices/system/cpu")
                    .join(format!("cpu{cpu}/topology/thread_siblings_list"));
                let core = read_trim(&sib)
                    .map(|s| parse_cpulist(&s).into_iter().min().unwrap_or(cpu))
                    .unwrap_or(cpu);
                if !seen_cores.contains(&core) {
                    seen_cores.push(core);
                    physical.push(core);
                }
            }
            physical.sort_unstable();
            let distances = read_trim(&dir.join("distance"))
                .map(|s| s.split_whitespace().filter_map(|v| v.parse().ok()).collect())
                .unwrap_or_default();
            let meminfo = dir.join("meminfo");
            nodes.push(Node {
                id,
                cpus,
                physical,
                mem_total_kb: meminfo_kb(&meminfo, "MemTotal"),
                mem_free_kb: meminfo_kb(&meminfo, "MemFree"),
                distances,
            });
        }
        Ok(Topology { nodes })
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Total physical cores across all nodes.
    pub fn n_physical(&self) -> usize {
        self.nodes.iter().map(|n| n.physical.len()).sum()
    }

    pub fn summary(&self) -> String {
        let mut s = String::new();
        for n in &self.nodes {
            s.push_str(&format!(
                "node {:2}: {:2} phys cores {} ({} logical)  mem {:6.1} GiB total {:6.1} GiB free  dist {:?}\n",
                n.id,
                n.physical.len(),
                collapse(&n.physical),
                n.cpus.len(),
                n.mem_total_kb as f64 / (1u64 << 20) as f64,
                n.mem_free_kb as f64 / (1u64 << 20) as f64,
                n.distances
            ));
        }
        s
    }
}

/// Collapse a sorted list into `a-b,c` form.
pub fn collapse(v: &[usize]) -> String {
    let mut parts = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let start = v[i];
        let mut end = start;
        while i + 1 < v.len() && v[i + 1] == end + 1 {
            i += 1;
            end = v[i];
        }
        if start == end {
            parts.push(format!("{start}"));
        } else {
            parts.push(format!("{start}-{end}"));
        }
        i += 1;
    }
    parts.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cpulist_parse_and_collapse() {
        let v = parse_cpulist("0-3,7,10-11");
        assert_eq!(v, vec![0, 1, 2, 3, 7, 10, 11]);
        assert_eq!(collapse(&v), "0-3,7,10-11");
    }
    #[test]
    fn discover_runs() {
        let t = Topology::discover().unwrap();
        assert!(t.n_nodes() >= 1);
        for n in &t.nodes {
            assert!(!n.physical.is_empty());
            assert!(n.physical.len() <= n.cpus.len());
        }
    }
}

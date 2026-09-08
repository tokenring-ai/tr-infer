//! /proc/self/status readings used for load reports.
use std::collections::BTreeMap;

pub fn status_kb() -> BTreeMap<String, u64> {
    let mut m = BTreeMap::new();
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for l in s.lines() {
            for key in ["VmRSS", "VmSwap", "VmLck", "VmHWM"] {
                if let Some(rest) = l.strip_prefix(key) {
                    let v = rest.trim_start_matches(':').split_whitespace().next().and_then(|v| v.parse().ok()).unwrap_or(0);
                    m.insert(key.to_string(), v);
                }
            }
        }
    }
    m
}

pub fn mem_summary() -> String {
    let s = status_kb();
    let g = |k: &str| *s.get(k).unwrap_or(&0) as f64 / (1u64 << 20) as f64;
    let thp = std::fs::read_to_string("/proc/self/smaps_rollup")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("AnonHugePages:").and_then(|r| r.split_whitespace().next()).and_then(|v| v.parse::<u64>().ok())))
        .map(|kb| kb as f64 / (1u64 << 20) as f64)
        .unwrap_or(0.0);
    let (minflt, majflt) = faults();
    format!("VmRSS {:.2} GiB  VmSwap {:.2} GiB  VmLck {:.2} GiB  THP {:.2} GiB  faults {minflt}/{majflt}", g("VmRSS"), g("VmSwap"), g("VmLck"), thp)
}

/// (minor, major) page faults of this process so far (/proc/self/stat fields 10 and 12).
pub fn faults() -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    // the command name (field 2) is parenthesised and may contain spaces: split after the last ')'
    let rest = s.rsplit_once(')').map(|(_, r)| r).unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    // rest[0] is field 3 (state); minflt is field 10 -> index 7, majflt field 12 -> index 9
    let g = |i: usize| f.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
    (g(7), g(9))
}

/// Keep freed heap memory in the process: `t` never trims the heap top back to the kernel, `m`
/// never serves large allocations with fresh mmaps (threshold 32 MiB, glibc's cap), so per-batch
/// allocations stop page-faulting — on a node with no free memory each fault is a direct-reclaim
/// stall and the pressure swaps the process's own pages out (22 K major faults per first batch).
pub fn pin_heap(mode: &str) {
    unsafe {
        if mode.contains('t') {
            libc::mallopt(libc::M_TRIM_THRESHOLD, i32::MAX);
        }
        if mode.contains('m') {
            libc::mallopt(libc::M_MMAP_THRESHOLD, 32 << 20);
        }
        // (M_TOP_PAD of 64 MiB was tried too: decode fell to 9 tok/s — do not.)
    }
}

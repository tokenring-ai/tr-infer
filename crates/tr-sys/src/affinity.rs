//! Thread pinning helpers.
use anyhow::{bail, Result};

/// Pin the calling thread to one logical CPU.
pub fn pin_to_cpu(cpu: usize) -> Result<()> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            bail!("sched_setaffinity({cpu}): {}", std::io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Logical CPU the calling thread is currently on.
pub fn current_cpu() -> usize {
    unsafe { libc::sched_getcpu() as usize }
}

/// Name the calling thread (visible in `top -H`, perf).
pub fn set_thread_name(name: &str) {
    let mut buf = [0u8; 16];
    let b = name.as_bytes();
    let n = b.len().min(15);
    buf[..n].copy_from_slice(&b[..n]);
    unsafe {
        libc::pthread_setname_np(libc::pthread_self(), buf.as_ptr() as *const libc::c_char);
    }
}

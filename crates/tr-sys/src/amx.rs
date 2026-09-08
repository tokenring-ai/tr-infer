//! AMX enablement (arch_prctl) and raw tile instructions via inline asm.
use std::arch::asm;

const ARCH_GET_XCOMP_PERM: libc::c_long = 0x1022;
const ARCH_REQ_XCOMP_PERM: libc::c_long = 0x1023;
const XFEATURE_XTILEDATA: libc::c_ulong = 18;

/// Request AMX tile data permission for this process. Returns true when usable.
pub fn enable_amx() -> bool {
    unsafe {
        let mut mask: libc::c_ulong = 0;
        if libc::syscall(libc::SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &mut mask as *mut _) != 0 {
            return false;
        }
        if mask & (1 << XFEATURE_XTILEDATA) != 0 {
            return true;
        }
        if libc::syscall(libc::SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0 {
            return false;
        }
        libc::syscall(libc::SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &mut mask as *mut _) == 0
            && mask & (1 << XFEATURE_XTILEDATA) != 0
    }
}

/// 64-byte `ldtilecfg` structure.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TileConfig {
    pub palette: u8,
    pub start_row: u8,
    _reserved0: [u8; 14],
    pub colsb: [u16; 16],
    pub rows: [u8; 16],
}
const _: () = assert!(std::mem::size_of::<TileConfig>() == 64);

impl TileConfig {
    pub const fn new() -> Self {
        Self { palette: 1, start_row: 0, _reserved0: [0; 14], colsb: [0; 16], rows: [0; 16] }
    }
    pub const fn set(mut self, tile: usize, rows: u8, colsb: u16) -> Self {
        self.rows[tile] = rows;
        self.colsb[tile] = colsb;
        self
    }
    /// # Safety: AMX must be enabled for the process.
    #[inline(always)]
    pub unsafe fn load(&self) {
        asm!("ldtilecfg [{0}]", in(reg) self as *const _, options(nostack, readonly));
    }
}

#[inline(always)]
pub unsafe fn tile_release() {
    asm!("tilerelease", options(nostack, nomem));
}

/// tilezero tmmT
#[inline(always)]
pub unsafe fn tile_zero<const T: usize>() {
    asm!("tilezero tmm{t}", t = const T, options(nostack, nomem));
}
/// tileloadd tmmT, [base + stride]
#[inline(always)]
pub unsafe fn tile_load<const T: usize>(base: *const u8, stride: usize) {
    asm!("tileloadd tmm{t}, [{0} + {1}*1]", in(reg) base, in(reg) stride, t = const T, options(nostack, readonly));
}
/// tileloadd with the streaming (non-temporal) hint.
#[inline(always)]
pub unsafe fn tile_stream_load<const T: usize>(base: *const u8, stride: usize) {
    asm!("tileloaddt1 tmm{t}, [{0} + {1}*1]", in(reg) base, in(reg) stride, t = const T, options(nostack, readonly));
}
/// tilestored [base + stride], tmmT
#[inline(always)]
pub unsafe fn tile_store<const T: usize>(base: *mut u8, stride: usize) {
    asm!("tilestored [{0} + {1}*1], tmm{t}", in(reg) base, in(reg) stride, t = const T, options(nostack));
}

macro_rules! dp_op {
    ($name:ident, $instr:literal) => {
        /// tmmC += tmmA x tmmB
        #[inline(always)]
        pub unsafe fn $name<const C: usize, const A: usize, const B: usize>() {
            asm!(concat!($instr, " tmm{c}, tmm{a}, tmm{b}"), c = const C, a = const A, b = const B, options(nostack, nomem));
        }
    };
}
dp_op!(tdpbf16ps, "tdpbf16ps");
dp_op!(tdpbsud, "tdpbsud");
dp_op!(tdpbusd, "tdpbusd");
dp_op!(tdpbssd, "tdpbssd");
dp_op!(tdpbuud, "tdpbuud");

#[cfg(test)]
mod tests {
    use super::*;

    /// 16x32 bf16 A times 32x16 bf16 B (VNNI pairs) -> 16x16 f32 C, checked against scalar.
    #[test]
    fn bf16_tile_matmul_matches_scalar() {
        if !enable_amx() {
            eprintln!("AMX not available; skipping");
            return;
        }
        let m = 16usize;
        let k = 32usize;
        let n = 16usize;
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 5) % 11) as f32 - 5.0).collect();
        let to_bf16 = |x: f32| -> u16 { half::bf16::from_f32(x).to_bits() };
        // A tile: 16 rows x 64 bytes (32 bf16), row-major.
        let a_t: Vec<u16> = a.iter().map(|&x| to_bf16(x)).collect();
        // B tile: 16 rows (k pairs) x 64 bytes: row r holds for each n the pair (b[2r][n], b[2r+1][n]).
        let mut b_t = vec![0u16; 16 * 32];
        for r in 0..16 {
            for j in 0..n {
                b_t[r * 32 + 2 * j] = to_bf16(b[(2 * r) * n + j]);
                b_t[r * 32 + 2 * j + 1] = to_bf16(b[(2 * r + 1) * n + j]);
            }
        }
        let mut c = vec![0f32; m * n];
        unsafe {
            let cfg = TileConfig::new().set(0, 16, 64).set(1, 16, 64).set(2, 16, 64);
            cfg.load();
            tile_zero::<2>();
            tile_load::<0>(a_t.as_ptr() as *const u8, 64);
            tile_load::<1>(b_t.as_ptr() as *const u8, 64);
            tdpbf16ps::<2, 0, 1>();
            tile_store::<2>(c.as_mut_ptr() as *mut u8, 64);
            tile_release();
        }
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * n + j];
                }
                assert!((c[i * n + j] - s).abs() < 1e-3, "c[{i}][{j}]={} want {s}", c[i * n + j]);
            }
        }
    }
}

//! AMX feasibility + throughput probe: request XTILEDATA, configure tiles, run tdpbf16ps against
//! a scalar reference, then measure tdpbf16ps / tileloadd throughput and the core clock.
use std::arch::asm;
use std::time::Instant;

#[repr(C, align(64))]
struct TileCfg {
    palette: u8,
    start_row: u8,
    _res: [u8; 14],
    colsb: [u16; 16],
    rows: [u8; 16],
}

unsafe fn request_amx() -> bool {
    let ret: i64;
    // arch_prctl(ARCH_REQ_XCOMP_PERM = 0x1023, XFEATURE_XTILEDATA = 18)
    asm!("syscall", inlateout("rax") 158i64 => ret, in("rdi") 0x1023i64, in("rsi") 18i64, lateout("rcx") _, lateout("r11") _, options(nostack));
    ret == 0
}

unsafe fn tile_config(rows: u8, colsb: u16) {
    let mut cfg = TileCfg { palette: 1, start_row: 0, _res: [0; 14], colsb: [0; 16], rows: [0; 16] };
    for i in 0..8 {
        cfg.colsb[i] = colsb;
        cfg.rows[i] = rows;
    }
    asm!("ldtilecfg [{0}]", in(reg) &cfg as *const TileCfg, options(nostack));
}
unsafe fn tile_release() {
    asm!("tilerelease", options(nostack));
}

fn bf16(x: f32) -> u16 {
    let b = x.to_bits();
    let r = ((b >> 16) & 1) + 0x7FFF;
    ((b + r) >> 16) as u16
}
fn f32_of(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn clock_ghz() -> f64 {
    // dependent add chain: 1 cycle per add
    // imul chain: 3 cycles each
    let n = 200_000_000u64;
    let t0 = Instant::now();
    let mut x: u64 = 3;
    unsafe {
        asm!(
            "2:",
            "imul {x}, {x}",
            "imul {x}, {x}",
            "dec {n}",
            "jnz 2b",
            x = inout(reg) x, n = inout(reg) n => _, options(nostack)
        );
    }
    let dt = t0.elapsed().as_secs_f64();
    std::hint::black_box(x);
    6.0 * n as f64 / dt / 1e9
}

fn main() {
    unsafe {
        if !request_amx() {
            println!("arch_prctl(ARCH_REQ_XCOMP_PERM) failed");
            return;
        }
        println!("AMX permission granted; clock ~{:.2} GHz (dependent-add chain)", clock_ghz());
        tile_config(16, 64);
        // A: 16 x 32 bf16, B: 16 rows (k pairs) x 16 n x 2, C: 16 x 16 f32
        let mut a_raw = vec![0u16; 16 * 32 + 64];
        let mut bt_raw = vec![0u16; 16 * 32 + 64];
        let al = |v: &mut Vec<u16>| -> &'static mut [u16] { let p = v.as_mut_ptr(); let off = (64 - (p as usize % 64)) % 64 / 2; std::slice::from_raw_parts_mut(p.add(off), 16 * 32) };
        let a = al(&mut a_raw);
        let bt = al(&mut bt_raw);
        let mut af = vec![0f32; 16 * 32];
        let mut bf = vec![0f32; 32 * 16]; // [k][n]
        let mut seed = 9u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as u32 % 20000) as f32 / 10000.0 - 1.0
        };
        for m in 0..16 {
            for k in 0..32 {
                let v = f32_of(bf16(rnd()));
                af[m * 32 + k] = v;
                a[m * 32 + k] = bf16(v);
            }
        }
        for k in 0..32 {
            for n in 0..16 {
                let v = f32_of(bf16(rnd()));
                bf[k * 16 + n] = v;
                bt[(k / 2) * 32 + n * 2 + (k % 2)] = bf16(v);
            }
        }
        let mut c = vec![0f32; 16 * 16];
        asm!(
            "tilezero tmm0",
            "tileloadd tmm1, [{a} + {sa}]",
            "tileloadd tmm2, [{b} + {sb}]",
            "tdpbf16ps tmm0, tmm1, tmm2",
            "tilestored [{c} + {sc}], tmm0",
            a = in(reg) a.as_ptr(), sa = in(reg) 64usize,
            b = in(reg) bt.as_ptr(), sb = in(reg) 64usize,
            c = in(reg) c.as_mut_ptr(), sc = in(reg) 64usize,
            options(nostack)
        );
        let mut maxerr = 0f32;
        for m in 0..16 {
            for n in 0..16 {
                let mut s = 0f32;
                for k in 0..32 {
                    s += af[m * 32 + k] * bf[k * 16 + n];
                }
                maxerr = maxerr.max((s - c[m * 16 + n]).abs());
            }
        }
        println!("tdpbf16ps vs reference: max abs err {maxerr:.2e}  c[0]={}", c[0]);
        // throughput: 4 C tiles, 2 A x 2 B, reload A/B from L1 each step
        let iters = 2_000_000u64;
        let t0 = Instant::now();
        asm!(
            "tilezero tmm0", "tilezero tmm1", "tilezero tmm2", "tilezero tmm3",
            "2:",
            "tileloadd tmm4, [{a} + {s}]",
            "tileloadd tmm6, [{b} + {s}]",
            "tdpbf16ps tmm0, tmm4, tmm6",
            "tileloadd tmm7, [{b} + {s}]",
            "tdpbf16ps tmm1, tmm4, tmm7",
            "tileloadd tmm5, [{a} + {s}]",
            "tdpbf16ps tmm2, tmm5, tmm6",
            "tdpbf16ps tmm3, tmm5, tmm7",
            "dec {n}",
            "jnz 2b",
            "tilestored [{c} + {s}], tmm0",
            a = in(reg) a.as_ptr(), b = in(reg) bt.as_ptr(), c = in(reg) c.as_mut_ptr(), s = in(reg) 64usize,
            n = inout(reg) iters => _, options(nostack)
        );
        let dt = t0.elapsed().as_secs_f64();
        let macs = iters as f64 * 4.0 * 16.0 * 16.0 * 32.0;
        println!("2x2 tile loop (L1 operands): {:.1} ns per 4 tdp, {:.0} GMAC/s", dt * 1e9 / iters as f64, macs / dt / 1e9);
        // pure tdp throughput (no loads), and loads only
        let iters = 4_000_000u64;
        let t0 = Instant::now();
        asm!(
            "2:",
            "tdpbf16ps tmm0, tmm4, tmm6",
            "tdpbf16ps tmm1, tmm4, tmm7",
            "tdpbf16ps tmm2, tmm5, tmm6",
            "tdpbf16ps tmm3, tmm5, tmm7",
            "dec {n}",
            "jnz 2b",
            n = inout(reg) iters => _, options(nostack)
        );
        let dt = t0.elapsed().as_secs_f64();
        println!("pure tdpbf16ps: {:.1} ns per tdp, {:.0} GMAC/s", dt * 1e9 / (4 * iters) as f64, (iters * 4 * 8192) as f64 / dt / 1e9);
        let t0 = Instant::now();
        asm!(
            "2:",
            "tileloadd tmm4, [{a} + {s}]",
            "tileloadd tmm6, [{b} + {s}]",
            "tileloadd tmm7, [{b} + {s}]",
            "tileloadd tmm5, [{a} + {s}]",
            "dec {n}",
            "jnz 2b",
            a = in(reg) a.as_ptr(), b = in(reg) bt.as_ptr(), s = in(reg) 64usize,
            n = inout(reg) iters => _, options(nostack)
        );
        let dt = t0.elapsed().as_secs_f64();
        println!("pure tileloadd (L1): {:.1} ns per load", dt * 1e9 / (4 * iters) as f64);
        // same with operands streaming from a 1 MiB buffer (L2)
        let mut big_raw = vec![0u16; 512 * 1024 + 64];
        let big = { let p = big_raw.as_mut_ptr(); let off = (64 - (p as usize % 64)) % 64 / 2; std::slice::from_raw_parts_mut(p.add(off), 512 * 1024) };
        let iters = 1_000_000u64;
        let t0 = Instant::now();
        let mut off = 0usize;
        for _ in 0..iters {
            let p = big.as_ptr().add(off);
            asm!(
                "tileloadd tmm4, [{a} + {s}]",
                "tileloadd tmm6, [{b} + {s}]",
                "tdpbf16ps tmm0, tmm4, tmm6",
                "tileloadd tmm7, [{b2} + {s}]",
                "tdpbf16ps tmm1, tmm4, tmm7",
                "tileloadd tmm5, [{a2} + {s}]",
                "tdpbf16ps tmm2, tmm5, tmm6",
                "tdpbf16ps tmm3, tmm5, tmm7",
                a = in(reg) p, a2 = in(reg) p.add(512), b = in(reg) p.add(1024), b2 = in(reg) p.add(1536), s = in(reg) 64usize,
                options(nostack)
            );
            off = (off + 2048) % (512 * 1024 - 2048);
        }
        let dt = t0.elapsed().as_secs_f64();
        let macs = iters as f64 * 4.0 * 16.0 * 16.0 * 32.0;
        println!("2x2 tile loop (1 MiB stream, 4 KiB per step): {:.1} ns per 4 tdp, {:.0} GMAC/s", dt * 1e9 / iters as f64, macs / dt / 1e9);
        tile_release();
    }
    engine_like(2560, 7);
    engine_like(1280, 2);
    engine_like(320, 8);
    engine_like(2560, 1);
}

#[allow(dead_code)]
pub fn engine_like(k: usize, strips: usize) {
    use tr_format::codec::{pack_ref, Codec};
    use tr_kernels::amx::{gemm_bf16, rows_to_bf16, strip_lines, unpack_strips, Line};
    use tr_kernels::gemv::gemm_tq;
    use tr_kernels::quant::QAct;
    let m = 256usize;
    let rows = strips * 16;
    let c = Codec::new(4, 32);
    let nb = k / 32;
    let q: Vec<u32> = (0..rows * k).map(|i| (i * 7 % 16) as u32).collect();
    let d: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32(0.01).to_bits()).collect();
    let mn: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32(-0.08).to_bits()).collect();
    let packed = pack_ref(rows, k, c, &q, &d, &mn);
    let x: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
    let mut xh = vec![Line([0; 32]); m * k / 32];
    let mut lines = vec![Line([0; 32]); strips * strip_lines(k)];
    let ldy = 2048;
    let mut y = vec![0f32; m * ldy];
    unsafe {
        let xh16 = std::slice::from_raw_parts_mut(xh.as_mut_ptr() as *mut u16, m * k);
        rows_to_bf16(&x, k, 0, m, xh16);
        for _ in 0..2 {
            unpack_strips(packed.as_ptr(), k, c, strips, lines.as_mut_ptr());
            gemm_bf16(lines.as_ptr(), k, strips, xh.as_ptr() as *const u16, m, y.as_mut_ptr(), ldy, 0);
        }
        let reps = 20;
        let t0 = Instant::now();
        for _ in 0..reps {
            unpack_strips(packed.as_ptr(), k, c, strips, lines.as_mut_ptr());
        }
        let tu = t0.elapsed().as_secs_f64() / reps as f64;
        let t0 = Instant::now();
        for _ in 0..reps {
            gemm_bf16(lines.as_ptr(), k, strips, xh.as_ptr() as *const u16, m, y.as_mut_ptr(), ldy, 0);
        }
        let tg = t0.elapsed().as_secs_f64() / reps as f64;
        let macs = (m * rows * k) as f64;
        println!("engine-like K={k} m={m} strips={strips}: unpack {:.1} us, gemm {:.1} us ({:.0} GMAC/s), total {:.0} GMAC/s  y0={}", tu * 1e6, tg * 1e6, macs / tg / 1e9, macs / (tu + tg) / 1e9, y[0]);
        // same with a padded A row stride (k + 32) to break 4 KiB aliasing
        let kp = k + 32;
        let mut xp = vec![Line([0; 32]); m * kp / 32];
        let xp16 = std::slice::from_raw_parts_mut(xp.as_mut_ptr() as *mut u16, m * kp);
        for r in 0..m {
            rows_to_bf16(&x[r * k..(r + 1) * k], k, 0, 1, &mut xp16[r * kp..r * kp + k]);
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            tr_kernels::amx::gemm_bf16_stride(lines.as_ptr(), k, strips, xp.as_ptr() as *const u16, kp, m, y.as_mut_ptr(), ldy, 0);
        }
        let tp = t0.elapsed().as_secs_f64() / reps as f64;
        println!("gemm with A stride k+32: {:.1} us ({:.0} GMAC/s)  y0={}", tp * 1e6, macs / tp / 1e9, y[0]);
        // VNNI reference on the same shape
        let mut xq = QAct::zeros(m, k, 32);
        xq.quantize(&x);
        let t0 = Instant::now();
        for _ in 0..reps {
            gemm_tq(packed.as_ptr(), k, c, 0, strips, xq.as_ref(), y.as_mut_ptr(), ldy, false);
        }
        let tv = t0.elapsed().as_secs_f64() / reps as f64;
        println!("VNNI gemm_tq same shape: {:.1} us ({:.0} GMAC/s)  y0={}", tv * 1e6, macs / tv / 1e9, y[0]);
    }
}

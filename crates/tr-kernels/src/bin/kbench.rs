//! Single-core kernel microbenchmarks: TQ GEMV streaming bandwidth per codec.
use std::time::Instant;
use tr_format::codec::{pack_ref, Codec};
use tr_kernels::gemv::gemv_tq;
use tr_kernels::quant::QAct;

fn main() {
    if std::env::args().nth(1).as_deref() == Some("moe") {
        moe_shape();
        return;
    }
    let rows = 8192usize;
    let k = 2560usize;
    let nmat = 12; // > L3 (105 MiB) in total
    for &(bits, kb, m) in &[(4u8, 32usize, 1usize), (5, 32, 1), (8, 32, 1), (5, 16, 1), (4, 32, 4), (4, 32, 8)] {
        let c = Codec::new(bits, kb);
        let nb = k / kb;
        let mut seed = 1u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let q: Vec<u32> = (0..rows * k).map(|_| rnd() % (1 << bits)).collect();
        let d: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32(0.01).to_bits()).collect();
        let mn: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32(-0.1).to_bits()).collect();
        let one = pack_ref(rows, k, c, &q, &d, &mn);
        let mats: Vec<Vec<u8>> = (0..nmat).map(|_| one.clone()).collect();
        let x: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        let mut xq = QAct::zeros(m, k, kb);
        xq.quantize(&x);
        let mut y = vec![0f32; m * rows];
        let bytes = one.len();
        // warm
        unsafe { gemv_tq(mats[0].as_ptr(), k, c, 0, rows / 16, xq.as_ref(), y.as_mut_ptr(), rows, false) };
        let reps = 3;
        let t0 = Instant::now();
        for _ in 0..reps {
            for mat in &mats {
                unsafe { gemv_tq(mat.as_ptr(), k, c, 0, rows / 16, xq.as_ref(), y.as_mut_ptr(), rows, false) };
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        let gbs = (bytes * nmat * reps) as f64 / dt / 1e9;
        let weights = (rows * k * nmat * reps) as f64;
        println!("TQ{bits}_{kb} m={m}: {:6.2} GB/s  {:6.1} Gweight/s  ({:.1} MiB/matrix)  y0={}", gbs, weights / dt / 1e9, bytes as f64 / 1048576.0, y[0]);
    }
}

/// MoE routed-expert shape: per tile an expert's gate (or up) is 80 rows x K 2560 (5 strips);
/// the batch path calls `gemm_tq` once per (expert, gate|up) with the group's m rows.
pub fn moe_shape() {
    use tr_kernels::gemv::gemm_tq;
    let (k, rows) = (2560usize, 80usize);
    let c = Codec::new(4, 32);
    let nb = k / 32;
    let mut seed = 1u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let q: Vec<u32> = (0..rows * k).map(|_| rnd() % 16).collect();
    let d: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32(0.01).to_bits()).collect();
    let mn: Vec<u16> = (0..rows * nb).map(|_| half::f16::from_f32(-0.1).to_bits()).collect();
    let one = pack_ref(rows, k, c, &q, &d, &mn);
    let nmat = 2048; // 2048 x 115 KB = 235 MB pool
    let pool: Vec<u8> = (0..nmat).flat_map(|_| one.iter().copied()).collect();
    for m in [1usize, 5, 8, 10, 16] {
        let x: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        let mut xq = QAct::zeros(m, k, 32);
        xq.quantize(&x);
        let mut y = vec![0f32; m * rows];
        let reps = 2;
        let t0 = Instant::now();
        for _ in 0..reps {
            for i in 0..nmat {
                unsafe { gemm_tq(pool.as_ptr().add(i * one.len()), k, c, 0, rows / 16, xq.as_ref(), y.as_mut_ptr(), rows, false) };
            }
        }
        let dt = t0.elapsed().as_secs_f64() / (reps * nmat) as f64;
        let gbs = one.len() as f64 / dt / 1e9;
        let gmac = (m * rows * k) as f64 / dt / 1e9;
        println!("moe TQ4_32 80x2560 m={m:2}: VNNI {:6.2} us/expert  {:5.1} GB/s  {:6.1} GMAC/s", dt * 1e6, gbs, gmac);
        if tr_kernels::amx::init() {
            let mut y2 = vec![0f32; m * rows];
            let t0 = Instant::now();
            for _ in 0..reps {
                for i in 0..nmat {
                    unsafe { tr_kernels::amx_i8::gemm_tq_i8(pool.as_ptr().add(i * one.len()), k, c, 0, rows / 16, xq.as_ref(), y2.as_mut_ptr(), rows, None, false) };
                }
            }
            let dt = t0.elapsed().as_secs_f64() / (reps * nmat) as f64;
            assert_eq!(y, y2);
            println!("                          AMX-INT8 {:6.2} us/expert  {:5.1} GB/s  {:6.1} GMAC/s", dt * 1e6, one.len() as f64 / dt / 1e9, (m * rows * k) as f64 / dt / 1e9);
        }
    }
}

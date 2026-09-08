//! Single-core GDN recurrence microbenchmark: state layout [dk][dv] with 16-col chunk tasks
//! (stride dv) versus chunk-major [dv/16][dk][16] (stride 16), 256 tokens, 4 tasks per core.
use std::time::Instant;
use tr_kernels::gdn::{gdn_chunk_seq, gdn_step16, gdn_step_cols};

fn main() {
    let (dk, dv, m, tasks) = (128usize, 128usize, 256usize, 4usize);
    let mut seed = 7u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as u32 % 20000) as f32 / 10000.0 - 1.0
    };
    let q: Vec<f32> = (0..m * dk).map(|_| rnd() * 0.09).collect();
    let k: Vec<f32> = (0..m * dk).map(|_| rnd() * 0.09).collect();
    let v: Vec<f32> = (0..m * dv).map(|_| rnd()).collect();
    let g: Vec<f32> = (0..m).map(|_| -0.05 * rnd().abs()).collect();
    let beta: Vec<f32> = (0..m).map(|_| 0.5 + 0.4 * rnd()).collect();
    let scale = (dk as f32).powf(-0.5);
    let mut y = vec![0f32; m * dv];
    let reps = 5;
    // layout A: [dk][dv], task = 16-col chunk
    let mut st_a = vec![0f32; tasks * dk * dv];
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t0 = Instant::now();
        for task in 0..tasks {
            let ch = task % (dv / 16);
            let state = &mut st_a[task * dk * dv..(task + 1) * dk * dv];
            for mi in 0..m {
                gdn_step_cols(state, dk, dv, &q[mi * dk..(mi + 1) * dk], &k[mi * dk..(mi + 1) * dk], &v[mi * dv..(mi + 1) * dv], g[mi], beta[mi], scale, &mut y[mi * dv..(mi + 1) * dv], ch * 16, ch * 16 + 16);
            }
        }
        best = best.min(t0.elapsed().as_secs_f64());
    }
    println!("layout [dk][dv] stride {dv}: {:8.1} us for {tasks} tasks x {m} tokens = {:6.1} ns/token/task  y={}", best * 1e6, best * 1e9 / (m * tasks) as f64, y[5]);
    // layout B: chunk-major, task = contiguous [dk][16]
    let mut st_b = vec![0f32; tasks * dk * 16];
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t0 = Instant::now();
        for task in 0..tasks {
            let ch = task % (dv / 16);
            let state = &mut st_b[task * dk * 16..(task + 1) * dk * 16];
            for mi in 0..m {
                gdn_step_cols(state, dk, 16, &q[mi * dk..(mi + 1) * dk], &k[mi * dk..(mi + 1) * dk], &v[mi * dv + ch * 16..mi * dv + ch * 16 + 16], g[mi], beta[mi], scale, &mut y[mi * dv + ch * 16..mi * dv + ch * 16 + 16], 0, 16);
            }
        }
        best = best.min(t0.elapsed().as_secs_f64());
    }
    let mut st_c = vec![0f32; tasks * dk * 16];
    let mut y2 = vec![0f32; m * dv];
    let mut best_c = f64::MAX;
    for _ in 0..reps {
        let t0 = Instant::now();
        for task in 0..tasks {
            let ch = task % (dv / 16);
            let state = &mut st_c[task * dk * 16..(task + 1) * dk * 16];
            for mi in 0..m {
                gdn_step16(state, dk, &q[mi * dk..(mi + 1) * dk], &k[mi * dk..(mi + 1) * dk], &v[mi * dv + ch * 16..mi * dv + ch * 16 + 16], g[mi], beta[mi], scale, &mut y2[mi * dv + ch * 16..mi * dv + ch * 16 + 16]);
            }
        }
        best_c = best_c.min(t0.elapsed().as_secs_f64());
    }
    println!("gdn_step16 (8 acc chains):    {:8.1} us for {tasks} tasks x {m} tokens = {:6.1} ns/token/task  y={}", best_c * 1e6, best_c * 1e9 / (m * tasks) as f64, y2[5]);
    let mut st_d = vec![0f32; tasks * dk * 16];
    let mut y3 = vec![0f32; m * dv];
    let mut best_d = f64::MAX;
    for _ in 0..reps {
        let t0 = Instant::now();
        for task in 0..tasks {
            let ch = task % (dv / 16);
            let state = &mut st_d[task * dk * 16..(task + 1) * dk * 16];
            gdn_chunk_seq(state, dk, m, &q, dk, &k, dk, &v[ch * 16..], dv, &g, &beta, scale, &mut y3[ch * 16..], dv);
        }
        best_d = best_d.min(t0.elapsed().as_secs_f64());
    }
    println!("gdn_chunk_seq (fused):        {:8.1} us for {tasks} tasks x {m} tokens = {:6.1} ns/token/task  y={}", best_d * 1e6, best_d * 1e9 / (m * tasks) as f64, y3[5]);
    println!("layout chunk-major stride 16: {:8.1} us for {tasks} tasks x {m} tokens = {:6.1} ns/token/task  y={}", best * 1e6, best * 1e9 / (m * tasks) as f64, y[5]);
}

#[allow(dead_code)]
fn unused() {}

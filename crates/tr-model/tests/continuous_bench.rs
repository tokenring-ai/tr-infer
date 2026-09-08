//! Explicit full-model benchmark; not part of the continuous correctness suite.
use tr_model::{exec::Model, sequence::BatchInput, weights::LoadOptions};
#[test]
#[ignore = "loads the full model and benchmarks all NUMA tiles"]
fn benchmark_continuous_scaling() {
    use std::time::Instant;
    let pack = std::env::var("TR_BENCH_PACK").unwrap_or("/opt/llm/tr-infer/flashnext-v3".into());
    let mut m =
        Model::load_with(std::path::Path::new(&pack), 1024, None, &LoadOptions { ple_mmap: true, batch_max: 256, ..Default::default() })
            .unwrap();
    let prompt: Vec<_> = (1000..1032).collect();
    // Replay the complete trace once so file-backed PLE faults do not bias the first mode.
    m.step_batch(&prompt);
    for t in 2000..2026 {
        m.step(t, None);
    }
    m.reset();
    m.step_batch(&prompt);
    for t in 2000..2002 {
        m.step(t, None);
    }
    let mut times = Vec::new();
    for t in 2002..2026 {
        let start = Instant::now();
        m.step(t, None);
        times.push(start.elapsed().as_secs_f64());
    }
    let mut results = vec![benchmark_record("serial", 1, times)];
    m.enable_sequences(16, 256).unwrap();
    for n in [1, 2, 4, 8, 16] {
        let mut times = Vec::new();
        for pass in 0..2 {
            let mut sequences = Vec::new();
            for i in 0..n {
                let seq = m.allocate_sequence().unwrap();
                let prompt: Vec<_> = (0..32).map(|t| 1000 + t + i as u32 * 73).collect();
                m.forward_batch(&[BatchInput { sequence: seq, tokens: &prompt, logits: false }]).unwrap();
                sequences.push(seq);
            }
            for step in 0..26 {
                let tokens: Vec<_> = (0..n).map(|i| [2000 + step + i as u32 * 137]).collect();
                let inputs: Vec<_> =
                    sequences.iter().zip(&tokens).map(|(&sequence, token)| BatchInput { sequence, tokens: token, logits: true }).collect();
                let start = Instant::now();
                m.forward_batch(&inputs).unwrap();
                if pass == 1 && step >= 2 {
                    times.push(start.elapsed().as_secs_f64());
                }
            }
            for id in sequences {
                m.release_sequence(id).unwrap();
            }
        }
        results.push(benchmark_record("continuous", n, times));
    }
    let result = serde_json::json!({"pack": pack, "note": "complete trace warmed once; teacher-forced distinct tokens; includes full logit transfer, excludes sampling/network/prefill", "results": results});
    let path = std::env::var("TR_BENCH_OUTPUT").unwrap_or("/tmp/tr-continuous-benchmark.json".into());
    std::fs::write(&path, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    eprintln!("{}", serde_json::to_string_pretty(&result).unwrap());
}
fn benchmark_record(mode: &str, n: usize, mut times: Vec<f64>) -> serde_json::Value {
    let seconds: f64 = times.iter().sum();
    let count = times.len();
    times.sort_by(f64::total_cmp);
    serde_json::json!({"mode": mode, "sequences": n, "tokens_per_second": (count * n) as f64 / seconds,
        "iteration_p50_ms": times[count / 2] * 1000.0, "iteration_p95_ms": times[(count * 95 / 100).min(count - 1)] * 1000.0})
}

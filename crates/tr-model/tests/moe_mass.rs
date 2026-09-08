//! Cumulative-mass expert routing on the subset pack at $TR_TEST_PACK (default
//! /opt/llm/tr-infer/test-l03-v4). Needs the box (a pool over all 8 tiles):
//! `cargo test --release -p tr-model --test moe_mass -- --ignored --nocapture`.
use std::path::PathBuf;
use tr_kernels::router::{MassBasis, MoePolicy};
use tr_model::exec::Model;
use tr_model::weights::LoadOptions;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}
fn argmax(a: &[f32]) -> usize {
    a.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }).0
}

#[test]
#[ignore]
fn mass_policy_matches_topk_and_decode_path() {
    let pack = PathBuf::from(std::env::var("TR_TEST_PACK").unwrap_or("/opt/llm/tr-infer/test-l03-v4".into()));
    if !pack.join("manifest.json").exists() {
        eprintln!("no test pack at {}; skipping", pack.display());
        return;
    }
    let mut model = Model::load_with(&pack, 512, None, &LoadOptions { batch_max: 16, ..Default::default() }).expect("load");
    let k = model.cfg.n_expert_used;
    let prompt: Vec<u32> = vec![1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000];
    let tail: Vec<u32> = vec![9000, 10000, 11000];
    // fixed top-k: batch prefill + decode steps
    model.reset();
    let ref_b = model.step_batch(&prompt);
    let mut ref_d = Vec::new();
    for &t in &tail {
        ref_d.push(model.step(t, None));
    }
    // min = max = k with any mass is the same arithmetic: identical logits
    model.set_moe(Some(MoePolicy { min: k, max: k, mass: 0.5, basis: MassBasis::All })).unwrap();
    let s0 = model.moe_stats.snapshot();
    model.reset();
    let same_b = model.step_batch(&prompt);
    assert_eq!(same_b, ref_b, "batch path: min=max=k must equal top-k exactly");
    for (i, &t) in tail.iter().enumerate() {
        assert_eq!(model.step(t, None), ref_d[i], "decode path: min=max=k must equal top-k exactly");
    }
    assert!((model.moe_stats.mean_since(s0) - k as f64).abs() < 1e-9);
    // a real mass target: fewer experts on average; batch and decode paths agree with each other
    // (both route the same way) and stay in the neighbourhood of the fixed top-k output
    let mut min_mean = k as f64;
    for &(basis, mass) in &[(MassBasis::All, 0.05f32), (MassBasis::All, 0.1), (MassBasis::Top, 0.5), (MassBasis::Top, 0.8), (MassBasis::Top, 0.95)] {
        model.set_moe(Some(MoePolicy { min: 1, max: k, mass, basis })).unwrap();
        let s1 = model.moe_stats.snapshot();
        model.reset();
        let mb = model.step_batch(&prompt);
        let mean_b = model.moe_stats.mean_since(s1);
        // decode path over the same prompt (token by token) must give the same last logits, up to the
        // batch-vs-decode activation quantisation drift (verify.rs tolerance)
        model.reset();
        let mut md = Vec::new();
        for &t in &prompt {
            md = model.step(t, None);
        }
        let scale = mb.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d = max_abs_diff(&mb, &md);
        let d_top = max_abs_diff(&mb, &ref_b);
        eprintln!("{basis:?} mass {mass}: {mean_b:.2} experts/token (k = {k}); batch vs decode max|diff| {d:.4}, vs top-k {d_top:.4} (scale {scale:.2}); argmax {} / {} / {}", argmax(&mb), argmax(&md), argmax(&ref_b));
        assert!(mean_b <= k as f64);
        min_mean = min_mean.min(mean_b);
        assert!(d < 0.06 * scale + 0.05, "batch and decode paths disagree under the mass policy");
        assert_eq!(argmax(&mb), argmax(&md));
    }
    assert!(min_mean < k as f64 - 0.5, "the sweep never selected fewer experts: variable counts untested");
    // per-token counts vary inside one batch: the grouping must handle rows of different length
    model.set_moe(Some(MoePolicy { min: 1, max: k, mass: 0.6, basis: MassBasis::Top })).unwrap();
    model.reset();
    let _ = model.step_batch(&prompt);
    model.set_moe(None).unwrap();
    model.reset();
    assert_eq!(model.step_batch(&prompt), ref_b, "top-k restored");
}

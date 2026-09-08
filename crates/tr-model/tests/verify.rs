//! Speculative verify/commit consistency on the subset pack at $TR_TEST_PACK (default
//! /opt/llm/tr-infer/test-l03-v4: layers 0 (GDN + PLE) and 3 (attention), 16 experts).
//! Needs the box (a pool over all 8 tiles): `cargo test --release -p tr-model --test verify -- --ignored --nocapture`.
use std::path::PathBuf;
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
fn verify_commit_matches_sequential_steps() {
    let pack = PathBuf::from(std::env::var("TR_TEST_PACK").unwrap_or("/opt/llm/tr-infer/test-l03-v4".into()));
    if !pack.join("manifest.json").exists() {
        eprintln!("no test pack at {}; skipping", pack.display());
        return;
    }
    let k = 3usize;
    let mut model = Model::load_with(&pack, 512, None, &LoadOptions { batch_max: 16, spec_k: k, ..Default::default() }).expect("load");
    assert_eq!(model.n_ckpt(), k + 1);
    let prompt: Vec<u32> = vec![1000, 2000, 3000, 4000, 5000, 6000, 7000];
    let drafts: Vec<u32> = vec![8000, 9000, 10000, 11000];
    let probe = 12000u32;
    // reference: decode path token by token
    model.reset();
    model.step_batch(&prompt);
    let mut ref_rows = Vec::new();
    for &t in &drafts {
        ref_rows.push(model.step(t, None));
    }
    // batch-path references (the verify path's own numerics, up to the per-row vs chunked GDN kernel):
    // row i = last logits of step_batch(drafts[..=i]); probe j = logits of `probe` after that prefix
    let mut ref_rows_b = Vec::new();
    let mut ref_probe = Vec::new();
    for j in 0..drafts.len() {
        model.reset();
        model.step_batch(&prompt);
        ref_rows_b.push(model.step_batch(&drafts[..=j]));
        ref_probe.push(model.step(probe, None));
    }
    for j in 0..drafts.len() {
        model.reset();
        model.step_batch(&prompt);
        let rows = model.verify(&drafts);
        assert_eq!(rows.len(), drafts.len());
        assert_eq!(model.n_past, prompt.len(), "verify must not advance n_past");
        for (i, row) in rows.iter().enumerate() {
            let scale = ref_rows[i].iter().fold(0f32, |m, &v| m.max(v.abs()));
            let d_dec = max_abs_diff(row, &ref_rows[i]);
            let d_bat = max_abs_diff(row, &ref_rows_b[i]);
            eprintln!("j={j} row {i}: max|diff| vs decode {d_dec:.4}, vs batch {d_bat:.4} (scale {scale:.2}), argmax {} / {} / {}", argmax(row), argmax(&ref_rows[i]), argmax(&ref_rows_b[i]));
            // single-level int8 (batch) vs pair int8 (decode) activations: a few percent drift
            assert!(d_dec < 0.06 * scale + 0.05, "row {i} logits far from the decode path");
            if i > 0 {
                // i = 0 delegates to the decode path in step_batch; rows >= 1 share the batch arithmetic
                // except for the LM head's activation quantisation (pair int8 in the decode head, single
                // level in the batched head): identical with TR_HIPREC=0, ~1 % apart otherwise
                assert!(d_bat < 0.02 * scale + 0.01, "row {i} logits differ from the prefill path");
            }
        }
        model.commit(j);
        assert_eq!(model.n_past, prompt.len() + j + 1);
        let after = model.step(probe, None);
        let d = max_abs_diff(&after, &ref_probe[j]);
        let scale = ref_probe[j].iter().fold(0f32, |m, &v| m.max(v.abs()));
        let wrong: Vec<f32> = (0..drafts.len()).filter(|&o| o != j).map(|o| max_abs_diff(&after, &ref_probe[o])).collect();
        eprintln!("j={j}: probe after commit: max|diff| {d:.4} (scale {scale:.2}); vs other prefixes {wrong:?}");
        let tol = if j == 0 { 0.06 * scale + 0.05 } else { 0.01 * scale + 0.01 };
        assert!(d < tol, "state after commit({j}) differs from the prefix reference");
        assert_eq!(argmax(&after), argmax(&ref_probe[j]));
    }
}

//! Prefix-cache state round trip on the subset pack at $TR_TEST_PACK (default
//! /opt/llm/tr-infer/test-l03-v4: layers 0 (GDN + PLE) and 3 (attention), 16 experts):
//! export rows + snapshot, wreck the state, import, and the next step must be bit-identical.
//! Needs the box: `cargo test --release -p tr-model --test snapshot -- --ignored --nocapture`.
//! With `TR_TEST_MTP=<overlay dir>` the same check runs with the draft head loaded.
use std::path::PathBuf;
use tr_model::exec::Model;
use tr_model::weights::LoadOptions;

fn run(pack: &PathBuf, mtp: Option<PathBuf>) {
    let spec_k = if mtp.is_some() { 2 } else { 0 };
    let mut model = Model::load_with(pack, 512, None, &LoadOptions { batch_max: 16, spec_k, mtp, ..Default::default() }).expect("load");
    // 37 tokens: not a multiple of the QSA block (4), so ik_raw carries a partial block
    let prompt: Vec<u32> = (0..37u32).map(|i| 1000 + i * 37).collect();
    let probe = [5000u32, 5001, 5002];
    model.reset();
    for ch in prompt.chunks(16) {
        model.step_batch(ch);
    }
    assert_eq!(model.n_past, 37);
    // export: rows in two chunks + snapshot (with 3 caller floats)
    let n = model.n_past;
    let cut = 20;
    let mut r0 = vec![0u8; model.rows_bytes(0, cut)];
    let mut r1 = vec![0u8; model.rows_bytes(cut, n)];
    model.export_rows(0, cut, &mut r0);
    model.export_rows(cut, n, &mut r1);
    let extra = [1.5f32, -2.0, 3.25];
    let mut snap = vec![0u8; model.snapshot_bytes(extra.len())];
    model.export_snapshot(&mut snap, &extra);
    eprintln!("rows {} + {} bytes, snapshot {} bytes", r0.len(), r1.len(), snap.len());
    // reference continuation
    let mut reference = Vec::new();
    for &t in &probe {
        reference.push(model.step(t, None));
    }
    // wreck the state with an unrelated prefill, then restore
    model.reset();
    let junk: Vec<u32> = (0..50u32).map(|i| 3000 + i * 11).collect();
    for ch in junk.chunks(16) {
        model.step_batch(ch);
    }
    assert_eq!(model.n_past, 50);
    model.import_rows(0, cut, &r0);
    model.import_rows(cut, n, &r1);
    let host = model.import_snapshot(&snap).expect("import");
    assert_eq!(host.n_past, n);
    assert_eq!(model.n_past, n);
    assert_eq!(host.prev_tokens, prompt[n - 2..].to_vec());
    assert_eq!(host.extra, extra.to_vec());
    assert_eq!(host.rope_delta, 0);
    for (i, &t) in probe.iter().enumerate() {
        let got = model.step(t, None);
        let diff = got.iter().zip(&reference[i]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("probe {i}: max|diff| {diff}");
        assert_eq!(got, reference[i], "probe {i} logits differ after restore");
    }
    // a second export of the same positions must be byte-identical (nothing was disturbed)
    let mut again = vec![0u8; model.rows_bytes(0, cut)];
    model.export_rows(0, cut, &mut again);
    assert_eq!(again, r0);
    // sizes: rows scale with positions, the snapshot with the model only
    assert_eq!(model.rows_bytes(0, 8) * 2, model.rows_bytes(0, 16));
    assert!(model.rows_bytes(3, 4) < model.rows_bytes(0, 4), "block keys only for completed blocks");
    // a corrupt trailer is refused
    let mut bad = snap.clone();
    let off = snap.len() - 4 * (3 + 4 + 2);
    bad[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
    assert!(model.import_snapshot(&bad).is_err());
}

#[test]
#[ignore]
fn export_import_round_trip_is_exact() {
    let pack = PathBuf::from(std::env::var("TR_TEST_PACK").unwrap_or("/opt/llm/tr-infer/test-l03-v4".into()));
    if !pack.join("manifest.json").exists() {
        eprintln!("no test pack at {}; skipping", pack.display());
        return;
    }
    run(&pack, None);
    if let Ok(m) = std::env::var("TR_TEST_MTP") {
        run(&pack, Some(PathBuf::from(m)));
    }
}

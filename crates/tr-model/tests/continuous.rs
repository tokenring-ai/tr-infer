//! Hardware integration: TR_TEST_PACK defaults to the local GDN/PLE + attention subset.
use tr_model::{exec::Model, sequence::BatchInput, weights::LoadOptions};

#[test]
#[ignore = "requires the eight-tile Xeon and subset pack"]
fn packed_sequences_fork_and_recycle() {
    let pack = std::env::var("TR_TEST_PACK").unwrap_or("/opt/llm/tr-infer/test-l03-v4".into());
    let mut m = Model::load_with(std::path::Path::new(&pack), 2304, Some(2), &LoadOptions { batch_max: 64, ..Default::default() }).unwrap();
    m.enable_sequences(4, 128).unwrap();
    let a = m.allocate_sequence().unwrap();
    let b = m.allocate_sequence().unwrap();
    let reference = m.allocate_sequence().unwrap();
    let p: Vec<_> = (0..67).map(|i| 1000 + i).collect();
    for chunk in p.chunks(32) {
        m.forward_batch(&[
            BatchInput { sequence: a, tokens: chunk, logits: false },
            BatchInput { sequence: b, tokens: &[500], logits: false },
        ])
        .unwrap();
        m.forward_batch(&[BatchInput { sequence: reference, tokens: chunk, logits: false }]).unwrap();
    }
    let fork = m.fork_sequence(a).unwrap();
    let free = m.free_page_count();
    let out = m
        .forward_batch(&[
            BatchInput { sequence: b, tokens: &[800, 900], logits: true },
            BatchInput { sequence: a, tokens: &[44, 55], logits: true },
            BatchInput { sequence: fork, tokens: &[66, 77], logits: true },
        ])
        .unwrap();
    let expected = m.forward_batch(&[BatchInput { sequence: reference, tokens: &[44, 55], logits: true }]).unwrap();
    let x = out[1].logits.as_ref().unwrap();
    let y = expected[0].logits.as_ref().unwrap();
    let diff = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let scale = y.iter().map(|x| x.abs()).fold(0f32, f32::max);
    assert!(diff < 0.02 * scale + 0.01, "packed sequence drift {diff}, scale {scale}");
    assert!(m.free_page_count() < free, "fork append must copy its partial shared page");
    assert_eq!(m.sequence_position(a).unwrap(), 69);
    assert_eq!(m.sequence_position(b).unwrap(), 5);
    for id in [a, b, reference, fork] {
        m.release_sequence(id).unwrap();
    }
    assert_eq!(m.free_page_count(), m.page_capacity());
    let fresh = m.allocate_sequence().unwrap();
    assert!(m.sequence_position(a).is_err());
    assert_eq!(m.sequence_position(fresh).unwrap(), 0);
}

#[test]
#[ignore = "requires the eight-tile Xeon and subset pack"]
fn sparse_paged_state_roundtrips_legacy_snapshots_all_kv_types() {
    use tr_model::state::KvType;
    let pack = std::env::var("TR_TEST_PACK").unwrap_or("/opt/llm/tr-infer/test-l03-v4".into());
    for kv in [KvType::F16, KvType::Bf16, KvType::F32] {
        let mut m =
            Model::load_with(std::path::Path::new(&pack), 2304, Some(2), &LoadOptions { kv, batch_max: 64, ..Default::default() }).unwrap();
        let prompt: Vec<_> = (0..2117).map(|i| 1000 + i % 997).collect();
        let mut logits = Vec::new();
        for chunk in prompt.chunks(64) {
            logits = m.step_batch(chunk);
        }
        let mut rows = vec![0; m.rows_bytes(0, prompt.len())];
        let mut snapshot = vec![0; m.snapshot_bytes(logits.len())];
        m.export_rows(0, prompt.len(), &mut rows);
        m.export_snapshot(&mut snapshot, &logits);
        let expected = m.step_batch(&[70, 71]);
        m.enable_sequences(4, 192).unwrap();
        let sequence = m.allocate_sequence().unwrap();
        let other = m.allocate_sequence().unwrap();
        // Allocate interleaved page IDs to exercise fragmented physical storage.
        for end in (64..=2176).step_by(64) {
            m.prepare_sequence(other, end).unwrap();
            m.prepare_sequence(sequence, end.min(prompt.len())).unwrap();
        }
        m.with_sequence(sequence, |m| {
            m.import_rows(0, prompt.len(), &rows);
            m.import_snapshot(&snapshot).unwrap();
            let mut roundtrip = vec![0; rows.len()];
            m.export_rows(0, prompt.len(), &mut roundtrip);
            assert_eq!(roundtrip, rows, "paged row serialization changed for {kv:?}");
            let mut roundtrip = vec![0; snapshot.len()];
            m.export_snapshot(&mut roundtrip, &logits);
            assert_eq!(roundtrip, snapshot, "snapshot serialization changed for {kv:?}");
        })
        .unwrap();
        let fork = m.fork_sequence(sequence).unwrap();
        let got = m
            .forward_batch(&[
                BatchInput { sequence, tokens: &[70, 71], logits: true },
                BatchInput { sequence: other, tokens: &[99, 98], logits: true },
                BatchInput { sequence: fork, tokens: &[80, 81], logits: true },
            ])
            .unwrap();
        let actual = got[0].logits.as_ref().unwrap();
        let scale = expected.iter().map(|x| x.abs()).fold(0f32, f32::max);
        let diff = actual.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(diff < 0.02 * scale + 0.01, "{kv:?}: paged sparse logits drift {diff}, scale {scale}");
        // A failed call must not advance positions or consume pages.
        let before = (m.sequence_position(sequence).unwrap(), m.free_page_count());
        assert!(m
            .forward_batch(&[BatchInput { sequence, tokens: &[1], logits: true }, BatchInput { sequence, tokens: &[2], logits: true }])
            .is_err());
        assert_eq!(before, (m.sequence_position(sequence).unwrap(), m.free_page_count()));
        for id in [sequence, other, fork] {
            m.release_sequence(id).unwrap();
        }
        assert_eq!(m.free_page_count(), m.page_capacity());
    }
}

#[test]
#[ignore = "requires the eight-tile Xeon and subset pack"]
fn forked_teacher_forced_logits_and_exact_page_budget() {
    let pack = std::env::var("TR_TEST_PACK").unwrap_or("/opt/llm/tr-infer/test-l03-v4".into());
    let mut m = Model::load_with(std::path::Path::new(&pack), 128, Some(4), &LoadOptions { batch_max: 16, ..Default::default() }).unwrap();
    // Two physical pages: one shared initial page, then exactly one COW allocation.
    m.enable_sequences(3, 2).unwrap();
    assert_eq!(m.page_capacity(), 2);
    let a = m.allocate_sequence().unwrap();
    m.forward_batch(&[BatchInput { sequence: a, tokens: &[100, 200, 300], logits: false }]).unwrap();
    let b = m.fork_sequence(a).unwrap();
    let mut worst = 0f32;
    for i in 0..32 {
        let token = [1000 + i];
        let result = m
            .forward_batch(&[
                BatchInput { sequence: a, tokens: &token, logits: true },
                BatchInput { sequence: b, tokens: &token, logits: true },
            ])
            .unwrap();
        let x = result[0].logits.as_ref().unwrap();
        let y = result[1].logits.as_ref().unwrap();
        let scale = y.iter().map(|x| x.abs()).fold(0f32, f32::max);
        let diff = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        worst = worst.max(diff);
        assert!(diff < 0.001 * scale + 0.0001, "step {i}: identical fork state drift {diff}, scale {scale}");
    }
    eprintln!("same-input fork maximum absolute logit difference: {worst}");
    assert_eq!(m.free_page_count(), 0);
    m.release_sequence(a).unwrap();
    m.release_sequence(b).unwrap();
    assert_eq!(m.free_page_count(), 2);
}


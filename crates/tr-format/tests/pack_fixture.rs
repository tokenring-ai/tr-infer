//! Reads the subset pack at $TR_TEST_PACK (default /opt/llm/tr-infer/test-l03) and compares the
//! Rust reference dequant against fixtures dumped by tools/oracle/dump_tq_fixture.py.
use std::path::PathBuf;
use tr_format::codec::dequant_ref;
use tr_format::Manifest;

#[test]
fn manifest_loads_and_dequant_matches_python() {
    let pack = PathBuf::from(std::env::var("TR_TEST_PACK").unwrap_or("/opt/llm/tr-infer/test-l03".into()));
    if !pack.join("manifest.json").exists() {
        eprintln!("no test pack at {}; skipping", pack.display());
        return;
    }
    let m = Manifest::load(&pack).expect("manifest");
    assert_eq!(m.n_tiles, 8);
    let fix = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/fixtures/tq");
    let idx_path = fix.join("index.json");
    if !idx_path.exists() {
        eprintln!("no fixtures at {}; run tools/oracle/dump_tq_fixture.py", fix.display());
        return;
    }
    let idx: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(idx_path).unwrap()).unwrap();
    for e in idx {
        let name = e["name"].as_str().unwrap();
        let tile = e["tile"].as_u64().unwrap() as usize;
        let rows = e["rows"].as_u64().unwrap() as usize;
        let k = e["k"].as_u64().unwrap() as usize;
        let t = m.tensor(name, Some(tile)).expect(name);
        let mut buf = vec![0u8; t.stride];
        use std::os::unix::fs::FileExt;
        std::fs::File::open(m.file_path(&t.file)).unwrap().read_exact_at(&mut buf, t.offset).unwrap();
        let got = dequant_ref(&buf, t.rows, t.k, t.codec());
        let want_bytes = std::fs::read(fix.join(e["file"].as_str().unwrap())).unwrap();
        let want: Vec<f32> = want_bytes.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(want.len(), rows * k);
        for r in 0..rows {
            for c in 0..k {
                let (g, w) = (got[r * t.k + c], want[r * k + c]);
                assert!(g == w, "{name} tile {tile} [{r}][{c}]: {g} != {w}");
            }
        }
        eprintln!("{name} tile {tile}: {rows}x{k} exact");
    }
}

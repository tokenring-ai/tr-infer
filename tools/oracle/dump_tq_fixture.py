"""Dump reference f32 dequant of some tensors of a pack (via trpack.codec.unpack_tq) for the Rust test."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from trpack.codec import unpack_tq

pack = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
m = json.loads((pack / "manifest.json").read_text())
picks = [("blk.0.ffn_gate_exps.weight", 0), ("blk.0.ffn_down_exps.weight", 3), ("blk.0.attn_qkv.weight", 1), ("blk.0.hc_attn_down.weight", 7), ("blk.3.attn_q.weight", 2), ("output.weight", 5)]
index = []
for name, tile in picks:
    t = next(x for x in m["tensors"] if x["name"] == name and x["tile"] == tile)
    with open(pack / t["file"], "rb") as f:
        f.seek(t["offset"]); buf = np.frombuffer(f.read(t["stride"]), np.uint8)  # first item (expert 0)
    w = unpack_tq(buf, t["rows"], t["k"], t["bits"], t["kb"])[:32]  # first 32 rows
    fn = out / (name.replace("/", "_") + f".t{tile}.f32")
    w.astype(np.float32).tofile(fn)
    index.append({"name": name, "tile": tile, "rows": int(w.shape[0]), "k": int(w.shape[1]), "file": fn.name})
(out / "index.json").write_text(json.dumps(index))
print("wrote", len(index), "fixtures to", out)

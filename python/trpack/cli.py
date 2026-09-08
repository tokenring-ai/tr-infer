from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="trpack")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="convert a GGUF into a tr-infer pack v2")
    p.add_argument("--gguf", default="/opt/llm/Qwen3.8-Flash-Next-UD-Q4_K_XL")
    p.add_argument("--out", required=True, help="destination directory (real filesystem, not /tmp)")
    p.add_argument("--tiles", type=int, default=8)
    p.add_argument("--layers", help="comma list / ranges of layers to pack (tests), e.g. 0,3 or 0-3")
    p.add_argument("--experts", type=int, help="pack only the first N experts (tests)")
    p.add_argument("--procs", type=int, default=6)
    p.add_argument("--fast-hash", action="store_true", help="hash only the first 16 MiB of each shard")
    p.add_argument("--down-q8-to-q5", action="store_true", help="requantize Q8_0 down_exps layers to 5-bit (lossy, -0.25 GiB/tile)")
    p.add_argument("--skip-ple", action="store_true")
    p.add_argument("--force", action="store_true")
    m = sub.add_parser("mtp", help="pack the MTP draft head (BF16 safetensors) as an overlay of an existing pack")
    m.add_argument("--safetensors", default="/opt/llm/Qwen3.8-Flash-Next-W4A16-AutoRound/model_extra_tensors.safetensors")
    m.add_argument("--base", required=True, help="base pack directory (the overlay references its hash)")
    m.add_argument("--out", required=True, help="destination directory (real filesystem, not /tmp)")
    m.add_argument("--expert-bits", default="4,4,5", help="gate,up,down bits for the MTP experts (default 4,4,5 like the base pack)")
    m.add_argument("--force", action="store_true")
    v = sub.add_parser("vision", help="pack the vision encoder (llama.cpp mmproj GGUF) as an overlay for --vision")
    v.add_argument("--mmproj", default="/opt/llm/Qwen3.8-Flash-Next-UD-Q4_K_XL/mmproj-F16.gguf")
    v.add_argument("--out", required=True, help="destination directory (real filesystem, not /tmp)")
    v.add_argument("--tiles", type=int, default=8)
    v.add_argument("--force", action="store_true")
    q = sub.add_parser("inspect", help="summarise a pack manifest")
    q.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "pack":
        from .writer import pack
        layers = None
        if a.layers:
            layers = []
            for part in a.layers.split(","):
                if "-" in part:
                    x, y = part.split("-")
                    layers.extend(range(int(x), int(y) + 1))
                else:
                    layers.append(int(part))
        pack(Path(a.gguf), Path(a.out), n_tiles=a.tiles, layers=layers, n_experts=a.experts, procs=a.procs,
             fast_hash=a.fast_hash, down_q8_to_q5=a.down_q8_to_q5, skip_ple=a.skip_ple, force=a.force)
    elif a.cmd == "mtp":
        from .mtp import pack_mtp
        eb = tuple(int(x) for x in a.expert_bits.split(","))
        assert len(eb) == 3 and all(b in (4, 5, 8) for b in eb), eb
        pack_mtp(Path(a.safetensors), Path(a.base), Path(a.out), force=a.force, expert_bits=eb)
    elif a.cmd == "vision":
        from .vision import pack_vision
        pack_vision(Path(a.mmproj), Path(a.out), n_tiles=a.tiles, force=a.force)
    elif a.cmd == "inspect":
        m = json.loads((Path(a.dir) / "manifest.json").read_text())
        print(f"format {m['format_version']} hash {m['hash']} tiles {m['n_tiles']} layers {len(m['layer_ids'])} experts {m['n_expert_packed']}")
        for f in m["files"]:
            print(f"  {f['name']:12s} {f['bytes'] / 2**30:7.2f} GiB")
        per_tile = {}
        for t in m["tensors"]:
            per_tile.setdefault(t["tile"], 0)
            per_tile[t["tile"]] += t["nbytes"]
        for k, v in sorted(per_tile.items(), key=lambda kv: (kv[0] is None, kv[0])):
            print(f"  tile {k}: {v / 2**30:.2f} GiB")
        print(f"  tensors: {len(m['tensors'])}")


if __name__ == "__main__":
    main()

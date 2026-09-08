import json, os
from pathlib import Path
import pytest
from trpack.ggufio import GgufModel
from trpack.tokenizer import build_tokenizer_json

GGUF = Path(os.environ.get("TRPACK_GGUF", "/opt/llm/Qwen3.8-Flash-Next-UD-Q4_K_XL"))
HF = Path("/opt/llm/Qwen3.8-Flash-Next-NVFP4/tokenizer.json")


@pytest.mark.skipif(not (GGUF.exists() and HF.exists()), reason="model or HF tokenizer not present")
def test_matches_hf_tokenizer_json():
    m = GgufModel(GGUF)
    toks, types, merges = m.tokenizer_lists()
    ours = build_tokenizer_json(toks, types, merges, m.kv.get("tokenizer.ggml.pre", "qwen35"))
    hf = json.loads(HF.read_text())
    assert ours["model"]["vocab"] == hf["model"]["vocab"]
    assert ours["model"]["merges"] == hf["model"]["merges"]
    assert ours["added_tokens"] == hf["added_tokens"]
    assert ours["pre_tokenizer"] == hf["pre_tokenizer"]
    assert ours["normalizer"] == hf["normalizer"]
    assert ours["decoder"] == hf["decoder"]

"""Emit a HuggingFace tokenizer.json (byte-level BPE) from GGUF vocab/merges/token types."""
from __future__ import annotations

import json

# llama.cpp src/llama-vocab.cpp `qwen35` pre-tokenizer (the HF original form).
QWEN35_REGEX = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
NORMAL, UNKNOWN, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 2, 3, 4, 5, 6
# The GGUF converter marks these as control tokens, but the HF tokenizer.json has special=false for
# them (they may appear in normal text, e.g. tool-call markup and FIM markers). Mirror HF.
NON_SPECIAL_ADDED = {"<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
                     "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
                     "<think>", "</think>"}


def build_tokenizer_json(tokens: list[str], types: list[int], merges: list[str], pre: str = "qwen35") -> dict:
    if pre != "qwen35":
        raise ValueError(f"unsupported pre-tokenizer {pre!r}; add its regex")
    vocab = {}
    added = []
    for i, (t, ty) in enumerate(zip(tokens, types)):
        if ty in (NORMAL, BYTE):
            vocab[t] = i
        elif ty in (CONTROL, USER_DEFINED):
            # HF marks control tokens special; user-defined ones (<tool_call>, <|fim_*|>) are not.
            added.append({"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": t not in NON_SPECIAL_ADDED})
        # UNUSED / UNKNOWN padding ids are left out (they never tokenize; logits for them are ignored)
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN35_REGEX}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        ]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": merges},
    }


def dumps(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)

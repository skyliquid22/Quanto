from __future__ import annotations

import json
from pathlib import Path

from research.utils.validation_harness import hash_json_normalized, hash_jsonl_normalized


def test_hash_json_normalized_stable(tmp_path: Path):
    sample = {"z": 1, "a": {"nested": [3, 2, 1]}}
    path_a = tmp_path / "payload_a.json"
    path_b = tmp_path / "payload_b.json"
    path_a.write_text(json.dumps(sample, indent=4), encoding="utf-8")
    path_b.write_text(json.dumps(sample, separators=(",", ":")), encoding="utf-8")
    assert hash_json_normalized(path_a) == hash_json_normalized(path_b)


def test_hash_jsonl_normalized_stable(tmp_path: Path):
    lines = [
        {"value": 1, "text": " alpha "},
        {"value": 2, "payload": {"b": 2, "a": 1}},
    ]
    path_a = tmp_path / "payload_a.jsonl"
    path_b = tmp_path / "payload_b.jsonl"
    formatted = "\n".join(f"  {json.dumps(entry, separators=(',', ': '))}  " for entry in lines)
    compact = "\n".join(json.dumps(entry, separators=(",", ":")) for entry in lines)
    path_a.write_text(formatted + "\n", encoding="utf-8")
    path_b.write_text("\n" + compact, encoding="utf-8")
    assert hash_jsonl_normalized(path_a) == hash_jsonl_normalized(path_b)

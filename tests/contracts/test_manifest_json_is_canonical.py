from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import guard
    sys.path.insert(0, str(PROJECT_ROOT))

from infra.validation.manifest import persist_manifest


def test_manifest_json_is_canonical(tmp_path) -> None:
    manifest = {
        "run_id": "canonical-test",
        "domain": "equity_ohlcv",
        "schema_version": "1.0",
        "source_vendor": "polygon",
        "input_file_hashes": ["sha256:abc"],
        "total_records": 10,
        "valid_records": 10,
        "invalid_records": 0,
        "validation_status": "passed",
        "creation_timestamp": "2024-01-01T00:00:00+00:00",
        "errors": [],
    }
    path = persist_manifest(dict(manifest), base_dir=tmp_path)
    written_text = path.read_text(encoding="utf-8")
    expected = json.dumps(manifest, indent=2, sort_keys=True)
    assert written_text == expected

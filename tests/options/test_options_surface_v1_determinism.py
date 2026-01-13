from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

from infra.normalization.lineage import compute_file_hash
from infra.ingestion.adapters import OptionsSurfaceIngestionRequest, OptionsSurfaceStorage
from infra.normalization.options_surface import normalize_ivol_surface


def _load_fixture() -> list[dict[str, object]]:
    fixture_path = Path("tests/fixtures/options_surface_sample.json")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_options_surface_v1_determinism(tmp_path):
    raw_records = _load_fixture()
    rows = normalize_ivol_surface(raw_records, allowed_symbols=("AAA", "BBB"), start_date="2024-01-02", end_date="2024-01-03")
    request = OptionsSurfaceIngestionRequest(
        symbols=("AAA", "BBB"),
        start_date=date.fromisoformat("2024-01-02"),
        end_date=date.fromisoformat("2024-01-03"),
        vendor="ivolatility",
        options={},
        vendor_params={},
    )
    base_dir = tmp_path / "derived" / "options_surface_v1"
    storage = OptionsSurfaceStorage(base_dir=base_dir)
    manifest_one = storage.persist(
        rows,
        request=request,
        run_id="surface-test",
        endpoint="mock-endpoint",
        params={"symbols": ["AAA", "BBB"], "from": "2024-01-02", "to": "2024-01-03"},
        created_at="2024-01-15T00:00:00+00:00",
    )
    file_hashes_one = {entry["path"]: compute_file_hash(entry["path"]) for entry in manifest_one["files_written"]}
    manifest_bytes_one = Path(manifest_one["manifest_path"]).read_bytes()

    shutil.rmtree(base_dir, ignore_errors=True)
    storage = OptionsSurfaceStorage(base_dir=base_dir)
    manifest_two = storage.persist(
        rows,
        request=request,
        run_id="surface-test",
        endpoint="mock-endpoint",
        params={"symbols": ["AAA", "BBB"], "from": "2024-01-02", "to": "2024-01-03"},
        created_at="2024-01-15T00:00:00+00:00",
    )
    manifest_bytes_two = Path(manifest_two["manifest_path"]).read_bytes()
    file_hashes_two = {entry["path"]: compute_file_hash(entry["path"]) for entry in manifest_two["files_written"]}

    assert manifest_bytes_one == manifest_bytes_two
    assert file_hashes_one == file_hashes_two

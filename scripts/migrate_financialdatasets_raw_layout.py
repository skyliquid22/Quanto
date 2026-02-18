"""Migrate FinancialDatasets raw layout to policy-driven CSV layouts."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from infra.paths import raw_root
from infra.storage.financialdatasets_policy import DomainPolicy, load_financialdatasets_policies
from infra.storage.raw_writer import RawFinancialDatasetsWriter

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

try:  # pragma: no cover - optional dependency
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate FinancialDatasets raw layout to CSV policies.")
    parser.add_argument("--raw-root", help="Override raw root directory.")
    parser.add_argument("--config", help="Override data_sources.yml path.")
    parser.add_argument("--dry-run", action="store_true", help="Preview migration without writing files.")
    parser.add_argument("--promote", action="store_true", help="Promote staged files and remove legacy layout.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing target files on promote.")
    parser.add_argument(
        "--report-path",
        help="Optional path to write migration report JSON. Defaults to raw/financialdatasets/.migration_report.json",
    )
    return parser.parse_args()


def _staging_root(vendor_root: Path) -> Path:
    return vendor_root / ".migration_staging"


def _report_path(vendor_root: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    return vendor_root / ".migration_report.json"


def _read_old_records(path: Path) -> List[Dict[str, Any]]:
    if pd is None:
        raise RuntimeError("pandas is required to migrate FinancialDatasets raw data")
    if path.suffix == ".parquet":
        if pq is not None:
            table = pq.read_table(path)
            return table.to_pylist()
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        if frame is None or frame.empty:
            return []
        return frame.to_dict(orient="records")
    return []


def _compute_file_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _count_csv_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _format_date(value: str | datetime | None, kind: str) -> str:
    if not value:
        return ""
    if pd is None:
        return str(value)
    if kind == "datetime":
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        return parsed.isoformat() if pd.notna(parsed) else ""
    parsed = pd.to_datetime(value, utc=True, errors="coerce").date()
    return parsed.isoformat() if parsed else ""


def _old_layout_files(vendor_root: Path, domain: str) -> List[Path]:
    domain_root = vendor_root / domain
    if not domain_root.exists():
        return []
    files: List[Path] = []
    for path in domain_root.rglob("*.*"):
        if ".migration_staging" in path.parts or "manifests" in path.parts:
            continue
        if path.suffix not in {".parquet", ".csv"}:
            continue
        rel = path.relative_to(domain_root)
        if domain == "insider_trades":
            if len(rel.parts) >= 4:
                files.append(path)
        else:
            if len(rel.parts) >= 4:
                files.append(path)
    return files


def _parse_old_partition(path: Path, vendor_root: Path, domain: str) -> tuple[str, str] | None:
    try:
        rel = path.relative_to(vendor_root / domain)
    except ValueError:
        rel = Path(*path.parts[path.parts.index(domain) + 1 :]) if domain in path.parts else None
        if rel is None:
            return None
    parts = rel.parts
    if len(parts) < 4:
        return None
    ticker = parts[0]
    year, month = parts[1], parts[2]
    day = parts[3].split(".")[0]
    return ticker, f"{year}-{month}-{day}"


def _prepare_records(
    records: Sequence[Mapping[str, Any]],
    ticker: str,
    date_text: str,
    policy: DomainPolicy,
) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for record in records:
        row = dict(record)
        row.setdefault("ticker", ticker)
        row.setdefault("symbol", ticker)
        if policy.policy == "snapshot_single_csv" and not row.get("as_of_date"):
            row["as_of_date"] = _format_date(date_text, "date")
        if policy.date_priority:
            for key in policy.date_priority:
                if row.get(key):
                    continue
                row[key] = _format_date(date_text, policy.date_kind)
                break
        prepared.append(row)
    return prepared


def _sort_company_facts(records: List[Dict[str, Any]], policy: DomainPolicy) -> List[Dict[str, Any]]:
    key = policy.date_priority[0] if policy.date_priority else "as_of_date"
    return sorted(records, key=lambda row: str(row.get(key) or ""))


def _write_timeseries_yearly(
    writer: RawFinancialDatasetsWriter,
    vendor_root: Path,
    domain: str,
    policy: DomainPolicy,
    frame: "pd.DataFrame",
    dedupe: bool,
) -> List[Dict[str, Any]]:
    frame["__year"] = frame["__date_key"].str.slice(0, 4)
    outputs: List[Dict[str, Any]] = []
    for (ticker, year), group in frame.groupby(["ticker", "__year"]):
        path = vendor_root / domain / ticker / f"{year}.csv"
        outputs.append(writer._append_csv(path, group, policy, domain, dedupe=dedupe))
    return outputs


def _write_timeseries_unsharded(
    writer: RawFinancialDatasetsWriter,
    vendor_root: Path,
    domain: str,
    policy: DomainPolicy,
    frame: "pd.DataFrame",
    dedupe: bool,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for ticker, group in frame.groupby("ticker"):
        path = vendor_root / domain / f"{ticker}.csv"
        outputs.append(writer._append_csv(path, group, policy, domain, dedupe=dedupe))
    return outputs


def migrate_to_staging(
    *,
    raw_root_path: Path,
    policies: Mapping[str, DomainPolicy],
    dry_run: bool,
) -> Dict[str, Any]:
    if pd is None:
        raise RuntimeError("pandas is required for migration")
    vendor_root = raw_root_path / "financialdatasets"
    staging_root = _staging_root(vendor_root)
    report: Dict[str, Any] = {
        "vendor": "financialdatasets",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "phase": "dry_run" if dry_run else "stage",
        "domains": {},
        "file_map": [],
        "rows_skipped": [],
    }
    if not dry_run:
        staging_root.mkdir(parents=True, exist_ok=True)
    writer = RawFinancialDatasetsWriter(
        base_path=staging_root,
        policy_map=policies,
        allow_old_layout=True,
        vendor_prefix=False,
    )

    for domain, policy in policies.items():
        old_files = _old_layout_files(vendor_root, domain)
        if not old_files:
            continue
        domain_stats = {
            "files_read": 0,
            "rows_read": 0,
            "rows_read_raw": 0,
            "files_written": 0,
            "rows_written": 0,
        }
        all_records: List[Dict[str, Any]] = []
        for path in sorted(old_files):
            partition = _parse_old_partition(path, vendor_root, domain)
            if not partition:
                continue
            ticker, date_text = partition
            records = _read_old_records(path)
            domain_stats["files_read"] += 1
            domain_stats["rows_read_raw"] += len(records)
            mapped = _map_old_path_to_new(
                old_path=path,
                vendor_root=vendor_root,
                domain=domain,
                policy=policy,
            )
            staged_path = None
            if mapped is not None:
                try:
                    staged_path = staging_root / mapped.relative_to(vendor_root)
                except ValueError:
                    staged_path = mapped
            report["file_map"].append(
                {
                    "domain": domain,
                    "old_path": str(path),
                    "old_hash": _compute_file_hash(path),
                    "new_path": str(staged_path) if staged_path else None,
                }
            )
            prepared = _prepare_records(records, ticker, date_text, policy)
            if policy.policy == "snapshot_single_csv":
                all_records.extend(prepared)
                continue

            frame = writer._records_to_frame(prepared)
            frame = writer._ensure_ticker(frame)
            frame = writer._normalize_date_columns(frame, policy)
            before_drop = len(frame)
            frame = writer._filter_missing_date(frame, policy, domain)
            after_drop = len(frame)
            dropped = before_drop - after_drop
            if dropped:
                report["rows_skipped"].append(
                    {
                        "domain": domain,
                        "reason": "missing_date",
                        "rows_read": before_drop,
                        "rows_written": after_drop,
                    }
                )
            domain_stats["rows_read"] += after_drop
            if frame.empty or dry_run:
                continue
            if policy.policy == "timeseries_csv_yearly":
                _write_timeseries_yearly(writer, staging_root, domain, policy, frame, dedupe=False)
            else:
                _write_timeseries_unsharded(writer, staging_root, domain, policy, frame, dedupe=False)

        if policy.policy == "snapshot_single_csv" and all_records:
            ordered = _sort_company_facts(all_records, policy)
            unique_tickers = {row.get("ticker") for row in ordered}
            domain_stats["rows_read_raw"] = len(ordered)
            domain_stats["rows_read"] = len(unique_tickers)
            if not dry_run:
                writer._write_snapshot_single(staging_root, domain, policy, ordered)
                if len(ordered) != len(unique_tickers):
                    report["rows_skipped"].append(
                        {
                            "domain": domain,
                            "reason": "company_facts_upsert",
                            "rows_read": len(ordered),
                            "rows_written": len(unique_tickers),
                        }
                    )

        if not dry_run:
            domain_root = staging_root / domain
            csv_files = list(domain_root.rglob("*.csv")) if domain_root.exists() else []
            domain_stats["files_written"] = len(csv_files)
            domain_stats["rows_written"] = sum(_count_csv_records(path) for path in csv_files)

        if dry_run and domain_stats["rows_read"] == 0:
            domain_stats["rows_read"] = domain_stats["rows_read_raw"]
        report["domains"][domain] = domain_stats
    if not dry_run:
        new_hash_map = {}
        for entry in report["file_map"]:
            new_path = entry.get("new_path")
            if not new_path:
                continue
            path = Path(new_path)
            if not path.exists():
                continue
            if new_path not in new_hash_map:
                new_hash_map[new_path] = _compute_file_hash(path)
            entry["new_hash"] = new_hash_map[new_path]
    return report


def _map_old_path_to_new(
    *,
    old_path: Path,
    vendor_root: Path,
    domain: str,
    policy: DomainPolicy,
) -> Path | None:
    try:
        rel = old_path.relative_to(vendor_root / domain)
    except ValueError:
        if domain in old_path.parts:
            rel = Path(*old_path.parts[old_path.parts.index(domain) + 1 :])
        else:
            return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    ticker = parts[0]
    year = parts[1]
    if policy.policy == "snapshot_single_csv":
        return vendor_root / domain / "Facts.csv"
    if policy.policy in {"snapshot_table_csv", "timeseries_csv_unsharded"}:
        return vendor_root / domain / f"{ticker}.csv"
    if policy.policy == "timeseries_csv_yearly":
        return vendor_root / domain / ticker / f"{year}.csv"
    return None


def _update_manifests(vendor_root: Path, policies: Mapping[str, DomainPolicy]) -> List[Dict[str, Any]]:
    updates: List[Dict[str, Any]] = []
    for domain, policy in policies.items():
        manifest_dir = vendor_root / domain / "manifests"
        if not manifest_dir.exists():
            continue
        for manifest_path in manifest_dir.glob("*.json"):
            payload = json.loads(manifest_path.read_text())
            old_files = payload.get("files_written", [])
            new_files: List[Dict[str, Any]] = []
            if old_files:
                new_paths = []
                for entry in old_files:
                    raw_path = Path(entry.get("path", ""))
                    mapped = _map_old_path_to_new(
                        old_path=raw_path,
                        vendor_root=vendor_root,
                        domain=domain,
                        policy=policy,
                    )
                    if mapped is not None:
                        new_paths.append(mapped)
                unique_paths = sorted({str(path) for path in new_paths})
            else:
                symbols = payload.get("symbols", [])
                unique_paths = []
                for symbol in symbols:
                    if policy.policy == "snapshot_single_csv":
                        unique_paths.append(str(vendor_root / domain / "Facts.csv"))
                    elif policy.policy in {"snapshot_table_csv", "timeseries_csv_unsharded"}:
                        unique_paths.append(str(vendor_root / domain / f"{symbol}.csv"))
                    elif policy.policy == "timeseries_csv_yearly":
                        for csv_path in (vendor_root / domain / symbol).glob("*.csv"):
                            unique_paths.append(str(csv_path))
            for path_str in sorted(set(unique_paths)):
                path = Path(path_str)
                if not path.exists():
                    continue
                new_files.append(
                    {
                        "path": path_str,
                        "hash": _compute_file_hash(path),
                        "records": _count_csv_records(path),
                    }
                )
            payload["files_written"] = new_files
            manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            updates.append(
                {
                    "manifest_path": str(manifest_path),
                    "files_written": new_files,
                }
            )
    return updates


def promote_staging(
    *,
    raw_root_path: Path,
    policies: Mapping[str, DomainPolicy],
    force: bool,
    report_path: Path | None = None,
) -> Dict[str, Any]:
    vendor_root = raw_root_path / "financialdatasets"
    staging_root = _staging_root(vendor_root)
    if not staging_root.exists():
        raise RuntimeError("No staging output found. Run migration without --promote first.")
    report_path = report_path or _report_path(vendor_root, None)
    if not report_path.exists():
        raise RuntimeError("No migration report found. Run migration without --promote first.")
    report = json.loads(report_path.read_text())
    if report.get("phase") != "stage":
        raise RuntimeError("Migration report must be generated from the staging phase before promote.")
    expected_domains = report.get("domains", {})
    for domain, stats in expected_domains.items():
        staged_domain = staging_root / domain
        if not staged_domain.exists():
            if stats.get("rows_written", 0) > 0:
                raise RuntimeError(f"Staging output missing for domain '{domain}'.")
            continue
        csv_files = list(staged_domain.rglob("*.csv"))
        staged_rows = sum(_count_csv_records(path) for path in csv_files)
        if staged_rows != stats.get("rows_written", 0):
            raise RuntimeError(
                f"Staging row count mismatch for {domain}: "
                f"expected {stats.get('rows_written', 0)}, found {staged_rows}."
            )
    report_out: Dict[str, Any] = {
        "vendor": "financialdatasets",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "phase": "promote",
        "files_removed": [],
        "manifest_updates": [],
    }

    for domain in policies:
        staged_domain = staging_root / domain
        if not staged_domain.exists():
            continue
        target_domain = vendor_root / domain
        for path in staged_domain.rglob("*.csv"):
            rel = path.relative_to(staged_domain)
            target = target_domain / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not force:
                raise RuntimeError(f"Target file {target} already exists. Use --force to overwrite.")
            shutil.move(str(path), str(target))

    # Clean up empty staging dirs
    for path in sorted(staging_root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

    # Remove legacy files
    for domain in policies:
        legacy_paths = _old_layout_files(vendor_root, domain)
        for path in legacy_paths:
            path.unlink(missing_ok=True)
            report_out["files_removed"].append(str(path))

    report_out["manifest_updates"] = _update_manifests(vendor_root, policies)
    return report_out


def main() -> int:
    args = _parse_args()
    raw_root_path = Path(args.raw_root).expanduser() if args.raw_root else raw_root()
    policies = load_financialdatasets_policies(args.config)
    if not policies:
        raise RuntimeError("No FinancialDatasets policies found in data_sources.yml")

    if args.promote:
        report = promote_staging(
            raw_root_path=raw_root_path,
            policies=policies,
            force=args.force,
            report_path=_report_path(raw_root_path / "financialdatasets", args.report_path),
        )
    else:
        report = migrate_to_staging(raw_root_path=raw_root_path, policies=policies, dry_run=args.dry_run)

    report_path = _report_path(raw_root_path / "financialdatasets", args.report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

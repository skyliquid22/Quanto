# Task ID: T45

## Goal
Refactor **FinancialDatasets raw storage** to use domain‑specific storage policies (CSV / snapshot tables / TS layouts) and **migrate existing raw data in place**. This applies **only** to vendor `financialdatasets`. Old layouts must **not** be read; incompatible layouts should raise clear errors.

## Background
Current raw ingestion stores non‑time‑series data using a time‑series shard path (e.g., `company_facts/AAPL/YYYY/MM/DD.parquet`). This is incorrect and makes inspection + canonicalization fragile. We need explicit storage policies per domain and deterministic migration.

## Scope
**Vendor: financialdatasets only.**
Other vendors remain unchanged.

---

## Storage Policies
Define storage policies and enforce them in raw writers + pipelines.

### Policy Types
1. **timeseries_parquet_sharded** (legacy; for future use)
   - Path: `raw/<vendor>/<domain>/<TICKER>/YYYY/MM/DD.parquet`
   - **Not used in this task**, but keep policy defined for later domains.

2. **timeseries_csv_unsharded**
   - Path: `raw/<vendor>/<domain>/<TICKER>.csv`
   - All dates stored in a single CSV per ticker.
   - New writes **append** rows (dedupe by domain‑specific keys).

3. **timeseries_csv_yearly**
   - Path: `raw/<vendor>/<domain>/<TICKER>/<YYYY>.csv`
   - New writes append (dedupe by domain‑specific keys).

4. **snapshot_table_csv**
   - Path: `raw/<vendor>/<domain>/<TICKER>.csv`
   - Append new snapshots; canonicalization uses **latest row**.

5. **snapshot_single_csv** (special case: company facts)
   - Path: `raw/<vendor>/company_facts/Facts.csv`
   - One file for all tickers; each row is a ticker's latest facts.
   - New writes **overwrite rows for the same ticker** (upsert semantics).
   - **Concurrency: ingestion into `company_facts` must be serialized.** Do not run parallel ingestion jobs for this domain. The writer must acquire a file lock (`fcntl.flock` or equivalent) before read‑modify‑write. If locking is not feasible in the initial implementation, document this as a known limitation and enforce single‑threaded ingestion for this domain via a config guard.

## Domain → Policy Mapping (financialdatasets)
- `company_facts` → **snapshot_single_csv** (`Facts.csv`)
- `financial_metrics` → **snapshot_table_csv** (per‑ticker CSV, append)
- `financial_metrics_snapshot` → **snapshot_table_csv** (per‑ticker CSV, append)
- `financial_statements` → **timeseries_csv_unsharded** (per‑ticker CSV, append)
- `institutional_ownership` → **timeseries_csv_unsharded** (per‑ticker CSV, append)
- `news` → **timeseries_csv_unsharded** (per‑ticker CSV, append)
- `insider_trades` → **timeseries_csv_yearly** (per‑ticker, per‑year CSV)

---

## Requirements

### R1 — Policy registry in `configs/data_sources.yml`
Add a per‑domain raw storage policy mapping for vendor `financialdatasets`, e.g.
```yaml
vendors:
  - name: financialdatasets
    domains: [company_facts, financial_metrics, financial_metrics_snapshot, financial_statements, institutional_ownership, news, insider_trades]
    raw_storage_policy:
      company_facts: snapshot_single_csv
      financial_metrics: snapshot_table_csv
      financial_metrics_snapshot: snapshot_table_csv
      financial_statements: timeseries_csv_unsharded
      institutional_ownership: timeseries_csv_unsharded
      news: timeseries_csv_unsharded
      insider_trades: timeseries_csv_yearly
```

### R2 — Raw writer refactor (financialdatasets only)
Create a **policy‑aware writer** for FinancialDatasets raw domains:
- `infra/storage/raw_writer.py` or a new `infra/storage/financialdatasets_raw_writer.py` that:
  - selects storage layout based on policy
  - writes CSV with deterministic column ordering
  - supports append + dedupe on keys (see below)

**Dedup keys** (minimum):
- `company_facts`: `ticker`
- `financial_metrics` / `financial_metrics_snapshot`: `ticker`, `as_of_date` or `report_period` (whichever exists)
- `financial_statements`: `ticker`, `report_date`, `statement_type`
- `institutional_ownership`: `ticker`, `report_period`, `investor`
- `news`: `ticker`, `published_at` or `date`, `title` (fallback to `url` if present)
- `insider_trades`: `ticker`, `filing_date`, `name`, `transaction_date`, `transaction_value`, `transaction_shares`, `security_title`

### R3 — Pipeline updates
Update ingestion pipelines so that FinancialDatasets domains use the new policy writer:
- For **company_facts**: write to `Facts.csv`, overwrite rows by ticker.
- For **snapshot_table_csv**: append new rows; do not attempt to rebuild per‑day shards.
- For **timeseries_csv_unsharded**: append new rows per ticker.
- For **timeseries_csv_yearly**: append to `TICKER/YYYY.csv`.

### R4 — Canonicalization updates
Update canonical builders to read **only the new policy paths**. If the old sharded parquet layout is detected, **raise a descriptive error** instructing to run migration.

**Important:** the error must only trigger when old‑format files are **actually found on disk**. The absence of new‑format files alone is **not** an error — it simply means no data has been ingested yet. Do not conflate "old layout exists" with "new layout doesn't exist."

- `fundamentals` canonicalization should read from:
  - `financial_statements` CSVs (unsharded)
  - `fundamentals` raw if still used (no change) but for financialdatasets we only use CSV
- `insiders` canonicalization should read from yearly CSVs (`insider_trades/<TICKER>/<YYYY>.csv`)

### R5 — Manifests
Existing manifests must be updated to reflect new file paths after migration.

**Ingestion manifests:**
- Manifest content and schema remain unchanged — only file paths within manifests must point to the new layout paths.
- The migration script (R6) must rewrite manifest paths as part of the migration.
- New ingestion runs must produce manifests referencing the new layout paths.

### R6 — Two‑phase migration script (financialdatasets only)
Create a script:
- `scripts/migrate_financialdatasets_raw_layout.py`

**Phase 1 — Migrate (default behavior):**
- Scans existing `raw/financialdatasets/**` and rewrites into the new layout in a **staging directory** (`raw/financialdatasets/.migration_staging/`).
- Rewrites manifest file paths to reflect new layout.
- Produces a migration report (JSON) listing:
  - domains migrated
  - file counts read / written (per domain)
  - row counts read / written (per domain) — must match exactly
  - manifest files updated
  - any rows skipped or malformed (with reasons)
- Does **not** delete or modify original files.

**Dry run mode (`--dry-run`):**
- Produces the migration report without writing any files.
- Must be run before Phase 1 to preview the migration scope.

**Phase 2 — Promote and cleanup (`--promote`):**
- Validates that staging output exists and row counts match the report.
- Moves staged files from `.migration_staging/` into their final paths.
- Removes old‑layout files only after promotion succeeds for each domain.
- Produces a cleanup report (JSON) listing files removed.

**Force flag (`--force`):**
- Allows `--promote` to overwrite files that already exist at the target path.
- Without `--force`, promotion aborts if any target file already exists.

**Expected workflow:**
```
# 1. Preview
python -m scripts.migrate_financialdatasets_raw_layout --dry-run

# 2. Migrate to staging
python -m scripts.migrate_financialdatasets_raw_layout

# 3. Inspect staging output manually if desired

# 4. Promote and cleanup
python -m scripts.migrate_financialdatasets_raw_layout --promote
```

### R7 — Validation / Safety
- Do not ingest into new layout if old layout files exist and migration not completed: raise error with instructions to run migration.
- The old‑layout detection check must scan for actual old‑format files on disk (e.g., `domain/TICKER/YYYY/MM/DD.parquet` paths). The absence of new‑format files alone must **never** trigger this error.

---

## Tests
Add tests in `tests/storage/` and `tests/ingestion/`:
1. **Policy writer**: verify each domain writes to correct path (CSV layout) with dedupe.
2. **Company facts upsert**: writing a second row for same ticker replaces prior row.
3. **Insider yearly CSV**: writes to `TICKER/YYYY.csv` and appends.
4. **Insider dedup**: two distinct transactions from the same insider on the same filing date with the same value/shares but different `transaction_date` or `security_title` are **not** deduped (verifies expanded dedup keys).
5. **Migration script (dry‑run)**: produces report without writing files.
6. **Migration script (phase 1)**: simulate old sharded parquet/CSV input → verify staging output path + row counts match input exactly.
7. **Migration script (phase 2 / promote)**: verify staging files moved to final paths, old files removed, and manifest paths updated.
8. **Canonical reader rejects old layout**: explicit error raised only when old shard paths are actually detected on disk. Absent new‑format files with no old files present → no error.
9. **Manifest paths**: verify ingestion manifests reference new layout paths after migration and for new ingestion runs.

## Acceptance Criteria
- FinancialDatasets raw storage uses CSV layouts per policy.
- `company_facts/Facts.csv` is written and upserts by ticker.
- Old sharded paths are **not read** by canonicalization.
- Migration script supports `--dry-run`, default (phase 1 → staging), and `--promote` (phase 2 → final + cleanup).
- Row counts in migration report match exactly between source and target per domain.
- Manifests are updated to reflect new file paths (both during migration and for new ingestion).
- `insider_trades` dedup keys include `transaction_date` and `security_title` to prevent false dedup.
- `company_facts` writer enforces serialized access (file lock or documented single‑threaded constraint).
- Old‑layout detection does not false‑positive when new‑layout files are simply absent.
- Tests pass.

## Out of Scope
- Non‑financialdatasets vendors
- Changing canonical schemas beyond required parsing updates
- Automated manifest versioning or manifest schema changes
- Canonical builders for `company_facts`, `financial_metrics`, `institutional_ownership`, and `news` (future follow‑up)

## TRACE
- inputs_used: DATA_SPEC.md, existing raw writer + canonical builder
- decisions: policy‑driven raw layout; CSV for snapshots and time series; vendor‑specific enforcement; two‑phase migration with staging; expanded insider dedup keys; manifest path migration
- checks: unit tests + migration report + row count validation

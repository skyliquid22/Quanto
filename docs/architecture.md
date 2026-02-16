# Architecture Deep Dive

This page describes the end-to-end research pipeline, why each layer is separated, and how configuration files drive deterministic, reproducible outputs.

## Purpose & Guarantees

- **Deterministic outputs**: Same inputs + same config ⇒ identical artifacts.
- **Reproducibility**: All runs are driven by explicit config files and saved metadata.
- **Auditability**: Manifests, evaluation artifacts, and promotion reports capture evidence.
- **OOS evaluation**: Metrics are computed on out-of-sample windows only.

## Pipeline Overview (Layers)

The stack is intentionally layered to keep vendor ingestion, reconciliation, and research logic decoupled:

1. **Raw Ingestion** → vendor adapters pull source data into `raw/`.
2. **Canonical Reconciliation** → multi-vendor merges into `canonical/`.
3. **Derived Datasets** → computed datasets aligned to canonical.
4. **Feature Sets** → versioned observations for training/eval.
5. **Training** → PPO/SAC (spec-driven), checkpoints + metadata.
6. **Evaluation (OOS)** → metrics, timeseries, regime slices.
7. **Qualification / Promotion** → hard/soft gates, evidence reports.
8. **Shadow Replay** → deterministic execution evidence.

### Diagram: Pipeline Layers (PNG)

Add a PNG here showing the eight layers in a left-to-right flow:

![Pipeline layers](assets/architecture_pipeline_layers.png)

## Stage Deep Dives

Each layer follows the same design pattern: inputs → transforms → outputs → config knobs → failure modes.

### 1) Raw Ingestion

**Inputs**
- Vendor APIs or flat files
- Ingest config files under `configs/ingest/`

**Transformations**
- Vendor-specific normalization
- Raw schema validation
- Deterministic manifests

**Outputs**
- `.quanto_data/raw/<vendor>/<domain>/...`
- Manifest JSON under `raw/.../manifests/`

**Config Knobs**
- Ingest configs (symbols, date ranges, vendor params, mode)

**Failure Modes**
- Missing API credentials
- Vendor outages/timeouts
- Schema validation errors

### 2) Canonical Reconciliation

**Inputs**
- Raw shards from one or more vendors
- `configs/data_sources.yml`

**Transformations**
- Vendor priority merge
- Deduplication and validation gates
- Optional anomaly checks

**Outputs**
- `.quanto_data/canonical/<domain>/...`
- Reconciliation manifests + metrics

**Config Knobs**
- Data source priority + per-domain settings

**Failure Modes**
- Missing raw inputs for requested range
- Validation failures (missing required fields)

### 3) Derived Datasets

**Inputs**
- Canonical datasets

**Transformations**
- Derived fields aligned to canonical calendars
- Domain-specific QC

**Outputs**
- `.quanto_data/derived/<domain>/...`

**Config Knobs**
- Domain-specific derivation scripts

**Failure Modes**
- Coverage gaps in canonical
- Derived feature validation errors

### 4) Feature Sets

**Inputs**
- Canonical + derived datasets
- Feature set definitions (versioned)

**Transformations**
- Deterministic joins with coverage flags
- Fill/clip policies

**Outputs**
- In-memory feature panels used in training/eval
- Optional cached feature artifacts

**Config Knobs**
- Feature set names in experiment specs

**Failure Modes**
- Missing columns or mismatched schemas
- Coverage flags below thresholds

### 5) Training

**Inputs**
- Experiment spec (model, features, policy params)
- Feature panel

**Transformations**
- PPO/SAC training loop
- Reward shaping (registry-defined)

**Outputs**
- `.quanto_data/experiments/<eid>/runs/training/`
- Metadata including reward version and data split

**Config Knobs**
- Experiment spec `policy_params`, `reward_version`, `feature_set`

**Failure Modes**
- Missing canonical shards
- Invalid policy params

### 6) Evaluation (OOS)

**Inputs**
- Trained policy or spec
- OOS test window

**Transformations**
- Metrics, timeseries, regime slices

**Outputs**
- `evaluation/metrics.json`
- `evaluation/timeseries.json`
- `evaluation/regime_slices.json`

**Config Knobs**
- `evaluation_split` in experiment spec
- Regime feature set override

**Failure Modes**
- Missing test window data
- Metrics computation errors

### 7) Qualification / Promotion

**Inputs**
- Candidate + baseline evaluation artifacts

**Transformations**
- Gate checks (hard/soft)
- Comparison deltas

**Outputs**
- `promotion/qualification_report.json`
- Promotion records

**Config Knobs**
- Promotion criteria definitions

**Failure Modes**
- Missing baseline artifacts
- Gate criteria misconfiguration

### 8) Shadow Replay

**Inputs**
- Promoted experiment
- Replay window

**Transformations**
- Deterministic execution simulation
- Evidence generation

**Outputs**
- `.quanto_data/shadow/<eid>/<replay_id>/`
- `metrics_sim.json` + logs

**Config Knobs**
- `--execution-mode`
- Replay window flags

**Failure Modes**
- Missing replay window data
- Execution simulation errors

## Config-Driven Architecture

The entire system is driven by explicit configs:

- **Ingest**: `configs/ingest/*.yml`
- **Canonical**: `configs/data_sources.yml`
- **Experiments**: `configs/experiments/*.yml`
- **Sweeps**: `configs/sweeps/*.yml`

Each run writes the config and resolved metadata into artifacts so results can be reproduced exactly.

### Diagram: Config Drives Everything (PNG)

Add a PNG showing config files flowing into each pipeline stage:

![Config-driven architecture](assets/architecture_config_driven.png)

## Diagnostics & Evidence

- **Manifests** capture input coverage, record counts, and validation results.
- **Evaluation artifacts** store metrics and timeseries for audit.
- **Promotion reports** encode gate decisions and deltas.
- **Shadow replays** provide deterministic execution evidence.

## Artifact Map (Quick Reference)

Add a PNG table that maps stages to output paths:

![Artifact map](assets/architecture_artifact_map.png)

Key paths:
- `.quanto_data/raw/<vendor>/<domain>/...`
- `.quanto_data/canonical/<domain>/...`
- `.quanto_data/derived/<domain>/...`
- `.quanto_data/experiments/<eid>/evaluation/metrics.json`
- `.quanto_data/experiments/<eid>/promotion/qualification_report.json`
- `.quanto_data/shadow/<eid>/<replay_id>/metrics_sim.json`

## Extensibility Points

- **New vendors**: add adapters under `infra/ingestion/adapters/`.
- **New feature sets**: register in `research/features/feature_registry.py`.
- **New rewards/policies**: reward registry + trainer wiring.
- **New diagnostics**: add monitoring/report scripts under `scripts/` and `research/monitoring/`.

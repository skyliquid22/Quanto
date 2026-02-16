from __future__ import annotations

from cli.commands.specs import CommandSpec, ParamSpec

spec = CommandSpec(
    name="ingest",
    description="Ingest raw vendor data into .quanto_data/raw using a config file.",
    module="scripts.ingest",
    usage="ingest --config <path> --domain <domain> [--mode auto|rest|flat_file] [--run-id <id>] [--data-root <path>] [--force] [--dry-run]",
    params=(
        ParamSpec("--config", "path", "Ingestion config file (YAML or JSON).", required=True),
        ParamSpec("--domain", "str", "Domain to ingest (e.g., equity_ohlcv).", required=True),
        ParamSpec(
            "--mode",
            "str",
            "Force ingestion mode (auto, rest, flat_file).",
            default="auto",
        ),
        ParamSpec("--run-id", "str", "Optional deterministic run id."),
        ParamSpec("--data-root", "path", "Override QUANTO data root."),
        ParamSpec("--force", "flag", "Overwrite existing manifest for run-id."),
        ParamSpec("--dry-run", "flag", "Resolve routing and print summary without writing data."),
    ),
    returns="JSON summary to stdout; raw files + manifest written under .quanto_data/raw/<vendor>/<domain>/.",
    example="ingest --config configs/ingest/polygon_equity_backfill.yml --domain equity_ohlcv --mode rest",
)

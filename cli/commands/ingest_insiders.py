from __future__ import annotations

from cli.commands.specs import CommandSpec, ParamSpec

spec = CommandSpec(
    name="ingest-insiders",
    description="Ingest insider trades into .quanto_data/raw using a config file.",
    module="scripts.ingest_insiders",
    usage="ingest-insiders --config <path> [--run-id <id>] [--data-root <path>] [--force] [--dry-run]",
    params=(
        ParamSpec("--config", "path", "Ingestion config file (YAML or JSON).", required=True),
        ParamSpec("--run-id", "str", "Optional deterministic run id."),
        ParamSpec("--data-root", "path", "Override QUANTO data root."),
        ParamSpec("--force", "flag", "Overwrite existing manifest for run-id."),
        ParamSpec("--dry-run", "flag", "Resolve config and print summary without writing data."),
    ),
    returns="JSON summary to stdout; raw files + manifest written under .quanto_data/raw/financialdatasets/insider_trades/.",
    example="ingest-insiders --config configs/ingest/financialdatasets_insider_smoke.yml --run-id insider_trades-smoke",
)

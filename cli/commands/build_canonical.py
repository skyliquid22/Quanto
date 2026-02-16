from __future__ import annotations

from cli.commands.specs import CommandSpec, ParamSpec

spec = CommandSpec(
    name="build-canonical",
    description="Build canonical datasets from raw shards for selected domains.",
    module="scripts.build_canonical_datasets",
    usage="build-canonical --start-date YYYY-MM-DD --end-date YYYY-MM-DD [--config <path>] [--domains <domain ...>] [--run-id <id>] [--raw-root <path>] [--canonical-root <path>] [--manifest-root <path>] [--metrics-root <path>]",
    params=(
        ParamSpec("--config", "path", "Data sources config file (YAML or JSON).", default="configs/data_sources.yml"),
        ParamSpec("--domains", "list[str]", "Optional domain list (default: all configured)."),
        ParamSpec("--start-date", "date", "Inclusive start date (YYYY-MM-DD).", required=True),
        ParamSpec("--end-date", "date", "Inclusive end date (YYYY-MM-DD).", required=True),
        ParamSpec("--run-id", "str", "Optional deterministic run id."),
        ParamSpec("--raw-root", "path", "Override raw data root."),
        ParamSpec("--canonical-root", "path", "Override canonical output root."),
        ParamSpec("--manifest-root", "path", "Override validation manifest root."),
        ParamSpec("--metrics-root", "path", "Override reconciliation metrics root."),
    ),
    returns="JSON manifest summary to stdout; canonical shards written under .quanto_data/canonical/<domain>/.",
    example="build-canonical --start-date 2022-01-01 --end-date 2025-12-31 --domains equity_ohlcv",
)

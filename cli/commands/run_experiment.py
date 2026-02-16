from __future__ import annotations

from cli.commands.specs import CommandSpec, ParamSpec

spec = CommandSpec(
    name="run-experiment",
    description="Run an experiment spec and produce evaluation artifacts (OOS).",
    module="scripts.run_experiment",
    usage="run-experiment --spec <path> [--force]",
    params=(
        ParamSpec("--spec", "path", "Experiment spec file (YAML or JSON).", required=True),
        ParamSpec("--force", "flag", "Re-run even if experiment ID exists (registry reset)."),
    ),
    returns="JSON payload to stdout with experiment_id and artifact paths.",
    example="run-experiment --spec configs/experiments/core_v1_regime_slices_ppo.json",
)

from __future__ import annotations

from cli.commands.specs import CommandSpec, ParamSpec

spec = CommandSpec(
    name="monitor",
    description="Generate a metrics/plot report for an experiment.",
    module="scripts.monitor_experiment",
    usage="monitor --experiment-id <id> [--output-dir <path>] [--strict]",
    params=(
        ParamSpec("--experiment-id", "str", "Experiment ID to summarize.", required=True),
        ParamSpec("--output-dir", "path", "Optional plot output directory."),
        ParamSpec("--strict", "flag", "Fail if optional artifacts are missing."),
    ),
    returns="ASCII report to stdout; plots saved when --output-dir is provided.",
    example="monitor --experiment-id <EXPERIMENT_ID> --output-dir monitoring/cli",
)

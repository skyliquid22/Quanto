from __future__ import annotations

from cli.commands.specs import CommandSpec, ParamSpec

spec = CommandSpec(
    name="run-shadow",
    description="Replay a promoted experiment through shadow execution.",
    module="scripts.run_shadow",
    usage="run-shadow --experiment-id <id> --replay --start-date YYYY-MM-DD --end-date YYYY-MM-DD [--max-steps N] [--reset|--resume] [--execution-mode none|sim|alpaca_paper] [--output-dir <path>]",
    params=(
        ParamSpec("--experiment-id", "str", "Experiment identifier to execute.", required=True),
        ParamSpec("--replay", "flag", "Enable historical replay mode."),
        ParamSpec("--live", "flag", "Enable live mode (unsupported in v1)."),
        ParamSpec("--start-date", "date", "Inclusive start date for replay."),
        ParamSpec("--end-date", "date", "Inclusive end date for replay."),
        ParamSpec("--max-steps", "int", "Optional cap on steps processed."),
        ParamSpec("--reset", "flag", "Delete run directory before replay."),
        ParamSpec("--resume", "flag", "Resume a previous run directory."),
        ParamSpec("--registry-root", "path", "Override experiment registry root."),
        ParamSpec("--promotion-root", "path", "Override promotion record root."),
        ParamSpec("--output-dir", "path", "Override shadow output directory."),
        ParamSpec("--qualification-replay", "flag", "Allow replay for qualification evidence."),
        ParamSpec("--qualification-reason", "str", "Reason recorded with qualification replay."),
        ParamSpec(
            "--execution-mode",
            "str",
            "Execution mode (none, sim, alpaca_paper).",
        ),
    ),
    returns="Status line to stdout; replay artifacts under .quanto_data/shadow/<experiment_id>/<replay_id>/.",
    example="run-shadow --experiment-id <EXPERIMENT_ID> --replay --start-date 2024-01-01 --end-date 2024-12-31",
)

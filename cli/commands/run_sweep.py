from __future__ import annotations

from cli.commands.specs import CommandSpec, ParamSpec

spec = CommandSpec(
    name="run-sweep",
    description="Expand and run a sweep spec across multiple experiments.",
    module="scripts.run_sweep",
    usage="run-sweep --sweep <path> [--force]",
    params=(
        ParamSpec("--sweep", "path", "Sweep spec file (YAML or JSON).", required=True),
        ParamSpec("--force", "flag", "Re-run experiments even if registry already has results."),
    ),
    returns="Console summary plus artifacts under .quanto_data/experiments/sweeps/<sweep_name>/.",
    example="run-sweep --sweep configs/sweeps/core_v1_primary_regime_baselines.yml",
)

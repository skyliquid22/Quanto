from __future__ import annotations

from cli.commands.specs import CommandSpec

spec = CommandSpec(
    name="run-experiment",
    description="Run an experiment spec and produce evaluation artifacts (OOS).",
    module="scripts.run_experiment",
)

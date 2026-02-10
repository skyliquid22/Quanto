from __future__ import annotations

from cli.commands.specs import CommandSpec

spec = CommandSpec(
    name="monitor",
    description="Generate a metrics/plot report for an experiment.",
    module="scripts.monitor_experiment",
)

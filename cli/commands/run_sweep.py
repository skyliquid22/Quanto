from __future__ import annotations

from cli.commands.specs import CommandSpec

spec = CommandSpec(
    name="run-sweep",
    description="Expand and run a sweep spec across multiple experiments.",
    module="scripts.run_sweep",
)

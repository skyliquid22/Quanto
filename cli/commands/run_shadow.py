from __future__ import annotations

from cli.commands.specs import CommandSpec

spec = CommandSpec(
    name="run-shadow",
    description="Replay a promoted experiment through shadow execution.",
    module="scripts.run_shadow",
)

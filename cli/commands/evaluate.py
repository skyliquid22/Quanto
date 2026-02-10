from __future__ import annotations

from cli.commands.specs import CommandSpec

spec = CommandSpec(
    name="evaluate",
    description="Evaluate a policy/checkpoint over a specified window.",
    module="scripts.evaluate_agent",
)

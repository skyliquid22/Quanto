from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from cli.commands.specs import CommandContext, CommandResult, CommandSpec


def _resolve_data_root(repo_root: Path) -> Path:
    override = os.environ.get("QUANTO_DATA_ROOT")
    if override:
        return Path(override).expanduser()
    return repo_root / ".quanto_data"


def run(args: Sequence[str], context: CommandContext) -> CommandResult:
    failures: list[str] = []
    warnings: list[str] = []

    if not context.python_path:
        failures.append("PYTHON env var is not set.")
    else:
        python_path = Path(context.python_path)
        if not python_path.exists():
            failures.append(f"PYTHON path does not exist: {context.python_path}")
        elif not os.access(context.python_path, os.X_OK):
            failures.append(f"PYTHON is not executable: {context.python_path}")

    research_dir = context.repo_root / "research"
    if not research_dir.exists():
        failures.append("Repository root not detected (missing research/).")

    data_root = _resolve_data_root(context.repo_root)
    if not data_root.exists():
        warnings.append(f"Data root does not exist: {data_root}")

    if failures:
        lines = ["Doctor checks failed:"] + [f"- {entry}" for entry in failures]
        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- {entry}" for entry in warnings)
        message = "\n".join(lines)
        return CommandResult(exit_code=2, stdout="", stderr=message)

    lines = ["Doctor checks passed."]
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {entry}" for entry in warnings)
    message = "\n".join(lines)
    return CommandResult(exit_code=0, stdout=message, stderr="")


spec = CommandSpec(
    name="doctor",
    description="Check PYTHON env, repo root, and data root availability.",
    handler=run,
)

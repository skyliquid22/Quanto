from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class CommandContext:
    repo_root: Path
    python_path: str


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


CommandHandler = Callable[[Sequence[str], CommandContext], CommandResult]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    module: Optional[str] = None
    handler: Optional[CommandHandler] = None

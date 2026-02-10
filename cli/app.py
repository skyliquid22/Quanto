from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Sequence

import typer

try:
    import readline  # noqa: F401
except Exception:  # pragma: no cover - Windows fallback
    readline = None  # type: ignore[assignment]

from cli.commands.registry import COMMAND_SPECS
from cli.commands.specs import CommandContext, CommandResult, CommandSpec

app = typer.Typer(add_completion=False)

_COMMAND_MAP: Dict[str, CommandSpec] = {spec.name: spec for spec in COMMAND_SPECS}


def _repo_root() -> Path:
    current = Path.cwd().resolve()
    while True:
        if (current / "research").exists():
            return current
        if current.parent == current:
            return Path.cwd().resolve()
        current = current.parent


def _python_path() -> str:
    return os.environ.get("PYTHON", "").strip()


def _help_text() -> str:
    lines = ["Commands:"]
    for spec in COMMAND_SPECS:
        lines.append(f"  {spec.name:<14} {spec.description}")
    return "\n".join(lines)


def _print_help() -> None:
    typer.echo(_help_text())


def _ensure_log_dir(repo_root: Path) -> Path:
    path = repo_root / "monitoring" / "cli"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log_path(repo_root: Path, command_name: str) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return _ensure_log_dir(repo_root) / f"{stamp}_{command_name}.log"


def _write_log(path: Path, command_line: str, result: CommandResult) -> None:
    payload = [
        f"command: {command_line}",
        f"exit_code: {result.exit_code}",
        "--- stdout ---",
        result.stdout.rstrip(),
        "--- stderr ---",
        result.stderr.rstrip(),
        "",
    ]
    path.write_text("\n".join(payload), encoding="utf-8")


def _run_python_module(spec: CommandSpec, args: Sequence[str], context: CommandContext) -> CommandResult:
    if not context.python_path:
        message = "PYTHON env var is required. Example: export PYTHON=/usr/bin/python3"
        return CommandResult(exit_code=2, stdout="", stderr=message)
    if not spec.module:
        return CommandResult(exit_code=2, stdout="", stderr="Command is missing a module binding.")
    cmd = [context.python_path, "-m", spec.module, *args]
    result = subprocess.run(
        cmd,
        cwd=str(context.repo_root),
        env={**os.environ, "PYTHONPATH": str(context.repo_root)},
        capture_output=True,
        text=True,
    )
    return CommandResult(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)


def _run_command(spec: CommandSpec, args: Sequence[str], context: CommandContext) -> CommandResult:
    if spec.handler is not None:
        return spec.handler(args, context)
    return _run_python_module(spec, args, context)


def _command_help(spec: CommandSpec) -> str:
    if spec.module:
        return f"{spec.name}: {spec.description}\nForwards to: python -m {spec.module}"
    return f"{spec.name}: {spec.description}"


def _handle_line(raw: str, context: CommandContext) -> None:
    trimmed = raw.strip()
    if not trimmed:
        return
    if trimmed in {"exit", "quit"}:
        raise typer.Exit()
    if trimmed in {"help", "-h", "--help"}:
        _print_help()
        return
    tokens = shlex.split(trimmed)
    if not tokens:
        return
    name = tokens[0]
    args = tokens[1:]
    spec = _COMMAND_MAP.get(name)
    if spec is None:
        typer.echo(f"Unknown command: {name}")
        _print_help()
        return
    if any(flag in {"-h", "--help"} for flag in args):
        typer.echo(_command_help(spec))
        return
    result = _run_command(spec, args, context)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if spec.module:
        command_line = f"{context.python_path or 'PYTHON'} -m {spec.module} {' '.join(args)}".strip()
    else:
        command_line = " ".join([name, *args]).strip()
    log_path = _log_path(context.repo_root, name)
    _write_log(log_path, command_line, result)
    if result.exit_code != 0:
        typer.echo(f"Command exited with code {result.exit_code}", err=True)


def _configure_readline() -> None:
    if readline is None:
        return
    commands = sorted(_COMMAND_MAP.keys())

    def completer(text: str, state: int) -> str | None:
        matches = [cmd for cmd in commands if cmd.startswith(text)]
        if state < len(matches):
            return matches[state]
        return None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")


def _prompt(context: CommandContext) -> None:
    _configure_readline()
    while True:
        try:
            line = input("quanto> ")
        except EOFError:
            break
        _handle_line(line, context)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        typer.echo("This CLI uses interactive mode. Run `quanto` and type commands.", err=True)
        raise typer.Exit(code=2)
    context = CommandContext(repo_root=_repo_root(), python_path=_python_path())
    _prompt(context)


def _register_blocked_command(spec: CommandSpec) -> None:
    context_settings = {"allow_extra_args": True, "ignore_unknown_options": True}

    @app.command(name=spec.name, help=spec.description, context_settings=context_settings)
    def _blocked(_: typer.Context) -> None:  # type: ignore[valid-type]
        typer.echo("This CLI uses interactive mode. Run `quanto` and type commands.", err=True)
        raise typer.Exit(code=2)


for _spec in COMMAND_SPECS:
    _register_blocked_command(_spec)


if __name__ == "__main__":
    app()

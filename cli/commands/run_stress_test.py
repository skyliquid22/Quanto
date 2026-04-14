from __future__ import annotations

from cli.commands.specs import CommandSpec, ParamSpec

spec = CommandSpec(
    name="run-stress-test",
    description="Run a stress test suite against an experiment spec and evaluate pass/fail gates.",
    module="scripts.run_stress_test",
    usage="run-stress-test --stress-config <path> [--out-dir <path>]",
    params=(
        ParamSpec("--stress-config", "path", "Stress test config file (YAML).", required=True),
        ParamSpec("--out-dir", "path", "Root directory for run artifacts (default: runs/stress/)."),
    ),
    returns="JSON payload to stdout with overall_status and per-scenario gate results. Exits 1 on fail.",
    example="run-stress-test --stress-config configs/stress/example_stress_test.yml",
)

from __future__ import annotations

from cli.commands.build_canonical import spec as build_canonical_spec
from cli.commands.doctor import spec as doctor_spec
from cli.commands.evaluate import spec as evaluate_spec
from cli.commands.ingest import spec as ingest_spec
from cli.commands.monitor import spec as monitor_spec
from cli.commands.run_experiment import spec as run_experiment_spec
from cli.commands.run_shadow import spec as run_shadow_spec
from cli.commands.run_sweep import spec as run_sweep_spec
from cli.commands.specs import CommandSpec

COMMAND_SPECS: tuple[CommandSpec, ...] = (
    ingest_spec,
    build_canonical_spec,
    run_experiment_spec,
    run_sweep_spec,
    evaluate_spec,
    run_shadow_spec,
    monitor_spec,
    doctor_spec,
)

from __future__ import annotations

from cli.commands.baseline_allowlist_cmd import spec as baseline_allowlist_spec
from cli.commands.build_canonical import spec as build_canonical_spec
from cli.commands.build_regime_thresholds import spec as build_regime_thresholds_spec
from cli.commands.compare import spec as compare_spec
from cli.commands.doctor import spec as doctor_spec
from cli.commands.evaluate import spec as evaluate_spec
from cli.commands.ingest import spec as ingest_spec
from cli.commands.ingest_equity import spec as ingest_equity_spec
from cli.commands.ingest_fundamentals import spec as ingest_fundamentals_spec
from cli.commands.ingest_insiders import spec as ingest_insiders_spec
from cli.commands.ingest_options import spec as ingest_options_spec
from cli.commands.monitor import spec as monitor_spec
from cli.commands.promote import spec as promote_spec
from cli.commands.qualify import spec as qualify_spec
from cli.commands.report_data_health import spec as report_data_health_spec
from cli.commands.run_experiment import spec as run_experiment_spec
from cli.commands.run_paper import spec as run_paper_spec
from cli.commands.run_paper_scheduled import spec as run_paper_scheduled_spec
from cli.commands.run_shadow import spec as run_shadow_spec
from cli.commands.run_stress_test import spec as run_stress_test_spec
from cli.commands.run_sweep import spec as run_sweep_spec
from cli.commands.show_traces_cmd import spec as show_traces_spec
from cli.commands.specs import CommandSpec
from cli.commands.train_ppo import spec as train_ppo_spec
from cli.commands.train_sac import spec as train_sac_spec

COMMAND_SPECS: tuple[CommandSpec, ...] = (
    # ingestion
    ingest_spec,
    ingest_insiders_spec,
    ingest_equity_spec,
    ingest_fundamentals_spec,
    ingest_options_spec,
    # data / features
    build_canonical_spec,
    build_regime_thresholds_spec,
    report_data_health_spec,
    # experiments
    run_experiment_spec,
    run_sweep_spec,
    evaluate_spec,
    compare_spec,
    promote_spec,
    qualify_spec,
    baseline_allowlist_spec,
    # training
    train_ppo_spec,
    train_sac_spec,
    # simulation / paper
    run_shadow_spec,
    run_stress_test_spec,
    run_paper_spec,
    run_paper_scheduled_spec,
    # observability
    show_traces_spec,
    monitor_spec,
    doctor_spec,
)

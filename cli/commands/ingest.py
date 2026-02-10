from __future__ import annotations

from cli.commands.specs import CommandSpec

spec = CommandSpec(
    name="ingest",
    description="Ingest raw vendor data into .quanto_data/raw using a config file.",
    module="scripts.ingest",
)

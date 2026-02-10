from __future__ import annotations

from cli.commands.specs import CommandSpec

spec = CommandSpec(
    name="build-canonical",
    description="Build canonical datasets from raw shards for selected domains.",
    module="scripts.build_canonical_datasets",
)

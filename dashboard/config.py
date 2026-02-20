"""Dashboard configuration and data-root resolution."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_TTL_SECONDS = 300

COLOR_GOOD = "#4CAF50"
COLOR_BAD = "#EF5350"
COLOR_WARN = "#FFA726"
COLOR_MUTED = "#888888"
COLOR_BG = "#0F1117"
COLOR_BG_SECONDARY = "#161B22"
COLOR_ROW_ALT = "rgba(255,255,255,0.03)"

HEATMAP_RED = "#5C2626"
HEATMAP_NEUTRAL = "#1A1A2E"
HEATMAP_GREEN = "#1E3A2F"


def _load_quanto_yaml(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    if yaml is None:
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_data_root() -> Path:
    env_root = os.environ.get("QUANTO_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    for candidate in (Path.cwd(), PROJECT_ROOT):
        for name in (".quanto.yml", ".quanto.yaml"):
            cfg = _load_quanto_yaml(candidate / name)
            if cfg and isinstance(cfg, Mapping):
                data_root = cfg.get("data_root") or cfg.get("quanto_data_root")
                if data_root:
                    return Path(str(data_root)).expanduser()
    return Path.cwd() / ".quanto_data"


__all__ = [
    "CACHE_TTL_SECONDS",
    "COLOR_GOOD",
    "COLOR_BAD",
    "COLOR_WARN",
    "COLOR_MUTED",
    "COLOR_BG",
    "COLOR_BG_SECONDARY",
    "COLOR_ROW_ALT",
    "HEATMAP_RED",
    "HEATMAP_NEUTRAL",
    "HEATMAP_GREEN",
    "resolve_data_root",
]

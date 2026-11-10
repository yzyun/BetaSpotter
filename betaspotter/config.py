"""Load rule thresholds: packaged defaults, optionally overridden by a user file."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).parent / "rules" / "config.yaml"


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict:
    cfg = yaml.safe_load(DEFAULT_PATH.read_text())
    if path is not None:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    return cfg

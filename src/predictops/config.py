"""Configuration loading.

All modules call `load_config()` to get settings from configs/config.yaml.
Relative paths in the config are resolved against the project root, so code
works no matter which directory it is launched from.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Read the YAML config once and cache it."""
    cfg_path = Path(path or os.environ.get("PREDICTOPS_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(relative: str) -> Path:
    """Turn a config-relative path (e.g. 'data/raw/x.csv') into an absolute Path."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def sensor_columns(cfg: dict[str, Any] | None = None) -> list[str]:
    cfg = cfg or load_config()
    return cfg["dataset"]["analog_sensors"] + cfg["dataset"]["digital_sensors"]

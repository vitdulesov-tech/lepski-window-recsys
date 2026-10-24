"""Загрузка конфигурации эксперимента из TOML (стандартный tomllib, Python >= 3.11)."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.toml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    # Абсолютные пути к данным, чтобы скрипты можно было запускать из любой директории.
    for key in ("raw_dir", "processed_dir"):
        cfg["general"][key] = str(PROJECT_ROOT / cfg["general"][key])
    return cfg


def processed_path(cfg: dict[str, Any], name: str) -> Path:
    return Path(cfg["general"]["processed_dir"]) / name

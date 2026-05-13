"""Config loading for Inference-Lens.

A thin YAML wrapper so every module gets settings from the same source
without hard-coding paths or hyperparameters anywhere.

Usage:
    from inference_lens.utils.config import load_config

    cfg = load_config()
    seed = cfg["project"]["seed"]
    cache_file = cfg.get("features", "cache_file")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin wrapper around a loaded YAML config dict."""

    def __init__(self, path: str | Path = "configs/base.yaml"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            self._cfg: dict = yaml.safe_load(f)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested key access with a safe default.

        Example: cfg.get("features", "cache_file", default="data/processed/features.parquet")
        """
        node = self._cfg
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def __getitem__(self, key: str) -> Any:
        return self._cfg[key]

    def __repr__(self) -> str:
        return f"Config({list(self._cfg.keys())})"


def load_config(path: str | Path = "configs/base.yaml") -> Config:
    """Load and return a Config from a YAML file."""
    return Config(path)

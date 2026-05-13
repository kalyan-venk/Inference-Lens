"""Logging setup for Inference-Lens.

Call setup_logging() once at the start of any script or notebook to get
consistent, timestamped log output across all modules.

Usage:
    from inference_lens.utils.logging import setup_logging
    setup_logging()
"""
from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Quiet down noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.WARNING)

"""Logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str = "INFO", trace_dir: Path | None = None) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)

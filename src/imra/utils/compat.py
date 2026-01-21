"""Python version compatibility utilities."""

from __future__ import annotations

import sys
from datetime import datetime, timezone

# Python 3.11+ exposes a UTC singleton; fall back to timezone.utc for older
try:
    # Available in Python 3.11+
    from datetime import UTC  # type: ignore
except Exception:
    UTC = timezone.utc


def utc_now() -> datetime:
    """Get current UTC datetime, compatible with Python 3.10+."""
    return datetime.now(UTC)

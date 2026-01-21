"""Trace data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from imra.utils.compat import utc_now


@dataclass(slots=True)
class TraceEvent:
    kind: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class TraceBuffer:
    task_id: str
    created_at: datetime
    events: list[TraceEvent] = field(default_factory=list)

    def append(self, event: TraceEvent) -> None:
        self.events.append(event)

    def to_markdown(self) -> str:
        lines = [f"# Trace for {self.task_id}"]
        for idx, event in enumerate(self.events):
            lines.append(f"{idx + 1}. **{event.kind}** – {event.payload}")
        return "\n".join(lines)

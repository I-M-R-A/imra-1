"""Markdown trace formatter."""

from __future__ import annotations

from imra.tracing.schemas import TraceBuffer


class MarkdownFormatter:
    def format(self, trace: TraceBuffer) -> str:
        return trace.to_markdown()

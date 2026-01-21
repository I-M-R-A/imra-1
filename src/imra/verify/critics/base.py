"""Base critic definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from imra.tracing.schemas import TraceBuffer


class VerifyCritic(ABC):
    """Score reasoning traces and emit directives."""

    @abstractmethod
    def score(self, trace: TraceBuffer) -> float:
        """Return confidence score between 0 and 1."""

    def directive(self, trace: TraceBuffer) -> str:
        score = self.score(trace)
        if score < 0.5:
            return "refine"
        return "accept"

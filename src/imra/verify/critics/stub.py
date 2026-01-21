"""Minimal VERIFY critic for demos."""

from __future__ import annotations

from statistics import mean

from imra.tracing.schemas import TraceBuffer
from imra.verify.critics.base import VerifyCritic


class StubVerifyCritic(VerifyCritic):
    def score(self, trace: TraceBuffer) -> float:
        payload_scores = []
        for event in trace.events:
            if event.kind == "reason.step" and isinstance(
                event.payload.get("output"), (int, float)
            ):
                payload_scores.append(0.1)
            if event.kind == "verify.score" and "confidence" in event.payload:
                payload_scores.append(float(event.payload["confidence"]))
        if not payload_scores:
            return 0.5
        return min(max(mean(payload_scores), 0.0), 1.0)

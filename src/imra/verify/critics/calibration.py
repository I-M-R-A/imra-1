"""Calibration-aware VERIFY critic with confidence scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from imra.tracing.schemas import TraceBuffer
from imra.verify.critics.base import VerifyCritic


@dataclass
class CalibrationStats:
    """Tracks prediction vs. outcome for calibration measurement."""

    predictions: list[float] = field(default_factory=list)
    outcomes: list[float] = field(default_factory=list)  # 1.0 = correct, 0.0 = wrong

    def add(self, confidence: float, correct: bool) -> None:
        self.predictions.append(confidence)
        self.outcomes.append(1.0 if correct else 0.0)

    def expected_calibration_error(self, n_bins: int = 10) -> float:
        """Compute ECE across confidence bins."""
        if not self.predictions:
            return 0.0

        bins = [[] for _ in range(n_bins)]
        for conf, out in zip(self.predictions, self.outcomes, strict=False):
            idx = min(int(conf * n_bins), n_bins - 1)
            bins[idx].append((conf, out))

        ece = 0.0
        total = len(self.predictions)
        for bin_data in bins:
            if not bin_data:
                continue
            avg_conf = sum(c for c, _ in bin_data) / len(bin_data)
            avg_acc = sum(o for _, o in bin_data) / len(bin_data)
            ece += (len(bin_data) / total) * abs(avg_conf - avg_acc)

        return ece

    def brier_score(self) -> float:
        """Compute Brier score (MSE of probability estimates)."""
        if not self.predictions:
            return 0.0
        return sum(
            (p - o) ** 2 for p, o in zip(self.predictions, self.outcomes, strict=False)
        ) / len(self.predictions)


class CalibrationCritic(VerifyCritic):
    """
    A VERIFY critic that scores traces based on multiple factors
    and maintains calibration statistics for self-improvement.
    """

    def __init__(
        self,
        coherence_weight: float = 0.4,
        completeness_weight: float = 0.3,
        efficiency_weight: float = 0.3,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.coherence_weight = coherence_weight
        self.completeness_weight = completeness_weight
        self.efficiency_weight = efficiency_weight
        self.confidence_threshold = confidence_threshold
        self.calibration = CalibrationStats()
        self._step_budget = 100  # Expected max steps for efficiency scoring

    def score(self, trace: TraceBuffer) -> float:
        """Compute overall confidence score from trace analysis."""
        coherence = self._score_coherence(trace)
        completeness = self._score_completeness(trace)
        efficiency = self._score_efficiency(trace)

        raw_score = (
            self.coherence_weight * coherence
            + self.completeness_weight * completeness
            + self.efficiency_weight * efficiency
        )

        return max(0.0, min(1.0, raw_score))

    def directive(self, trace: TraceBuffer) -> str:
        """Decide whether to accept, refine, or rollback."""
        score = self.score(trace)

        if self._has_errors(trace):
            return "rollback"

        if score < self.confidence_threshold:
            return "refine"

        return "accept"

    def detailed_analysis(self, trace: TraceBuffer) -> dict:
        """Return breakdown of all scoring factors."""
        coherence = self._score_coherence(trace)
        completeness = self._score_completeness(trace)
        efficiency = self._score_efficiency(trace)

        return {
            "overall_score": self.score(trace),
            "coherence": coherence,
            "completeness": completeness,
            "efficiency": efficiency,
            "directive": self.directive(trace),
            "error_count": self._count_errors(trace),
            "step_count": self._count_steps(trace),
            "has_result": self._has_final_result(trace),
        }

    def record_outcome(self, trace: TraceBuffer, correct: bool) -> None:
        """Record outcome for calibration tracking."""
        score = self.score(trace)
        self.calibration.add(score, correct)

    def get_calibration_metrics(self) -> dict:
        """Return current calibration statistics."""
        return {
            "ece": self.calibration.expected_calibration_error(),
            "brier": self.calibration.brier_score(),
            "n_samples": len(self.calibration.predictions),
        }

    def _score_coherence(self, trace: TraceBuffer) -> float:
        """Score logical consistency of the trace."""
        if not trace.events:
            return 0.0

        steps = [e for e in trace.events if e.kind.startswith("reason.")]
        if not steps:
            return 0.5

        errors = self._count_errors(trace)
        coherence = 1.0 - (errors * 0.3)

        has_gradients = any("gradients" in e.payload and e.payload["gradients"] for e in steps)

        if has_gradients:
            coherence += 0.1

        return max(0.0, min(1.0, coherence))

    def _score_completeness(self, trace: TraceBuffer) -> float:
        """Score whether trace has all expected components."""
        score = 0.0

        has_sketch = any(e.kind == "perceive.sketch" for e in trace.events)
        if has_sketch:
            score += 0.3

        reason_steps = [e for e in trace.events if e.kind == "reason.step"]
        if reason_steps:
            score += 0.4

        if self._has_final_result(trace):
            score += 0.3

        return score

    def _score_efficiency(self, trace: TraceBuffer) -> float:
        """Score computational efficiency."""
        steps = self._count_steps(trace)

        if steps == 0:
            return 0.5

        ratio = steps / self._step_budget
        efficiency = max(0.0, 1.0 - ratio)

        return efficiency

    def _has_errors(self, trace: TraceBuffer) -> bool:
        return any(e.kind == "reason.error" for e in trace.events)

    def _count_errors(self, trace: TraceBuffer) -> int:
        return sum(1 for e in trace.events if e.kind == "reason.error")

    def _count_steps(self, trace: TraceBuffer) -> int:
        return sum(1 for e in trace.events if e.kind == "reason.step")

    def _has_final_result(self, trace: TraceBuffer) -> bool:
        for event in reversed(trace.events):
            if event.kind == "reason.step" and "output" in event.payload:
                return True
            if event.kind == "loop.directive" and event.payload.get("action") == "accept":
                return True
        return False


class AnomalyDetector:
    """Detect anomalous patterns in reasoning traces."""

    def __init__(self, threshold: float = 2.0) -> None:
        self.threshold = threshold
        self._value_history: list[float] = []

    def check(self, trace: TraceBuffer) -> list[dict]:
        """Return list of detected anomalies."""
        anomalies = []

        values = []
        for event in trace.events:
            if event.kind == "reason.step" and "output" in event.payload:
                val = event.payload["output"]
                if isinstance(val, (int, float)):
                    values.append(val)

        if len(values) < 2:
            return anomalies

        for i in range(1, len(values)):  # Check for sudden jumps
            prev, curr = values[i - 1], values[i]
            if abs(prev) > 1e-6:
                ratio = abs(curr / prev)
                if ratio > 100 or ratio < 0.01:
                    anomalies.append(
                        {
                            "type": "value_jump",
                            "step": i,
                            "prev": prev,
                            "curr": curr,
                            "ratio": ratio,
                        }
                    )

        for i, v in enumerate(values):  # Check for NaN or Inf
            if math.isnan(v) or math.isinf(v):
                anomalies.append(
                    {
                        "type": "invalid_value",
                        "step": i,
                        "value": v,
                    }
                )

        return anomalies

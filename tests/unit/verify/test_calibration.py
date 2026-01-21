"""Tests for CalibrationCritic and related components."""

import pytest

from imra.tracing.schemas import TraceBuffer, TraceEvent
from imra.utils.compat import utc_now
from imra.verify.critics.calibration import AnomalyDetector, CalibrationCritic, CalibrationStats


class TestCalibrationStats:
    """Tests for calibration statistics computation."""

    def test_empty_stats(self) -> None:
        stats = CalibrationStats()
        assert stats.expected_calibration_error() == 0.0
        assert stats.brier_score() == 0.0

    def test_add_prediction(self) -> None:
        stats = CalibrationStats()
        stats.add(confidence=0.8, correct=True)
        stats.add(confidence=0.8, correct=False)

        assert len(stats.predictions) == 2
        assert len(stats.outcomes) == 2

    def test_ece_perfectly_calibrated(self) -> None:
        """ECE should be 0 for perfect calibration."""
        stats = CalibrationStats()
        for _ in range(8):
            stats.add(0.8, True)
        for _ in range(2):
            stats.add(0.8, False)

        ece = stats.expected_calibration_error(n_bins=1)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_brier_score_perfect(self) -> None:
        """Brier score should be 0 for perfect predictions."""
        stats = CalibrationStats()
        stats.add(1.0, True)
        stats.add(0.0, False)

        assert stats.brier_score() == 0.0

    def test_brier_score_worst(self) -> None:
        """Brier score should be 1 for worst predictions."""
        stats = CalibrationStats()
        stats.add(1.0, False)
        stats.add(0.0, True)

        assert stats.brier_score() == 1.0


class TestCalibrationCritic:
    """Tests for CalibrationCritic."""

    def test_score_empty_trace(self) -> None:
        critic = CalibrationCritic()
        trace = TraceBuffer(task_id="empty", created_at=utc_now())

        score = critic.score(trace)
        assert 0.0 <= score <= 1.0

    def test_score_with_reason_events(self) -> None:
        critic = CalibrationCritic()
        trace = TraceBuffer(task_id="test", created_at=utc_now())
        trace.append(
            TraceEvent(
                kind="reason.step",
                payload={"op": "add", "inputs": [3, 5], "output": 8},
            )
        )
        trace.append(
            TraceEvent(
                kind="reason.step",
                payload={"op": "mul", "inputs": [8, 2], "output": 16},
            )
        )

        score = critic.score(trace)
        assert 0.0 <= score <= 1.0

    def test_directive_accept(self) -> None:
        critic = CalibrationCritic(confidence_threshold=0.5)
        trace = TraceBuffer(task_id="test", created_at=utc_now())

        for i in range(5):
            trace.append(
                TraceEvent(
                    kind="reason.step",
                    payload={"op": "add", "inputs": [i, 1], "output": i + 1},
                )
            )

        directive = critic.directive(trace)
        assert directive in ("accept", "refine", "rollback")

    def test_detailed_analysis(self) -> None:
        critic = CalibrationCritic()
        trace = TraceBuffer(task_id="analysis", created_at=utc_now())
        trace.append(TraceEvent(kind="reason.step", payload={"output": 5}))

        analysis = critic.detailed_analysis(trace)

        assert "coherence" in analysis
        assert "completeness" in analysis
        assert "efficiency" in analysis
        assert "overall_score" in analysis
        assert "directive" in analysis


class TestAnomalyDetector:
    """Tests for AnomalyDetector."""

    def test_no_anomalies(self) -> None:
        detector = AnomalyDetector()
        trace = TraceBuffer(task_id="test", created_at=utc_now())
        trace.append(TraceEvent(kind="reason.step", payload={"output": 1.0}))
        trace.append(TraceEvent(kind="reason.step", payload={"output": 2.0}))
        trace.append(TraceEvent(kind="reason.step", payload={"output": 3.0}))

        anomalies = detector.check(trace)
        assert len(anomalies) == 0

    def test_detect_jump(self) -> None:
        detector = AnomalyDetector()
        trace = TraceBuffer(task_id="test", created_at=utc_now())
        trace.append(TraceEvent(kind="reason.step", payload={"output": 1.0}))
        trace.append(TraceEvent(kind="reason.step", payload={"output": 1000.0}))  # Large jump

        anomalies = detector.check(trace)
        assert len(anomalies) >= 1
        assert any(a["type"] == "value_jump" for a in anomalies)

    def test_detect_invalid_value(self) -> None:
        detector = AnomalyDetector()
        trace = TraceBuffer(task_id="test", created_at=utc_now())
        trace.append(TraceEvent(kind="reason.step", payload={"output": 1.0}))
        trace.append(TraceEvent(kind="reason.step", payload={"output": float("nan")}))

        anomalies = detector.check(trace)
        assert len(anomalies) >= 1
        assert any(a["type"] == "invalid_value" for a in anomalies)

    def test_empty_trace(self) -> None:
        detector = AnomalyDetector()
        trace = TraceBuffer(task_id="empty", created_at=utc_now())

        anomalies = detector.check(trace)
        assert len(anomalies) == 0

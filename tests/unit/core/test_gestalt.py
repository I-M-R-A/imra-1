#!/usr/bin/env python3

"""Unit tests for the Gestalt Coherence Comparator.

Tests the coherence scoring and failure mode detection.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))  # Adjust path to src

from imra.core.gestalt import (
    CoherenceAnalysis,
    CoherenceLevel,
    FailureMode,
    GestaltComparator,
    analyze_trace,
)


class TestCoherenceLevel:
    """Tests for coherence level classification thresholds."""

    def test_broken_coherence(self):
        """Scores < 0.3 should be BROKEN."""
        comparator = GestaltComparator()
        assert comparator.BROKEN_THRESHOLD == 0.3

    def test_uncertain_coherence(self):
        """Scores 0.3-0.6 should be UNCERTAIN."""
        comparator = GestaltComparator()
        assert comparator.UNCERTAIN_THRESHOLD == 0.6

    def test_acceptable_coherence(self):
        """Scores 0.6-0.8 should be ACCEPTABLE."""
        comparator = GestaltComparator()
        assert comparator.ACCEPTABLE_THRESHOLD == 0.8


class TestCoherenceScoring:
    """Tests for coherence score calculation."""

    def test_weights_sum_to_one(self):
        """Coherence weights should sum to 1.0."""
        comparator = GestaltComparator()
        total = (
            comparator.ENTROPY_WEIGHT + comparator.ERROR_DENSITY_WEIGHT + comparator.MISMATCH_WEIGHT
        )
        assert total == pytest.approx(1.0)

    def test_high_coherence_scenario(self):
        """Perfect reasoning should give high coherence."""
        comparator = GestaltComparator()

        mock_trace = MagicMock()  # Create a mock Trace object for testing
        mock_trace.discussion = MagicMock()
        mock_trace.discussion.turns = [
            MagicMock(stance="agree"),
            MagicMock(stance="agree"),
            MagicMock(stance="agree"),
        ]
        mock_trace.discussion.consensus_reached = True
        mock_trace.total_errors_detected = 0
        mock_trace.total_corrections_made = 0
        mock_trace.final_confidence = 0.9
        mock_trace.final_answer = "42"

        analysis = comparator.analyze(mock_trace)

        assert analysis.coherence_score >= 0.6
        assert analysis.coherence_level in [
            CoherenceLevel.ACCEPTABLE,
            CoherenceLevel.COHERENT,
        ]  # It should be at least ACCEPTABLE

    def test_low_coherence_from_errors(self):
        """Many uncorrected errors should reduce coherence."""
        comparator = GestaltComparator()

        mock_trace = MagicMock()
        mock_trace.discussion = MagicMock()
        mock_trace.discussion.turns = [
            MagicMock(stance="disagree"),
            MagicMock(stance="disagree"),
        ]
        mock_trace.discussion.consensus_reached = False
        mock_trace.total_errors_detected = 10
        mock_trace.total_corrections_made = 1
        mock_trace.final_confidence = 0.2
        mock_trace.final_answer = "I don't know"

        analysis = comparator.analyze(mock_trace)

        assert analysis.coherence_score < 0.8


class TestFailureModeDetection:
    """Tests for failure mode classification."""

    def test_meta_retreat_detection(self):
        """Should detect meta-retreat in answers."""
        comparator = GestaltComparator()

        meta_retreat_answers = [
            "We need to analyze this further",
            "I would recommend a different method",
            "The methodology should be revised",
            "Not applicable to this case",
            "Cannot determine the answer",
        ]

        for answer in meta_retreat_answers:
            result = comparator._detect_meta_retreat(answer)
            assert result is True, f"Failed to detect meta-retreat in: {answer}"

    def test_non_meta_retreat_detection(self):
        """Should not flag concrete answers as meta-retreat."""
        comparator = GestaltComparator()

        concrete_answers = ["42", "The answer is 5", "x = 3.14", "Yes", "No"]

        for answer in concrete_answers:
            result = comparator._detect_meta_retreat(answer)
            assert result is False, f"Incorrectly flagged as meta-retreat: {answer}"


class TestCoherenceAnalysis:
    """Tests for CoherenceAnalysis output."""

    def test_analysis_to_dict(self):
        """Analysis should serialize to dict correctly."""
        analysis = CoherenceAnalysis(
            coherence_score=0.75,
            coherence_level=CoherenceLevel.ACCEPTABLE,
            discussion_entropy=0.2,
            error_density=0.1,
            confidence_mismatch=0.15,
            failure_mode=FailureMode.HEALTHY,
            diagnosis="Reasoning appears sound",
            recommendation="Accept answer with monitoring",
            total_errors=1,
            total_corrections=1,
            final_confidence=0.85,
            consensus_reached=True,
            meta_retreat_detected=False,
        )

        d = analysis.to_dict()

        assert d["coherence_score"] == 0.75
        assert d["coherence_level"] == "acceptable"
        assert d["diagnosis"]["failure_mode"] == "healthy"
        assert "components" in d
        assert "evidence" in d

    def test_analysis_to_human_readable(self):
        """Analysis should generate readable text."""
        analysis = CoherenceAnalysis(
            coherence_score=0.75,
            coherence_level=CoherenceLevel.ACCEPTABLE,
            discussion_entropy=0.2,
            error_density=0.1,
            confidence_mismatch=0.15,
            failure_mode=FailureMode.HEALTHY,
            diagnosis="Reasoning appears sound",
            recommendation="Accept answer",
            total_errors=0,
            total_corrections=0,
            final_confidence=0.9,
            consensus_reached=True,
            meta_retreat_detected=False,
        )

        text = analysis.to_human_readable()

        assert "COHERENCE" in text
        assert "0.75" in text or "75" in text
        assert "ACCEPTABLE" in text
        assert "HEALTHY" in text


class TestAnalyzeTraceFunction:
    """Tests for the convenience analyze_trace function."""

    def test_analyze_trace_returns_analysis(self):
        """analyze_trace should return a CoherenceAnalysis."""
        mock_trace = MagicMock()
        mock_trace.discussion = None
        mock_trace.total_errors_detected = 0
        mock_trace.total_corrections_made = 0
        mock_trace.final_confidence = 0.5
        mock_trace.final_answer = "test"

        analysis = analyze_trace(mock_trace)

        assert isinstance(analysis, CoherenceAnalysis)
        assert 0.0 <= analysis.coherence_score <= 1.0
        assert isinstance(analysis.coherence_level, CoherenceLevel)
        assert isinstance(analysis.failure_mode, FailureMode)

    def test_analyze_trace_handles_no_discussion(self):
        """Should handle traces without discussion gracefully."""
        mock_trace = MagicMock()
        mock_trace.discussion = None
        mock_trace.total_errors_detected = 0
        mock_trace.total_corrections_made = 0
        mock_trace.final_confidence = 0.7
        mock_trace.final_answer = "42"

        # Should not raise
        analysis = analyze_trace(mock_trace)
        assert analysis is not None


class TestFailureModeEnum:
    """Tests for FailureMode enum values."""

    def test_all_failure_modes_exist(self):
        """All expected failure modes should be defined."""
        expected_modes = [
            "VERIFICATION_PARADOX",
            "CIRCULAR_REASONING",
            "OVERCONFIDENT_ERROR",
            "META_RETREAT",
            "INCOMPLETE_REASONING",
            "FORMAT_MISMATCH",
            "HEALTHY",
        ]

        for mode in expected_modes:
            assert hasattr(FailureMode, mode), f"Missing failure mode: {mode}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""Gestalt Comparator - Coherence Detection for Multi-Agent Reasoning

This module implements the "logical smoke detector" that identifies when
reasoning is broken even when individual components appear confident.

Based on MAE paper insights: High confidence + high entropy = broken gestalt.

The Gestalt Comparator computes a coherence score by measuring:
1. Discussion Entropy (Shannon entropy of stances)
2. Error Density (errors detected vs corrections made)
3. Confidence-Consensus Mismatch (overconfidence without agreement)

COHERENCE SCORE:
  0.0-0.3: BROKEN (recommend reject/retry)
  0.3-0.6: UNCERTAIN (recommend human review)
  0.6-0.8: ACCEPTABLE (but monitor)
  0.8-1.0: COHERENT (high trust)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .trace import CognitiveTrace, TeamDiscussion


class CoherenceLevel(Enum):
    """Coherence categories for reasoning quality."""

    BROKEN = "broken"  # 0.0 - 0.3: Logic is fractured, reject answer
    UNCERTAIN = "uncertain"  # 0.3 - 0.6: Needs human review
    ACCEPTABLE = "acceptable"  # 0.6 - 0.8: Likely correct but monitor
    COHERENT = "coherent"  # 0.8 - 1.0: High confidence in logic


class FailureMode(Enum):
    """Specific diagnostic patterns."""

    VERIFICATION_PARADOX = "verification_paradox"  # Found errors but high confidence
    CIRCULAR_REASONING = "circular_reasoning"  # High entropy, no convergence
    OVERCONFIDENT_ERROR = "overconfident_error"  # Confidence=1.0 but errors detected
    META_RETREAT = "meta_retreat"  # Discussing methodology instead of solving
    INCOMPLETE_REASONING = "incomplete_reasoning"  # No conclusion reached
    FORMAT_MISMATCH = "format_mismatch"  # Answer type doesn't match question
    HEALTHY = "healthy"  # No issues detected


@dataclass
class CoherenceAnalysis:
    """Complete coherence diagnosis for a reasoning trace."""

    # Core metric

    coherence_score: float  # 0.0-1.0 (higher is better)
    coherence_level: CoherenceLevel

    discussion_entropy: float  # 0.0-1.0 (Shannon entropy normalized)
    error_density: float  # 0.0-∞ (errors/corrections)
    confidence_mismatch: float  # 0.0-1.0 ( |confidence - consensus| )

    failure_mode: FailureMode
    diagnosis: str  # Human-readable explanation
    recommendation: str  # What to do about it (maybe LLM action prompt)

    total_errors: int
    total_corrections: int
    final_confidence: float
    consensus_reached: bool
    meta_retreat_detected: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "coherence_score": round(self.coherence_score, 3),
            "coherence_level": self.coherence_level.value,
            "components": {
                "discussion_entropy": round(self.discussion_entropy, 3),
                "error_density": round(self.error_density, 3),
                "confidence_mismatch": round(self.confidence_mismatch, 3),
            },
            "diagnosis": {
                "failure_mode": self.failure_mode.value,
                "explanation": self.diagnosis,
                "recommendation": self.recommendation,
            },
            "evidence": {
                "total_errors": self.total_errors,
                "total_corrections": self.total_corrections,
                "final_confidence": self.final_confidence,
                "consensus_reached": self.consensus_reached,
                "meta_retreat_detected": self.meta_retreat_detected,
            },
        }

    def to_human_readable(self) -> str:
        """Generate human-readable coherence report."""
        level_indicator = {
            CoherenceLevel.BROKEN: "[BROKEN]",
            CoherenceLevel.UNCERTAIN: "[UNCERTAIN]",
            CoherenceLevel.ACCEPTABLE: "[OK]",
            CoherenceLevel.COHERENT: "[COHERENT]",
        }

        lines = [
            "\n" + "",
            f"{level_indicator[self.coherence_level]} GESTALT COHERENCE ANALYSIS",
            "",
            f"\nCoherence Score: {self.coherence_score:.3f} ({self.coherence_level.value.upper()})",
            f"Failure Mode: {self.failure_mode.value.upper().replace('_', ' ')}",
            "",
            "Component Breakdown:",
            f"  Discussion Entropy:     {self.discussion_entropy:.3f} (lower is better)",
            f"  Error Density:          {self.error_density:.3f} (lower is better)",
            f"  Confidence Mismatch:    {self.confidence_mismatch:.3f} (lower is better)",
            "",
            "Evidence:",
            f"  Errors Detected:        {self.total_errors}",
            f"  Corrections Made:       {self.total_corrections}",
            f"  Final Confidence:       {self.final_confidence:.2f}",
            f"  Consensus Reached:      {self.consensus_reached}",
            f"  Meta-Retreat:           {self.meta_retreat_detected}",
            "",
            "DIAGNOSIS:",
            f"  {self.diagnosis}",
            "",
            "RECOMMENDATION:",
            f"  {self.recommendation}",
            "",
        ]

        return "\n".join(lines)


class GestaltComparator:
    """The 'Logical Smoke Detector' for Multi-Agent Reasoning.

    This is a pure logic module (no LLM calls) that computes coherence
    by measuring mismatches between confidence and discussion quality.

    Key Insight from MAE paper:
    - High confidence + high entropy = broken gestalt
    - The system "sees" the right structure but renders it incorrectly
    """

    def __init__(self) -> None:
        """Initialize comparator with diagnostic thresholds."""
        self.ENTROPY_WEIGHT = 0.4
        self.ERROR_DENSITY_WEIGHT = 0.3
        self.MISMATCH_WEIGHT = 0.3

        self.BROKEN_THRESHOLD = 0.3
        self.UNCERTAIN_THRESHOLD = 0.6
        self.ACCEPTABLE_THRESHOLD = 0.8

        self.META_KEYWORDS = [
            "we need to",
            "should analyze",
            "would propose",
            "methodology",
            "approach should",
            "recommend",
            "different method",
            "not applicable",
            "cannot determine",
            "unable to solve",
            "beyond the scope",
            "meta-discussion",
            "out of scope",
        ]

    def analyze(self, trace: CognitiveTrace) -> CoherenceAnalysis:
        """Compute coherence analysis for a complete reasoning trace.

        Args:
            trace: Complete cognitive trace from IMRA-1 reasoning

        Returns:
            CoherenceAnalysis with score, diagnosis, and recommendation
        """
        discussion_entropy = self._compute_discussion_entropy(trace.discussion)
        error_density = self._compute_error_density(
            trace.total_errors_detected, trace.total_corrections_made
        )
        confidence_mismatch = self._compute_confidence_mismatch(
            trace.final_confidence,
            trace.discussion.consensus_reached if trace.discussion else False,
        )

        # Compute overall coherence score ( 0.0 - 1.0, higher is better )

        coherence_score = (
            self.ENTROPY_WEIGHT * (1.0 - discussion_entropy)
            + self.ERROR_DENSITY_WEIGHT * (1.0 - min(error_density, 1.0))
            + self.MISMATCH_WEIGHT * (1.0 - confidence_mismatch)
        )

        if coherence_score < self.BROKEN_THRESHOLD:
            level = CoherenceLevel.BROKEN
        elif coherence_score < self.UNCERTAIN_THRESHOLD:
            level = CoherenceLevel.UNCERTAIN
        elif coherence_score < self.ACCEPTABLE_THRESHOLD:
            level = CoherenceLevel.ACCEPTABLE
        else:
            level = CoherenceLevel.COHERENT

        meta_retreat = self._detect_meta_retreat(trace.final_answer)

        failure_mode = self._diagnose_failure_mode(
            trace.final_confidence,
            trace.total_errors_detected,
            trace.total_corrections_made,
            trace.discussion.consensus_reached if trace.discussion else False,
            discussion_entropy,
            meta_retreat,
        )

        diagnosis = self._generate_diagnosis(failure_mode, trace)

        recommendation = self._generate_recommendation(level, failure_mode)

        # Down below is the return statement for the analyze function, which kind of summarizes everything
        # that was computed above.
        # It creates and returns a CoherenceAnalysis object with all the relevant metrics and insights.

        return CoherenceAnalysis(
            coherence_score=coherence_score,
            coherence_level=level,
            discussion_entropy=discussion_entropy,
            error_density=error_density,
            confidence_mismatch=confidence_mismatch,
            failure_mode=failure_mode,
            diagnosis=diagnosis,
            recommendation=recommendation,
            total_errors=trace.total_errors_detected,
            total_corrections=trace.total_corrections_made,
            final_confidence=trace.final_confidence,
            consensus_reached=trace.discussion.consensus_reached if trace.discussion else False,
            meta_retreat_detected=meta_retreat,
        )

    def _compute_discussion_entropy(self, discussion: TeamDiscussion | None) -> float:
        """Compute Shannon entropy of discussion stances (normalized to 0-1).

        Low entropy (0.0) = all agents agree
        High entropy (1.0) = chaotic disagreement

        Args:
            discussion: Team discussion or None

        Returns:
            Normalized entropy (0.0-1.0)
        """
        if not discussion or not discussion.turns:
            return 0.5

        stance_counts = {}
        for turn in discussion.turns:
            stance = turn.stance.lower()
            stance_counts[stance] = stance_counts.get(stance, 0) + 1

        total_turns = len(discussion.turns)

        # Shannon entropy [ Defined as -Σ p(x) log2 p(x) where p(x) is the probability of each stance ]
        # Something to note about Shannon entropy is that, it quantifies the uncertainty or unpredictability in a set of outcomes.
        # In this context, the "outcomes" are the different stances taken by agents during the discussion.
        # A higher entropy value indicates a more diverse set of stances, suggesting disagreement or lack of consensus among the agents.
        # Conversely, a lower entropy value indicates that the agents are more aligned in their stances, suggesting agreement or consensus.

        entropy = 0.0
        for count in stance_counts.values():
            if count > 0:
                p = count / total_turns
                entropy -= p * math.log2(p)

        # Normalize to 0-1 (max entropy is log2(4) for 4 possible stances)

        max_entropy = math.log2(4)  # Agree, disagree, refine, uncertain
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(normalized_entropy, 1.0)

    def _compute_error_density(self, errors_detected: int, corrections_made: int) -> float:
        """Compute ratio of uncorrected errors.

        Low density (0.0) = all errors fixed
        High density (1.0+) = many errors unfixed

        Args:
            errors_detected: Total errors found
            corrections_made: Total corrections applied

        Returns:
            Error density (0.0-1.0+, clamped in final score)
        """
        if errors_detected == 0:
            return 0.0

        uncorrected = errors_detected - corrections_made

        density = uncorrected / max(errors_detected, 1)

        if errors_detected > 3:
            density *= 1.2
        return density

    def _compute_confidence_mismatch(
        self, final_confidence: float, consensus_reached: bool
    ) -> float:
        """Compute mismatch between confidence and consensus.

        Problem: High confidence (0.95+) without consensus = overconfidence

        Args:
            final_confidence: Final answer confidence (0.0-1.0)
            consensus_reached: Whether team reached consensus

        Returns:
            Mismatch score (0.0 - 1.0)
        """
        consensus_score = 1.0 if consensus_reached else 0.0

        mismatch = abs(final_confidence - consensus_score)

        if final_confidence >= 0.95 and not consensus_reached:
            mismatch = min(mismatch * 1.5, 1.0)

        return mismatch

    def _detect_meta_retreat(self, final_answer: str) -> bool:
        """Detect if final answer is meta-commentary instead of solution.

        Args:
            final_answer: The final answer text

        Returns:
            True if answer is meta-retreat
        """
        if not final_answer:
            return True

        answer_lower = str(final_answer).lower()

        return any(keyword in answer_lower for keyword in self.META_KEYWORDS)

    def _diagnose_failure_mode(
        self,
        confidence: float,
        errors: int,
        corrections: int,
        consensus: bool,
        entropy: float,
        meta_retreat: bool,
    ) -> FailureMode:
        """Identify specific failure pattern.

        Args:
            confidence: Final confidence score
            errors: Total errors detected
            corrections: Total corrections made
            consensus: Whether consensus reached
            entropy: Discussion entropy
            meta_retreat: Whether meta-retreat detected

        Returns:
            Specific failure mode
        """
        if meta_retreat:
            return FailureMode.META_RETREAT

        # Verification Paradox (errors found but high confidence)

        if errors > 0 and confidence >= 0.95:
            return FailureMode.VERIFICATION_PARADOX

        if confidence >= 0.99 and errors > corrections:
            return FailureMode.OVERCONFIDENT_ERROR

        if entropy > 0.7 and not consensus:
            return FailureMode.CIRCULAR_REASONING

        if confidence < 0.6 and not consensus and errors > 0:
            return FailureMode.INCOMPLETE_REASONING

        return FailureMode.HEALTHY

    def _generate_diagnosis(self, mode: FailureMode, trace: CognitiveTrace) -> str:
        """Generate human-readable diagnosis.

        Args:
            mode: Detected failure mode
            trace: Complete reasoning trace

        Returns:
            Human-readable diagnostic message
        """
        diagnoses = {
            FailureMode.META_RETREAT: (
                "System retreated to meta-discussion instead of solving. "
                "Final answer contains methodology keywords instead of concrete solution. "
                "This suggests the reasoning decoder cannot bridge abstract logic to specific answer."
            ),
            FailureMode.VERIFICATION_PARADOX: (
                f"Detected {trace.total_errors_detected} errors but reported "
                f"{trace.final_confidence:.0%} confidence. This paradox suggests "
                f"verification layer found problems but resolution layer ignored them."
            ),
            FailureMode.OVERCONFIDENT_ERROR: (
                f"Perfect confidence ({trace.final_confidence:.0%}) despite "
                f"{trace.total_errors_detected - trace.total_corrections_made} uncorrected errors. "
                f"The system is overconfident in a flawed solution."
            ),
            FailureMode.CIRCULAR_REASONING: (
                "High discussion entropy with no convergence. The team debated but "
                "couldn't reach consensus, suggesting circular or contradictory reasoning."
            ),
            FailureMode.INCOMPLETE_REASONING: (
                f"Low confidence ({trace.final_confidence:.0%}) with {trace.total_errors_detected} "
                f"errors detected. The reasoning path has gaps or contradictions that were identified "
                f"but not resolved."
            ),
            FailureMode.HEALTHY: (
                "No significant coherence issues detected. The reasoning appears sound, "
                "with good agreement between confidence, consensus, and error correction."
            ),
        }

        return diagnoses.get(mode, "Unknown failure mode")

    def _generate_recommendation(self, level: CoherenceLevel, mode: FailureMode) -> str:
        """Generate actionable recommendation.

        Args:
            level: Overall coherence level
            mode: Specific failure mode

        Returns:
            Recommendation for what to do
        """
        if level == CoherenceLevel.BROKEN:
            if mode == FailureMode.META_RETREAT:
                return "REJECT: Implement deep decoder to bridge abstract → concrete translation."
            elif mode == FailureMode.VERIFICATION_PARADOX:
                return "REJECT: Verification found errors but resolution ignored them. Retry with error enforcement."
            elif mode == FailureMode.OVERCONFIDENT_ERROR:
                return "REJECT: Overconfident in broken logic. Flag for manual review or retry."
            else:
                return "REJECT: Coherence score too low. Recommend retry or human review."

        elif level == CoherenceLevel.UNCERTAIN:
            return "REVIEW: Moderate coherence issues detected. Recommend human verification before accepting."

        elif level == CoherenceLevel.ACCEPTABLE:
            return "ACCEPT WITH CAUTION: Coherence acceptable but monitor for patterns."

        else:
            return "ACCEPT: High coherence. Answer likely correct."


def analyze_trace(trace: CognitiveTrace) -> CoherenceAnalysis:
    """Convenience function to analyze a single trace.

    Args:
        trace: Complete cognitive trace

    Returns:
        Coherence analysis
    """
    comparator = GestaltComparator()
    return comparator.analyze(trace)

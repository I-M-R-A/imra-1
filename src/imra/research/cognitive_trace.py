"""IMRA-1 Cognitive Trace System.

The trace is the PRIMARY RESEARCH OUTPUT. It captures:
- What the AI "thought" at each step
- Where uncertainty arose and why
- How errors were detected and corrected
- The reasoning behind every decision
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CognitivePhase(Enum):
    """Phases of meta-cognitive processing."""

    PERCEPTION = "perception"
    INTUITION = "intuition"
    UNCERTAINTY_DETECTION = "uncertainty_detection"
    REASONING = "reasoning"
    SELF_CRITIQUE = "self_critique"
    ERROR_DETECTION = "error_detection"
    CORRECTION = "correction"
    REFLECTION = "reflection"
    CONSENSUS = "consensus"


class UncertaintyType(Enum):
    """Types of uncertainty the system can detect."""

    EPISTEMIC = "epistemic"
    ALEATORIC = "aleatoric"
    MODEL = "model"
    COMPUTATIONAL = "computational"
    SEMANTIC = "semantic"
    CONFIDENCE = "confidence"


@dataclass
class UncertaintyMarker:
    """A detected point of uncertainty in reasoning.

    This is research data: WHERE did the AI doubt itself and WHY?
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    uncertainty_type: str = ""
    location: str = ""
    description: str = ""
    severity: str = "medium"
    source_text: str = ""
    resolution: str | None = None
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "type": self.uncertainty_type,
            "location": self.location,
            "description": self.description,
            "severity": self.severity,
            "source_text": self.source_text,
            "resolution": self.resolution,
            "resolved": self.resolved,
        }


@dataclass
class ReasoningStep:
    """A single step in the reasoning process.

    Captures the AI's "thought" at one moment, including:
    - What it did
    - Why it did it
    - What it was unsure about
    - What it got wrong (if detected)
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    step_number: int = 0
    phase: str = ""
    actor: str = ""

    # The reasoning content
    input_context: str = ""
    raw_output: str = ""
    parsed_output: dict[str, Any] = field(default_factory=dict)

    # Meta-cognitive annotations
    stated_confidence: float = 0.0
    stated_reasoning: str = ""
    uncertainties: list[UncertaintyMarker] = field(default_factory=list)

    # Error tracking
    error_detected: bool = False
    error_description: str = ""
    correction_attempted: bool = False
    correction_description: str = ""

    # Timing
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "step_number": self.step_number,
            "phase": self.phase,
            "actor": self.actor,
            "input_context": self.input_context,
            "raw_output": self.raw_output,
            "parsed_output": self.parsed_output,
            "stated_confidence": self.stated_confidence,
            "stated_reasoning": self.stated_reasoning,
            "uncertainties": [u.to_dict() for u in self.uncertainties],
            "error_detected": self.error_detected,
            "error_description": self.error_description,
            "correction_attempted": self.correction_attempted,
            "correction_description": self.correction_description,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class CognitiveTrace:
    """The complete cognitive trace of a deliberation.

    THIS IS THE RESEARCH PRODUCT.

    It contains everything needed to study:
    - How the AI understood the problem
    - What approaches it considered
    - Where it was uncertain
    - What errors it made
    - How it corrected itself
    - What it "learned"

    This is human-interpretable data for studying AI cognition.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Problem context
    problem_statement: str = ""
    problem_category: str = ""
    expected_answer: Any = None

    # The reasoning trace
    steps: list[ReasoningStep] = field(default_factory=list)

    # Aggregated uncertainty analysis
    all_uncertainties: list[UncertaintyMarker] = field(default_factory=list)

    # Error and correction log
    errors_detected: list[dict[str, Any]] = field(default_factory=list)
    corrections_made: list[dict[str, Any]] = field(default_factory=list)

    # Final outcomes
    final_answer: Any = None
    final_confidence: float = 0.0
    answer_correct: bool | None = None

    # Meta-cognitive summary
    reasoning_summary: str = ""
    key_insights: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)

    # Research annotations (added during analysis)
    research_notes: list[str] = field(default_factory=list)

    # Timing
    total_duration_seconds: float = 0.0

    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step to the trace."""
        step.step_number = len(self.steps) + 1
        self.steps.append(step)
        self.all_uncertainties.extend(step.uncertainties)

        if step.error_detected:
            self.errors_detected.append(
                {
                    "step_id": step.id,
                    "step_number": step.step_number,
                    "phase": step.phase,
                    "description": step.error_description,
                }
            )

        if step.correction_attempted:
            self.corrections_made.append(
                {
                    "step_id": step.id,
                    "step_number": step.step_number,
                    "phase": step.phase,
                    "description": step.correction_description,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "problem": {
                "statement": self.problem_statement,
                "category": self.problem_category,
                "expected_answer": self.expected_answer,
            },
            "reasoning_trace": [s.to_dict() for s in self.steps],
            "uncertainty_analysis": {
                "total_uncertainties": len(self.all_uncertainties),
                "by_type": self._group_uncertainties_by_type(),
                "by_severity": self._group_uncertainties_by_severity(),
                "details": [u.to_dict() for u in self.all_uncertainties],
            },
            "error_analysis": {
                "total_errors": len(self.errors_detected),
                "total_corrections": len(self.corrections_made),
                "errors": self.errors_detected,
                "corrections": self.corrections_made,
            },
            "outcome": {
                "final_answer": self.final_answer,
                "final_confidence": self.final_confidence,
                "answer_correct": self.answer_correct,
            },
            "meta_cognitive_summary": {
                "reasoning_summary": self.reasoning_summary,
                "key_insights": self.key_insights,
                "failure_modes": self.failure_modes,
            },
            "research_notes": self.research_notes,
            "timing": {
                "total_duration_seconds": self.total_duration_seconds,
                "step_count": len(self.steps),
            },
        }

    def _group_uncertainties_by_type(self) -> dict[str, int]:
        """Group uncertainties by type for analysis."""
        groups: dict[str, int] = {}
        for u in self.all_uncertainties:
            t = u.uncertainty_type
            groups[t] = groups.get(t, 0) + 1
        return groups

    def _group_uncertainties_by_severity(self) -> dict[str, int]:
        """Group uncertainties by severity for analysis."""
        groups: dict[str, int] = {}
        for u in self.all_uncertainties:
            s = u.severity
            groups[s] = groups.get(s, 0) + 1
        return groups

    def to_human_readable(self) -> str:
        """Generate human-readable research log.

        This is what researchers read to understand AI cognition.
        """
        lines = []
        lines.append("")
        lines.append("IMRA-1 COGNITIVE TRACE - RESEARCH LOG")
        lines.append("")
        lines.append(f"Trace ID: {self.id}")
        lines.append(f"Generated: {self.created_at}")
        lines.append("")

        # Problem
        lines.append("")
        lines.append("PROBLEM")
        lines.append("")
        lines.append(self.problem_statement)
        if self.expected_answer:
            lines.append(f"Expected Answer: {self.expected_answer}")
        lines.append("")

        # Reasoning steps
        lines.append("")
        lines.append("REASONING TRACE")
        lines.append("")
        for step in self.steps:
            lines.append(f"\n[Step {step.step_number}] {step.phase.upper()} ({step.actor})")
            lines.append(f"Time: {step.timestamp} | Duration: {step.duration_seconds:.1f}s")
            lines.append(f"Confidence: {step.stated_confidence:.2f}")
            if step.stated_reasoning:
                lines.append(f"Reasoning: {step.stated_reasoning[:200]}...")
            if step.uncertainties:
                lines.append(f"Uncertainties flagged: {len(step.uncertainties)}")
                for u in step.uncertainties:
                    lines.append(f"  - [{u.severity.upper()}] {u.description}")
            if step.error_detected:
                lines.append(f"ERROR DETECTED: {step.error_description}")
            if step.correction_attempted:
                lines.append(f"CORRECTION: {step.correction_description}")

        # Uncertainty summary
        lines.append("")
        lines.append("")
        lines.append("UNCERTAINTY ANALYSIS")
        lines.append("")
        lines.append(f"Total uncertainty markers: {len(self.all_uncertainties)}")
        by_type = self._group_uncertainties_by_type()
        for t, count in by_type.items():
            lines.append(f"  {t}: {count}")

        # Error summary
        lines.append("")
        lines.append("")
        lines.append("ERROR ANALYSIS")
        lines.append("")
        lines.append(f"Errors detected: {len(self.errors_detected)}")
        lines.append(f"Corrections attempted: {len(self.corrections_made)}")

        # Outcome
        lines.append("")
        lines.append("")
        lines.append("OUTCOME")
        lines.append("")
        lines.append(f"Final Answer: {self.final_answer}")
        lines.append(f"Confidence: {self.final_confidence:.2f}")
        if self.answer_correct is not None:
            lines.append(f"Correct: {self.answer_correct}")

        # Research insights
        if self.key_insights:
            lines.append("")
            lines.append("")
            lines.append("KEY INSIGHTS")
            lines.append("")
            for insight in self.key_insights:
                lines.append(f"- {insight}")

        if self.failure_modes:
            lines.append("")
            lines.append("")
            lines.append("FAILURE MODES IDENTIFIED")
            lines.append("")
            for mode in self.failure_modes:
                lines.append(f"- {mode}")

        lines.append("")
        lines.append("")
        lines.append("END OF TRACE")
        lines.append("")

        return "\n".join(lines)

    def save(self, filepath: str) -> None:
        """Save trace to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def save_human_readable(self, filepath: str) -> None:
        """Save human-readable log to text file."""
        with open(filepath, "w") as f:
            f.write(self.to_human_readable())

    @classmethod
    def load(cls, filepath: str) -> CognitiveTrace:
        """Load trace from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        trace = cls()
        trace.id = data.get("id", trace.id)
        trace.created_at = data.get("created_at", trace.created_at)

        problem = data.get("problem", {})
        trace.problem_statement = problem.get("statement", "")
        trace.problem_category = problem.get("category", "")
        trace.expected_answer = problem.get("expected_answer")

        outcome = data.get("outcome", {})
        trace.final_answer = outcome.get("final_answer")
        trace.final_confidence = outcome.get("final_confidence", 0.0)
        trace.answer_correct = outcome.get("answer_correct")

        meta = data.get("meta_cognitive_summary", {})
        trace.reasoning_summary = meta.get("reasoning_summary", "")
        trace.key_insights = meta.get("key_insights", [])
        trace.failure_modes = meta.get("failure_modes", [])

        trace.research_notes = data.get("research_notes", [])

        return trace

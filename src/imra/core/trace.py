#!/usr/bin/env python3

"""IMRA-1: Integrated Meta-Reasoning Architecture

A research system for studying AI cognition through structured deliberation.

Three Interlocking Layers:
├── PERCEIVE (Intuition Layer)
│   └── Produces probabilistic sketch of approaches
├── REASON (Deliberation Layer)
│   └── Executes candidate paths, tracks uncertainty
└── VERIFY (Self-Reflection Layer)
    └── Identifies failures, flags contradictions

This is not "just code" - it's a system that defines how AI should
reason, reflect, and self-correct.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ReasoningPhase(Enum):
    PERCEIVE = "perceive"
    REASON = "reason"
    VERIFY = "verify"
    DISCUSS = "discuss"


class ErrorType(Enum):
    LOGICAL = "logical"
    COMPUTATIONAL = "computational"
    SEMANTIC = "semantic"
    ASSUMPTION = "assumption"
    OVERCONFIDENCE = "overconfidence"


class UncertaintyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class UncertaintyMarker:
    """Tracks where and why the system is uncertain."""

    location: str
    description: str
    level: UncertaintyLevel
    phase: ReasoningPhase
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: str | None = None


@dataclass
class ErrorRecord:
    """Records an error detected during reasoning."""

    error_type: ErrorType
    description: str
    phase_detected: ReasoningPhase
    phase_occurred: ReasoningPhase
    severity: str
    corrected: bool = False
    correction: str | None = None
    why_explanation: str | None = None  # VERIFY's reasoning for why this is an error
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DiscussionTurn:
    """A single turn in the team discussion."""

    speaker: str  # "perceive", "reason", "verify"
    speaker_model: str  # Which LLM
    message: str  # What they said
    stance: str  # "agree", "disagree", "refine", "uncertain"
    reasoning: str  # WHY they hold this position
    references_to: list[str] = field(
        default_factory=list
    )  # Which previous speakers they're responding to
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "speaker_model": self.speaker_model,
            "message": self.message,
            "stance": self.stance,
            "reasoning": self.reasoning,
            "references_to": self.references_to,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TeamDiscussion:
    """Records the collaborative discussion between all three modules."""

    turns: list[DiscussionTurn] = field(default_factory=list)
    consensus_reached: bool = False
    final_team_answer: str | None = None
    dissenting_opinions: list[str] = field(default_factory=list)
    discussion_duration_seconds: float = 0.0

    def add_turn(self, turn: DiscussionTurn) -> None:
        self.turns.append(turn)

    def to_dict(self) -> dict:
        return {
            "turns": [t.to_dict() for t in self.turns],
            "consensus_reached": self.consensus_reached,
            "final_team_answer": self.final_team_answer,
            "dissenting_opinions": self.dissenting_opinions,
            "discussion_duration_seconds": self.discussion_duration_seconds,
        }

    def to_human_readable(self) -> str:
        lines = ["\n" + "", "TEAM DISCUSSION", ""]
        for i, turn in enumerate(self.turns, 1):
            lines.append(f"\n[Turn {i}] {turn.speaker.upper()} ({turn.speaker_model})")
            lines.append(f"  Stance: {turn.stance}")
            lines.append(f"  Message: {turn.message[:300]}...")
            lines.append(f"  WHY: {turn.reasoning[:200]}...")

        lines.append(f"\n  Consensus Reached: {self.consensus_reached}")
        lines.append(f"  Final Team Answer: {self.final_team_answer}")
        if self.dissenting_opinions:
            lines.append(f"  Dissenting Opinions: {len(self.dissenting_opinions)}")
        lines.append(f"  Discussion Duration: {self.discussion_duration_seconds:.1f}s")
        return "\n".join(lines)


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""

    step_id: str
    phase: ReasoningPhase
    actor: str  # Which LLM/module
    action: str  # What was done
    input_data: str  # What it received
    output_data: str  # What it produced
    confidence: float  # 0.0 to 1.0
    duration_seconds: float
    uncertainties: list[UncertaintyMarker] = field(default_factory=list)
    errors_detected: list[ErrorRecord] = field(default_factory=list)
    alternative_paths: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["phase"] = self.phase.value
        d["timestamp"] = self.timestamp.isoformat()
        d["uncertainties"] = [
            {
                **asdict(u),
                "level": u.level.value,
                "phase": u.phase.value,
                "timestamp": u.timestamp.isoformat(),
            }
            for u in self.uncertainties
        ]
        d["errors_detected"] = [
            {
                **asdict(e),
                "error_type": e.error_type.value,
                "phase_detected": e.phase_detected.value,
                "phase_occurred": e.phase_occurred.value,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in self.errors_detected
        ]
        return d


@dataclass
class CognitiveTrace:
    """Complete audit trail of AI reasoning.

    This is the core research artifact - every run produces one of these,
    capturing the full reasoning process for analysis.
    """

    trace_id: str
    problem_id: str
    problem_text: str
    problem_category: str
    problem_difficulty: str
    expected_answer: Any

    steps: list[ReasoningStep] = field(default_factory=list)

    discussion: TeamDiscussion | None = None

    final_answer: Any = None
    final_confidence: float = 0.0
    is_correct: bool | None = None

    total_duration_seconds: float = 0.0
    total_uncertainties: int = 0
    total_errors_detected: int = 0
    total_corrections_made: int = 0
    recovery_rate: float = 0.0  # errors corrected / errors detected

    models_used: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_step(self, step: ReasoningStep) -> None:
        """Add a reasoning step and update metrics."""
        self.steps.append(step)
        self.total_uncertainties += len(step.uncertainties)
        self.total_errors_detected += len(step.errors_detected)
        self.total_corrections_made += sum(1 for e in step.errors_detected if e.corrected)

    def add_discussion(self, discussion: TeamDiscussion) -> None:
        """Add the team discussion."""
        self.discussion = discussion

    def finalize(self) -> None:
        """Calculate final metrics after all steps complete."""
        self.total_duration_seconds = sum(s.duration_seconds for s in self.steps)
        if self.discussion:
            self.total_duration_seconds += self.discussion.discussion_duration_seconds
        if self.total_errors_detected > 0:
            self.recovery_rate = self.total_corrections_made / self.total_errors_detected
        if self.expected_answer is not None and self.final_answer is not None:
            self.is_correct = str(self.final_answer).strip() == str(self.expected_answer).strip()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "trace_id": self.trace_id,
            "problem": {
                "id": self.problem_id,
                "text": self.problem_text,
                "category": self.problem_category,
                "difficulty": self.problem_difficulty,
                "expected_answer": self.expected_answer,
            },
            "reasoning_steps": [s.to_dict() for s in self.steps],
            "discussion": self.discussion.to_dict() if self.discussion else None,
            "outcome": {
                "final_answer": self.final_answer,
                "final_confidence": self.final_confidence,
                "is_correct": self.is_correct,
            },
            "metrics": {
                "total_duration_seconds": self.total_duration_seconds,
                "total_uncertainties": self.total_uncertainties,
                "total_errors_detected": self.total_errors_detected,
                "total_corrections_made": self.total_corrections_made,
                "recovery_rate": self.recovery_rate,
            },
            "models_used": self.models_used,
            "created_at": self.created_at.isoformat(),
        }
        return result

    def to_human_readable(self) -> str:
        """Generate human-readable trace for inspection."""
        lines = [
            "",
            f"COGNITIVE TRACE: {self.trace_id}",
            "",
            f"\nPROBLEM [{self.problem_category} / {self.problem_difficulty}]",
            f"{self.problem_text}",
            f"\nExpected: {self.expected_answer}",
            "\n" + "",
            "REASONING STEPS",
            "",
        ]

        for i, step in enumerate(self.steps, 1):
            lines.append(f"\n[Step {i}] {step.phase.value.upper()} ({step.actor})")
            lines.append(f"  Action: {step.action}")
            lines.append(f"  Confidence: {step.confidence:.2f}")
            lines.append(f"  Duration: {step.duration_seconds:.1f}s")

            if step.uncertainties:
                lines.append(f"  Uncertainties: {len(step.uncertainties)}")
                for u in step.uncertainties:
                    lines.append(f"    • [{u.level.value}] {u.description}")

            if step.errors_detected:
                lines.append(f"  Errors Found: {len(step.errors_detected)}")
                for e in step.errors_detected:
                    status = "[corrected]" if e.corrected else "[uncorrected]"
                    lines.append(f"    • [{e.error_type.value}] {e.description} ({status})")
                    if e.why_explanation:
                        lines.append(f"      WHY: {e.why_explanation}")
                    if e.correction:
                        lines.append(f"      CORRECTION: {e.correction}")

            if step.alternative_paths:
                lines.append(f"  Alternatives: {len(step.alternative_paths)}")

        # Add team discussion if present
        if self.discussion:
            lines.append(self.discussion.to_human_readable())

        lines.extend(
            [
                "\n" + "",
                "OUTCOME",
                "",
                f"\nFinal Answer: {self.final_answer}",
                f"Confidence: {self.final_confidence:.2f}",
                f"Correct: {self.is_correct}",
                "\nMetrics:",
                f"  Duration: {self.total_duration_seconds:.1f}s",
                f"  Uncertainties flagged: {self.total_uncertainties}",
                f"  Errors detected: {self.total_errors_detected}",
                f"  Corrections made: {self.total_corrections_made}",
                f"  Recovery rate: {self.recovery_rate:.1%}",
                "\n" + "",
            ]
        )

        return "\n".join(lines)

    def save(self, output_dir: Path) -> tuple[Path, Path]:
        """Save trace as both JSON and human-readable."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"trace_{self.trace_id}.json"
        txt_path = output_dir / f"trace_{self.trace_id}.txt"

        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        with open(txt_path, "w") as f:
            f.write(self.to_human_readable())

        return json_path, txt_path


def create_trace(problem: dict) -> CognitiveTrace:
    """Factory function to create a new CognitiveTrace."""
    return CognitiveTrace(
        trace_id=datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8],
        problem_id=problem.get("id", "unknown"),
        problem_text=problem.get("problem", ""),
        problem_category=problem.get("category", "unknown"),
        problem_difficulty=problem.get("difficulty", "unknown"),
        expected_answer=problem.get("expected"),
    )

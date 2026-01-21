"""Enhanced Multi-LLM Orchestrator for IMRA-1 Meta-Cognitive Framework.

This orchestrator implements the full IMRA-1 vision:
- PERCEIVE: Generates intuition sketches with explicit uncertainty
- REASON: Executes with step-by-step traces and self-assessment
- VERIFY: Acts as internal critic, builds consensus, prevents overconfidence
- REFLECT: Learns from the deliberation for future improvement

Every decision, sketch, and correction is logged for full auditability.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .meta_prompts import (
    get_perceive_prompt,
    get_reason_prompt,
    get_verify_consensus_prompt,
    get_verify_final_prompt,
)
from .ollama_client import OllamaClient


@dataclass
class Sketch:
    """An intuition sketch from a PERCEIVE agent."""

    author: str
    sketch: dict[str, Any]
    confidence: float = 0.0
    uncertainty_flags: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExecutionResult:
    """Result from REASON executing a sketch."""

    sketch_author: str
    result: dict[str, Any]
    answer: Any = None
    confidence: float = 0.0
    difficulties: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class VerificationResult:
    """Result from VERIFY analyzing sketches or executions."""

    phase: str
    result: dict[str, Any]
    accepted: bool = False
    confidence: float = 0.0
    red_flags: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TraceEvent:
    """A single event in the deliberation trace."""

    phase: str
    actor: str
    payload: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0


@dataclass
class DeliberationTrace:
    """Full trace of a deliberation session — the audit log."""

    problem: str = ""
    events: list[TraceEvent] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""
    total_duration_seconds: float = 0.0

    def add(self, phase: str, actor: str, payload: dict[str, Any], duration: float = 0.0) -> None:
        self.events.append(
            TraceEvent(phase=phase, actor=actor, payload=payload, duration_seconds=duration)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": self.total_duration_seconds,
            "events": [
                {
                    "phase": e.phase,
                    "actor": e.actor,
                    "payload": e.payload,
                    "timestamp": e.timestamp,
                    "duration_seconds": e.duration_seconds,
                }
                for e in self.events
            ],
        }


class MetaCognitiveOrchestrator:
    """Orchestrates multi-LLM deliberation with full meta-cognitive capabilities.

    This is the heart of IMRA-1: an AI system that thinks about its own thinking,
    flags uncertainty, deliberates between alternatives, and learns from outcomes.
    """

    def __init__(self, models: dict[str, str], verbose: bool = True) -> None:
        """Initialize orchestrator with role-to-model mapping.

        Args:
            models: Dict mapping roles to model names, e.g.:
                {
                    "perceive1": "qwen2.5-coder:3b",
                    "perceive2": "qwen2.5-coder:7b",
                    "reason": "codellama:latest",
                    "verify": "qwen2.5-coder:7b"
                }
            verbose: Whether to print progress updates
        """
        self.models = models
        self.clients = {role: OllamaClient(model=name) for role, name in models.items()}
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """Print progress if verbose mode is on."""
        if self.verbose:
            print(f"[IMRA-1] {datetime.now().strftime('%H:%M:%S')} | {message}")

    def _call(self, role: str, prompt: str) -> tuple[str, float]:
        """Call an LLM and return response with duration."""
        self._log(f"Calling {role.upper()} ({self.models[role]})...")
        start = time.time()
        response = self.clients[role].send_prompt(prompt)
        duration = time.time() - start
        self._log(f"  {role.upper()} responded in {duration:.1f}s")
        return response, duration

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling common issues."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            return {"raw_response": text, "parse_error": True}

    def perceive(self, problem: str, trace: DeliberationTrace) -> tuple[Sketch, Sketch]:
        """Generate two intuition sketches for the problem.

        PERCEIVE1 generates an initial sketch.
        PERCEIVE2 generates an alternative, potentially challenging the first.
        """
        self._log("PERCEIVE PHASE: Generating intuition sketches...")

        # First sketch
        prompt1 = get_perceive_prompt(problem)
        raw1, dur1 = self._call("perceive1", prompt1)
        parsed1 = self._parse_json(raw1)

        sketch1 = Sketch(
            author="perceive1",
            sketch=parsed1,
            confidence=(
                float(parsed1.get("confidence", 0.0))
                if isinstance(parsed1.get("confidence"), (int, float))
                else 0.0
            ),
            uncertainty_flags=(
                parsed1.get("uncertainty_flags", []) if isinstance(parsed1, dict) else []
            ),
        )
        trace.add(
            "perceive_sketch",
            "perceive1",
            {"sketch": sketch1.sketch, "confidence": sketch1.confidence},
            dur1,
        )

        # Second sketch (given first as context)
        prompt2 = get_perceive_prompt(problem, first_sketch=sketch1.sketch)
        raw2, dur2 = self._call("perceive2", prompt2)
        parsed2 = self._parse_json(raw2)

        sketch2 = Sketch(
            author="perceive2",
            sketch=parsed2,
            confidence=(
                float(parsed2.get("confidence", 0.0))
                if isinstance(parsed2.get("confidence"), (int, float))
                else 0.0
            ),
            uncertainty_flags=(
                parsed2.get("uncertainty_flags", []) if isinstance(parsed2, dict) else []
            ),
        )
        trace.add(
            "perceive_sketch_alternative",
            "perceive2",
            {"sketch": sketch2.sketch, "confidence": sketch2.confidence},
            dur2,
        )

        self._log(f"  Sketch 1 confidence: {sketch1.confidence:.2f}")
        self._log(f"  Sketch 2 confidence: {sketch2.confidence:.2f}")

        return sketch1, sketch2

    def verify_consensus(
        self, problem: str, sketch1: Sketch, sketch2: Sketch, trace: DeliberationTrace
    ) -> VerificationResult:
        """VERIFY analyzes both sketches and builds consensus.

        This is the "self-doubt" mechanism — challenging both approaches.
        """
        self._log("VERIFY PHASE: Building consensus...")

        prompt = get_verify_consensus_prompt(problem, sketch1.sketch, sketch2.sketch)
        raw, dur = self._call("verify", prompt)
        parsed = self._parse_json(raw)

        result = VerificationResult(
            phase="consensus",
            result=parsed,
            accepted=True,
            confidence=(
                float(parsed.get("overall_confidence", 0.0))
                if isinstance(parsed.get("overall_confidence"), (int, float))
                else 0.0
            ),
            red_flags=parsed.get("red_flags", []) if isinstance(parsed, dict) else [],
        )
        trace.add(
            "verify_consensus",
            "verify",
            {"analysis": parsed, "confidence": result.confidence, "red_flags": result.red_flags},
            dur,
        )

        if result.red_flags:
            self._log(f"  RED FLAGS: {result.red_flags}")

        return result

    def reason_execute(
        self, problem: str, sketch: Sketch, trace: DeliberationTrace
    ) -> ExecutionResult:
        """REASON executes the chosen sketch step-by-step."""
        self._log(f"REASON PHASE: Executing sketch from {sketch.author}...")

        prompt = get_reason_prompt(problem, sketch.sketch)
        raw, dur = self._call("reason", prompt)
        parsed = self._parse_json(raw)

        result = ExecutionResult(
            sketch_author=sketch.author,
            result=parsed,
            answer=parsed.get("final_answer") if isinstance(parsed, dict) else None,
            confidence=(
                float(parsed.get("answer_confidence", 0.0))
                if isinstance(parsed.get("answer_confidence"), (int, float))
                else 0.0
            ),
            difficulties=(
                parsed.get("difficulties_encountered", []) if isinstance(parsed, dict) else []
            ),
        )
        trace.add(
            "reason_execute",
            "reason",
            {
                "sketch_author": sketch.author,
                "execution": parsed,
                "final_answer": result.answer,
                "confidence": result.confidence,
            },
            dur,
        )

        if result.difficulties:
            self._log(f"  Difficulties encountered: {len(result.difficulties)}")

        return result

    def verify_final(self, problem: str, trace: DeliberationTrace) -> VerificationResult:
        """Final verification of the entire deliberation."""
        self._log("VERIFY PHASE: Final verification...")

        prompt = get_verify_final_prompt(problem, trace.to_dict())
        raw, dur = self._call("verify", prompt)
        parsed = self._parse_json(raw)

        accepted = False
        if isinstance(parsed, dict):
            answer_verif = parsed.get("answer_verification", {})
            accepted = (
                answer_verif.get("answer_accepted", False)
                if isinstance(answer_verif, dict)
                else False
            )

        result = VerificationResult(
            phase="final",
            result=parsed,
            accepted=accepted,
            confidence=(
                float(parsed.get("final_confidence", 0.0))
                if isinstance(parsed.get("final_confidence"), (int, float))
                else 0.0
            ),
            red_flags=[],
        )
        trace.add(
            "verify_final",
            "verify",
            {"verification": parsed, "accepted": result.accepted, "confidence": result.confidence},
            dur,
        )

        grade = "?"
        if isinstance(parsed, dict):
            dq = parsed.get("deliberation_quality", {})
            if isinstance(dq, dict):
                grade = dq.get("overall_grade", "?")

        self._log(f"  Deliberation grade: {grade}")
        self._log(f"  Answer accepted: {result.accepted}")

        return result

    def run_deliberation(self, problem: str, max_iterations: int = 1) -> dict[str, Any]:
        """Run a full meta-cognitive deliberation on the problem.

        This is the main entry point. It:
        1. Generates diverse intuition sketches (PERCEIVE)
        2. Builds consensus and flags concerns (VERIFY)
        3. Executes the best approach (REASON)
        4. Performs final verification (VERIFY)
        5. Returns full audit trace

        Args:
            problem: The problem to solve
            max_iterations: Max deliberation cycles (for revision loops)

        Returns:
            Dict with trace, final answer, and meta-learning insights
        """
        self._log("")
        self._log("IMRA-1 META-COGNITIVE DELIBERATION")
        self._log("")
        self._log(f"Problem: {problem[:100]}...")

        start_time = time.time()
        trace = DeliberationTrace(problem=problem)

        # Phase 1: PERCEIVE - Generate intuition sketches
        sketch1, sketch2 = self.perceive(problem, trace)

        # Phase 2: VERIFY - Build consensus
        consensus = self.verify_consensus(problem, sketch1, sketch2, trace)

        # Phase 3: REASON - Execute best approach
        # Choose sketch based on consensus
        chosen_sketch = sketch1
        if isinstance(consensus.result, dict):
            rec = consensus.result.get("recommended_approach", {})
            if isinstance(rec, dict) and rec.get("primary_sketch") == "sketch2":
                chosen_sketch = sketch2
            elif isinstance(rec, dict) and rec.get("primary_sketch") == "hybrid":
                self._log("  Using hybrid approach (defaulting to sketch1 as base)")

        execution = self.reason_execute(problem, chosen_sketch, trace)

        # Phase 4: VERIFY - Final verification
        final_verify = self.verify_final(problem, trace)

        # Complete trace
        total_duration = time.time() - start_time
        trace.end_time = datetime.now().isoformat()
        trace.total_duration_seconds = total_duration

        self._log("")
        self._log(f"DELIBERATION COMPLETE in {total_duration:.1f}s")
        self._log(f"Final answer: {execution.answer}")
        self._log(f"Final confidence: {final_verify.confidence:.2f}")
        self._log("")

        return {
            "problem": problem,
            "final_answer": execution.answer,
            "final_confidence": final_verify.confidence,
            "answer_accepted": final_verify.accepted,
            "trace": trace.to_dict(),
            "meta_learning": (
                final_verify.result.get("meta_learning", {})
                if isinstance(final_verify.result, dict)
                else {}
            ),
            "deliberation_quality": (
                final_verify.result.get("deliberation_quality", {})
                if isinstance(final_verify.result, dict)
                else {}
            ),
        }


def run_olympiad_test(
    orchestrator: MetaCognitiveOrchestrator, problem: str, expected: Any = None
) -> dict[str, Any]:
    """Run a single olympiad-level test problem.

    Args:
        orchestrator: The meta-cognitive orchestrator
        problem: The problem statement
        expected: Expected answer (optional, for validation)

    Returns:
        Dict with result and comparison to expected
    """
    result = orchestrator.run_deliberation(problem)

    outcome = {"problem": problem, "result": result, "expected": expected}

    if expected is not None:
        outcome["matches_expected"] = str(result.get("final_answer")) == str(expected)

    return outcome

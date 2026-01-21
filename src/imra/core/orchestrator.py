#!/usr/bin/env python3

"""IMRA-1 Orchestrator

Coordinates the four reasoning layers and manages the full cognitive loop:
PERCEIVE → REASON → VERIFY → DISCUSS → (optional REFLECT & RETRY)

This is where the magic happens - structured reasoning with full auditability.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .gestalt import CoherenceAnalysis, analyze_trace
from .layers import (
    DiscussLayer,
    LayerConfig,
    LayerOutput,
    PerceiveLayer,
    ReasonLayer,
    ResolveLayer,
    VerifyLayer,
)
from .trace import CognitiveTrace, create_trace


@dataclass
class OrchestratorConfig:
    """Configuration for the reasoning orchestrator."""

    perceive_model: str
    reason_model: str
    verify_model: str
    output_dir: Path = Path("research_data/traces")
    max_retries: int = 2
    verify_threshold: float = 0.6  # Below this, retry
    enable_reflection: bool = True
    enable_discussion: bool = True
    timeout_seconds: float = 300.0


class ReasoningOrchestrator:
    """Orchestrates the four-layer reasoning process.

    The orchestrator:
    1. Runs PERCEIVE to get initial analysis
    2. Runs REASON to solve the problem
    3. Runs VERIFY to check the solution
    4. Runs DISCUSS for team collaboration
    5. Optionally retries if confidence is low
    6. Logs everything in CognitiveTrace
    """

    def __init__(self, config: OrchestratorConfig, llm_client: Any) -> None:
        self.config = config
        self.llm = llm_client

        self.perceive = PerceiveLayer(LayerConfig(model_name=config.perceive_model), llm_client)
        self.reason = ReasonLayer(LayerConfig(model_name=config.reason_model), llm_client)
        self.verify = VerifyLayer(LayerConfig(model_name=config.verify_model), llm_client)
        self.discuss = DiscussLayer(
            LayerConfig(model_name=config.perceive_model),
            LayerConfig(model_name=config.reason_model),
            LayerConfig(model_name=config.verify_model),
            llm_client,
        )
        self.resolve = ResolveLayer(LayerConfig(model_name=config.verify_model), llm_client)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def solve(self, problem: dict) -> CognitiveTrace:
        """Run the full reasoning loop on a problem.

        Args:
            problem: Dict with 'problem', 'expected', 'category', etc.

        Returns:
            CognitiveTrace with full audit trail
        """
        trace = create_trace(problem)
        trace.models_used = {
            "perceive": self.config.perceive_model,
            "reason": self.config.reason_model,
            "verify": self.config.verify_model,
        }

        print(f"\nSOLVING: {problem.get('id', 'unknown')}")

        print("\n[PERCEIVE] Analyzing problem...")
        perceive_output = self.perceive.process({"problem": problem["problem"]}, trace)
        print(f"  [+] Confidence: {perceive_output.confidence:.2f}")
        print(f"  [+] Alternatives: {len(perceive_output.alternatives)}")
        print(f"  [+] Uncertainties: {len(perceive_output.uncertainties)}")

        print("\n[REASON] Solving...")
        reason_output = self.reason.process(
            {"problem": problem["problem"], "perceive_output": perceive_output.content}, trace
        )

        proposed_answer = reason_output.raw_response.get("answer", "")
        print(f"  [+] Proposed answer: {proposed_answer}")
        print(f"  [+] Confidence: {reason_output.confidence:.2f}")

        print("\n[VERIFY] Checking solution...")
        verify_output = self.verify.process(
            {
                "problem": problem["problem"],
                "expected": problem.get("expected", "unknown"),
                "perceive_output": perceive_output.content,
                "reason_output": reason_output.content,
                "proposed_answer": proposed_answer,
            },
            trace,
        )

        is_valid = verify_output.raw_response.get("is_valid", False)
        corrected = verify_output.raw_response.get("corrected_answer")

        print(f"  [+] Valid: {is_valid}")
        print(f"  [+] Errors found: {len(verify_output.errors)}")
        if corrected:
            print(f"  [+] Corrected to: {corrected}")

        final_answer = corrected if corrected else proposed_answer
        final_confidence = verify_output.confidence

        # DISCUSSION (Team collaboration)

        discussion = None
        if self.config.enable_discussion:
            discussion = self.discuss.facilitate_discussion(
                problem_text=problem["problem"],
                perceive_output=perceive_output.content,
                reason_output=reason_output.content,
                verify_output=verify_output.content,
                proposed_answer=proposed_answer,
                trace=trace,
            )
            trace.add_discussion(discussion)

            # RESOLVE (Force concrete answer after discussion)
            # This breaks the "retreat to methodology" pattern

            resolved_answer, resolved_confidence, _ = self.resolve.resolve(
                problem_text=problem["problem"],
                proposed_answer=proposed_answer,
                discussion=discussion,
                trace=trace,
            )

            final_answer = resolved_answer
            final_confidence = resolved_confidence

            print(f"  [+] Final resolved answer: {final_answer[:100]}")
        else:
            if (
                discussion
                and discussion.final_team_answer
                and discussion.final_team_answer != proposed_answer
            ):
                final_answer = discussion.final_team_answer
                print(f"  [+] Team revised answer to: {final_answer}")

        retry_count = 0
        while (
            final_confidence < self.config.verify_threshold
            and retry_count < self.config.max_retries
            and self.config.enable_reflection
        ):
            retry_count += 1
            print(
                f"\n[REFLECT] Confidence too low ({final_confidence:.2f}), retry {retry_count}..."
            )

            reason_output = self.reason.process(
                {
                    "problem": problem["problem"],
                    "perceive_output": perceive_output.content,
                    "previous_attempt": reason_output.content,
                    "verification_feedback": verify_output.content,
                },
                trace,
            )

            proposed_answer = reason_output.raw_response.get("answer", "")

            verify_output = self.verify.process(
                {
                    "problem": problem["problem"],
                    "expected": problem.get("expected", "unknown"),
                    "perceive_output": perceive_output.content,
                    "reason_output": reason_output.content,
                    "proposed_answer": proposed_answer,
                },
                trace,
            )

            corrected = verify_output.raw_response.get("corrected_answer")
            final_answer = corrected if corrected else proposed_answer
            final_confidence = verify_output.confidence

        trace.final_answer = final_answer
        trace.final_confidence = final_confidence
        trace.finalize()

        print("\n[GESTALT] Computing coherence score...")
        coherence = analyze_trace(trace)

        original_confidence = trace.final_confidence

        calibrated_confidence = original_confidence * coherence.coherence_score
        trace.final_confidence = calibrated_confidence

        if coherence.coherence_score < 0.4 and not getattr(trace, "_coherence_retry_done", False):
            print(
                f"\n[METACOGNITION] Low coherence ({coherence.coherence_score:.3f}), attempting reflection..."
            )
            trace._coherence_retry_done = True  # Prevent infinite loops

            retry_trace = self._retry_with_coherence_feedback(
                problem, trace, coherence, perceive_output
            )

            if retry_trace:
                retry_coherence = analyze_trace(retry_trace)

                if retry_coherence.coherence_score > coherence.coherence_score:
                    print(
                        f"  [+] Retry improved coherence: {coherence.coherence_score:.3f} -> {retry_coherence.coherence_score:.3f}"
                    )
                    trace = retry_trace
                    coherence = retry_coherence
                    calibrated_confidence = trace.final_confidence * coherence.coherence_score
                    trace.final_confidence = calibrated_confidence
                else:
                    print(
                        f"  [-] Retry did not improve ({retry_coherence.coherence_score:.3f}), keeping original"
                    )

        if coherence.coherence_score < 0.25:
            print(
                f"\n[METACOGNITION] Very low coherence ({coherence.coherence_score:.3f}), marking as uncertain"
            )
            trace.final_answer = f"UNCERTAIN: {trace.final_answer}"
            trace.final_confidence = min(trace.final_confidence, 0.2)

        if coherence.failure_mode.value in ["verification_paradox", "circular_reasoning"]:
            print(
                f"\n[METACOGNITION] Critical failure mode detected: {coherence.failure_mode.value}"
            )
            trace.final_confidence = min(trace.final_confidence, 0.3)

        trace.coherence_score = coherence.coherence_score
        trace.coherence_level = coherence.coherence_level.value
        trace.failure_mode = coherence.failure_mode.value
        trace.confidence_calibrated = True
        trace.original_confidence = original_confidence

        print("\nRESULT")
        print(f"  Final Answer: {trace.final_answer}")
        print(f"  Expected: {trace.expected_answer}")
        print(f"  Correct: {trace.is_correct}")
        print(f"  Raw Confidence: {original_confidence:.2f}")
        print(f"  Calibrated Confidence: {trace.final_confidence:.2f}")
        print(f"  Duration: {trace.total_duration_seconds:.1f}s")
        print(f"  Errors detected: {trace.total_errors_detected}")
        print(f"  Corrections: {trace.total_corrections_made}")
        print(f"  Recovery rate: {trace.recovery_rate:.1%}")
        print(
            f"\n  COHERENCE: {coherence.coherence_score:.3f} ({coherence.coherence_level.value.upper()})"
        )
        print(f"  Diagnosis: {coherence.failure_mode.value.upper().replace('_', ' ')}")

        json_path, txt_path = trace.save(self.config.output_dir)

        coherence_path = self.config.output_dir / f"coherence_{trace.trace_id}.txt"
        with open(coherence_path, "w") as f:
            f.write(coherence.to_human_readable())

        coherence_json_path = self.config.output_dir / f"coherence_{trace.trace_id}.json"
        with open(coherence_json_path, "w") as f:
            json.dump(coherence.to_dict(), f, indent=2)

        print(f"\n  Saved: {json_path.name}")
        print(f"  Coherence: {coherence_path.name}")

        return trace

    def _retry_with_coherence_feedback(
        self,
        problem: dict,
        failed_trace: CognitiveTrace,
        coherence: CoherenceAnalysis,
        perceive_output: LayerOutput,
    ) -> CognitiveTrace | None:
        """Retry solving with explicit feedback about coherence failures.

        This is the heart of metacognition - using self-diagnosis to improve.

        Args:
            problem: The original problem dict
            failed_trace: The trace from the failed attempt
            coherence: The coherence analysis showing what went wrong
            perceive_output: Original perception (reused to save time)

        Returns:
            New trace if retry succeeded, None if it failed
        """
        try:
            print(f"  Failure mode: {coherence.failure_mode.value}")
            print("  Attempting targeted retry...")

            failure_guidance = self._get_failure_guidance(coherence)

            retry_trace = create_trace(problem)
            retry_trace.models_used = failed_trace.models_used
            retry_trace._coherence_retry_done = True

            reason_output = self.reason.process(
                {
                    "problem": problem["problem"],
                    "perceive_output": perceive_output.content,
                    "previous_attempt": failed_trace.final_answer,
                    "reflection": failure_guidance,
                },
                retry_trace,
            )

            proposed_answer = reason_output.raw_response.get("answer", "")

            verify_output = self.verify.process(
                {
                    "problem": problem["problem"],
                    "expected": problem.get("expected", "unknown"),
                    "perceive_output": perceive_output.content,
                    "reason_output": reason_output.content,
                    "proposed_answer": proposed_answer,
                },
                retry_trace,
            )

            corrected = verify_output.raw_response.get("corrected_answer")
            final_answer = corrected if corrected else proposed_answer

            retry_trace.final_answer = final_answer
            retry_trace.final_confidence = verify_output.confidence
            retry_trace.finalize()

            return retry_trace

        except Exception as e:
            print(f"  Retry failed: {e}")
            return None

    def _get_failure_guidance(self, coherence: CoherenceAnalysis) -> str:
        """Generate specific guidance based on detected failure mode."""
        mode = coherence.failure_mode.value

        guidance_map = {
            "verification_paradox": """
                CRITICAL: Your verification said the answer was wrong but found no specific errors.
                This is contradictory. Either:
                1. Find the SPECIFIC error in your reasoning, or
                2. Accept that the answer may be correct
                Do not say "invalid" without pointing to the exact mistake.
            """,
            "circular_reasoning": """
                CRITICAL: Your reasoning is circular - you're using the conclusion to prove itself.
                Break the loop by:
                1. Starting from first principles
                2. Each step must follow from previous steps, not from the desired answer
                3. Check: could your reasoning prove a DIFFERENT answer? If so, it's circular.
            """,
            "incomplete_reasoning": """
                Your reasoning has gaps. Make sure:
                1. Every step is justified
                2. You reach a CONCRETE answer (a number, yes/no, a specific value)
                3. Don't stop at "it depends" or "further analysis needed"
            """,
            "meta_retreat": """
                You retreated to discussing methodology instead of solving the problem.
                STOP explaining HOW to solve it and actually SOLVE it.
                Give a concrete, specific answer.
            """,
            "healthy": """
                The reasoning structure looks okay but the answer may still be wrong.
                Double-check your calculations and assumptions.
            """,
        }

        return guidance_map.get(mode, guidance_map["healthy"]).strip()


class ExperimentRunner:
    """Runs controlled experiments across problem sets.

    Supports:
    - Multiple problem sets
    - Variable LLM configurations
    - Aggregate metrics collection
    """

    def __init__(self, orchestrator: ReasoningOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.traces: list[CognitiveTrace] = []

    def run_problems(self, problems: list[dict]) -> dict:
        """Run all problems and collect metrics."""
        print(f"\nEXPERIMENT: {len(problems)} problems")

        start_time = time.time()

        for i, problem in enumerate(problems, 1):
            print(f"\n[{i}/{len(problems)}] {problem.get('id', 'unknown')}")
            try:
                trace = self.orchestrator.solve(problem)
                self.traces.append(trace)
            except Exception as e:
                print(f"  ERROR: {e}")

        total_time = time.time() - start_time

        return self._compute_metrics(total_time)

    def _compute_metrics(self, total_time: float) -> dict:
        """Compute aggregate metrics from all traces."""
        if not self.traces:
            return {}

        correct = sum(1 for t in self.traces if t.is_correct)
        total = len(self.traces)

        high_conf_correct = sum(1 for t in self.traces if t.final_confidence > 0.8 and t.is_correct)
        high_conf_total = sum(1 for t in self.traces if t.final_confidence > 0.8)

        errors_detected = sum(t.total_errors_detected for t in self.traces)
        errors_corrected = sum(t.total_corrections_made for t in self.traces)

        by_category = {}
        for t in self.traces:
            cat = t.problem_category
            if cat not in by_category:
                by_category[cat] = {"correct": 0, "total": 0}
            by_category[cat]["total"] += 1
            if t.is_correct:
                by_category[cat]["correct"] += 1

        metrics = {
            "total_problems": total,
            "correct_answers": correct,
            "accuracy": correct / total if total > 0 else 0,
            "total_time_seconds": total_time,
            "avg_time_per_problem": total_time / total if total > 0 else 0,
            "confidence_calibration": {
                "high_confidence_accuracy": (
                    high_conf_correct / high_conf_total if high_conf_total > 0 else 0
                ),
                "high_confidence_count": high_conf_total,
            },
            "error_metrics": {
                "total_detected": errors_detected,
                "total_corrected": errors_corrected,
                "overall_recovery_rate": (
                    errors_corrected / errors_detected if errors_detected > 0 else 0
                ),
            },
            "by_category": {
                cat: {
                    **data,
                    "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
                }
                for cat, data in by_category.items()
            },
        }

        return metrics

    def save_experiment(self, name: str, output_dir: Path) -> Path:
        """Save complete experiment results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"experiment_{name}_{timestamp}.json"

        results = {
            "experiment_name": name,
            "timestamp": timestamp,
            "config": {
                "perceive_model": self.orchestrator.config.perceive_model,
                "reason_model": self.orchestrator.config.reason_model,
                "verify_model": self.orchestrator.config.verify_model,
            },
            "metrics": self._compute_metrics(0),
            "traces": [t.trace_id for t in self.traces],
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        return filepath

"""Full PERCEIVE → REASON → VERIFY deliberation loop with learning signals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from imra.perceive.engines.base import PerceiveEngine
from imra.perceive.sketches import PerceptionRequest, ProgramSketch
from imra.reason.executor import ReasonExecutor, ReasonResult
from imra.tracing.schemas import TraceBuffer, TraceEvent
from imra.utils.compat import utc_now
from imra.verify.critics.base import VerifyCritic


@dataclass
class LoopResult:
    """Full result from a deliberation run."""

    success: bool
    confidence: float
    trace: TraceBuffer
    final_value: Any = None
    iterations: int = 0
    sketches_explored: int = 0
    accepted_sketch: ProgramSketch | None = None
    gradients: dict[str, float] = field(default_factory=dict)


@dataclass
class DeliberationConfig:
    """Configuration for deliberation behavior."""

    max_iterations: int = 10
    max_sketches_per_iteration: int = 5
    confidence_threshold: float = 0.7
    refinement_budget: int = 3
    collect_all_traces: bool = True
    variables: dict[str, float] = field(default_factory=dict)


class DifferentiableDeliberationLoop:
    """
    Orchestrates the full reasoning cycle:
    1. PERCEIVE proposes sketches
    2. REASON executes and computes gradients
    3. VERIFY scores and decides accept/refine/rollback
    4. Learning signals flow back to PERCEIVE
    """

    def __init__(
        self,
        perceive: PerceiveEngine,
        reason: ReasonExecutor,
        verify: VerifyCritic,
        config: DeliberationConfig | None = None,
        on_trace_complete: Callable[[TraceBuffer], None] | None = None,
    ) -> None:
        self.perceive = perceive
        self.reason = reason
        self.verify = verify
        self.config = config or DeliberationConfig()
        self.on_trace_complete = on_trace_complete
        self._all_traces: list[TraceBuffer] = []

    def run(self, request: PerceptionRequest) -> LoopResult:
        """Execute full deliberation loop."""
        created_at = request.metadata.get("created_at") or utc_now()
        trace = TraceBuffer(task_id=request.task_id, created_at=created_at)

        iterations = 0
        sketches_explored = 0
        best_result: ReasonResult | None = None
        best_sketch: ProgramSketch | None = None
        best_confidence = 0.0
        refinement_count = 0

        trace.append(
            TraceEvent(
                kind="loop.start",
                payload={
                    "task_id": request.task_id,
                    "max_iterations": self.config.max_iterations,
                    "confidence_threshold": self.config.confidence_threshold,
                },
            )
        )

        while iterations < self.config.max_iterations:
            iterations += 1

            trace.append(
                TraceEvent(
                    kind="loop.iteration",
                    payload={"iteration": iterations},
                )
            )

            # PERCEIVE: generate sketches
            sketches = list(self.perceive.propose(request))[
                : self.config.max_sketches_per_iteration
            ]

            for sketch in sketches:
                sketches_explored += 1

                trace.append(
                    TraceEvent(
                        kind="perceive.sketch",
                        payload={
                            "sketch_id": sketch.sketch_id,
                            "prior": sketch.prior,
                            "structure": sketch.structure,
                            "node_count": len(sketch.nodes),
                        },
                    )
                )

                # REASON: execute sketch
                variables = {**self.config.variables, **request.variables}
                result = self.reason.run(sketch, trace, variables=variables)

                # VERIFY: score the trace
                confidence = self.verify.score(trace)
                directive = self.verify.directive(trace)

                trace.append(
                    TraceEvent(
                        kind="verify.score",
                        payload={
                            "confidence": confidence,
                            "directive": directive,
                            "sketch_id": sketch.sketch_id,
                        },
                    )
                )

                # Track best result
                if result.success and confidence > best_confidence:
                    best_result = result
                    best_sketch = sketch
                    best_confidence = confidence

                # Handle directive
                if directive == "accept" and confidence >= self.config.confidence_threshold:
                    trace.append(
                        TraceEvent(
                            kind="loop.directive",
                            payload={"action": "accept", "final_confidence": confidence},
                        )
                    )

                    # Send positive reward to PERCEIVE
                    self.perceive.update(sketch, reward=confidence)

                    if self.on_trace_complete:
                        self.on_trace_complete(trace)

                    return LoopResult(
                        success=True,
                        confidence=confidence,
                        trace=trace,
                        final_value=result.value,
                        iterations=iterations,
                        sketches_explored=sketches_explored,
                        accepted_sketch=sketch,
                        gradients=result.gradients,
                    )

                elif directive == "refine" and refinement_count < self.config.refinement_budget:
                    refinement_count += 1
                    trace.append(
                        TraceEvent(
                            kind="loop.directive",
                            payload={"action": "refine", "refinement_count": refinement_count},
                        )
                    )
                    # Negative reward to discourage this sketch pattern
                    self.perceive.update(sketch, reward=-0.1)
                    # Continue to try more sketches

                elif directive == "rollback":
                    trace.append(
                        TraceEvent(
                            kind="loop.directive",
                            payload={"action": "rollback", "reason": "verify_rejected"},
                        )
                    )
                    # Strong negative reward
                    self.perceive.update(sketch, reward=-0.5)

            # If no sketch was accepted this iteration, check if we should stop
            if best_confidence >= self.config.confidence_threshold * 0.9:
                # Close enough, accept best
                break

        # Loop exhausted without definitive accept
        trace.append(
            TraceEvent(
                kind="loop.exhausted",
                payload={
                    "iterations": iterations,
                    "sketches_explored": sketches_explored,
                    "best_confidence": best_confidence,
                },
            )
        )

        if self.on_trace_complete:
            self.on_trace_complete(trace)

        if self.config.collect_all_traces:
            self._all_traces.append(trace)

        return LoopResult(
            success=best_result is not None and best_result.success,
            confidence=best_confidence,
            trace=trace,
            final_value=best_result.value if best_result else None,
            iterations=iterations,
            sketches_explored=sketches_explored,
            accepted_sketch=best_sketch,
            gradients=best_result.gradients if best_result else {},
        )

    def get_all_traces(self) -> list[TraceBuffer]:
        """Return all collected traces for analysis."""
        return self._all_traces.copy()

    def clear_traces(self) -> None:
        """Clear collected traces."""
        self._all_traces.clear()


class BatchDeliberationRunner:
    """Run deliberation over multiple problems and collect statistics."""

    def __init__(self, loop: DifferentiableDeliberationLoop) -> None:
        self.loop = loop
        self.results: list[LoopResult] = []

    def run_batch(self, requests: list[PerceptionRequest]) -> dict:
        """Execute batch and return aggregate statistics."""
        self.results.clear()

        for req in requests:
            result = self.loop.run(req)
            self.results.append(result)

        successes = sum(1 for r in self.results if r.success)
        total_iterations = sum(r.iterations for r in self.results)
        total_sketches = sum(r.sketches_explored for r in self.results)
        avg_confidence = (
            sum(r.confidence for r in self.results) / len(self.results) if self.results else 0
        )

        return {
            "total_problems": len(requests),
            "successes": successes,
            "success_rate": successes / len(requests) if requests else 0,
            "total_iterations": total_iterations,
            "avg_iterations": total_iterations / len(requests) if requests else 0,
            "total_sketches": total_sketches,
            "avg_sketches": total_sketches / len(requests) if requests else 0,
            "avg_confidence": avg_confidence,
        }

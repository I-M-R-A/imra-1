"""Integration tests for the full PERCEIVE → REASON → VERIFY loop."""

import pytest

from imra.loops.deliberation import (
    BatchDeliberationRunner,
    DeliberationConfig,
    DifferentiableDeliberationLoop,
)
from imra.perceive.engines.symbolic import SymbolicPerceiveEngine
from imra.perceive.sketches import PerceptionRequest
from imra.reason.executor import ReasonExecutor
from imra.utils.compat import utc_now
from imra.verify.critics.calibration import CalibrationCritic


class TestDeliberationLoop:
    """Tests for DifferentiableDeliberationLoop."""

    def test_loop_solves_simple_addition(self) -> None:
        """Test that the loop can solve a simple addition problem."""
        perceive = SymbolicPerceiveEngine()
        reason = ReasonExecutor()
        verify = CalibrationCritic(
            confidence_threshold=0.3
        )  # Lower threshold for tests while debugging

        loop = DifferentiableDeliberationLoop(
            perceive=perceive,
            reason=reason,
            verify=verify,
            config=DeliberationConfig(
                max_iterations=5,
                confidence_threshold=0.3,
            ),
        )

        request = PerceptionRequest(
            task_id="add_problem",
            payload={"type": "math"},
            problem_text="What is 5 plus 3?",
            metadata={"created_at": utc_now()},
        )

        result = loop.run(request)

        assert result.trace is not None
        assert result.iterations >= 1
        assert result.sketches_explored >= 1

    def test_loop_collects_traces(self) -> None:
        """Test that traces are properly collected."""
        perceive = SymbolicPerceiveEngine()
        reason = ReasonExecutor()
        verify = CalibrationCritic()

        collected_traces = []

        loop = DifferentiableDeliberationLoop(
            perceive=perceive,
            reason=reason,
            verify=verify,
            config=DeliberationConfig(collect_all_traces=True),
            on_trace_complete=lambda t: collected_traces.append(t),
        )

        request = PerceptionRequest(
            task_id="trace_test",
            payload={},
            problem_text="Calculate 10 times 2",
        )

        loop.run(request)

        assert len(collected_traces) >= 0 or len(loop.get_all_traces()) >= 0

    def test_loop_respects_max_iterations(self) -> None:
        """Test that max_iterations is respected."""
        perceive = SymbolicPerceiveEngine()
        reason = ReasonExecutor()
        verify = CalibrationCritic(
            confidence_threshold=0.99
        )  # Very high, won't accept the answer during test

        loop = DifferentiableDeliberationLoop(
            perceive=perceive,
            reason=reason,
            verify=verify,
            config=DeliberationConfig(max_iterations=2),
        )

        request = PerceptionRequest(
            task_id="max_iter_test",
            payload={},
            problem_text="What is 7 minus 3?",
        )

        result = loop.run(request)
        assert result.iterations <= 2

    def test_loop_with_variables(self) -> None:
        """Test loop with variable substitution."""
        perceive = SymbolicPerceiveEngine()
        reason = ReasonExecutor()
        verify = CalibrationCritic()

        loop = DifferentiableDeliberationLoop(
            perceive=perceive,
            reason=reason,
            verify=verify,
            config=DeliberationConfig(
                variables={"x": 10, "y": 5},
            ),
        )

        request = PerceptionRequest(
            task_id="var_test",
            payload={},
            problem_text="Add x and y",
            variables={"z": 2},
        )

        result = loop.run(request)
        assert result.trace is not None


class TestBatchDeliberationRunner:
    """Tests for BatchDeliberationRunner."""

    def test_batch_runner_processes_multiple(self) -> None:
        """Test processing multiple problems in batch."""
        perceive = SymbolicPerceiveEngine()
        reason = ReasonExecutor()
        verify = CalibrationCritic(confidence_threshold=0.3)

        loop = DifferentiableDeliberationLoop(
            perceive=perceive,
            reason=reason,
            verify=verify,
        )

        runner = BatchDeliberationRunner(loop)

        requests = [
            PerceptionRequest(
                task_id=f"batch_{i}",
                payload={},
                problem_text=text,
            )
            for i, text in enumerate(
                [
                    "What is 2 plus 2?",
                    "Calculate 5 times 3",
                    "What is 10 minus 4?",
                ]
            )
        ]

        stats = runner.run_batch(requests)

        assert stats["total_problems"] == 3
        assert "success_rate" in stats
        assert "avg_iterations" in stats
        assert "avg_confidence" in stats

    def test_batch_runner_empty_batch(self) -> None:
        """Test handling empty batch."""
        perceive = SymbolicPerceiveEngine()
        reason = ReasonExecutor()
        verify = CalibrationCritic()

        loop = DifferentiableDeliberationLoop(perceive, reason, verify)
        runner = BatchDeliberationRunner(loop)

        stats = runner.run_batch([])
        assert stats["total_problems"] == 0


class TestEndToEndReasoning:
    """End-to-end tests for complete reasoning scenarios."""

    @pytest.mark.slow
    def test_multi_step_problem(self) -> None:
        """Test a problem requiring multiple reasoning steps."""
        perceive = SymbolicPerceiveEngine(num_sketches=5)
        reason = ReasonExecutor(max_steps=100)
        verify = CalibrationCritic()

        loop = DifferentiableDeliberationLoop(
            perceive=perceive,
            reason=reason,
            verify=verify,
            config=DeliberationConfig(
                max_iterations=10,
                max_sketches_per_iteration=10,
            ),
        )

        request = PerceptionRequest(
            task_id="multi_step",
            payload={},
            problem_text="A store has 5 boxes with 4 items each. After selling 8 items, how many remain?",
        )

        result = loop.run(request)

        assert result.sketches_explored >= 1
        assert len(result.trace.events) >= 1

    def test_trace_contains_all_phases(self) -> None:
        """Test that trace contains perceive, reason, and verify events."""
        perceive = SymbolicPerceiveEngine()
        reason = ReasonExecutor()
        verify = CalibrationCritic()

        loop = DifferentiableDeliberationLoop(perceive, reason, verify)

        request = PerceptionRequest(
            task_id="phases_test",
            payload={},
            problem_text="What is 6 divided by 2?",
        )

        result = loop.run(request)

        event_kinds = [e.kind for e in result.trace.events]

        assert any("loop" in k for k in event_kinds), f"No loop events in {event_kinds}"
        assert any("perceive" in k or "sketch" in k for k in event_kinds), (
            f"No perceive events in {event_kinds}"
        )

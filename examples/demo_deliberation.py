#!/usr/bin/env python

"""
IMRA-1 Demo: Full Deliberation Loop

This script demonstrates the complete PERCEIVE -> REASON -> VERIFY
cycle solving arithmetic word problems.
"""

from __future__ import annotations

from imra.loops import BatchDeliberationRunner, DeliberationConfig, DifferentiableDeliberationLoop
from imra.perceive import PerceptionRequest, SymbolicPerceiveEngine
from imra.reason import ReasonExecutor
from imra.verify import CalibrationCritic


def print_trace_summary(result) -> None:
    """Print a readable summary of the reasoning trace."""
    print("\nTRACE SUMMARY")

    for event in result.trace.events:
        if event.kind == "perceive.sketch":
            sketch_id = event.payload.get("sketch_id", "?")
            prior = event.payload.get("prior", 0)
            print(f"\n PERCEIVE: Generated sketch '{sketch_id}' (prior={prior:.3f})")

        elif event.kind == "reason.step":
            op = event.payload.get("op", "?")
            inputs = event.payload.get("inputs", [])
            output = event.payload.get("output", "?")
            node_type = event.payload.get("type", "?")
            if node_type == "literal":
                print(f"    Literal: {output}")
            elif node_type == "variable":
                name = event.payload.get("name", "?")
                print(f"    Variable {name} = {output}")
            else:
                print(f"      {op.upper()}({inputs}) → {output}")

        elif event.kind == "verify.score":
            conf = event.payload.get("confidence", 0)
            directive = event.payload.get("directive", "?")
            print(f"\n[OK] VERIFY: confidence={conf:.3f}, directive='{directive}'")

        elif event.kind == "loop.directive":
            action = event.payload.get("action", "?")
            print(f"\n LOOP: {action.upper()}")


def demo_single_problem():
    """Demonstrate solving a single word problem."""
    print("\nIMRA-1 DEMO: Single Problem")

    perceive = SymbolicPerceiveEngine(num_sketches=5, temperature=0.3)
    reason = ReasonExecutor(max_steps=50)
    verify = CalibrationCritic()

    loop = DifferentiableDeliberationLoop(
        perceive=perceive,
        reason=reason,
        verify=verify,
        config=DeliberationConfig(
            max_iterations=5,
            confidence_threshold=0.4,
            max_sketches_per_iteration=5,
        ),
    )

    request = PerceptionRequest(
        task_id="demo_single",
        payload={"type": "arithmetic"},
        problem_text="Sarah has 15 stickers. She gives 7 to her friend. How many stickers does Sarah have now?",
    )

    print(f"\nProblem: {request.problem_text}")

    result = loop.run(request)

    print("\nResult:")
    print(f"   Success: {result.success}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Final Value: {result.final_value}")
    print(f"   Iterations: {result.iterations}")
    print(f"   Sketches Explored: {result.sketches_explored}")

    if result.gradients:
        print("\n   Gradients (for learning):")
        for k, v in list(result.gradients.items())[:5]:
            print(f"      {k}: {v:.4f}")

    print_trace_summary(result)

    return result


def demo_batch_problems():
    """Demonstrate batch processing multiple problems."""
    print("\nIMRA-1 DEMO: Batch Processing")

    perceive = SymbolicPerceiveEngine(num_sketches=3)
    reason = ReasonExecutor()
    verify = CalibrationCritic()

    loop = DifferentiableDeliberationLoop(
        perceive=perceive,
        reason=reason,
        verify=verify,
    )

    runner = BatchDeliberationRunner(loop)

    problems = [
        "What is 8 plus 5?",
        "A box has 20 items. 12 are removed. How many remain?",
        "Each shelf holds 6 books. There are 4 shelves. How many books total?",
        "Divide 24 by 3.",
        "If you have 100 and lose 37, what remains?",
    ]

    requests = [
        PerceptionRequest(
            task_id=f"batch_{i}",
            payload={},
            problem_text=text,
        )
        for i, text in enumerate(problems)
    ]

    print(f"\nProcessing {len(problems)} problems...")

    stats = runner.run_batch(requests)

    print("\n  Batch Results:")
    print(f"   Total Problems: {stats['total_problems']}")
    print(f"   Successes: {stats['successes']}")
    print(f"   Success Rate: {stats['success_rate']:.1%}")
    print(f"   Avg Iterations: {stats['avg_iterations']:.2f}")
    print(f"   Avg Sketches: {stats['avg_sketches']:.2f}")
    print(f"   Avg Confidence: {stats['avg_confidence']:.3f}")

    print("\n  Individual Results:")
    for _i, (problem, res) in enumerate(zip(problems, runner.results, strict=False)):
        status = "[+]" if res.success else "[-]"
        print(f"   {status} [{res.confidence:.2f}] {problem[:40]}... → {res.final_value}")

    return stats


def demo_calibration_tracking() -> None:
    """Demonstrate calibration statistics tracking."""
    print("\nIMRA-1 DEMO: Calibration Tracking")

    perceive = SymbolicPerceiveEngine()
    reason = ReasonExecutor()
    verify = CalibrationCritic()

    loop = DifferentiableDeliberationLoop(perceive, reason, verify)

    # Run problems and record outcomes
    test_cases = [
        ("What is 3 plus 4?", 7),
        ("What is 10 minus 2?", 8),
        ("What is 5 times 6?", 30),
    ]

    print("\nRunning problems with known answers...")

    for problem, expected in test_cases:
        request = PerceptionRequest(
            task_id=f"calib_{expected}",
            payload={},
            problem_text=problem,
        )
        result = loop.run(request)

        correct = result.final_value is not None and abs(result.final_value - expected) < 0.1
        verify.record_outcome(result.trace, correct)

        status = "[+]" if correct else "[-]"
        print(f"   {status} {problem} → got {result.final_value}, expected {expected}")

    metrics = verify.get_calibration_metrics()
    print("\n  Calibration Metrics:")
    print(f"   ECE (Expected Calibration Error): {metrics['ece']:.4f}")
    print(f"   Brier Score: {metrics['brier']:.4f}")
    print(f"   Samples: {metrics['n_samples']}")


if __name__ == "__main__":
    print("IMRA-1: Intuition Meta-Reasoning Architecture")
    print("Demo Script")

    demo_single_problem()
    demo_batch_problems()
    demo_calibration_tracking()

    print("\nDemo complete!")

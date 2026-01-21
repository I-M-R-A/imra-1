#!/usr/bin/env python3

"""IMRA-1 Olympiad Test Runner.

Runs olympiad-level problems through the full meta-cognitive deliberation
framework with real Ollama LLMs. No timeouts — quality over speed.

Usage:
    python -m tests.olympiad.runner              # Quick test (3 problems)
    python -m tests.olympiad.runner --full       # Full test suite
    python -m tests.olympiad.runner --problem comb_001  # Single problem
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))  # Add src to path

from imra.llm.meta_orchestrator import MetaCognitiveOrchestrator, run_olympiad_test
from tests.olympiad.problems import get_full_test_set, get_problem_by_id, get_quick_test_set

DEFAULT_MODELS = {
    "perceive1": os.getenv("IMRA_LLM_PERCEIVE1", "qwen2.5-coder:3b"),
    "perceive2": os.getenv("IMRA_LLM_PERCEIVE2", "qwen2.5-coder:7b"),
    "reason": os.getenv("IMRA_LLM_REASON", "codellama:latest"),
    "verify": os.getenv("IMRA_LLM_VERIFY", "qwen2.5-coder:7b"),
}


def run_test_suite(problems: list, models: dict, output_dir: Path) -> dict:
    """Run a set of problems through IMRA-1.

    Args:
        problems: List of problem dicts
        models: Role-to-model mapping
        output_dir: Directory to save results

    Returns:
        Summary dict with results
    """
    print()
    print("IMRA-1 OLYMPIAD TEST SUITE")
    print()
    print(f"Testing {len(problems)} problems")
    print(f"Models: {json.dumps(models, indent=2)}")
    print("Timeouts: UNLIMITED (quality testing)")
    print()

    orchestrator = MetaCognitiveOrchestrator(models=models, verbose=True)

    results = []
    passed = 0
    failed = 0

    for i, problem in enumerate(problems, 1):
        if problem is None:
            continue

        print()
        print(f"PROBLEM {i}/{len(problems)}: {problem['id']}")
        print(f"Category: {problem['category']} | Difficulty: {problem['difficulty']}")
        print()

        try:
            result = run_olympiad_test(
                orchestrator=orchestrator,
                problem=problem["problem"],
                expected=problem.get("expected"),
            )

            results.append(
                {
                    "problem_id": problem["id"],
                    "category": problem["category"],
                    "difficulty": problem["difficulty"],
                    "expected": problem.get("expected"),
                    "got": result["result"].get("final_answer"),
                    "confidence": result["result"].get("final_confidence"),
                    "accepted": result["result"].get("answer_accepted"),
                    "matches_expected": result.get("matches_expected"),
                    "trace": result["result"].get("trace"),
                    "meta_learning": result["result"].get("meta_learning"),
                    "deliberation_quality": result["result"].get("deliberation_quality"),
                }
            )

            if result.get("matches_expected"):
                passed += 1
                print(f"\n[PASS] Answer matches expected: {problem.get('expected')}")
            else:
                failed += 1
                print(
                    f"\n[FAIL] Expected: {problem.get('expected')}, Got: {result['result'].get('final_answer')}"
                )

        except Exception as e:
            failed += 1
            results.append({"problem_id": problem["id"], "error": str(e)})
            print(f"\n[ERROR] {e}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"olympiad_results_{timestamp}.json"

    summary = {
        "timestamp": timestamp,
        "models": models,
        "total_problems": len(problems),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / len(problems) if problems else 0,
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("TEST SUITE COMPLETE")
    print()
    print(f"Total: {len(problems)} | Passed: {passed} | Failed: {failed}")
    print(f"Pass rate: {summary['pass_rate'] * 100:.1f}%")
    print(f"Results saved to: {output_file}")
    print()

    return summary


def main():
    parser = argparse.ArgumentParser(description="IMRA-1 Olympiad Test Runner")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--problem", type=str, help="Run single problem by ID")
    parser.add_argument("--output", type=str, default="test_results", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.problem:
        problem = get_problem_by_id(args.problem)
        if problem is None:
            print(f"Problem '{args.problem}' not found")
            sys.exit(1)
        problems = [problem]
    elif args.full:
        problems = get_full_test_set()
    else:
        problems = get_quick_test_set()

    run_test_suite(problems, DEFAULT_MODELS, output_dir)


if __name__ == "__main__":
    main()

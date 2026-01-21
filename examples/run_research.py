#!/usr/bin/env python3

"""IMRA-1 Research Runner.

Runs problems through the research orchestrator and produces:
1. JSON traces (machine-readable research data)
2. Human-readable logs (for researcher analysis)
3. Summary statistics

This is the primary tool for conducting AI cognition research with IMRA-1.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imra.research import ResearchOrchestrator

DEFAULT_MODELS = {
    "perceive1": os.getenv("IMRA_PERCEIVE1", "codellama:latest"),
    "perceive2": os.getenv("IMRA_PERCEIVE2", "codellama:latest"),
    "reason": os.getenv("IMRA_REASON", "codellama:latest"),
    "verify": os.getenv("IMRA_VERIFY", "codellama:latest"),
}


RESEARCH_PROBLEMS = [
    {
        "id": "divisors_001",
        "problem": "How many positive divisors does 2^5 * 3^3 * 5^2 have? For a number N = p1^a1 * p2^a2 * ... the count is (a1+1)(a2+1)... Show your complete reasoning.",
        "expected": 72,
        "category": "number_theory",
    },
    {
        "id": "paths_001",
        "problem": "In a 4x4 grid, how many paths go from top-left to bottom-right using only right and down moves? This requires choosing 3 rights out of 6 total moves.",
        "expected": 20,
        "category": "combinatorics",
    },
    {
        "id": "mod_001",
        "problem": "What is 2^10 mod 7? Find the pattern: 2^1=2, 2^2=4, 2^3=8=1(mod7), 2^4=2, ... The cycle repeats every 3. Since 10=3*3+1, answer is 2^1 mod 7.",
        "expected": 2,
        "category": "modular_arithmetic",
    },
]


def run_research_session(problems: list, models: dict, output_dir: Path) -> dict:
    """Run a research session on multiple problems."""

    print()
    print("IMRA-1 RESEARCH SESSION")
    print()
    print(f"Problems: {len(problems)}")
    print(f"Models: {json.dumps(models, indent=2)}")
    print(f"Output: {output_dir}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = ResearchOrchestrator(models=models, verbose=True)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / f"session_{session_id}"
    session_dir.mkdir(exist_ok=True)

    results = []

    for i, prob in enumerate(problems, 1):
        print()
        print(f"PROBLEM {i}/{len(problems)}: {prob['id']}")
        print()

        try:
            trace = orchestrator.run_research_deliberation(
                problem=prob["problem"],
                expected_answer=prob.get("expected"),
                problem_category=prob.get("category", ""),
            )

            json_path = session_dir / f"{prob['id']}_trace.json"
            trace.save(str(json_path))

            log_path = session_dir / f"{prob['id']}_log.txt"
            trace.save_human_readable(str(log_path))

            results.append(
                {
                    "problem_id": prob["id"],
                    "expected": prob.get("expected"),
                    "got": trace.final_answer,
                    "correct": trace.answer_correct,
                    "confidence": trace.final_confidence,
                    "uncertainties": len(trace.all_uncertainties),
                    "errors_detected": len(trace.errors_detected),
                    "duration": trace.total_duration_seconds,
                    "trace_file": str(json_path),
                    "log_file": str(log_path),
                }
            )

            print(f"\nResult: {trace.final_answer} (expected: {prob.get('expected')})")
            print(f"Correct: {trace.answer_correct}")
            print(f"Uncertainties logged: {len(trace.all_uncertainties)}")

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback

            traceback.print_exc()
            results.append({"problem_id": prob["id"], "error": str(e)})

    summary = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "problem_count": len(problems),
        "results": results,
        "statistics": {
            "total": len(results),
            "correct": sum(1 for r in results if r.get("correct")),
            "incorrect": sum(1 for r in results if not r.get("correct")),
            "errors": sum(1 for r in results if "error" in r),
            "avg_confidence": (
                sum(r.get("confidence", 0) for r in results) / len(results) if results else 0
            ),
            "avg_uncertainties": (
                sum(r.get("uncertainties", 0) for r in results) / len(results) if results else 0
            ),
            "total_duration": sum(r.get("duration", 0) for r in results),
        },
    }

    summary_path = session_dir / "session_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("SESSION COMPLETE")
    print()
    print(f"Results saved to: {session_dir}")
    print(f"Correct: {summary['statistics']['correct']}/{summary['statistics']['total']}")
    print(f"Average confidence: {summary['statistics']['avg_confidence']:.2f}")
    print(f"Average uncertainties: {summary['statistics']['avg_uncertainties']:.1f}")
    print(f"Total duration: {summary['statistics']['total_duration']:.1f}s")
    print()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="IMRA-1 Research Runner")
    parser.add_argument("--output", default="research_data", help="Output directory")
    parser.add_argument("--problem", type=int, help="Run specific problem index (0-based)")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.problem is not None:
        problems = [RESEARCH_PROBLEMS[args.problem]]
    else:
        problems = RESEARCH_PROBLEMS

    run_research_session(problems, DEFAULT_MODELS, output_dir)


if __name__ == "__main__":
    main()

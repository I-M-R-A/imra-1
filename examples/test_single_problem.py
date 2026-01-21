#!/usr/bin/env python3

"""Quick single-problem test for IMRA-1.

Run with: python examples/test_single_problem.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imra.llm import MetaCognitiveOrchestrator

MODELS = {
    "perceive1": "codellama:latest",
    "perceive2": "codellama:latest",
    "reason": "codellama:latest",
    "verify": "codellama:latest",
}

PROBLEM = """How many positive divisors does the number N = 2^5 * 3^3 * 5^2 have?

HINT: For a number N = p1^a1 * p2^a2 * ... * pk^ak where p1, p2, ..., pk are distinct primes,
the number of positive divisors is (a1+1) * (a2+1) * ... * (ak+1).

Show your reasoning step by step."""  # Test problem with known answer


EXPECTED = 72  # (5+1) * (3+1) * (2+1) = 6 * 4 * 3 = 72


def main() -> None:
    print()
    print("IMRA-1 SINGLE PROBLEM TEST")
    print()
    print(f"Problem: {PROBLEM[:80]}...")
    print(f"Expected answer: {EXPECTED}")
    print("Models: codellama:latest (all roles)")
    print("Timeout: UNLIMITED")
    print()

    orch = MetaCognitiveOrchestrator(models=MODELS, verbose=True)

    result = orch.run_deliberation(PROBLEM)

    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"single_test_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    print()
    print("SUMMARY")
    print()
    print(f"Final Answer: {result.get('final_answer')}")
    print(f"Expected: {EXPECTED}")
    print(f"Confidence: {result.get('final_confidence')}")
    print(f"Accepted: {result.get('answer_accepted')}")

    got = result.get("final_answer")
    if str(got) == str(EXPECTED) or got == EXPECTED:
        print("STATUS: CORRECT")
    else:
        print(f"STATUS: INCORRECT (got {got}, expected {EXPECTED})")

    meta = result.get("meta_learning", {})
    if meta:
        print("\nMeta-Learning Insights:")
        for key, value in meta.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

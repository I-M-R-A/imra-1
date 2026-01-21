#!/usr/bin/env python3

"""IMRA-1 Demo: Metacognitive Reasoning Framework

This demo shows how IMRA-1 uses self-diagnosis to calibrate confidence
and retry when it detects reasoning failures.

Run with: python examples/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ollama

from imra.core.orchestrator import OrchestratorConfig, ReasoningOrchestrator


def main() -> None:
    print("IMRA-1: Metacognitive Reasoning Framework Demo")
    print()
    print("This demo shows IMRA-1's key feature: using self-diagnosis")
    print("to calibrate confidence and detect reasoning failures.")
    print()

    try:
        models = ollama.list()
        available = [m.model for m in models.models]
        print(f"Available models: {', '.join(available[:3])}...")
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama: {e}")
        print("Please ensure Ollama is running: ollama serve")
        return

    config = OrchestratorConfig(
        perceive_model="qwen2.5-coder:3b",
        reason_model="codellama:latest",
        verify_model="qwen2.5-coder:7b",
        output_dir=Path("demo_output"),
        enable_discussion=False,  # Faster for demo given time constraints
        max_retries=1,
    )

    orchestrator = ReasoningOrchestrator(config, ollama)

    # Test problem - simple enough to run quickly
    problem = {
        "id": "demo_problem",
        "problem": """What is the sum of all positive divisors of 12?

The positive divisors of 12 are: 1, 2, 3, 4, 6, 12.
Sum them up and give a single number as your answer.""",
        "expected": "28",
        "category": "arithmetic",
    }

    print("\nPROBLEM:")
    print(problem["problem"])
    print(f"\nExpected answer: {problem['expected']}")

    # Run the orchestrator
    print("\nRunning IMRA-1...")
    trace = orchestrator.solve(problem)

    # Show metacognitive features
    print("\nMETACOGNITIVE ANALYSIS")

    print(f"\nAnswer: {trace.final_answer}")
    print(f"Expected: {trace.expected_answer}")
    print(f"Correct: {trace.is_correct}")

    # Show confidence calibration
    if hasattr(trace, "original_confidence"):
        print("\nConfidence Calibration:")
        print(f"  Raw confidence:        {trace.original_confidence:.2f}")
        print(f"  Coherence score:       {getattr(trace, 'coherence_score', 'N/A')}")
        print(f"  Calibrated confidence: {trace.final_confidence:.2f}")

    # Show failure mode detection
    if hasattr(trace, "failure_mode"):
        print("\nFailure Mode Detection:")
        print(f"  Diagnosis: {trace.failure_mode.upper().replace('_', ' ')}")
        print(f"  Coherence Level: {getattr(trace, 'coherence_level', 'N/A').upper()}")

    print("\nKEY INSIGHT:")
    print("IMRA-1 doesn't just give an answer - it analyzes its own reasoning:")
    print("1. Computes coherence score from internal consistency")
    print("2. Detects failure modes (circular reasoning, verification paradox, etc.)")
    print("3. Calibrates confidence: final_conf = raw_conf * coherence")
    print("4. Retries with targeted feedback if coherence is low")
    print("5. Refuses to be confident when reasoning is incoherent")
    print()
    print("This is metacognition: thinking about thinking.")


if __name__ == "__main__":
    main()

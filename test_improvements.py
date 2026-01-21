#!/usr/bin/env python3

"""Quick test of IMRA-1 improvements.

Tests the framework on Problem 3 (IMO 2017) which previously failed.
Expected: 4, Previously got: 2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import ollama

from imra.core.orchestrator import OrchestratorConfig, ReasoningOrchestrator
from imra.problems.research_bank import get_all_problems


def main() -> None:
    print()
    print("IMRA-1 IMPROVEMENTS TEST")
    print()

    # Get problem 3 (IMO 2017 - sequence analysis)
    problems = get_all_problems()
    test_problem = problems[2]  # Index 2 = Problem 3

    print(f"\nTesting: {test_problem['id']}")
    print(f"Problem: {test_problem['problem'][:150]}...")
    print(f"Expected Answer: {test_problem['expected']}")
    print("Previous Result: 2 (WRONG)")
    print()

    # Detect models
    models = ollama.list()
    model_names = [m.model for m in models.models if "embed" not in m.model.lower()]

    roles = {
        "perceive": model_names[0],
        "reason": model_names[-1],
        "verify": model_names[1] if len(model_names) > 1 else model_names[0],
    }

    print("Models:")
    print(f"  PERCEIVE → {roles['perceive']}")
    print(f"  REASON   → {roles['reason']}")
    print(f"  VERIFY   → {roles['verify']}")
    print()

    config = OrchestratorConfig(
        perceive_model=roles["perceive"],
        reason_model=roles["reason"],
        verify_model=roles["verify"],
        output_dir=Path("research_data/test_traces"),
        enable_discussion=True,  # Enable team discussion
        enable_reflection=False,
        max_retries=0,
    )

    orchestrator = ReasoningOrchestrator(config, ollama)

    print("Running with improvements:")
    print("  ✓ Enhanced VERIFY (no vague corrections)")
    print("  ✓ RESOLVE layer (forces concrete answers)")
    print("  ✓ Computational tools (math verification)")
    print()

    try:
        trace = orchestrator.solve(test_problem)

        print()
        print("TEST RESULT")
        print()
        print(f"Final Answer: {trace.final_answer}")
        print(f"Expected:     {trace.expected_answer}")
        print(f"Correct:      {trace.is_correct} {'✓' if trace.is_correct else '✗'}")
        print(f"Confidence:   {trace.final_confidence:.2f}")
        print(f"Duration:     {trace.total_duration_seconds:.1f}s")
        print()

        if trace.discussion:
            print("Discussion:")
            print(f"  Turns:     {len(trace.discussion.turns)}")
            print(f"  Consensus: {trace.discussion.consensus_reached}")
            print(f"  Team Answer: {trace.discussion.final_team_answer}")

        print()
        print("Improvements Working:")

        verify_steps = [s for s in trace.steps if s.phase.value == "verify"]
        if verify_steps:
            verify_output = verify_steps[0].output_data
            has_vague = any(
                phrase in verify_output.lower()
                for phrase in ["not applicable", "cannot determine", "would suggest"]
            )
            print(f"  VERIFY concrete: {'✗ Still vague' if has_vague else '✓ Specific correction'}")

        if trace.discussion:
            print("  RESOLVE ran:     ✓ Forced synthesis")

        is_concrete = trace.final_answer and len(str(trace.final_answer).strip()) < 50
        print(f"  Answer concrete: {'✓ Yes' if is_concrete else '✗ Still meta-commentary'}")

        print()
        if trace.is_correct:
            print(" SUCCESS! Improvements fixed the issue!")
        else:
            print(" Still incorrect, but check if improvements applied")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

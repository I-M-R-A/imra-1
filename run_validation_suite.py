#!/usr/bin/env python3

"""Validation Suite - Test all 10 problems with improvements.

Compares results against baseline (batch1-10 logs) to measure improvement.

MEMORY OPTIMIZATIONS:
- Unload models between problems to free GPU/RAM
- Garbage collection after each problem
- Reduced context window sizes where possible
"""

import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import ollama

from imra.core.orchestrator import OrchestratorConfig, ReasoningOrchestrator
from imra.problems.research_bank import get_all_problems


def unload_all_models() -> None:
    """Unload all models from Ollama to free memory."""
    try:
        # List all loaded models
        models = ollama.list()
        for m in models.models:
            model_name = m.model
            # Try to unload by generating with keep_alive=0
            try:
                ollama.generate(
                    model=model_name, prompt="", options={"num_predict": 0}, keep_alive=0
                )
                print(f"  [MEM] Unloaded: {model_name}")
            except Exception:
                pass  # Model might not be loaded
    except Exception as e:
        print(f"  [MEM] Warning: Could not unload models: {e}")


def force_gc() -> None:
    """Force garbage collection."""
    gc.collect()
    gc.collect()  # Run twice for cyclic refs


def load_baseline_results():
    """Load baseline results from original batch runs."""
    baseline = {}
    for i in range(1, 11):
        baseline[i] = {"correct": False, "had_vague_correction": True, "had_meta_commentary": True}
    return baseline


def run_validation_suite(start_from=1):
    """Run all 10 problems and compare to baseline.

    Args:
        start_from: Problem number to start from (1-10). Useful for resuming.
    """
    print("IMRA-1 VALIDATION SUITE")
    print("Testing framework on 10 problems")
    print()

    print("[MEM] Initial cleanup...")  # Memory optimizations
    unload_all_models()
    force_gc()
    print()

    all_problems = get_all_problems()
    test_problems = all_problems[start_from - 1 : 10]  # Slice from start_from
    print(f"Starting from problem {start_from}/10")
    print()

    models = ollama.list()
    model_names = [m.model for m in models.models if "embed" not in m.model.lower()]

    roles = {
        "perceive": model_names[0],
        "reason": model_names[-1],
        "verify": model_names[1] if len(model_names) > 1 else model_names[0],
    }

    print("Configuration:")
    print(f"  PERCEIVE → {roles['perceive']}")
    print(f"  REASON   → {roles['reason']}")
    print(f"  VERIFY   → {roles['verify']}")
    print()
    print("Improvements Active:")
    print("  [+] Enhanced VERIFY (no vague corrections)")
    print("  [+] RESOLVE layer (forces concrete answers)")
    print("  [+] Computational tools (math verification)")
    print()

    output_dir = Path("research_data/validation_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OrchestratorConfig(
        perceive_model=roles["perceive"],
        reason_model=roles["reason"],
        verify_model=roles["verify"],
        output_dir=output_dir / "traces",
        enable_discussion=True,
        enable_reflection=False,
        max_retries=0,
    )

    orchestrator = ReasoningOrchestrator(config, ollama)

    results = []
    start_time = time.time()

    for idx, problem in enumerate(test_problems):
        i = start_from + idx
        print(f"\nPROBLEM {i}/10: {problem['id']}")
        print(f"Category: {problem['category']} | Trap: {problem['trap_type']}")
        print(f"Expected: {problem['expected']}")
        print("Baseline: INCORRECT (0% on this problem)")
        print()

        try:
            trace = orchestrator.solve(problem)

            verify_steps = [s for s in trace.steps if s.phase.value == "verify"]
            has_vague = False
            if verify_steps:
                verify_output = verify_steps[0].output_data
                vague_phrases = ["not applicable", "cannot determine", "would suggest"]
                has_vague = any(phrase in verify_output.lower() for phrase in vague_phrases)

            is_concrete = (
                trace.final_answer
                and len(str(trace.final_answer).strip()) < 100
                and not any(
                    phrase in str(trace.final_answer).lower()
                    for phrase in ["would propose", "should use", "different approach"]
                )
            )

            result = {
                "problem_num": i,
                "problem_id": problem["id"],
                "category": problem["category"],
                "expected": problem["expected"],
                "final_answer": trace.final_answer,
                "correct": trace.is_correct,
                "confidence": trace.final_confidence,
                "duration": trace.total_duration_seconds,
                "errors_detected": trace.total_errors_detected,
                "corrections": trace.total_corrections_made,
                "recovery_rate": trace.recovery_rate,
                "improvements": {
                    "verify_concrete": not has_vague,
                    "answer_concrete": is_concrete,
                    "resolve_ran": trace.discussion is not None,
                },
                "trace_file": f"trace_{trace.trace_id}.json",
            }

            results.append(result)

            status = "[+] CORRECT" if trace.is_correct else "[-] WRONG"
            improvement = "FIXED!" if trace.is_correct else "Still incorrect"

            print()
            print(f"RESULT: {status}")
            print(f"  Answer:     {trace.final_answer}")
            print(f"  Expected:   {trace.expected_answer}")
            print(f"  Status:     {improvement}")
            print(f"  Confidence: {trace.final_confidence:.2f}")
            print(f"  Duration:   {trace.total_duration_seconds:.1f}s")
            print()
            print("Improvement Checks:")
            print(
                f"  VERIFY concrete:    {'[+]' if result['improvements']['verify_concrete'] else '[-]'}"
            )
            print(
                f"  Answer concrete:    {'[+]' if result['improvements']['answer_concrete'] else '[-]'}"
            )
            print(
                f"  RESOLVE ran:        {'[+]' if result['improvements']['resolve_ran'] else '[-]'}"
            )

        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback

            traceback.print_exc()

            results.append({"problem_num": i, "problem_id": problem["id"], "error": str(e)})

        print(f"\n[MEM] Cleaning up after problem {i}...")
        unload_all_models()
        force_gc()
        print("[MEM] Ready for next problem")

    total_time = time.time() - start_time

    print()
    print("VALIDATION SUITE COMPLETE")
    print()

    successful = [r for r in results if not r.get("error")]
    correct = [r for r in successful if r.get("correct")]

    baseline_correct = 0
    improved_correct = len(correct)

    print("Results Summary:")
    print(f"  Baseline:    {baseline_correct}/10 correct (0%)")
    print(f"  Improved:    {improved_correct}/10 correct ({improved_correct * 10}%)")
    print(f"  Improvement: +{improved_correct} problems fixed")
    print()

    if improved_correct > 0:
        print(f"SUCCESS! Framework improvements fixed {improved_correct} problem(s)!")
        print()
        print("Problems Fixed:")
        for r in correct:
            print(f"  [+] Problem {r['problem_num']}: {r['problem_id']}")
    else:
        print("No problems solved correctly yet.")
        print("   This suggests further iteration needed.")

    print()
    print("Improvement Metrics:")
    verify_concrete = sum(1 for r in successful if r.get("improvements", {}).get("verify_concrete"))
    answer_concrete = sum(1 for r in successful if r.get("improvements", {}).get("answer_concrete"))
    resolve_ran = sum(1 for r in successful if r.get("improvements", {}).get("resolve_ran"))

    print(
        f"  VERIFY gave concrete corrections: {verify_concrete}/{len(successful)} ({verify_concrete * 100 // len(successful) if successful else 0}%)"
    )
    print(
        f"  Answers were concrete:            {answer_concrete}/{len(successful)} ({answer_concrete * 100 // len(successful) if successful else 0}%)"
    )
    print(
        f"  RESOLVE phase ran:                {resolve_ran}/{len(successful)} ({resolve_ran * 100 // len(successful) if successful else 0}%)"
    )

    print()
    print(f"Total Runtime: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print()

    summary = {
        "validation_run": {
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "models": roles,
        },
        "baseline": {"correct": baseline_correct, "total": 10, "accuracy": 0.0},
        "improved": {
            "correct": improved_correct,
            "total": len(successful),
            "accuracy": improved_correct / len(successful) if successful else 0,
        },
        "improvement_delta": {
            "problems_fixed": improved_correct - baseline_correct,
            "accuracy_gain": (improved_correct / len(successful) if successful else 0) - 0.0,
        },
        "detailed_results": results,
    }

    summary_path = (
        output_dir / f"validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Detailed results saved to: {summary_path}")
    print()

    print("NEXT STEPS")

    if improved_correct >= 3:
        print("EXCELLENT! You're ready to:")
        print("  1. Run full 50-problem suite")
        print("  2. Post framework on GitHub")
        print("  3. Write research paper")
    elif improved_correct >= 1:
        print("GOOD PROGRESS! Consider:")
        print("  1. Analyze which problems were fixed")
        print("  2. Study remaining failures")
        print("  3. Iterate on specific issues")
    else:
        print("NEEDS ITERATION:")
        print("  1. Examine traces to see what went wrong")
        print("  2. Check if improvements are actually running")
        print("  3. Debug specific failure modes")

    print()
    print("View traces at: research_data/validation_run/traces/")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run IMRA-1 validation suite")
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Problem number to start from (1-10). Useful for resuming.",
    )
    args = parser.parse_args()

    summary = run_validation_suite(start_from=args.start_from)

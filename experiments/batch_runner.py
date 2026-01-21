#!/usr/bin/env python3

"""IMRA-1 Batch Runner - Run problems in controlled batches with full orchestration."""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ollama

from imra.core.orchestrator import OrchestratorConfig, ReasoningOrchestrator
from imra.problems.research_bank import get_all_problems


def run_batch(start_idx: int, end_idx: int, output_dir: Path):
    """Run a batch of problems using the full orchestrator (PERCEIVE → REASON → VERIFY → DISCUSS)."""
    problems = get_all_problems()

    if end_idx <= start_idx:
        batch = problems[start_idx : start_idx + 1]
    else:
        batch = problems[start_idx:end_idx]

    print(f"\n{'#' * 70}")
    display_end = start_idx + len(batch)
    print(f"BATCH: Problems {start_idx + 1} to {display_end} ({len(batch)} problems)")
    print(f"{'#' * 70}")

    models = ollama.list()
    model_names = [m.model for m in models.models if "embed" not in m.model.lower()]

    roles = {
        "perceive": model_names[0],  # qwen2.5-coder:3b (fastest)
        "reason": model_names[-1],  # codellama:latest (different arch)
        "verify": (
            model_names[1] if len(model_names) > 1 else model_names[0]
        ),  # qwen2.5-coder:7b (medium)
    }

    print("\nRoles:")
    print(f"  PERCEIVE → {roles['perceive']}")
    print(f"  REASON   → {roles['reason']}")
    print(f"  VERIFY   → {roles['verify']}")

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
    for i, problem in enumerate(batch):
        prob_num = start_idx + i + 1
        print()
        print(f"[{prob_num}/50] {problem['id']}")
        print(f"Category: {problem['category']} | Trap: {problem['trap_type']}")
        print()
        print(f"Problem: {problem['problem'][:150]}...")
        print(f"Expected: {problem['expected']}")

        try:
            trace = orchestrator.solve(problem)

            status = "✓" if trace.is_correct else "✗"
            print(
                f"\n  RESULT: {status} Answer={trace.final_answer} (Expected={trace.expected_answer})"
            )
            print(
                f"  Duration: {trace.total_duration_seconds:.1f}s | Recovery: {trace.recovery_rate:.0%}"
            )

            if trace.discussion:
                print(
                    f"  Discussion: {len(trace.discussion.turns)} turns | Consensus: {trace.discussion.consensus_reached}"
                )

            results.append(
                {
                    "problem_id": problem["id"],
                    "category": problem["category"],
                    "trap_type": problem["trap_type"],
                    "correct": trace.is_correct,
                    "answer": trace.final_answer,
                    "expected": trace.expected_answer,
                    "confidence": trace.final_confidence,
                    "duration": trace.total_duration_seconds,
                    "errors_detected": trace.total_errors_detected,
                    "corrections": trace.total_corrections_made,
                    "discussion_consensus": (
                        trace.discussion.consensus_reached if trace.discussion else None
                    ),
                    "trace_file": f"trace_{trace.trace_id}.json",
                }
            )

        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback

            traceback.print_exc()
            results.append({"problem_id": problem["id"], "error": str(e)})

    batch_summary = {
        "batch": f"{start_idx + 1}-{end_idx}",
        "timestamp": datetime.now().isoformat(),
        "roles": roles,
        "results": results,
        "correct": sum(1 for r in results if r.get("correct", False)),
        "total": len(results),
    }

    summary_path = output_dir / f"batch_{start_idx + 1}_{end_idx}.json"
    with open(summary_path, "w") as f:
        json.dump(batch_summary, f, indent=2)

    correct = batch_summary["correct"]
    total = batch_summary["total"]
    print(f"\n{'#' * 70}")
    print(f"BATCH COMPLETE: {correct}/{total} correct ({correct / total * 100:.0f}%)")
    print(f"Saved: {summary_path}")
    print(f"{'#' * 70}")

    return batch_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="Start index (0-based)")
    parser.add_argument("--end", type=int, required=True, help="End index (exclusive)")
    parser.add_argument("--output", type=Path, default=Path("research_data"))
    args = parser.parse_args()

    run_batch(args.start, args.end, args.output)

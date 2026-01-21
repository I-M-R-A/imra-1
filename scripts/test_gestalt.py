#!/usr/bin/env python3

"""Test Gestalt Comparator on existing validation traces.

This script loads the 10 existing traces from the validation run
and computes coherence scores to validate the diagnostic capability.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # Adjust import path to src/

from datetime import datetime

from imra.core.gestalt import analyze_trace
from imra.core.trace import CognitiveTrace, DiscussionTurn, TeamDiscussion


def load_trace_from_json(trace_path: Path) -> CognitiveTrace:
    """Load a trace from JSON and reconstruct the CognitiveTrace object."""
    with open(trace_path) as f:
        data = json.load(f)

    trace = CognitiveTrace(
        trace_id=data["trace_id"],
        problem_id=data["problem"]["id"],
        problem_text=data["problem"]["text"],
        problem_category=data["problem"]["category"],
        problem_difficulty=data["problem"]["difficulty"],
        expected_answer=data["problem"]["expected_answer"],
    )

    if data.get("discussion"):
        disc_data = data["discussion"]
        discussion = TeamDiscussion()
        discussion.consensus_reached = disc_data.get("consensus_reached", False)
        discussion.final_team_answer = disc_data.get("final_team_answer")
        discussion.dissenting_opinions = disc_data.get("dissenting_opinions", [])
        discussion.discussion_duration_seconds = disc_data.get("discussion_duration_seconds", 0.0)

        for turn_data in disc_data.get("turns", []):
            turn = DiscussionTurn(
                speaker=turn_data["speaker"],
                speaker_model=turn_data["speaker_model"],
                message=turn_data["message"],
                stance=turn_data["stance"],
                reasoning=turn_data["reasoning"],
                references_to=turn_data.get("references_to", []),
            )
            discussion.add_turn(turn)

        trace.add_discussion(discussion)

    outcome = data.get("outcome", {})
    trace.final_answer = outcome.get("final_answer")
    trace.final_confidence = outcome.get("final_confidence", 0.0)
    trace.is_correct = outcome.get("is_correct", False)

    metrics = data.get("metrics", {})
    trace.total_duration_seconds = metrics.get("total_duration_seconds", 0.0)
    trace.total_uncertainties = metrics.get("total_uncertainties", 0)
    trace.total_errors_detected = metrics.get("total_errors_detected", 0)
    trace.total_corrections_made = metrics.get("total_corrections_made", 0)
    trace.recovery_rate = metrics.get("recovery_rate", 0.0)

    trace.models_used = data.get("models_used", {})

    return trace


def main() -> None:
    """Test Gestalt on existing validation traces."""
    traces_dir = Path("research_data/validation_run/traces")

    if not traces_dir.exists():
        print(f"  Traces directory not found: {traces_dir}")
        print("   Run validation suite first to generate traces.")
        return

    trace_files = sorted(traces_dir.glob("trace_*.json"))

    if not trace_files:
        print(f"  No trace files found in {traces_dir}")
        return

    print()
    print("GESTALT COMPARATOR TEST")
    print()
    print(f"\nLoading {len(trace_files)} validation traces...\n")

    results = []

    for trace_file in trace_files:
        print(f"Analyzing {trace_file.name}...")

        try:
            trace = load_trace_from_json(trace_file)
            coherence = analyze_trace(trace)

            results.append(
                {
                    "problem_id": trace.problem_id,
                    "expected": trace.expected_answer,
                    "final_answer": trace.final_answer,
                    "correct": trace.is_correct,
                    "confidence": trace.final_confidence,
                    "coherence_score": coherence.coherence_score,
                    "coherence_level": coherence.coherence_level.value,
                    "failure_mode": coherence.failure_mode.value,
                    "diagnosis": (
                        coherence.diagnosis[:100] + "..."
                        if len(coherence.diagnosis) > 100
                        else coherence.diagnosis
                    ),
                }
            )

            print(
                f"  Coherence: {coherence.coherence_score:.3f} ({coherence.coherence_level.value.upper()})"
            )
            print(f"  Failure Mode: {coherence.failure_mode.value.upper()}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    print()
    print("SUMMARY")
    print()

    if not results:
        print("No results to summarize.")
        return

    coherence_levels = {}
    failure_modes = {}

    for r in results:
        level = r["coherence_level"]
        mode = r["failure_mode"]

        coherence_levels[level] = coherence_levels.get(level, 0) + 1
        failure_modes[mode] = failure_modes.get(mode, 0) + 1

    print("\nCoherence Level Distribution:")
    for level in ["broken", "uncertain", "acceptable", "coherent"]:
        count = coherence_levels.get(level, 0)
        pct = count / len(results) * 100
        print(f"  {level.upper():12} : {count:2d} / {len(results)} ({pct:5.1f}%)")

    print("\nFailure Mode Distribution:")
    for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        print(f"  {mode.replace('_', ' ').upper():25} : {count:2d} / {len(results)} ({pct:5.1f}%)")

    print("\n" + "")
    print("OVERCONFIDENT ERRORS (Confidence >= 0.95, Coherence < 0.6)")
    print("")

    overconfident = [
        r
        for r in results
        if r["confidence"] >= 0.95 and r["coherence_score"] < 0.6 and not r["correct"]
    ]

    if overconfident:
        for r in overconfident:
            print(f"\n{r['problem_id']}:")
            print(f"  Confidence:  {r['confidence']:.2f}")
            print(f"  Coherence:   {r['coherence_score']:.3f}")
            print(f"  Failure:     {r['failure_mode'].replace('_', ' ').upper()}")
            print(f"  Diagnosis:   {r['diagnosis']}")
    else:
        print("\n✓ No overconfident errors detected (good!)")

    summary_path = Path("research_data/validation_run/gestalt_analysis.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_traces": len(results),
                "coherence_distribution": coherence_levels,
                "failure_mode_distribution": failure_modes,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n Saved analysis to: {summary_path}")


if __name__ == "__main__":
    main()

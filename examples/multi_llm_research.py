#!/usr/bin/env python3

"""IMRA-1 Multi-LLM Research Experiment.

This script runs comprehensive cognitive trace experiments using
DIFFERENT LLMs for different cognitive roles:

- PERCEIVE1: First perception (qwen2.5-coder:3b - fastest, initial interpretation)
- PERCEIVE2: Alternative perception (codellama:latest - different architecture)
- REASON: Main reasoning (qwen2.5-coder:7b - largest model, deep reasoning)
- VERIFY: Critical verification (codellama:latest - cross-model verification)

Using different LLMs allows us to study:
1. How different models perceive the same problem
2. Cross-model verification effectiveness
3. Model-specific failure patterns
4. Ensemble cognitive behavior

"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from imra.research.cognitive_trace import CognitiveTrace
from imra.research.orchestrator import ResearchOrchestrator


def detect_ollama_models() -> list[dict[str, str]]:
    """Detect all available Ollama models."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=30)
        models = []
        for line in result.stdout.strip().split("\n")[1:]:
            if line.strip():
                parts = line.split()
                if parts:
                    models.append(
                        {
                            "name": parts[0],
                            "size": parts[1] if len(parts) > 1 else "unknown",
                            "modified": " ".join(parts[2:]) if len(parts) > 2 else "unknown",
                        }
                    )
        return models
    except Exception as e:
        print(f"Error detecting Ollama models: {e}")
        return []


def assign_roles(models: list[dict[str, str]]) -> dict[str, str]:
    """Assign LLMs to cognitive roles based on capabilities.

    Strategy:
    - Smaller/faster models for initial perception
    - Larger models for deep reasoning
    - Different architectures for verification (cross-validation)
    """
    available = {m["name"] for m in models}

    # Default assignments (can be customized, if by any chance you consider using this framework outside research)

    role_preferences = {
        "perceive1": ["qwen2.5-coder:3b", "codellama:latest", "qwen2.5-coder:7b"],
        "perceive2": ["codellama:latest", "qwen2.5-coder:7b", "qwen2.5-coder:3b"],
        "reason": ["qwen2.5-coder:7b", "codellama:latest", "qwen2.5-coder:3b"],
        "verify": ["codellama:latest", "qwen2.5-coder:7b", "qwen2.5-coder:3b"],
    }

    assignments = {}
    for role, preferences in role_preferences.items():
        for model in preferences:
            if model in available:
                assignments[role] = model
                break

    fallback = list(available)[0] if available else "qwen2.5-coder:7b"
    for role in role_preferences:
        if role not in assignments:
            assignments[role] = fallback

    return assignments


def print_experiment_header(assignments: dict[str, str], problems: list[dict]) -> None:
    """Print experiment configuration."""
    print()
    print("IMRA-1 MULTI-LLM COGNITIVE RESEARCH EXPERIMENT")
    print()
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print("\nRole Assignments (DIFFERENT LLMs for DIFFERENT ROLES):")
    for role, model in assignments.items():
        print(f"  {role.upper():12} → {model}")
    print(f"\nProblems to Test: {len(problems)}")
    print()


def load_problems() -> list[dict[str, Any]]:
    """Load research problem set."""
    try:
        from tests.olympiad.problems import get_problem_summary, get_research_test_set

        summary = get_problem_summary()
        print("\nProblem Bank Summary:")
        print(f"  Total Problems: {summary['total']}")
        print(f"  By Category: {summary['by_category']}")
        print(f"  By Difficulty: {summary['by_difficulty']}")

        problems = get_research_test_set()
        print(f"\nUsing Research Test Set: {len(problems)} problems")
        return problems
    except ImportError as e:
        print(f"Error loading problems: {e}")
        return []


class ResearchSession:
    """Manages a complete research experiment session."""

    def __init__(self, output_dir: str = "research_data") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.traces: list[CognitiveTrace] = []
        self.metadata: dict[str, Any] = {}

    def save_trace(self, trace: CognitiveTrace, problem_id: str) -> None:
        """Save a single trace to file."""
        filename = f"trace_{self.session_id}_{problem_id}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(trace.to_dict(), f, indent=2, default=str)

        log_filename = f"trace_{self.session_id}_{problem_id}.txt"
        log_filepath = self.output_dir / log_filename
        with open(log_filepath, "w") as f:
            f.write(trace.to_human_readable())

        print(f"  Saved: {filepath.name}")

    def save_session_summary(self) -> None:
        """Save complete session summary."""
        summary = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": self.metadata,
            "total_problems": len(self.traces),
            "results": [],
        }

        for trace in self.traces:
            summary["results"].append(
                {
                    "problem_id": trace.metadata.get("problem_id", "unknown"),
                    "category": trace.problem_category,
                    "final_answer": trace.final_answer,
                    "expected_answer": trace.expected_answer,
                    "correct": trace.answer_correct,
                    "confidence": trace.final_confidence,
                    "duration_seconds": trace.total_duration_seconds,
                    "uncertainties_count": len(trace.all_uncertainties),
                    "errors_detected": len(trace.errors_detected),
                }
            )

        correct = sum(1 for t in self.traces if t.answer_correct)
        total = len(self.traces)
        avg_confidence = sum(t.final_confidence for t in self.traces) / total if total else 0
        avg_duration = sum(t.total_duration_seconds for t in self.traces) / total if total else 0

        summary["aggregate"] = {
            "accuracy": correct / total if total else 0,
            "correct_count": correct,
            "total_count": total,
            "avg_confidence": avg_confidence,
            "avg_duration_seconds": avg_duration,
            "total_duration_seconds": sum(t.total_duration_seconds for t in self.traces),
        }

        filepath = self.output_dir / f"session_{self.session_id}_summary.json"
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nSession Summary saved: {filepath}")
        return summary


def run_experiment():
    """Run the complete multi-LLM research experiment."""

    print("\n[Step 1] Detecting Ollama Models...")
    models = detect_ollama_models()
    if not models:
        print("ERROR: No Ollama models detected. Please ensure Ollama is running.")
        return

    print(f"Found {len(models)} models:")
    for m in models:
        if "embed" in m["name"].lower():
            print(f"  {m['name']} (embedding model - skipped)")
        else:
            print(f" {m['name']} ({m['size']})")

    reasoning_models = [m for m in models if "embed" not in m["name"].lower()]

    print("\n[Step 2] Assigning LLMs to Cognitive Roles...")
    assignments = assign_roles(reasoning_models)

    unique_models = set(assignments.values())

    if len(unique_models) == 1:
        print(f"\n WARNING: Only one model type available ({list(unique_models)[0]})")
        print("    For true multi-LLM research, install additional models:")
        print("    ollama pull codellama:latest")
        print("    ollama pull qwen2.5-coder:3b")
        print("    ollama pull qwen2.5-coder:7b")
    else:
        print(f"\n✓ Using {len(unique_models)} different models for cognitive roles:")

    for role, model in assignments.items():
        print(f"  {role.upper():12} → {model}")

    print("\n[Step 3] Loading Research Problem Set...")
    problems = load_problems()
    if not problems:
        print("ERROR: No problems loaded.")
        return

    print("\n[Step 4] Initializing Research Session...")
    session = ResearchSession()
    session.metadata = {
        "model_assignments": assignments,
        "available_models": [m["name"] for m in models],
        "experiment_type": "multi_llm_cognitive_trace",
    }

    print("\n[Step 5] Creating Multi-LLM Orchestrator...")
    orchestrator = ResearchOrchestrator(models=assignments, verbose=True)

    print_experiment_header(assignments, problems)

    print("\n[Step 6] Running Research Deliberations...")
    print()

    for i, problem in enumerate(problems, 1):
        if problem is None:
            print(f"\n[Problem {i}] SKIPPED - None value")
            continue

        problem_id = problem.get("id", f"problem_{i}")
        category = problem.get("category", "unknown")
        difficulty = problem.get("difficulty", "unknown")
        problem_text = problem.get("problem", "")
        expected = problem.get("expected")

        print()
        print(f"[Problem {i}/{len(problems)}] {problem_id}")
        print(f"Category: {category} | Difficulty: {difficulty}")
        print(f"Expected Answer: {expected}")
        print()

        try:
            trace = orchestrator.run_research_deliberation(
                problem=problem_text, expected_answer=expected, problem_category=category
            )

            trace.metadata["problem_id"] = problem_id
            trace.metadata["difficulty"] = difficulty
            trace.metadata["source"] = problem.get("source", "olympiad")

            session.traces.append(trace)
            session.save_trace(trace, problem_id)

            status = "✓ CORRECT" if trace.answer_correct else "✗ INCORRECT"
            print(f"\n[Result] {status}")
            print(f"  Answer: {trace.final_answer} (expected: {expected})")
            print(f"  Confidence: {trace.final_confidence:.2f}")
            print(f"  Duration: {trace.total_duration_seconds:.1f}s")
            print(f"  Uncertainties: {len(trace.all_uncertainties)}")
            print(f"  Errors Detected: {len(trace.errors_detected)}")

        except Exception as e:
            print(f"\n[ERROR] Problem {problem_id} failed: {e}")
            import traceback

            traceback.print_exc()

    print()
    print("[Step 7] Saving Session Summary...")
    summary = session.save_session_summary()

    print()
    print("EXPERIMENT COMPLETE")
    print()
    agg = summary.get("aggregate", {})
    print("\nFinal Statistics:")
    print(f"  Total Problems: {agg.get('total_count', 0)}")
    print(f"  Correct Answers: {agg.get('correct_count', 0)}")
    print(f"  Accuracy: {agg.get('accuracy', 0) * 100:.1f}%")
    print(f"  Average Confidence: {agg.get('avg_confidence', 0):.2f}")
    print(f"  Average Duration: {agg.get('avg_duration_seconds', 0):.1f}s")
    print(f"  Total Duration: {agg.get('total_duration_seconds', 0) / 60:.1f} minutes")
    print(f"\nAll traces saved to: {session.output_dir}/")
    print()

    return session


if __name__ == "__main__":
    os.environ["OLLAMA_TIMEOUT"] = "0"  # Unlimited timeout for LLMs calls (research mode)

    print()
    print("IMRA-1: Meta-Cognitive Research Framework")
    print("Multi-LLM Cognitive Trace Experiment")
    print()
    print("\nThis experiment will:")
    print("1. Detect all available Ollama LLMs")
    print("2. Assign DIFFERENT LLMs to DIFFERENT cognitive roles")
    print("3. Run 15 research-grade problems (medium to expert)")
    print("4. Collect detailed cognitive traces")
    print("5. Save all data for analysis")
    print("\n This will take 2-4 hours. Press Ctrl+C to abort.")
    print()

    try:
        session = run_experiment()
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(0)

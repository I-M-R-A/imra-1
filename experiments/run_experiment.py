#!/usr/bin/env python3

"""IMRA-1 Research Experiment Runner

This is the main entry point for running controlled experiments.

Features:
├── Auto-detect available LLMs
├── Assign roles (PERCEIVE/REASON/VERIFY) to different models
├── Run problem sets with full CognitiveTrace logging
├── Generate research-grade metrics and analysis
└── Support controlled experiments (vary one factor at a time)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imra.analysis.metrics import ResearchAnalyzer
from imra.core.orchestrator import ExperimentRunner, OrchestratorConfig, ReasoningOrchestrator
from imra.problems.research_bank import (
    LOGIC_PROBLEMS,
    OLYMPIAD_PROBLEMS,
    PROBABILITY_PROBLEMS,
    TRICKY_PROBLEMS,
    get_all_problems,
    get_research_subset,
)

# Import ollama

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


def detect_ollama_models() -> list[str]:
    """Detect available Ollama models (excluding embedding models)."""
    if not OLLAMA_AVAILABLE:
        print("Warning: ollama package not installed")
        return []

    try:
        models = ollama.list()
        available = []
        for m in models.get("models", []):
            name = m.get("name", "")

            if "embed" in name.lower():  # Exclude embedding models
                continue
            available.append(name)
        return available
    except Exception as e:
        print(f"Warning: Could not detect Ollama models: {e}")
        return []


def assign_roles(models: list[str]) -> dict[str, str]:
    """Assign models to cognitive roles.

    Strategy:
    - PERCEIVE: Fastest model (quick intuition)
    - REASON: Largest/best model (deep thinking)
    - VERIFY: Different architecture for cross-checking
    """
    if len(models) == 0:
        raise ValueError("No models available")

    if len(models) == 1:
        return {"perceive": models[0], "reason": models[0], "verify": models[0]}

    sorted_models = sorted(
        models, key=lambda m: ("7b" in m.lower(), "13b" in m.lower(), "3b" in m.lower())
    )

    architectures = {}
    for m in models:
        if "qwen" in m.lower():
            architectures.setdefault("qwen", []).append(m)
        elif "llama" in m.lower() or "codellama" in m.lower():
            architectures.setdefault("llama", []).append(m)
        elif "mistral" in m.lower():
            architectures.setdefault("mistral", []).append(m)
        else:
            architectures.setdefault("other", []).append(m)

    roles = {}
    roles["perceive"] = sorted_models[0]

    roles["reason"] = sorted_models[-1] if len(sorted_models) > 1 else sorted_models[0]

    reason_arch = None
    for arch, ms in architectures.items():
        if roles["reason"] in ms:
            reason_arch = arch
            break

    verify_candidates = []
    for arch, ms in architectures.items():
        if arch != reason_arch:
            verify_candidates.extend(ms)

    if verify_candidates:
        roles["verify"] = verify_candidates[0]
    else:
        roles["verify"] = sorted_models[-2] if len(sorted_models) > 2 else sorted_models[0]

    return roles


def run_experiment(
    problems: list[dict], roles: dict[str, str], output_dir: Path, experiment_name: str
) -> dict:
    """Run a full experiment with the given configuration."""

    print(f"\n{'#' * 70}")
    print(f"IMRA-1 RESEARCH EXPERIMENT: {experiment_name}")
    print(f"{'#' * 70}")
    print("\nRoles:")
    print(f"  PERCEIVE → {roles['perceive']}")
    print(f"  REASON   → {roles['reason']}")
    print(f"  VERIFY   → {roles['verify']}")
    print(f"\nProblems: {len(problems)}")
    print(f"Output: {output_dir}")

    config = OrchestratorConfig(
        perceive_model=roles["perceive"],
        reason_model=roles["reason"],
        verify_model=roles["verify"],
        output_dir=output_dir / "traces",
        max_retries=1,
        verify_threshold=0.5,
    )

    client = ollama.Client()
    orchestrator = ReasoningOrchestrator(config, client)
    runner = ExperimentRunner(orchestrator)

    start_time = time.time()
    metrics = runner.run_problems(problems)
    total_time = time.time() - start_time

    summary = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "perceive": roles["perceive"],
            "reason": roles["reason"],
            "verify": roles["verify"],
        },
        "problems_run": len(problems),
        "total_time_seconds": total_time,
        "metrics": metrics,
    }

    summary_path = output_dir / f"experiment_{experiment_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("EXPERIMENT COMPLETE")
    print()
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Accuracy: {metrics.get('accuracy', 0):.1%}")
    print(f"Saved: {summary_path}")

    return summary


def run_analysis(traces_dir: Path) -> None:
    """Run full analysis on collected traces."""
    print(f"\n{'#' * 70}")
    print("RUNNING ANALYSIS")
    print(f"{'#' * 70}")

    analyzer = ResearchAnalyzer(traces_dir)
    count = analyzer.load_traces()

    if count == 0:
        print(f"No traces found in {traces_dir}")
        return

    print(f"Loaded {count} traces")
    print(analyzer.generate_report())

    json_path, txt_path = analyzer.save_analysis(traces_dir / "full_analysis")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="IMRA-1 Research Experiment Runner")

    parser.add_argument(
        "--mode",
        choices=["full", "quick", "analysis"],
        default="quick",
        help="Mode: full (50 problems), quick (15 problems), analysis (analyze existing)",
    )
    parser.add_argument(
        "--category",
        choices=["all", "olympiad", "logic", "probability", "tricky"],
        default="all",
        help="Problem category to run",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("research_data"), help="Output directory"
    )
    parser.add_argument("--name", type=str, default=None, help="Experiment name")

    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "analysis":
        run_analysis(output_dir / "traces")
        return

    print("Detecting Ollama models...")
    models = detect_ollama_models()

    if not models:
        print("ERROR: No Ollama models found. Please install models with:")
        print("  ollama pull qwen2.5-coder:7b")
        sys.exit(1)

    print(f"Found {len(models)} models:")
    for m in models:
        print(f"  • {m}")

    roles = assign_roles(models)

    if args.category == "olympiad":
        problems = OLYMPIAD_PROBLEMS
    elif args.category == "logic":
        problems = LOGIC_PROBLEMS
    elif args.category == "probability":
        problems = PROBABILITY_PROBLEMS
    elif args.category == "tricky":
        problems = TRICKY_PROBLEMS
    else:
        if args.mode == "full":
            problems = get_all_problems()
        else:
            problems = get_research_subset(15)

    exp_name = args.name or f"{args.mode}_{args.category}_{datetime.now().strftime('%H%M%S')}"

    run_experiment(problems, roles, output_dir, exp_name)

    print("\nRunning post-experiment analysis...")
    run_analysis(output_dir / "traces")


if __name__ == "__main__":
    main()

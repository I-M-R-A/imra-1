#!/usr/bin/env python3

"""Plot Coherence vs Confidence scatter plot.

Visualizes the relationship between confidence and coherence,
highlighting the overconfidence paradox.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    """Generate coherence vs confidence plot."""
    analysis_path = Path("research_data/validation_run/gestalt_analysis.json")

    if not analysis_path.exists():
        print(f"  Analysis file not found: {analysis_path}")
        print("   Run scripts/test_gestalt.py first.")
        return

    with open(analysis_path) as f:
        data = json.load(f)

    results = data["results"]

    confidences = [r["confidence"] for r in results]
    coherences = [r["coherence_score"] for r in results]
    problem_ids = [r["problem_id"] for r in results]
    failure_modes = [r["failure_mode"] for r in results]
    correct = [r["correct"] for r in results]

    colors = {
        "verification_paradox": "#e74c3c",  # Red
        "meta_retreat": "#f39c12",  # Orange
        "circular_reasoning": "#9b59b6",  # Purple
        "overconfident_error": "#c0392b",  # Dark red
        "incomplete_reasoning": "#3498db",  # Blue
        "healthy": "#2ecc71",  # Green
    }

    point_colors = [colors.get(mode, "#95a5a6") for mode in failure_modes]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(
        confidences, coherences, c=point_colors, s=200, alpha=0.7, edgecolors="black", linewidth=1.5
    )

    for _i, (x, y, label) in enumerate(
        zip(confidences, coherences, problem_ids, strict=False)
    ):  # Annotate points
        ax.annotate(
            label.replace("_", " "),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="BROKEN threshold")
    ax.axhline(y=0.6, color="orange", linestyle="--", alpha=0.5, label="UNCERTAIN threshold")
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="ACCEPTABLE threshold")

    ax.axvline(x=0.95, color="purple", linestyle="--", alpha=0.5, label="High confidence (0.95)")

    ax.fill_between([0.95, 1.0], 0, 0.6, alpha=0.1, color="red", label="Overconfident Region")

    ax.set_xlabel("Confidence", fontsize=14, fontweight="bold")
    ax.set_ylabel("Coherence Score", fontsize=14, fontweight="bold")
    ax.set_title(
        "IMRA-1 Coherence vs Confidence Analysis\n(MAE-Inspired Gestalt Comparator)",
        fontsize=16,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors["verification_paradox"], label="Verification Paradox (70%)"),
        Patch(facecolor=colors["meta_retreat"], label="Meta-Retreat (30%)"),
        ax.get_lines()[0],  # BROKEN threshold
        ax.get_lines()[1],  # UNCERTAIN threshold
        ax.get_lines()[2],  # ACCEPTABLE threshold
        ax.get_lines()[3],  # High confidence
    ]

    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    textstr = f"Total Problems: {len(results)}\n"
    textstr += f"Correct: {sum(correct)}/{len(results)}\n"
    textstr += f"Overconfident Errors: {sum(1 for c, co in zip(confidences, coherences, strict=False) if c >= 0.95 and co < 0.6)}"

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    output_path = Path("research_data/validation_run/coherence_vs_confidence.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved plot to: {output_path}")

    pdf_path = output_path.with_suffix(".pdf")  # Save as PDF
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"✓ Saved PDF to: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()

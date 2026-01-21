#!/usr/bin/env python3

"""IMRA-1 Research Metrics & Analysis

Metrics That Matter (per research requirements):
├── Confidence vs Correctness - How well does AI know it's wrong?
├── Error Types - Logical, computational, semantic, assumption failures
├── Recovery Rate - How often does it detect and fix mistakes?
├── Problem Coverage - Reliability across problem types
└── Stemming Issues - Root causes of failures

This is what makes IMRA-1 auditable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ConfidenceCalibration:
    """Measures: How well does the AI know when it's wrong?"""

    # When confidence is high, is it actually right?
    high_confidence_accuracy: float = 0.0

    # When confidence is low, is it actually wrong?
    low_confidence_wrong_rate: float = 0.0

    # Correlation between confidence and correctness
    correlation: float = 0.0

    # Overconfidence rate (high conf on wrong answers)
    overconfidence_rate: float = 0.0

    # Underconfidence rate (low conf on right answers)
    underconfidence_rate: float = 0.0

    def assess(self, traces: list[dict]) -> ConfidenceCalibration:
        """Calculate calibration metrics from traces."""
        high_conf_right = 0
        high_conf_total = 0
        low_conf_wrong = 0
        low_conf_total = 0
        overconfident = 0
        underconfident = 0

        for t in traces:
            conf = t.get("outcome", {}).get("final_confidence", 0)
            correct = t.get("outcome", {}).get("is_correct", False)

            if conf > 0.7:
                high_conf_total += 1
                if correct:
                    high_conf_right += 1
                else:
                    overconfident += 1
            elif conf < 0.4:
                low_conf_total += 1
                if not correct:
                    low_conf_wrong += 1
                else:
                    underconfident += 1

        total = len(traces)

        self.high_confidence_accuracy = (
            high_conf_right / high_conf_total if high_conf_total > 0 else 0
        )
        self.low_confidence_wrong_rate = (
            low_conf_wrong / low_conf_total if low_conf_total > 0 else 0
        )
        self.overconfidence_rate = overconfident / total if total > 0 else 0
        self.underconfidence_rate = underconfident / total if total > 0 else 0

        return self


@dataclass
class ErrorAnalysis:
    """Tracks: What types of errors occur and are they caught?"""

    total_errors: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    detected_rate: float = 0.0
    corrected_rate: float = 0.0
    detection_by_type: dict[str, float] = field(default_factory=dict)

    def analyze(self, traces: list[dict]) -> ErrorAnalysis:
        """Analyze error patterns from traces."""
        errors_by_type = defaultdict(lambda: {"total": 0, "detected": 0, "corrected": 0})

        for t in traces:
            for step in t.get("reasoning_steps", []):
                for error in step.get("errors_detected", []):
                    etype = error.get("error_type", "unknown")
                    errors_by_type[etype]["total"] += 1
                    errors_by_type[etype]["detected"] += 1
                    if error.get("corrected", False):
                        errors_by_type[etype]["corrected"] += 1

        self.total_errors = sum(d["total"] for d in errors_by_type.values())
        self.by_type = {k: v["total"] for k, v in errors_by_type.items()}

        total_detected = sum(d["detected"] for d in errors_by_type.values())
        total_corrected = sum(d["corrected"] for d in errors_by_type.values())

        self.detected_rate = total_detected / self.total_errors if self.total_errors > 0 else 0
        self.corrected_rate = total_corrected / total_detected if total_detected > 0 else 0

        self.detection_by_type = {
            k: v["detected"] / v["total"] if v["total"] > 0 else 0
            for k, v in errors_by_type.items()
        }

        return self


@dataclass
class RecoveryAnalysis:
    """Tracks: How often does the system detect and fix its own mistakes?"""

    total_mistakes: int = 0
    mistakes_detected: int = 0
    mistakes_corrected: int = 0
    detection_rate: float = 0.0
    correction_rate: float = 0.0
    recovery_rate: float = 0.0

    def analyze(self, traces: list[dict]) -> RecoveryAnalysis:
        """Analyze recovery patterns."""
        for t in traces:
            metrics = t.get("metrics", {})
            self.mistakes_detected += metrics.get("total_errors_detected", 0)
            self.mistakes_corrected += metrics.get("total_corrections_made", 0)

        if self.mistakes_detected > 0:
            self.correction_rate = self.mistakes_corrected / self.mistakes_detected

        return self


@dataclass
class ProblemCoverage:
    """Tracks: How reliable is the system across problem types?"""

    by_category: dict[str, dict] = field(default_factory=dict)
    by_difficulty: dict[str, dict] = field(default_factory=dict)
    by_trap_type: dict[str, dict] = field(default_factory=dict)
    weakest_categories: list[str] = field(default_factory=list)
    strongest_categories: list[str] = field(default_factory=list)

    def analyze(self, traces: list[dict]) -> ProblemCoverage:
        """Analyze coverage and performance by problem attributes."""
        by_cat = defaultdict(lambda: {"total": 0, "correct": 0})
        by_diff = defaultdict(lambda: {"total": 0, "correct": 0})
        defaultdict(lambda: {"total": 0, "correct": 0})

        for t in traces:
            problem = t.get("problem", {})
            correct = t.get("outcome", {}).get("is_correct", False)

            cat = problem.get("category", "unknown")
            diff = problem.get("difficulty", "unknown")

            by_cat[cat]["total"] += 1
            by_diff[diff]["total"] += 1

            if correct:
                by_cat[cat]["correct"] += 1
                by_diff[diff]["correct"] += 1

        self.by_category = {
            cat: {**data, "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0}
            for cat, data in by_cat.items()
        }

        self.by_difficulty = {
            diff: {**data, "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0}
            for diff, data in by_diff.items()
        }

        sorted_cats = sorted(self.by_category.items(), key=lambda x: x[1]["accuracy"])
        self.weakest_categories = [cat for cat, _ in sorted_cats[:3]]
        self.strongest_categories = [cat for cat, _ in sorted_cats[-3:]]

        return self


@dataclass
class StemmingIssue:
    """A root cause pattern identified across multiple failures."""

    name: str
    description: str
    frequency: float  # How often it appears
    severity: str  # low, medium, high, critical
    examples: list[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class StemmingAnalysis:
    """Identifies root causes of failures across all traces."""

    issues: list[StemmingIssue] = field(default_factory=list)

    def analyze(self, traces: list[dict]) -> StemmingAnalysis:
        """Identify stemming issues (root causes) from trace patterns."""
        issues = []
        incorrect_traces = [t for t in traces if not t.get("outcome", {}).get("is_correct", True)]

        if not incorrect_traces:
            return self

        total_incorrect = len(incorrect_traces)

        overconfident = [
            t for t in incorrect_traces if t.get("outcome", {}).get("final_confidence", 0) > 0.8
        ]
        if overconfident:
            issues.append(
                StemmingIssue(
                    name="Systematic Overconfidence",
                    description="High confidence (>0.8) on incorrect answers",
                    frequency=len(overconfident) / total_incorrect,
                    severity="critical" if len(overconfident) > total_incorrect * 0.5 else "high",
                    examples=[t["problem"]["id"] for t in overconfident[:3]],
                    recommendation="Implement adversarial verification; train on confidence calibration",
                )
            )

        # VERIFY approving wrong answers in some cases due to blind spots

        verify_failures = []
        for t in incorrect_traces:
            for step in t.get("reasoning_steps", []):
                if step.get("phase") == "verify" and step.get("confidence", 0) > 0.6:
                    verify_failures.append(t)
                    break

        if verify_failures:
            issues.append(
                StemmingIssue(
                    name="Verification Blind Spot",
                    description="VERIFY phase approved incorrect solutions",
                    frequency=len(verify_failures) / total_incorrect,
                    severity="critical",
                    examples=[t["problem"]["id"] for t in verify_failures[:3]],
                    recommendation="Use cross-model verification; add explicit error-seeking prompts",
                )
            )

        # Intuitive traps (metacognition failures)

        metacog_failures = [
            t for t in incorrect_traces if t.get("problem", {}).get("category") == "metacognition"
        ]
        if metacog_failures:
            issues.append(
                StemmingIssue(
                    name="Intuitive Trap Vulnerability",
                    description="Falling for problems designed to test reflexive vs deliberate thinking",
                    frequency=len(metacog_failures) / total_incorrect,
                    severity="high",
                    examples=[t["problem"]["id"] for t in metacog_failures[:3]],
                    recommendation="Add explicit 'check your intuition' step; use System 2 forcing",
                )
            )

        no_uncertainty_failures = []
        for t in incorrect_traces:
            total_uncertainties = t.get("metrics", {}).get("total_uncertainties", 0)
            if total_uncertainties == 0:
                no_uncertainty_failures.append(t)

        if no_uncertainty_failures:
            issues.append(
                StemmingIssue(
                    name="Uncertainty Blindness",
                    description="No uncertainty markers on incorrect answers",
                    frequency=len(no_uncertainty_failures) / total_incorrect,
                    severity="high",
                    examples=[t["problem"]["id"] for t in no_uncertainty_failures[:3]],
                    recommendation="Explicitly prompt for uncertainty; require doubt expression",
                )
            )

        # Self-correction failure to fix detected errors
        # Errors detected but no corrections made

        no_correction = []
        for t in incorrect_traces:
            metrics = t.get("metrics", {})
            errors = metrics.get("total_errors_detected", 0)
            corrections = metrics.get("total_corrections_made", 0)
            if errors > 0 and corrections == 0:
                no_correction.append(t)

        if no_correction:
            issues.append(
                StemmingIssue(
                    name="Self-Correction Failure",
                    description="Errors detected but not corrected",
                    frequency=len(no_correction) / total_incorrect,
                    severity="high",
                    examples=[t["problem"]["id"] for t in no_correction[:3]],
                    recommendation="Implement mandatory retry on error detection; add correction phase",
                )
            )

        self.issues = sorted(issues, key=lambda x: -x.frequency)
        return self


class ResearchAnalyzer:
    """Complete analysis suite for IMRA-1 research."""

    def __init__(self, traces_dir: Path) -> None:
        self.traces_dir = Path(traces_dir)
        self.traces: list[dict] = []

    def load_traces(self) -> int:
        """Load all trace files from directory."""
        self.traces = []
        for f in self.traces_dir.glob("trace_*.json"):
            with open(f) as fp:
                self.traces.append(json.load(fp))
        return len(self.traces)

    def full_analysis(self) -> dict:
        """Run complete analysis suite."""
        if not self.traces:
            self.load_traces()

        confidence = ConfidenceCalibration().assess(self.traces)
        errors = ErrorAnalysis().analyze(self.traces)
        recovery = RecoveryAnalysis().analyze(self.traces)
        coverage = ProblemCoverage().analyze(self.traces)
        stemming = StemmingAnalysis().analyze(self.traces)

        # Summary stats

        total = len(self.traces)
        correct = sum(1 for t in self.traces if t.get("outcome", {}).get("is_correct", False))

        return {
            "summary": {
                "total_problems": total,
                "correct": correct,
                "accuracy": correct / total if total > 0 else 0,
                "timestamp": datetime.now().isoformat(),
            },
            "confidence_calibration": asdict(confidence),
            "error_analysis": asdict(errors),
            "recovery_analysis": asdict(recovery),
            "problem_coverage": asdict(coverage),
            "stemming_issues": [asdict(i) for i in stemming.issues],
        }

    def generate_report(self) -> str:
        """Generate human-readable research report."""
        analysis = self.full_analysis()
        summary = analysis["summary"]

        lines = [
            "",
            "IMRA-1 RESEARCH ANALYSIS REPORT",
            f"Generated: {summary['timestamp']}",
            "",
            "EXECUTIVE SUMMARY",
            "",
            f"Total Problems: {summary['total_problems']}",
            f"Correct: {summary['correct']}",
            f"Accuracy: {summary['accuracy']:.1%}",
            "",
            "CONFIDENCE CALIBRATION",
            "",
        ]

        cal = analysis["confidence_calibration"]
        lines.extend(
            [
                f"High-confidence accuracy: {cal['high_confidence_accuracy']:.1%}",
                f"Overconfidence rate: {cal['overconfidence_rate']:.1%}",
                f"Underconfidence rate: {cal['underconfidence_rate']:.1%}",
            ]
        )

        lines.extend(
            [
                "",
                "RECOVERY ANALYSIS",
                "",
            ]
        )
        rec = analysis["recovery_analysis"]
        lines.extend(
            [
                f"Mistakes detected: {rec['mistakes_detected']}",
                f"Mistakes corrected: {rec['mistakes_corrected']}",
                f"Correction rate: {rec['correction_rate']:.1%}",
            ]
        )

        lines.extend(
            [
                "",
                "PROBLEM COVERAGE",
                "",
            ]
        )
        cov = analysis["problem_coverage"]
        lines.append("By Category:")
        for cat, data in sorted(cov["by_category"].items(), key=lambda x: -x[1]["accuracy"]):
            lines.append(f"  {cat}: {data['accuracy']:.1%} ({data['correct']}/{data['total']})")

        lines.extend(
            [
                "",
                "STEMMING ISSUES (Root Causes)",
                "",
            ]
        )

        for i, issue in enumerate(analysis["stemming_issues"], 1):
            lines.extend(
                [
                    f"\n{i}. {issue['name']} [{issue['severity'].upper()}]",
                    f"   Frequency: {issue['frequency']:.1%} of failures",
                    f"   {issue['description']}",
                    f"   Examples: {', '.join(issue['examples'])}",
                    f"   → {issue['recommendation']}",
                ]
            )

        lines.extend(["", "END REPORT", ""])

        return "\n".join(lines)

    def save_analysis(self, output_path: Path) -> tuple[Path, Path]:
        """Save analysis as both JSON and report."""
        analysis = self.full_analysis()
        report = self.generate_report()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        json_path = output_path.with_suffix(".json")
        txt_path = output_path.with_suffix(".txt")

        with open(json_path, "w") as f:
            json.dump(analysis, f, indent=2)

        with open(txt_path, "w") as f:
            f.write(report)

        return json_path, txt_path


if __name__ == "__main__":
    import sys

    traces_dir = Path("research_data/traces")
    if len(sys.argv) > 1:
        traces_dir = Path(sys.argv[1])

    analyzer = ResearchAnalyzer(traces_dir)
    count = analyzer.load_traces()

    if count == 0:
        print(f"No traces found in {traces_dir}")
        sys.exit(1)

    print(analyzer.generate_report())

    json_path, txt_path = analyzer.save_analysis(traces_dir / "analysis")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")

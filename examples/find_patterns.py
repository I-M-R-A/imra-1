#!/usr/bin/env python3

"""IMRA-1 Automated Pattern Finder.

This is the REAL framework - it automatically:
1. Reads ALL traces
2. Finds common failure patterns
3. Identifies root causes (stemming issues)
4. Categorizes errors by type
5. Produces actionable research insights

You don't read traces manually - the framework finds patterns FOR you.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class PatternFinder:
    """Automatically finds stemming issues across all traces."""

    def __init__(self, data_dir: str = "research_data") -> None:
        self.data_dir = Path(data_dir)
        self.traces: list[dict] = []

    def load_traces(self) -> int:
        """Load all traces recursively."""
        for f in self.data_dir.rglob("*.json"):
            if "report" in f.name:
                continue
            try:
                with open(f) as file:
                    trace = json.load(file)
                    trace["_file"] = str(f)
                    self.traces.append(trace)
            except (json.JSONDecodeError, OSError):
                pass
        return len(self.traces)

    def analyze(self) -> dict[str, Any]:
        """Run complete analysis and find patterns."""
        print()
        print("IMRA-1 AUTOMATED PATTERN ANALYSIS")
        print()

        n = self.load_traces()
        print(f"\nLoaded {n} traces for analysis\n")

        if n == 0:
            print("No traces found!")
            return {}

        correct = []
        incorrect = []

        for t in self.traces:
            outcome = t.get("outcome", {})
            if outcome.get("answer_correct"):
                correct.append(t)
            else:
                incorrect.append(t)

        print(f"[+] Correct answers: {len(correct)}")
        print(f"[-] Incorrect answers: {len(incorrect)}")
        print(f"Accuracy: {len(correct) / n * 100:.1f}%\n")

        print()
        print("STEMMING ISSUES (Root Causes of Failures)")
        print()

        issues = self._find_stemming_issues(incorrect)

        for i, issue in enumerate(issues, 1):
            print(f"\n--- Issue #{i}: {issue['name']} ---")
            print(f"Frequency: {issue['count']} traces ({issue['percentage']:.0f}% of failures)")
            print(f"Description: {issue['description']}")
            print("Evidence:")
            for ev in issue["evidence"][:3]:
                print(f"  • {ev}")

        print()
        print("CONFIDENCE CALIBRATION ANALYSIS")
        print()

        conf_analysis = self._analyze_confidence()
        print(f"\nCorrect answers - Mean confidence: {conf_analysis['correct_conf']:.2f}")
        print(f"Incorrect answers - Mean confidence: {conf_analysis['incorrect_conf']:.2f}")
        print(f"\nCalibration Status: {conf_analysis['calibration_status']}")

        print()
        print("COGNITIVE PHASE ANALYSIS")
        print()

        phase_analysis = self._analyze_phases()
        for phase, stats in phase_analysis.items():
            print(f"\n{phase.upper()}:")
            print(f"  Mean duration: {stats['mean_duration']:.1f}s")
            print(f"  Errors detected: {stats['errors_detected']}")
            print(f"  Corrections made: {stats['corrections_made']}")

        print()
        print("PROBLEM TYPE VULNERABILITY")
        print()

        type_analysis = self._analyze_by_type()
        for ptype, stats in type_analysis.items():
            status = "[+]" if stats["accuracy"] > 0.5 else "[-]"
            print(f"\n{status} {ptype}:")
            print(
                f"  Accuracy: {stats['accuracy'] * 100:.0f}% ({stats['correct']}/{stats['total']})"
            )
            if stats["common_failure"]:
                print(f"  Common failure: {stats['common_failure']}")

        print()
        print("ACTIONABLE RECOMMENDATIONS")
        print()

        recommendations = self._generate_recommendations(issues, conf_analysis, type_analysis)
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['title']}")
            print(f"   Problem: {rec['problem']}")
            print(f"   Solution: {rec['solution']}")

        return {
            "summary": {
                "total_traces": n,
                "correct": len(correct),
                "incorrect": len(incorrect),
                "accuracy": len(correct) / n,
            },
            "stemming_issues": issues,
            "confidence_analysis": conf_analysis,
            "phase_analysis": phase_analysis,
            "type_analysis": type_analysis,
            "recommendations": recommendations,
        }

    def _find_stemming_issues(self, incorrect: list[dict]) -> list[dict]:
        """Automatically identify root causes of failures."""
        issues = []

        if not incorrect:
            return issues

        semantic_failures = []
        for t in incorrect:
            problem = t.get("problem", {}).get("statement", "") or t.get("problem_statement", "")
            if any(
                phrase in problem.lower()
                for phrase in ["more than", "less than", "difference", "sum of"]
            ):
                reasoning = json.dumps(t.get("reasoning_trace", []))
                if "equation" not in reasoning.lower() or "variable" not in reasoning.lower():
                    semantic_failures.append(
                        {
                            "problem": problem[:80],
                            "issue": "Did not set up algebraic equation for relational language",
                        }
                    )

        if semantic_failures:
            issues.append(
                {
                    "name": "Semantic Parsing Failure",
                    "count": len(semantic_failures),
                    "percentage": len(semantic_failures) / len(incorrect) * 100,
                    "description": "AI misinterpreted relational language (e.g., 'X more than Y') as absolute values instead of algebraic constraints.",
                    "evidence": [f["problem"] for f in semantic_failures],
                }
            )

        verification_failures = []
        for t in incorrect:
            steps = t.get("reasoning_trace", [])
            for step in steps:
                if step.get("phase") == "verification":
                    parsed = step.get("parsed_output", {})
                    overall = parsed.get("overall_assessment", {})
                    if overall.get("answer_likely_correct"):
                        verification_failures.append(
                            {
                                "problem": t.get("problem", {}).get("statement", "")[:80],
                                "issue": "Verifier said 'correct' on wrong answer",
                            }
                        )

        if verification_failures:
            issues.append(
                {
                    "name": "Verification Blind Spot",
                    "count": len(verification_failures),
                    "percentage": len(verification_failures) / len(incorrect) * 100,
                    "description": "VERIFY phase approved incorrect answers, failing to catch errors.",
                    "evidence": [f["issue"] for f in verification_failures],
                }
            )

        overconfident = []
        for t in incorrect:
            conf = t.get("outcome", {}).get("final_confidence", 0)
            if conf >= 0.9:
                overconfident.append(
                    {"problem": t.get("problem", {}).get("statement", "")[:60], "confidence": conf}
                )

        if overconfident:
            issues.append(
                {
                    "name": "Systematic Overconfidence",
                    "count": len(overconfident),
                    "percentage": len(overconfident) / len(incorrect) * 100,
                    "description": "AI reported high confidence (≥0.9) on incorrect answers.",
                    "evidence": [
                        f"Confidence {f['confidence']:.1f} on: {f['problem']}"
                        for f in overconfident
                    ],
                }
            )

        method_errors = []
        for t in incorrect:
            steps = t.get("reasoning_trace", [])
            for step in steps:
                if step.get("phase") == "perception":
                    approach = step.get("parsed_output", {}).get("proposed_approach", {})
                    if approach.get("method"):
                        method_errors.append(
                            {
                                "method": approach.get("method"),
                                "problem": t.get("problem", {}).get("statement", "")[:60],
                            }
                        )

        if method_errors:
            methods = defaultdict(int)
            for m in method_errors:
                methods[m["method"]] += 1
            most_common = max(methods.items(), key=lambda x: x[1])

            issues.append(
                {
                    "name": "Flawed Approach Selection",
                    "count": len(method_errors),
                    "percentage": len(method_errors) / len(incorrect) * 100,
                    "description": f"AI chose approaches that led to errors. Most common: '{most_common[0]}'",
                    "evidence": [f"{m['method']}: {m['problem']}" for m in method_errors[:3]],
                }
            )

        return issues

    def _analyze_confidence(self) -> dict:
        """Analyze confidence calibration."""
        correct_confs = []
        incorrect_confs = []

        for t in self.traces:
            conf = t.get("outcome", {}).get("final_confidence", 0)
            if t.get("outcome", {}).get("answer_correct"):
                correct_confs.append(conf)
            else:
                incorrect_confs.append(conf)

        correct_mean = sum(correct_confs) / len(correct_confs) if correct_confs else 0
        incorrect_mean = sum(incorrect_confs) / len(incorrect_confs) if incorrect_confs else 0

        if incorrect_mean > 0.8:
            status = "[!]  POORLY CALIBRATED - High confidence on wrong answers"
        elif correct_mean > incorrect_mean + 0.2:
            status = "[+] WELL CALIBRATED - Higher confidence on correct answers"
        else:
            status = "[!]  UNCALIBRATED - No correlation between confidence and correctness"

        return {
            "correct_conf": correct_mean,
            "incorrect_conf": incorrect_mean,
            "calibration_status": status,
        }

    def _analyze_phases(self) -> dict:
        """Analyze cognitive phases."""
        phases = defaultdict(lambda: {"durations": [], "errors": 0, "corrections": 0})

        for t in self.traces:
            for step in t.get("reasoning_trace", []):
                phase = step.get("phase", "unknown")
                dur = step.get("duration_seconds", 0)
                phases[phase]["durations"].append(dur)
                if step.get("error_detected"):
                    phases[phase]["errors"] += 1
                if step.get("correction_attempted"):
                    phases[phase]["corrections"] += 1

        result = {}
        for phase, data in phases.items():
            result[phase] = {
                "mean_duration": (
                    sum(data["durations"]) / len(data["durations"]) if data["durations"] else 0
                ),
                "errors_detected": data["errors"],
                "corrections_made": data["corrections"],
            }

        return result

    def _analyze_by_type(self) -> dict:
        """Analyze performance by problem type."""
        types = defaultdict(lambda: {"correct": 0, "total": 0, "failures": []})

        for t in self.traces:
            ptype = t.get("problem", {}).get("category", "unknown")
            types[ptype]["total"] += 1
            if t.get("outcome", {}).get("answer_correct"):
                types[ptype]["correct"] += 1
            else:
                types[ptype]["failures"].append(t.get("problem", {}).get("statement", "")[:50])

        result = {}
        for ptype, data in types.items():
            result[ptype] = {
                "correct": data["correct"],
                "total": data["total"],
                "accuracy": data["correct"] / data["total"] if data["total"] else 0,
                "common_failure": data["failures"][0] if data["failures"] else None,
            }

        return result

    def _generate_recommendations(self, issues, conf_analysis, type_analysis) -> list[dict]:
        """Generate actionable recommendations."""
        recs = []

        for issue in issues:
            if issue["name"] == "Semantic Parsing Failure":
                recs.append(
                    {
                        "title": "Add Equation Extraction Step",
                        "problem": "AI misinterprets relational language in word problems",
                        "solution": "Insert a dedicated phase that explicitly extracts algebraic equations before solving",
                    }
                )
            elif issue["name"] == "Verification Blind Spot":
                recs.append(
                    {
                        "title": "Use Cross-Architecture Verification",
                        "problem": "VERIFY phase fails to catch errors made by REASON phase",
                        "solution": "Use a different LLM family for verification, or add numerical re-computation",
                    }
                )
            elif issue["name"] == "Systematic Overconfidence":
                recs.append(
                    {
                        "title": "Implement Confidence Calibration",
                        "problem": "Confidence scores don't correlate with correctness",
                        "solution": "Train/prompt models to output uncertainty, or use external calibration",
                    }
                )

        for ptype, stats in type_analysis.items():
            if stats["accuracy"] < 0.5 and stats["total"] > 0:
                recs.append(
                    {
                        "title": f"Improve {ptype.title()} Handling",
                        "problem": f"Low accuracy ({stats['accuracy'] * 100:.0f}%) on {ptype} problems",
                        "solution": f"Add specialized prompts or tools for {ptype} problem type",
                    }
                )

        return recs


def main() -> None:
    finder = PatternFinder()
    results = finder.analyze()

    output = Path("research_data/pattern_analysis.json")
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nFull analysis saved to: {output}")
    print()
    print("THIS IS THE FRAMEWORK:")
    print()
    print("""
1. Run experiments → Traces collected automatically
2. Run this analyzer → Patterns found automatically
3. Read the output → Stemming issues identified
4. Apply fixes → Re-run experiments
5. Compare results → Measure improvement

You don't read 4 JSON files manually.
The framework reads 100s of traces and tells you:
  - What's failing
  - Why it's failing
  - How to fix it
""")


if __name__ == "__main__":
    main()

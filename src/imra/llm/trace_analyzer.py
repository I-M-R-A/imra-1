"""Trace Analyzer for IMRA-1 Deliberation Traces.

Analyzes deliberation traces to extract insights about:
- Meta-cognitive quality (uncertainty flagging, self-correction)
- Reasoning patterns (which strategies work)
- System calibration (confidence vs accuracy)
- Learning opportunities

This is how IMRA-1 improves over time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TraceAnalyzer:
    """Analyzes deliberation traces for meta-learning."""

    def __init__(self, trace: dict[str, Any]) -> None:
        self.trace = trace
        self.events = trace.get("events", [])

    def get_phase_events(self, phase: str) -> list[dict[str, Any]]:
        """Get all events for a specific phase."""
        return [e for e in self.events if e.get("phase") == phase]

    def get_actor_events(self, actor: str) -> list[dict[str, Any]]:
        """Get all events from a specific actor."""
        return [e for e in self.events if e.get("actor") == actor]

    def analyze_uncertainty_flagging(self) -> dict[str, Any]:
        """Analyze how well the system flagged uncertainty."""
        perceive_events = self.get_phase_events("perceive_sketch") + self.get_phase_events(
            "perceive_sketch_alternative"
        )

        total_flags = []
        for event in perceive_events:
            payload = event.get("payload", {})
            sketch = payload.get("sketch", {})
            flags = sketch.get("uncertainty_flags", [])
            if flags:
                total_flags.extend(flags)

        high_severity = [f for f in total_flags if f.get("severity") == "high"]
        medium_severity = [f for f in total_flags if f.get("severity") == "medium"]
        low_severity = [f for f in total_flags if f.get("severity") == "low"]

        return {
            "total_flags": len(total_flags),
            "high_severity": len(high_severity),
            "medium_severity": len(medium_severity),
            "low_severity": len(low_severity),
            "flags": total_flags,
        }

    def analyze_sketch_diversity(self) -> dict[str, Any]:
        """Analyze whether sketches were genuinely diverse."""
        perceive1 = self.get_phase_events("perceive_sketch")
        perceive2 = self.get_phase_events("perceive_sketch_alternative")

        if not perceive1 or not perceive2:
            return {"diverse": False, "reason": "Missing sketches"}

        sketch1 = perceive1[0].get("payload", {}).get("sketch", {})
        sketch2 = perceive2[0].get("payload", {}).get("sketch", {})

        critique = sketch2.get("critique_of_first_approach", {})
        has_critique = bool(critique)

        approach1 = sketch1.get("proposed_approach", {}).get("strategy", "")
        approach2 = sketch2.get("alternative_approach", {}).get("strategy", "")

        return {
            "sketch1_confidence": sketch1.get("confidence", 0),
            "sketch2_confidence": sketch2.get("confidence", 0),
            "critique_provided": has_critique,
            "critique_content": critique,
            "approach1": approach1[:100] if approach1 else None,
            "approach2": approach2[:100] if approach2 else None,
        }

    def analyze_execution_quality(self) -> dict[str, Any]:
        """Analyze how well REASON executed the sketch."""
        exec_events = self.get_phase_events("reason_execute")

        if not exec_events:
            return {"executed": False}

        exec_data = exec_events[0].get("payload", {})
        execution = exec_data.get("execution", {})

        return {
            "executed": True,
            "final_answer": exec_data.get("final_answer"),
            "confidence": exec_data.get("confidence", 0),
            "difficulties": execution.get("difficulties_encountered", []),
            "intermediate_checks": execution.get("intermediate_checks", []),
            "sketch_assessment": execution.get("sketch_quality_assessment", {}),
        }

    def analyze_verification_quality(self) -> dict[str, Any]:
        """Analyze how well VERIFY performed its role."""
        consensus = self.get_phase_events("verify_consensus")
        final = self.get_phase_events("verify_final")

        result = {"consensus_phase": False, "final_phase": False}

        if consensus:
            c_data = consensus[0].get("payload", {})
            result["consensus_phase"] = True
            result["consensus_confidence"] = c_data.get("confidence", 0)
            result["red_flags"] = c_data.get("red_flags", [])
            analysis = c_data.get("analysis", {})
            result["divergence_points"] = analysis.get("divergence_points", [])

        if final:
            f_data = final[0].get("payload", {})
            result["final_phase"] = True
            result["final_accepted"] = f_data.get("accepted", False)
            result["final_confidence"] = f_data.get("confidence", 0)
            verification = f_data.get("verification", {})
            result["deliberation_grade"] = verification.get("deliberation_quality", {}).get(
                "overall_grade"
            )
            result["meta_learning"] = verification.get("meta_learning", {})

        return result

    def get_timing_analysis(self) -> dict[str, Any]:
        """Analyze time spent in each phase."""
        phase_times = {}
        for event in self.events:
            phase = event.get("phase", "unknown")
            duration = event.get("duration_seconds", 0)
            if phase in phase_times:
                phase_times[phase] += duration
            else:
                phase_times[phase] = duration

        total = sum(phase_times.values())

        return {
            "phase_durations": phase_times,
            "total_duration": total,
            "slowest_phase": max(phase_times, key=phase_times.get) if phase_times else None,
        }

    def get_full_analysis(self) -> dict[str, Any]:
        """Get complete analysis of the trace."""
        return {
            "uncertainty_analysis": self.analyze_uncertainty_flagging(),
            "sketch_diversity": self.analyze_sketch_diversity(),
            "execution_quality": self.analyze_execution_quality(),
            "verification_quality": self.analyze_verification_quality(),
            "timing": self.get_timing_analysis(),
        }


def analyze_trace_file(filepath: Path) -> dict[str, Any]:
    """Load and analyze a saved trace file."""
    with open(filepath) as f:
        data = json.load(f)

    trace = data.get("trace", data)
    analyzer = TraceAnalyzer(trace)
    return analyzer.get_full_analysis()


def compare_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare multiple traces to find patterns."""
    analyses = [TraceAnalyzer(t).get_full_analysis() for t in traces]

    avg_confidence = (
        sum(a["execution_quality"].get("confidence", 0) for a in analyses) / len(analyses)
        if analyses
        else 0
    )

    avg_uncertainty_flags = (
        sum(a["uncertainty_analysis"]["total_flags"] for a in analyses) / len(analyses)
        if analyses
        else 0
    )

    return {
        "trace_count": len(traces),
        "average_confidence": avg_confidence,
        "average_uncertainty_flags": avg_uncertainty_flags,
        "analyses": analyses,
    }

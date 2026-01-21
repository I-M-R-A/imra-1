"""IMRA-1 Research Orchestrator.

This orchestrator is designed for RESEARCH, not just problem-solving.
The goal is to produce rich cognitive traces that reveal:
- How AI reasons
- Where AI is uncertain
- How AI fails and self-corrects
- What patterns emerge in AI cognition

The traces are the research data product.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from imra.llm.ollama_client import OllamaClient
from imra.research.cognitive_trace import CognitiveTrace, ReasoningStep, UncertaintyMarker

# Research-focused prompts that explicitly ask for uncertainty and reasoning
PERCEIVE_RESEARCH_PROMPT = """You are part of a research study on AI cognition. Your responses will be analyzed by researchers.

PROBLEM TO ANALYZE:
{problem}

YOUR TASK: Generate your initial understanding and approach.

BE EXPLICIT ABOUT YOUR THINKING:
1. How do you interpret this problem? What is it really asking?
2. What relevant knowledge/patterns do you recognize?
3. What approach would you take? Why?
4. WHERE ARE YOU UNCERTAIN? Be specific about what you don't know or aren't sure about.
5. What assumptions are you making? Why?

RESPOND IN JSON:
{{
  "problem_interpretation": "your understanding of what's asked",
  "recognized_patterns": ["pattern1", "pattern2"],
  "proposed_approach": {{
    "method": "your approach",
    "reasoning": "why this approach",
    "steps": ["step1", "step2"]
  }},
  "uncertainties": [
    {{
      "type": "epistemic|model|semantic|confidence",
      "what": "what you're uncertain about",
      "why": "why you're uncertain",
      "severity": "low|medium|high|critical"
    }}
  ],
  "assumptions": [
    {{"assumption": "what you assume", "risk_if_wrong": "what happens if wrong"}}
  ],
  "confidence": 0.0 to 1.0,
  "confidence_explanation": "why this confidence level"
}}

IMPORTANT: Researchers need to understand your reasoning. Be transparent about doubts."""


REASON_RESEARCH_PROMPT = """You are part of a research study on AI cognition. Your step-by-step reasoning will be analyzed.

PROBLEM:
{problem}

APPROACH TO EXECUTE:
{approach}

YOUR TASK: Execute this approach step-by-step, showing ALL your work.

FOR EACH STEP:
- State what you're doing and why
- Show the calculation/logic
- Flag any uncertainty or confusion
- If something seems wrong, say so

RESPOND IN JSON:
{{
  "execution_steps": [
    {{
      "step_number": 1,
      "action": "what you did",
      "reasoning": "why you did it",
      "result": "outcome",
      "uncertainty": null or {{"what": "...", "severity": "low|medium|high"}},
      "possible_error": null or "description of potential mistake"
    }}
  ],
  "final_answer": "your computed answer",
  "answer_confidence": 0.0 to 1.0,
  "reasoning_quality_self_assessment": {{
    "clear_steps": true/false,
    "verified_intermediate": true/false,
    "spotted_issues": ["any problems noticed"]
  }}
}}

Show your work. Flag doubts. This is research data."""


VERIFY_RESEARCH_PROMPT = """You are the CRITIC in a research study on AI cognition. Your job is to find problems.

PROBLEM:
{problem}

REASONING TO VERIFY:
{reasoning}

YOUR TASK: Critically examine this reasoning. Find weaknesses.

LOOK FOR:
1. Logical errors or gaps
2. Computational mistakes
3. Unjustified assumptions
4. Missing edge cases
5. Overconfidence

RESPOND IN JSON:
{{
  "verification_results": {{
    "logic_check": {{"passed": true/false, "issues": ["list issues"]}},
    "computation_check": {{"passed": true/false, "issues": ["list issues"]}},
    "assumption_check": {{"passed": true/false, "issues": ["list issues"]}},
    "completeness_check": {{"passed": true/false, "issues": ["list issues"]}}
  }},
  "errors_found": [
    {{"type": "logical|computational|assumption", "description": "...", "severity": "low|medium|high|critical"}}
  ],
  "suggested_corrections": [
    {{"error_ref": "which error", "correction": "how to fix"}}
  ],
  "overall_assessment": {{
    "reasoning_valid": true/false,
    "answer_likely_correct": true/false,
    "confidence_appropriate": true/false,
    "confidence": 0.0 to 1.0
  }},
  "meta_observations": {{
    "reasoning_style": "description of how the AI reasoned",
    "notable_patterns": ["patterns observed"],
    "cognitive_biases_detected": ["any biases noticed"]
  }}
}}

Be critical. Researchers need to know where AI reasoning fails."""


REFLECTION_RESEARCH_PROMPT = """You are generating RESEARCH INSIGHTS about AI cognition.

COMPLETE DELIBERATION:
{trace}

YOUR TASK: Analyze this reasoning process for research purposes.

ANALYZE:
1. How did the AI approach this problem? What strategy did it use?
2. Where did uncertainty arise? Was it appropriate?
3. What errors occurred? Why did they happen?
4. What does this tell us about AI reasoning?

RESPOND IN JSON:
{{
  "reasoning_analysis": {{
    "approach_taken": "description",
    "approach_effectiveness": "assessment",
    "key_decision_points": ["moments where reasoning could have gone differently"]
  }},
  "uncertainty_analysis": {{
    "appropriate_uncertainty": ["where doubt was justified"],
    "missing_uncertainty": ["where doubt should have been but wasn't"],
    "overconfidence_instances": ["where confidence was too high"]
  }},
  "error_analysis": {{
    "errors_made": ["list of errors"],
    "error_causes": ["why errors happened"],
    "error_detection": "how well were errors caught?"
  }},
  "cognitive_insights": {{
    "reasoning_patterns": ["patterns observed"],
    "failure_modes": ["types of failures"],
    "strengths": ["what worked well"]
  }},
  "research_conclusions": [
    "insight 1 about AI cognition",
    "insight 2 about AI cognition"
  ]
}}

This produces research findings. Be analytical and insightful."""


class ResearchOrchestrator:
    """Orchestrator designed to produce research-grade cognitive traces.

    The primary output is the CognitiveTrace - a detailed record of
    AI reasoning that researchers can analyze to understand AI cognition.
    """

    def __init__(self, models: dict[str, str], verbose: bool = True) -> None:
        self.models = models
        self.clients = {role: OllamaClient(model=name) for role, name in models.items()}
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[IMRA-1 Research] {datetime.now().strftime('%H:%M:%S')} | {message}")

    def _call_llm(self, role: str, prompt: str) -> tuple[str, float]:
        """Call LLM and return response with timing."""
        self._log(f"Calling {role.upper()} ({self.models[role]})...")
        start = time.time()
        response = self.clients[role].send_prompt(prompt)
        duration = time.time() - start
        self._log(f"  Response received in {duration:.1f}s")
        return response, duration

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM output."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            return {"raw": text, "parse_error": True}

    def _extract_uncertainties(self, parsed: dict[str, Any]) -> list[UncertaintyMarker]:
        """Extract uncertainty markers from parsed LLM output."""
        markers = []
        uncertainties = parsed.get("uncertainties", [])

        if isinstance(uncertainties, list):
            for u in uncertainties:
                if isinstance(u, dict):
                    markers.append(
                        UncertaintyMarker(
                            uncertainty_type=u.get("type", "unknown"),
                            description=u.get("what", u.get("description", "")),
                            severity=u.get("severity", "medium"),
                            source_text=json.dumps(u),
                        )
                    )

        return markers

    def run_research_deliberation(
        self, problem: str, expected_answer: Any = None, problem_category: str = ""
    ) -> CognitiveTrace:
        """Run a deliberation and produce a research-grade cognitive trace.

        This is the main entry point. It produces a CognitiveTrace that
        contains everything needed for research analysis.
        """
        self._log("")
        self._log("IMRA-1 RESEARCH DELIBERATION")
        self._log("")
        self._log(f"Problem: {problem[:80]}...")

        start_time = time.time()
        trace = CognitiveTrace(
            problem_statement=problem,
            problem_category=problem_category,
            expected_answer=expected_answer,
        )

        # PHASE 1: PERCEPTION - Initial understanding
        self._log("\n--- PHASE 1: PERCEPTION ---")
        perceive_prompt = PERCEIVE_RESEARCH_PROMPT.format(problem=problem)
        perceive_raw, perceive_dur = self._call_llm("perceive1", perceive_prompt)
        perceive_parsed = self._parse_json(perceive_raw)

        perceive_step = ReasoningStep(
            phase="perception",
            actor="perceive1",
            input_context=problem,
            raw_output=perceive_raw,
            parsed_output=perceive_parsed,
            stated_confidence=(
                float(perceive_parsed.get("confidence", 0))
                if isinstance(perceive_parsed.get("confidence"), (int, float))
                else 0
            ),
            stated_reasoning=perceive_parsed.get("confidence_explanation", ""),
            uncertainties=self._extract_uncertainties(perceive_parsed),
            duration_seconds=perceive_dur,
        )
        trace.add_step(perceive_step)

        # PHASE 2: ALTERNATIVE PERCEPTION - Second view
        self._log("\n--- PHASE 2: ALTERNATIVE PERCEPTION ---")
        alt_prompt = (
            PERCEIVE_RESEARCH_PROMPT.format(problem=problem)
            + f"\n\nNOTE: A previous analysis suggested: {json.dumps(perceive_parsed.get('proposed_approach', {}), indent=2)}\n\nProvide an ALTERNATIVE perspective or approach."
        )
        alt_raw, alt_dur = self._call_llm("perceive2", alt_prompt)
        alt_parsed = self._parse_json(alt_raw)

        alt_step = ReasoningStep(
            phase="alternative_perception",
            actor="perceive2",
            input_context=problem,
            raw_output=alt_raw,
            parsed_output=alt_parsed,
            stated_confidence=(
                float(alt_parsed.get("confidence", 0))
                if isinstance(alt_parsed.get("confidence"), (int, float))
                else 0
            ),
            stated_reasoning=alt_parsed.get("confidence_explanation", ""),
            uncertainties=self._extract_uncertainties(alt_parsed),
            duration_seconds=alt_dur,
        )
        trace.add_step(alt_step)

        # PHASE 3: REASONING - Execute the approach
        self._log("\n--- PHASE 3: REASONING ---")
        approach = perceive_parsed.get("proposed_approach", {})
        reason_prompt = REASON_RESEARCH_PROMPT.format(
            problem=problem, approach=json.dumps(approach, indent=2)
        )
        reason_raw, reason_dur = self._call_llm("reason", reason_prompt)
        reason_parsed = self._parse_json(reason_raw)

        # Check for errors in execution
        execution_steps = reason_parsed.get("execution_steps", [])
        errors_in_execution = []
        for step in execution_steps:
            if isinstance(step, dict) and step.get("possible_error"):
                errors_in_execution.append(step.get("possible_error"))

        reason_step = ReasoningStep(
            phase="reasoning",
            actor="reason",
            input_context=json.dumps(approach),
            raw_output=reason_raw,
            parsed_output=reason_parsed,
            stated_confidence=(
                float(reason_parsed.get("answer_confidence", 0))
                if isinstance(reason_parsed.get("answer_confidence"), (int, float))
                else 0
            ),
            stated_reasoning=json.dumps(reason_parsed.get("reasoning_quality_self_assessment", {})),
            uncertainties=[],
            error_detected=len(errors_in_execution) > 0,
            error_description="; ".join(errors_in_execution) if errors_in_execution else "",
            duration_seconds=reason_dur,
        )

        # Extract uncertainties from execution steps
        for step in execution_steps:
            if isinstance(step, dict) and step.get("uncertainty"):
                u = step["uncertainty"]
                reason_step.uncertainties.append(
                    UncertaintyMarker(
                        uncertainty_type="computational",
                        location=f"Step {step.get('step_number', '?')}",
                        description=u.get("what", str(u)),
                        severity=u.get("severity", "medium"),
                    )
                )

        trace.add_step(reason_step)

        # PHASE 4: VERIFICATION - Critical analysis
        self._log("\n--- PHASE 4: VERIFICATION ---")
        verify_prompt = VERIFY_RESEARCH_PROMPT.format(
            problem=problem,
            reasoning=json.dumps({"approach": approach, "execution": reason_parsed}, indent=2),
        )
        verify_raw, verify_dur = self._call_llm("verify", verify_prompt)
        verify_parsed = self._parse_json(verify_raw)

        # Extract detected errors
        errors_found = verify_parsed.get("errors_found", [])
        corrections = verify_parsed.get("suggested_corrections", [])

        verify_step = ReasoningStep(
            phase="verification",
            actor="verify",
            input_context="Full reasoning trace",
            raw_output=verify_raw,
            parsed_output=verify_parsed,
            stated_confidence=(
                float(verify_parsed.get("overall_assessment", {}).get("confidence", 0))
                if isinstance(
                    verify_parsed.get("overall_assessment", {}).get("confidence"), (int, float)
                )
                else 0
            ),
            stated_reasoning=json.dumps(verify_parsed.get("meta_observations", {})),
            error_detected=len(errors_found) > 0,
            error_description=json.dumps(errors_found) if errors_found else "",
            correction_attempted=len(corrections) > 0,
            correction_description=json.dumps(corrections) if corrections else "",
            duration_seconds=verify_dur,
        )
        trace.add_step(verify_step)

        # PHASE 5: REFLECTION - Research insights
        self._log("\n--- PHASE 5: REFLECTION ---")
        reflect_prompt = REFLECTION_RESEARCH_PROMPT.format(
            trace=json.dumps(trace.to_dict(), indent=2, default=str)
        )
        reflect_raw, reflect_dur = self._call_llm("verify", reflect_prompt)
        reflect_parsed = self._parse_json(reflect_raw)

        reflect_step = ReasoningStep(
            phase="reflection",
            actor="verify",
            input_context="Complete trace",
            raw_output=reflect_raw,
            parsed_output=reflect_parsed,
            stated_reasoning=json.dumps(reflect_parsed.get("cognitive_insights", {})),
            duration_seconds=reflect_dur,
        )
        trace.add_step(reflect_step)

        # Finalize trace
        trace.final_answer = reason_parsed.get("final_answer")
        trace.final_confidence = (
            float(verify_parsed.get("overall_assessment", {}).get("confidence", 0))
            if isinstance(
                verify_parsed.get("overall_assessment", {}).get("confidence"), (int, float)
            )
            else 0
        )

        if expected_answer is not None:
            trace.answer_correct = str(trace.final_answer) == str(expected_answer)

        # Extract research insights
        trace.key_insights = reflect_parsed.get("research_conclusions", [])
        trace.failure_modes = reflect_parsed.get("cognitive_insights", {}).get("failure_modes", [])
        trace.reasoning_summary = reflect_parsed.get("reasoning_analysis", {}).get(
            "approach_taken", ""
        )

        trace.total_duration_seconds = time.time() - start_time

        self._log("\n" + "")
        self._log("DELIBERATION COMPLETE")
        self._log(f"Duration: {trace.total_duration_seconds:.1f}s")
        self._log(f"Final Answer: {trace.final_answer}")
        self._log(f"Confidence: {trace.final_confidence:.2f}")
        self._log(f"Uncertainties Logged: {len(trace.all_uncertainties)}")
        self._log(f"Errors Detected: {len(trace.errors_detected)}")
        self._log("")

        return trace

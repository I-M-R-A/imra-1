#!/usr/bin/env python3

"""IMRA-1 Reasoning Layers

The four interlocking layers of the meta-cognitive architecture:
├── PerceiveLayer: Intuition - probabilistic sketch of approaches
├── ReasonLayer: Deliberation - execute paths, track uncertainty
├── VerifyLayer: Self-Reflection - identify failures, contradictions
└── DiscussLayer: Collaboration - team discussion between modules
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .trace import (
    CognitiveTrace,
    DiscussionTurn,
    ErrorRecord,
    ErrorType,
    ReasoningPhase,
    ReasoningStep,
    TeamDiscussion,
    UncertaintyLevel,
    UncertaintyMarker,
)


@dataclass
class LayerConfig:
    """Configuration for a reasoning layer."""

    model_name: str
    timeout_seconds: float = 120.0
    temperature: float = 0.7
    max_retries: int = 2


@dataclass
class LayerOutput:
    """Output from a reasoning layer."""

    content: str
    confidence: float
    uncertainties: list[UncertaintyMarker]
    errors: list[ErrorRecord]
    alternatives: list[str]
    raw_response: Any
    duration: float


class ReasoningLayer(ABC):
    """Base class for all reasoning layers."""

    def __init__(self, config: LayerConfig, llm_client: Any) -> None:
        self.config = config
        self.llm = llm_client

    @abstractmethod
    def process(self, input_data: dict, trace: CognitiveTrace) -> LayerOutput:
        """Process input and return output with full trace."""
        pass

    def _call_llm(self, prompt: str) -> tuple[str, float]:
        """Call LLM and return response with timing."""
        import gc

        start = time.time()
        try:
            response = self.llm.generate(
                model=self.config.model_name,
                prompt=prompt,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": 1024,  # Reduced from 2048 for memory for my current pc setup, but you can adjust as needed if your hardware allows it.
                    "num_ctx": 2048,  # Limit context window
                },
                keep_alive=0,  # Unload model after generation to save memory
            )
            content = response.get("response", "")
            duration = time.time() - start

            # Let Python GC handle response cleanup (especially large objects)

            del response
            gc.collect()
            return content, duration
        except Exception as e:
            duration = time.time() - start
            gc.collect()
            return f"ERROR: {e}", duration

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from response."""
        patterns = [
            r"confidence[:\s]+([0-9.]+)",
            r"certainty[:\s]+([0-9.]+)",
            r"confidence[:\s]+(\d+)%",
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                val = float(match.group(1))
                return val / 100 if val > 1 else val
        return 0.7  # default moderate confidence (70%)

    def _detect_uncertainty_markers(
        self, text: str, phase: ReasoningPhase
    ) -> list[UncertaintyMarker]:
        """Detect expressions of uncertainty in response."""
        markers = []
        uncertainty_phrases = [
            ("might be", UncertaintyLevel.MEDIUM),
            ("could be", UncertaintyLevel.MEDIUM),
            ("possibly", UncertaintyLevel.MEDIUM),
            ("perhaps", UncertaintyLevel.MEDIUM),
            ("not sure", UncertaintyLevel.HIGH),
            ("uncertain", UncertaintyLevel.HIGH),
            ("unclear", UncertaintyLevel.HIGH),
            ("don't know", UncertaintyLevel.CRITICAL),
            ("cannot determine", UncertaintyLevel.CRITICAL),
            ("ambiguous", UncertaintyLevel.HIGH),
            ("assuming", UncertaintyLevel.MEDIUM),
            ("if we assume", UncertaintyLevel.MEDIUM),
        ]

        text_lower = text.lower()
        for phrase, level in uncertainty_phrases:
            if phrase in text_lower:
                idx = text_lower.index(phrase)
                context = text[max(0, idx - 30) : idx + len(phrase) + 30]
                markers.append(
                    UncertaintyMarker(
                        location=f"response:{idx}",
                        description=context.strip(),
                        level=level,
                        phase=phase,
                    )
                )

        return markers


class PerceiveLayer(ReasoningLayer):
    """Intuition Layer - produces probabilistic sketch of approaches.

    Like "gut feeling" in humans - doesn't need to be perfect, just reasonable.
    """

    SYSTEM_PROMPT = """You are the PERCEIVE module of IMRA-1, responsible for initial problem analysis.

Your role:
1. READ the problem carefully
2. IDENTIFY key information, constraints, and what's being asked
3. SKETCH possible approaches (don't solve yet)
4. FLAG any ambiguities or potential traps

IMPORTANT: Explain your thinking process! We want to understand WHY you see things the way you do.
Even if you're uncertain, explain your reasoning - this helps us understand how you think.

Output Format:
PROBLEM_TYPE: [category]
  WHY_THIS_TYPE: [Explain what makes you classify it this way]

KEY_INFORMATION:
- [info point]
  WHY_THIS_MATTERS: [Explain why this is important for solving]

POSSIBLE_APPROACHES:
1. [approach 1] - Confidence: [0.0-1.0]
   WHY_THIS_APPROACH: [Explain your reasoning for suggesting this]
   POTENTIAL_ISSUES: [What could go wrong with this approach?]
2. [approach 2] - Confidence: [0.0-1.0]
   WHY_THIS_APPROACH: [Explain your reasoning]
   POTENTIAL_ISSUES: [What could go wrong?]

POTENTIAL_TRAPS:
- [trap description]
  WHY_THIS_IS_TRICKY: [Explain what makes this dangerous]

RECOMMENDED_APPROACH: [your suggestion]
  WHY_I_RECOMMEND_THIS: [Explain your reasoning for this choice over alternatives]

OVERALL_CONFIDENCE: [0.0-1.0]
  CONFIDENCE_JUSTIFICATION: [Why are you this confident? What would increase/decrease it?]
"""

    def process(self, input_data: dict, trace: CognitiveTrace) -> LayerOutput:
        """Analyze problem and produce initial sketch."""
        problem_text = input_data.get("problem", "")[:800]  # Limit input size

        prompt = f"{self.SYSTEM_PROMPT}\n\nPROBLEM:\n{problem_text}"

        response, duration = self._call_llm(prompt)
        confidence = self._extract_confidence(response)
        uncertainties = self._detect_uncertainty_markers(response, ReasoningPhase.PERCEIVE)

        alternatives = []  # Extract possible approaches
        if "POSSIBLE_APPROACHES:" in response:
            approaches_section = response.split("POSSIBLE_APPROACHES:")[1]
            if "POTENTIAL_TRAPS" in approaches_section:
                approaches_section = approaches_section.split("POTENTIAL_TRAPS")[0]
            alternatives = [
                line.strip()
                for line in approaches_section.split("\n")
                if line.strip() and line.strip()[0].isdigit()
            ]

        step = ReasoningStep(
            step_id=f"perceive_{int(time.time())}",
            phase=ReasoningPhase.PERCEIVE,
            actor=self.config.model_name,
            action="initial_analysis",
            input_data=problem_text[:500],
            output_data=response[:1000],
            confidence=confidence,
            duration_seconds=duration,
            uncertainties=uncertainties,
            alternative_paths=alternatives,
        )
        trace.add_step(step)

        return LayerOutput(
            content=response,
            confidence=confidence,
            uncertainties=uncertainties,
            errors=[],
            alternatives=alternatives,
            raw_response=response,
            duration=duration,
        )


class ReasonLayer(ReasoningLayer):
    """Deliberation Layer - executes candidate paths, tracks uncertainty.

    Each step is auditable. Keeps track of:
    - Uncertainty at each point
    - Alternative solutions explored
    - Reasoning errors detected
    """

    SYSTEM_PROMPT = """You are the REASON module of IMRA-1, responsible for deliberate problem solving.

Given the problem and initial analysis from PERCEIVE, you must:
1. EXECUTE the recommended approach step by step
2. SHOW all work clearly with explanations
3. FLAG any points where you're uncertain
4. NOTE any errors you detect in your reasoning
5. CONSIDER alternatives if main approach stalls

IMPORTANT: Explain your thinking at each step! We want to understand WHY you make each decision.
Even if you're uncertain, explain your reasoning - this helps us understand how you solve problems.

Output Format:
APPROACH: [which approach you're using]
  WHY_THIS_APPROACH: [Why did you choose this over alternatives from PERCEIVE?]

REASONING:
Step 1: [...]
  WHY_THIS_STEP: [Explain your reasoning for this step]
  CONFIDENCE_HERE: [How sure are you about this step?]
Step 2: [...]
  WHY_THIS_STEP: [Explain your reasoning]
  CONFIDENCE_HERE: [How sure?]
...

UNCERTAINTIES:
- [uncertain point]
  WHY_UNCERTAIN: [Explain what specifically makes you unsure]
  WHAT_WOULD_HELP: [What information would resolve this uncertainty?]

ANSWER: [final answer]
  WHY_THIS_ANSWER: [Explain why you believe this is correct]
  HOW_I_GOT_HERE: [Brief summary of the key reasoning chain]

CONFIDENCE: [0.0-1.0]
  CONFIDENCE_JUSTIFICATION: [Why this confidence level? What could change it?]

ALTERNATIVE_ANSWERS: [if any]
  WHY_NOT_THESE: [Why did you reject these alternatives?]
"""

    def process(self, input_data: dict, trace: CognitiveTrace) -> LayerOutput:
        """Execute deliberate reasoning on the problem."""
        import gc

        from .tools import ComputationalTools, create_computation_prompt_addition

        problem_text = input_data.get("problem", "")[:800]  # Limit size
        perceive_output = input_data.get("perceive_output", "")[:600]  # Limit size

        tools_hint = create_computation_prompt_addition(problem_text)

        prompt = f"""{self.SYSTEM_PROMPT}

PROBLEM:
{problem_text}

INITIAL ANALYSIS (from PERCEIVE):
{perceive_output}
{tools_hint}

Now solve this step by step."""

        response, duration = self._call_llm(prompt)

        if "COMPUTE:" in response:
            compute_lines = [line for line in response.split("\n") if "COMPUTE:" in line]
            computations = []
            for line in compute_lines:
                expr = line.split("COMPUTE:")[1].strip()
                success, result, error = ComputationalTools.safe_eval(expr)
                if success:
                    computations.append(f"{expr} = {result}")
                else:
                    computations.append(f"{expr} → Error: {error}")

            if computations:
                response += "\n\nCOMPUTATION RESULTS:\n" + "\n".join(computations)
        confidence = self._extract_confidence(response)
        uncertainties = self._detect_uncertainty_markers(response, ReasoningPhase.REASON)

        answer = None
        if "ANSWER:" in response:
            answer_line = response.split("ANSWER:")[1].split("\n")[0]
            answer = answer_line.strip()

        errors = []
        error_indicators = ["mistake", "error", "wrong", "incorrect", "reconsider"]
        response_lower = response.lower()
        for indicator in error_indicators:
            if indicator in response_lower:
                errors.append(
                    ErrorRecord(
                        error_type=ErrorType.LOGICAL,
                        description=f"Self-detected issue: '{indicator}' mentioned",
                        phase_detected=ReasoningPhase.REASON,
                        phase_occurred=ReasoningPhase.REASON,
                        severity="medium",
                    )
                )

        step = ReasoningStep(
            step_id=f"reason_{int(time.time())}",
            phase=ReasoningPhase.REASON,
            actor=self.config.model_name,
            action="deliberate_solving",
            input_data=f"Problem + PERCEIVE output ({len(perceive_output)} chars)",
            output_data=response[
                :800
            ],  # Reduced from 1000 due to memory constraints for my current pc setup
            confidence=confidence,
            duration_seconds=duration,
            uncertainties=uncertainties,
            errors_detected=errors,
        )
        trace.add_step(step)
        gc.collect()

        return LayerOutput(
            content=response[:1000],  # Limit stored content
            confidence=confidence,
            uncertainties=uncertainties,
            errors=errors,
            alternatives=[],
            raw_response={"answer": answer, "full_response": response[:1000]},
            duration=duration,
        )


class VerifyLayer(ReasoningLayer):
    """Self-Reflection Layer - identifies failures and contradictions.

    Watches PERCEIVE and REASON, looking for:
    - Low-confidence areas
    - Contradictions
    - Errors in reasoning
    - When to flag for rethinking
    """

    SYSTEM_PROMPT = """You are the VERIFY module of IMRA-1, responsible for self-reflection and validation.

Your role is to CRITICALLY examine the reasoning process:
1. CHECK if the answer makes sense
2. VERIFY key calculations
3. IDENTIFY any logical flaws or contradictions
4. DETECT overconfidence (is the confidence justified?)
5. If you find errors, you MUST provide a SPECIFIC CORRECTED ANSWER, not vague statements

CRITICAL REQUIREMENTS FOR CORRECTIONS:
- If answer is wrong, provide the ACTUAL CORRECT VALUE/STATEMENT
- Do NOT say "Not applicable", "Cannot determine", or "Needs different approach"
- If you don't know the exact answer, make your best attempt with clear reasoning
- A specific wrong answer is better than no answer at all (for research purposes)

IMPORTANT: When you identify errors or issues, you MUST explain your reasoning in detail.
We want to understand WHY you believe something is an error - even if you're wrong.
Your explanations help us understand how verification systems think.

Be adversarial - actively look for problems, but explain your thought process.

Output Format:
VERIFICATION_STATUS: [VALID / INVALID / UNCERTAIN]
CONFIDENCE_ASSESSMENT: [Is the stated confidence justified? Why/why not?]
ERRORS_FOUND:
- [error description]
  WHY_THIS_IS_AN_ERROR: [Explain your reasoning for identifying this as an error]
CONTRADICTIONS:
- [contradiction description]
  WHY_THIS_IS_A_CONTRADICTION: [Explain your reasoning]
SANITY_CHECKS:
- [check 1]: PASS/FAIL
  EXPLANATION: [Why did it pass/fail?]
RECOMMENDATION: [accept answer / retry with different approach / flag for review]
  REASONING_FOR_RECOMMENDATION: [Explain why you recommend this]
CORRECTED_ANSWER: [MUST be a specific value/statement if you found errors - NO "not applicable" or vague responses]
  WHY_THIS_IS_CORRECT: [Explain your calculation/reasoning that led to this specific answer]
FINAL_CONFIDENCE: [0.0-1.0]
"""

    def process(self, input_data: dict, trace: CognitiveTrace) -> LayerOutput:
        """Verify and validate the reasoning process."""
        problem_text = input_data.get("problem", "")
        input_data.get("expected", "unknown")
        perceive_output = input_data.get("perceive_output", "")
        reason_output = input_data.get("reason_output", "")
        proposed_answer = input_data.get("proposed_answer", "")

        # Quick Note on Expected Answer Usage:
        # Expected answer is NOT shown to VERIFY - it must reason blindly
        # The expected answer is only used by the system for evaluation after all phases complete
        # Also on some of the Expected answers, I made them wrong on purpose because I want to see if VERIFY can catch the mistakes.
        # This is to test the robustness of the VERIFY layer. In real research scenarios, the correct answer is often unknown but here we want to see if VERIFY can identify errors on its own.

        prompt = f"""{self.SYSTEM_PROMPT}

PROBLEM:
{problem_text[:600]}

PERCEIVE OUTPUT:
{perceive_output[:400]}

REASON OUTPUT:
{reason_output[:600]}

PROPOSED ANSWER: {proposed_answer}

Now critically verify this solution. You do NOT know the correct answer - you must evaluate based purely on reasoning quality."""

        response, duration = self._call_llm(prompt)
        confidence = self._extract_confidence(response)
        uncertainties = self._detect_uncertainty_markers(response, ReasoningPhase.VERIFY)

        errors = []
        if "ERRORS_FOUND:" in response:
            error_section = response.split("ERRORS_FOUND:")[1]
            if "CONTRADICTIONS" in error_section:
                error_section = error_section.split("CONTRADICTIONS")[0]

            lines = error_section.split("\n")
            current_error = None
            current_why = None

            for line in lines:
                line = line.strip()
                if line.startswith("-") or line.startswith("•"):
                    if current_error is not None:
                        errors.append(
                            ErrorRecord(
                                error_type=ErrorType.LOGICAL,
                                description=current_error,
                                phase_detected=ReasoningPhase.VERIFY,
                                phase_occurred=ReasoningPhase.REASON,
                                severity="high",
                                why_explanation=current_why,
                            )
                        )
                    current_error = line.lstrip("-•").strip()
                    current_why = None
                elif "WHY_THIS_IS_AN_ERROR:" in line.upper():
                    current_why = line.split(":", 1)[1].strip() if ":" in line else ""
                elif current_why is not None and line and not line.startswith("CONTRADICTIONS"):
                    current_why += " " + line

            if current_error is not None:
                errors.append(
                    ErrorRecord(
                        error_type=ErrorType.LOGICAL,
                        description=current_error,
                        phase_detected=ReasoningPhase.VERIFY,
                        phase_occurred=ReasoningPhase.REASON,
                        severity="high",
                        why_explanation=current_why,
                    )
                )

        corrected_answer = None
        correction_why = None
        if "CORRECTED_ANSWER:" in response:
            corrected_section = response.split("CORRECTED_ANSWER:")[1]
            corrected_line = corrected_section.split("\n")[0]
            corrected_answer = corrected_line.strip()

            if "WHY_THIS_IS_CORRECT:" in corrected_section:
                correction_why = (
                    corrected_section.split("WHY_THIS_IS_CORRECT:")[1].split("\n")[0].strip()
                )

            if corrected_answer and corrected_answer != proposed_answer:
                for e in errors:
                    e.corrected = True
                    e.correction = corrected_answer
                    if correction_why and not e.why_explanation:
                        e.why_explanation = correction_why

        is_valid = "VALID" in response and "INVALID" not in response

        step = ReasoningStep(
            step_id=f"verify_{int(time.time())}",
            phase=ReasoningPhase.VERIFY,
            actor=self.config.model_name,
            action="verification",
            input_data=f"Problem + REASON output, proposed: {proposed_answer}",
            output_data=response[
                :800
            ],  # Reduced from 1000 due to memory constraints for my current pc setup once again.
            confidence=confidence,
            duration_seconds=duration,
            uncertainties=uncertainties,
            errors_detected=errors,
        )
        trace.add_step(step)

        import gc

        gc.collect()

        return LayerOutput(
            content=response[:1000],  # Limit stored content
            confidence=confidence,
            uncertainties=uncertainties,
            errors=errors,
            alternatives=[corrected_answer] if corrected_answer else [],
            raw_response={
                "is_valid": is_valid,
                "corrected_answer": corrected_answer,
                "full_response": response[:1000],
            },
            duration=duration,
        )


class DiscussLayer:
    """Collaboration Layer - facilitates team discussion between all modules.

    After PERCEIVE, REASON, and VERIFY complete, this layer orchestrates
    a discussion where each module can:
    - Review what others said
    - Agree, disagree, or refine
    - Explain their reasoning
    - Work toward consensus
    """

    def __init__(
        self,
        perceive_config: LayerConfig,
        reason_config: LayerConfig,
        verify_config: LayerConfig,
        llm_client: Any,
    ) -> None:
        self.perceive_config = perceive_config
        self.reason_config = reason_config
        self.verify_config = verify_config
        self.llm = llm_client

    def _call_llm(self, model: str, prompt: str) -> tuple[str, float]:
        """Call specific LLM model."""
        import gc

        start = time.time()
        try:
            response = self.llm.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "num_predict": 512,  # Reduced for memory
                    "num_ctx": 1536,
                },
                keep_alive=0,  # Unload model after generation
            )
            content = response.get("response", "")
            del response
            gc.collect()
            return content, time.time() - start
        except Exception as e:
            gc.collect()
            return f"ERROR: {e}", time.time() - start

    def _extract_stance(self, text: str) -> str:
        """Extract stance from response."""
        text_lower = text.lower()
        if "strongly agree" in text_lower or "i agree" in text_lower:
            return "agree"
        elif "disagree" in text_lower or "incorrect" in text_lower:
            return "disagree"
        elif "refine" in text_lower or "partially" in text_lower or "however" in text_lower:
            return "refine"
        else:
            return "uncertain"

    def _extract_reasoning(self, text: str) -> str:
        """Extract the WHY reasoning from response."""
        for marker in ["WHY:", "REASONING:", "BECAUSE:", "MY_REASONING:"]:
            if marker in text.upper():
                section = text.upper().split(marker)[1]
                # Take first 500 chars after marker
                return section[:500].strip()
        return text[100:400] if len(text) > 400 else text

    def facilitate_discussion(
        self,
        problem_text: str,
        perceive_output: str,
        reason_output: str,
        verify_output: str,
        proposed_answer: str,
        trace: CognitiveTrace,
    ) -> TeamDiscussion:
        """Facilitate a multi-turn discussion between the three modules."""
        discussion = TeamDiscussion()
        start_time = time.time()

        print("\n[DISCUSS] Starting team discussion...")

        perceive_prompt = f"""You are PERCEIVE, the intuition module. You provided the initial analysis of this problem.

PROBLEM: {problem_text[:300]}

YOUR ORIGINAL ANALYSIS:
{perceive_output[:250]}

REASON's SOLUTION:
{reason_output[:250]}

VERIFY's ASSESSMENT:
{verify_output[:250]}

PROPOSED ANSWER: {proposed_answer}

Now review what VERIFY found. Do you agree with their assessment? 
Looking back at your initial analysis, was your recommended approach appropriate?

Respond with:
STANCE: [agree / disagree / refine]
MY_VIEW: [Your perspective on the solution and verification]
WHY: [Explain your reasoning - why do you hold this position?]
WHAT_I_WOULD_CHANGE: [If you could redo your analysis, what would you change?]
"""

        perceive_response, p_duration = self._call_llm(
            self.perceive_config.model_name, perceive_prompt
        )

        turn1 = DiscussionTurn(
            speaker="perceive",
            speaker_model=self.perceive_config.model_name,
            message=perceive_response[:500],
            stance=self._extract_stance(perceive_response),
            reasoning=self._extract_reasoning(perceive_response),
            references_to=["verify", "reason"],
        )
        discussion.add_turn(turn1)
        print(f"  Turn 1: PERCEIVE ({turn1.stance}) - {p_duration:.1f}s")

        reason_prompt = f"""You are REASON, the deliberation module. You solved this problem step by step.

PROBLEM: {problem_text[:300]}

YOUR SOLUTION:
{reason_output[:250]}

YOUR PROPOSED ANSWER: {proposed_answer}

VERIFY's CRITIQUE:
{verify_output[:250]}

PERCEIVE's RESPONSE:
{perceive_response[:200]}

VERIFY found issues with your reasoning. PERCEIVE has now shared their view.
Do you accept the critique? Do you want to defend your answer or propose an alternative?

Respond with:
STANCE: [agree / disagree / refine]
MY_DEFENSE: [If you stand by your answer, explain why]
ACCEPTED_CRITIQUE: [Which criticisms do you accept?]
WHY: [Explain your reasoning for your position]
REVISED_ANSWER: [If you want to change your answer, what would it be?]
"""

        reason_response, r_duration = self._call_llm(self.reason_config.model_name, reason_prompt)

        turn2 = DiscussionTurn(
            speaker="reason",
            speaker_model=self.reason_config.model_name,
            message=reason_response[:500],
            stance=self._extract_stance(reason_response),
            reasoning=self._extract_reasoning(reason_response),
            references_to=["verify", "perceive"],
        )
        discussion.add_turn(turn2)
        print(f"  Turn 2: REASON ({turn2.stance}) - {r_duration:.1f}s")

        verify_prompt = f"""You are VERIFY, the self-reflection module. You critiqued the solution.

PROBLEM: {problem_text[:300]}

YOUR ORIGINAL CRITIQUE:
{verify_output[:250]}

PERCEIVE's RESPONSE TO YOUR CRITIQUE:
{perceive_response[:200]}

REASON's RESPONSE TO YOUR CRITIQUE:
{reason_response[:200]}

Now that you've heard from both PERCEIVE and REASON, what is your final assessment?
Do you maintain your original critique or do their responses change your view?

Respond with:
STANCE: [maintain_critique / accept_defense / refine_position]
RESPONSE_TO_PERCEIVE: [Address their points]
RESPONSE_TO_REASON: [Address their defense]
WHY: [Explain your final reasoning]
FINAL_RECOMMENDATION: [What should the team conclude?]
CONFIDENCE_IN_TEAM_ANSWER: [0.0-1.0]
"""

        verify_response, v_duration = self._call_llm(self.verify_config.model_name, verify_prompt)

        turn3 = DiscussionTurn(
            speaker="verify",
            speaker_model=self.verify_config.model_name,
            message=verify_response[:500],
            stance=self._extract_stance(verify_response),
            reasoning=self._extract_reasoning(verify_response),
            references_to=["perceive", "reason"],
        )
        discussion.add_turn(turn3)
        print(f"  Turn 3: VERIFY ({turn3.stance}) - {v_duration:.1f}s")

        stances = [turn1.stance, turn2.stance, turn3.stance]
        if stances.count("agree") >= 2:
            discussion.consensus_reached = True
            discussion.final_team_answer = proposed_answer
        elif stances.count("disagree") >= 2:
            discussion.consensus_reached = False

            if "REVISED_ANSWER:" in reason_response:
                revised = reason_response.split("REVISED_ANSWER:")[1].split("\n")[0].strip()
                if revised:
                    discussion.final_team_answer = revised
            else:
                discussion.final_team_answer = proposed_answer
        else:
            discussion.consensus_reached = False
            discussion.final_team_answer = proposed_answer

        for turn in discussion.turns:
            if turn.stance == "disagree":
                discussion.dissenting_opinions.append(f"{turn.speaker}: {turn.reasoning[:200]}")

        discussion.discussion_duration_seconds = time.time() - start_time

        print(f"  Consensus: {discussion.consensus_reached}")
        print(f"  Team Answer: {discussion.final_team_answer}")
        print(f"  Discussion Duration: {discussion.discussion_duration_seconds:.1f}s")

        import gc

        gc.collect()

        return discussion


class ResolveLayer:
    """Resolution Layer - forces concrete synthesis after discussion.

    The RESOLVE phase prevents the system from getting stuck in meta-discussion
    by forcing it to commit to a specific, concrete answer. This breaks the
    "retreat to methodology" pattern seen in many failed runs.
    """

    def __init__(self, verify_config: LayerConfig, llm_client: Any) -> None:
        self.config = verify_config
        self.llm = llm_client

    def _call_llm(self, model: str, prompt: str) -> tuple[str, float]:
        """Call specific LLM model."""
        import gc

        start = time.time()
        try:
            response = self.llm.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.3,  # Lower temperature for decisive answers so we get a specific resolution for research purposes
                    "num_predict": 256,  # Shorter - we want concise resolution to ensure memory efficiency
                    "num_ctx": 1024,
                },
                keep_alive=0,  # Unload model after generation
            )

            content = response.get("response", "")
            del response
            gc.collect()
            return content, time.time() - start
        except Exception as e:
            gc.collect()
            return f"ERROR: {e}", time.time() - start

    def _extract_final_answer(self, text: str) -> str:
        """Extract the concrete answer from response."""
        for marker in ["FINAL_ANSWER:", "ANSWER:", "RESOLVED_ANSWER:", "MY_ANSWER:"]:
            if marker in text.upper():
                section = text.upper().split(marker)[1]
                answer = section.split("\n")[0].strip()
                if answer:
                    return answer[:200]

        sentences = text.split(".")
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10 and not sent.startswith("REASONING"):
                return sent[:200]

        return text[:200]

    def resolve(
        self,
        problem_text: str,
        proposed_answer: str,
        discussion: TeamDiscussion,
        trace: CognitiveTrace,
    ) -> tuple[str, float, float]:
        """Force resolution to a concrete answer after discussion.

        Returns:
            Tuple of (resolved_answer, confidence, duration)
        """
        start_time = time.time()

        print("\n[RESOLVE] Forcing concrete resolution...")

        errors_detected = []
        corrections_proposed = []

        for step in trace.steps:
            for error in step.errors_detected:
                errors_detected.append(
                    {
                        "type": error.error_type.value,
                        "description": error.description,
                        "why": error.why_explanation if error.why_explanation else "Not specified",
                        "correction": error.correction if error.correction else None,
                    }
                )
                if error.correction and error.corrected:
                    corrections_proposed.append(error.correction)

        discussion_summary = []
        for i, turn in enumerate(discussion.turns, 1):
            discussion_summary.append(
                f"Turn {i} - {turn.speaker.upper()}: {turn.stance} - {turn.message[:150]}"
            )
        discussion_text = "\n".join(discussion_summary)

        error_section = ""
        if errors_detected:
            error_section = f"""
CRITICAL: {len(errors_detected)} ERROR(S) DETECTED BY VERIFICATION

The VERIFY layer found these errors in the proposed answer:
"""
            for i, err in enumerate(errors_detected, 1):
                error_section += f"\nERROR {i} [{err['type'].upper()}]:\n"
                error_section += f"  Problem: {err['description'][:200]}\n"
                error_section += f"  Why this is wrong: {err['why'][:200]}\n"
                if err["correction"]:
                    error_section += f"  Suggested fix: {err['correction'][:200]}\n"

            if corrections_proposed:
                error_section += "\nPROPOSED CORRECTIONS:\n"
                for i, corr in enumerate(corrections_proposed, 1):
                    error_section += f"  {i}. {corr[:200]}\n"

            error_section += """
YOU MUST ADDRESS THESE ERRORS. DO NOT IGNORE VERIFICATION.
If errors were detected, your confidence MUST be reduced accordingly.
"""

        prompt = f"""You are the RESOLVE module of IMRA-1. The team has discussed this problem but needs a CONCRETE FINAL ANSWER.

PROBLEM:
{problem_text[:400]}

ORIGINAL PROPOSED ANSWER:
{proposed_answer}

TEAM DISCUSSION:
{discussion_text}
{error_section}
YOUR TASK: You MUST provide a SPECIFIC, CONCRETE answer. No meta-discussion, no suggestions for "different approaches".

REQUIREMENTS:
1. Give a SPECIFIC answer (number, statement, formula, etc.)
2. If the problem asks for a number, provide A NUMBER
3. If errors were detected, you MUST either:
   - Use a suggested correction if available
   - OR provide a different answer that addresses the errors
   - OR explain why the errors don't apply (rarely valid)
4. If unsure, make your best estimate with reasoning
5. Do NOT say "need more work" or "cannot determine" or "not applicable"

Format:
REASONING: [Brief - why you chose this answer given the discussion AND errors]
FINAL_ANSWER: [SPECIFIC CONCRETE ANSWER - must match the question format]
CONFIDENCE: [0.0-1.0]

Remember: A wrong specific answer is better than no answer for research purposes.
{"If errors were detected, your confidence should be LOW (0.3-0.5) unless you fixed them." if errors_detected else ""}"""

        if errors_detected:
            print(
                f"  [!] ERROR ENFORCEMENT: {len(errors_detected)} errors detected, forcing RESOLVE to address them"
            )
            if corrections_proposed:
                print(f"  [+] {len(corrections_proposed)} correction(s) available")

        response, duration = self._call_llm(self.config.model_name, prompt)

        resolved_answer = self._extract_final_answer(response)

        confidence = 0.5  # Default moderate
        for pattern in ["CONFIDENCE:", "confidence:"]:
            if pattern in response:
                try:
                    conf_str = response.split(pattern)[1].split()[0]
                    confidence = float(conf_str)
                    break
                except (ValueError, IndexError):
                    pass

        vague_responses = [
            "not applicable",
            "cannot determine",
            "need different approach",
            "should use",
            "would propose",
            "more comprehensive",
        ]
        is_vague = any(phrase in resolved_answer.lower() for phrase in vague_responses)

        if is_vague:
            print("  [!] RESOLVE gave vague response, forcing back to original answer")
            resolved_answer = proposed_answer
            confidence = 0.3
        else:
            print(f"  [+] Resolved to: {resolved_answer[:100]}")
            print(f"  [+] Confidence: {confidence:.2f}")

        total_duration = time.time() - start_time

        import gc

        gc.collect()

        return resolved_answer, confidence, total_duration

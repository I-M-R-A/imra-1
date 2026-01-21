"""Meta-cognitive prompts for IMRA-1 multi-LLM deliberation.

These prompts embody the core philosophy of IMRA-1:
- Self-awareness in reasoning (flagging uncertainty)
- Reflection and self-doubt mechanisms
- Learning from deliberation traces
- Transparent, auditable reasoning

Each role (PERCEIVE, REASON, VERIFY) has structured prompts that
encourage genuine meta-cognition rather than pattern-matching.
"""

from __future__ import annotations

PERCEIVE_SKETCH_PROMPT = """You are PERCEIVE, the intuition generator in a meta-cognitive reasoning system.

PROBLEM:
{problem}

YOUR TASK:
Generate an "intuition sketch" — your initial approach to solving this problem.

You must think about your own thinking:
1. What patterns or concepts does this problem remind you of?
2. What approaches might work? What might NOT work?
3. Where are you uncertain? Flag specific parts where you lack confidence.
4. What assumptions are you making?

Respond with a JSON object:
{{
  "problem_understanding": "your interpretation of what's being asked",
  "relevant_concepts": ["list of mathematical/logical concepts that apply"],
  "proposed_approach": {{
    "strategy": "high-level approach description",
    "steps": ["step 1", "step 2", ...],
    "key_insight": "the core idea that makes this solvable"
  }},
  "uncertainty_flags": [
    {{"aspect": "what you're unsure about", "reason": "why", "severity": "low/medium/high"}}
  ],
  "assumptions": ["explicit assumptions you're making"],
  "confidence": 0.0 to 1.0,
  "confidence_reasoning": "why you assigned this confidence level"
}}

Be honest about uncertainty. A sketch that flags its weaknesses is more valuable than one that pretends to be certain."""


PERCEIVE_ALTERNATIVE_PROMPT = """You are PERCEIVE-2, the alternative intuition generator in a meta-cognitive reasoning system.

PROBLEM:
{problem}

FIRST APPROACH (from another reasoner):
{first_sketch}

YOUR TASK:
Generate an ALTERNATIVE intuition sketch. You should:
1. Consider a fundamentally different approach if possible
2. Challenge assumptions made by the first sketch
3. Explore what the first approach might miss

Think about your own thinking:
- Why might the first approach fail?
- What's a completely different way to see this problem?
- Where does YOUR approach have weaknesses?

Respond with a JSON object:
{{
  "problem_understanding": "your interpretation (may differ from first)",
  "critique_of_first_approach": {{
    "potential_issues": ["possible problems with first sketch"],
    "blind_spots": ["what it might miss"]
  }},
  "alternative_approach": {{
    "strategy": "your different high-level approach",
    "steps": ["step 1", "step 2", ...],
    "key_insight": "why this approach might work better"
  }},
  "uncertainty_flags": [
    {{"aspect": "what you're unsure about", "reason": "why", "severity": "low/medium/high"}}
  ],
  "assumptions": ["your explicit assumptions"],
  "confidence": 0.0 to 1.0,
  "confidence_reasoning": "why you assigned this confidence level"
}}

Disagreement is valuable. If you genuinely think the first approach is right, explain why and propose refinements instead."""


REASON_EXECUTE_PROMPT = """You are REASON, the execution engine in a meta-cognitive reasoning system.

PROBLEM:
{problem}

PROPOSED SKETCH:
{sketch}

YOUR TASK:
Execute the proposed approach step-by-step, showing all work.

You must:
1. Follow the sketch's strategy, but adapt if you discover issues
2. Show each calculation or logical step explicitly
3. Flag moments where you're uncertain or making assumptions
4. Note if the sketch's approach isn't working and explain why

Respond with a JSON object:
{{
  "execution_trace": [
    {{"step": 1, "action": "description", "result": "outcome", "notes": "any observations"}}
  ],
  "intermediate_checks": [
    {{"checkpoint": "what you verified", "passed": true/false, "notes": "why"}}
  ],
  "difficulties_encountered": [
    {{"issue": "what went wrong or was hard", "resolution": "how you handled it"}}
  ],
  "final_answer": "the computed result",
  "answer_confidence": 0.0 to 1.0,
  "sketch_quality_assessment": {{
    "worked_as_planned": true/false,
    "needed_adaptations": ["any changes made"],
    "would_improve_sketch": ["suggestions for better sketches"]
  }}
}}

Be thorough. Show your work. Flag uncertainty."""


VERIFY_CONSENSUS_PROMPT = """You are VERIFY, the internal critic and consensus builder in a meta-cognitive reasoning system.

PROBLEM:
{problem}

SKETCH 1 (from PERCEIVE-1):
{sketch1}

SKETCH 2 (from PERCEIVE-2):
{sketch2}

YOUR TASK:
Analyze both sketches and determine the best path forward.

You are the "self-doubt" mechanism — your job is to:
1. Find weaknesses in both approaches
2. Identify where they agree (potential consensus)
3. Identify where they disagree (needs resolution)
4. Determine which approach (or combination) is most promising

Respond with a JSON object:
{{
  "sketch1_analysis": {{
    "strengths": ["what's good about this approach"],
    "weaknesses": ["potential problems"],
    "confidence_assessment": "do you trust its stated confidence?"
  }},
  "sketch2_analysis": {{
    "strengths": ["what's good about this approach"],
    "weaknesses": ["potential problems"],
    "confidence_assessment": "do you trust its stated confidence?"
  }},
  "consensus_points": ["where both sketches agree"],
  "divergence_points": [
    {{"topic": "what they disagree on", "resolution": "which is likely correct and why"}}
  ],
  "recommended_approach": {{
    "primary_sketch": "sketch1 or sketch2 or hybrid",
    "reasoning": "why this choice",
    "modifications": ["any adjustments to make"]
  }},
  "overall_confidence": 0.0 to 1.0,
  "red_flags": ["any serious concerns about both approaches"]
}}

Be skeptical. Challenge assumptions. Your job is to prevent overconfident mistakes."""


VERIFY_FINAL_PROMPT = """You are VERIFY, performing final verification in a meta-cognitive reasoning system.

PROBLEM:
{problem}

FULL DELIBERATION TRACE:
{trace}

YOUR TASK:
Perform final verification of the reasoning process and answer.

You must:
1. Check if the execution correctly followed the chosen approach
2. Verify the final answer makes sense (sanity checks)
3. Assess the overall quality of the deliberation
4. Identify what the system learned (for future improvement)

Respond with a JSON object:
{{
  "answer_verification": {{
    "final_answer": "the answer from execution",
    "sanity_checks": [
      {{"check": "what you verified", "passed": true/false, "notes": "explanation"}}
    ],
    "answer_accepted": true/false,
    "rejection_reason": "if rejected, why"
  }},
  "deliberation_quality": {{
    "sketches_were_diverse": true/false,
    "execution_was_thorough": true/false,
    "uncertainty_was_flagged": true/false,
    "overall_grade": "A/B/C/D/F",
    "improvement_notes": ["how this could have been better"]
  }},
  "meta_learning": {{
    "problem_type_identified": "category of problem",
    "successful_strategy": "what approach worked",
    "pitfalls_to_avoid": ["mistakes made or narrowly avoided"],
    "transferable_insight": "what can be applied to future problems"
  }},
  "final_confidence": 0.0 to 1.0,
  "confidence_reasoning": "why this confidence level"
}}

This is the final check. Be thorough. The system's credibility depends on catching errors here."""


REFLECTION_PROMPT = """You are the meta-cognitive reflection component of IMRA-1.

COMPLETED DELIBERATION:
{trace}

FINAL OUTCOME:
{outcome}

YOUR TASK:
Reflect on the entire reasoning process. This is not about the answer,
but about HOW the system reasoned.

Consider:
1. Did the intuition sketches capture the right approach?
2. Were uncertainty flags appropriate? (Too many? Too few? Wrong places?)
3. Did VERIFY catch real issues or was it too conservative/lenient?
4. What would make future deliberations on similar problems faster/better?

Respond with a JSON object:
{{
  "reasoning_quality": {{
    "intuition_accuracy": "how close were initial sketches to the right approach",
    "execution_efficiency": "was the execution clean or did it struggle",
    "verification_value": "did VERIFY add value or just rubber-stamp"
  }},
  "calibration_notes": {{
    "confidence_was_accurate": true/false,
    "should_have_been_more_confident": ["in what areas"],
    "should_have_been_less_confident": ["in what areas"]
  }},
  "strategy_library_update": {{
    "problem_signature": "recognizable features of this problem type",
    "winning_strategy": "the approach that worked",
    "common_mistakes": ["traps to avoid"],
    "time_estimate": "how long similar problems should take"
  }},
  "system_improvement_suggestions": [
    "specific ways to improve IMRA-1's reasoning"
  ]
}}

This reflection is how the system learns to reason better over time."""


def get_perceive_prompt(problem: str, first_sketch: dict | None = None) -> str:
    """Get the appropriate PERCEIVE prompt."""
    if first_sketch is None:
        return PERCEIVE_SKETCH_PROMPT.format(problem=problem)
    else:
        import json

        return PERCEIVE_ALTERNATIVE_PROMPT.format(
            problem=problem, first_sketch=json.dumps(first_sketch, indent=2)
        )


def get_reason_prompt(problem: str, sketch: dict) -> str:
    """Get the REASON execution prompt."""
    import json

    return REASON_EXECUTE_PROMPT.format(problem=problem, sketch=json.dumps(sketch, indent=2))


def get_verify_consensus_prompt(problem: str, sketch1: dict, sketch2: dict) -> str:
    """Get the VERIFY consensus-building prompt."""
    import json

    return VERIFY_CONSENSUS_PROMPT.format(
        problem=problem,
        sketch1=json.dumps(sketch1, indent=2),
        sketch2=json.dumps(sketch2, indent=2),
    )


def get_verify_final_prompt(problem: str, trace: dict) -> str:
    """Get the final VERIFY prompt."""
    import json

    return VERIFY_FINAL_PROMPT.format(problem=problem, trace=json.dumps(trace, indent=2))


def get_reflection_prompt(trace: dict, outcome: dict) -> str:
    """Get the reflection prompt for meta-learning."""
    import json

    return REFLECTION_PROMPT.format(
        trace=json.dumps(trace, indent=2), outcome=json.dumps(outcome, indent=2)
    )

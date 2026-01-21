#!/usr/bin/env python3
"""Simple IMRA-1 Research Test.

Runs a single problem with minimal prompts to test the research trace system.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imra.llm.ollama_client import OllamaClient
from imra.research.cognitive_trace import CognitiveTrace, ReasoningStep, UncertaintyMarker


def simple_research_run() -> None:
    """Run a simple research trace."""

    print()
    print("IMRA-1 SIMPLE RESEARCH TEST")
    print()

    problem = "How many positive divisors does 2^5 * 3^3 * 5^2 have?"
    expected = 72

    print(f"Problem: {problem}")
    print(f"Expected: {expected}")
    print()

    trace = CognitiveTrace(
        problem_statement=problem, expected_answer=expected, problem_category="number_theory"
    )

    client = OllamaClient(model="codellama:latest")

    print("--- Step 1: PERCEIVE ---")
    prompt1 = f"""Problem: {problem}

Task: Understand this problem and plan a solution approach.

Respond in JSON:
{{
  "interpretation": "what the problem is asking",
  "approach": "how to solve it",
  "uncertainties": ["things you are unsure about"],
  "confidence": 0.0 to 1.0
}}"""

    start = time.time()
    raw1 = client.send_prompt(prompt1)
    dur1 = time.time() - start

    print(f"Response ({dur1:.1f}s): {raw1[:200]}...")

    try:
        parsed1 = json.loads(raw1)
    except json.JSONDecodeError:
        parsed1 = {"raw": raw1}

    step1 = ReasoningStep(
        phase="perception",
        actor="perceive",
        input_context=problem,
        raw_output=raw1,
        parsed_output=parsed1,
        stated_confidence=(
            float(parsed1.get("confidence", 0))
            if isinstance(parsed1.get("confidence"), (int, float))
            else 0
        ),
        duration_seconds=dur1,
    )

    # Extract uncertainties
    for u in parsed1.get("uncertainties", []):
        step1.uncertainties.append(
            UncertaintyMarker(uncertainty_type="epistemic", description=str(u), severity="medium")
        )

    trace.add_step(step1)

    # STEP 2: Execute reasoning
    print("\n--- Step 2: REASON ---")
    prompt2 = f"""Problem: {problem}

Approach: {parsed1.get("approach", "Use the divisor formula")}

Execute this step by step. Show your work.

Respond in JSON:
{{
  "steps": [
    {{"step": 1, "action": "what you did", "result": "outcome"}}
  ],
  "final_answer": "your answer",
  "confidence": 0.0 to 1.0,
  "possible_errors": ["any mistakes you might have made"]
}}"""

    start = time.time()
    raw2 = client.send_prompt(prompt2)
    dur2 = time.time() - start

    print(f"Response ({dur2:.1f}s): {raw2[:200]}...")

    try:
        parsed2 = json.loads(raw2)
    except json.JSONDecodeError:
        parsed2 = {"raw": raw2}

    step2 = ReasoningStep(
        phase="reasoning",
        actor="reason",
        input_context=prompt2,
        raw_output=raw2,
        parsed_output=parsed2,
        stated_confidence=(
            float(parsed2.get("confidence", 0))
            if isinstance(parsed2.get("confidence"), (int, float))
            else 0
        ),
        duration_seconds=dur2,
    )

    # Check for self-reported errors
    errors = parsed2.get("possible_errors", [])
    if errors:
        step2.error_detected = True
        step2.error_description = "; ".join(str(e) for e in errors)

    trace.add_step(step2)

    # STEP 3: Verify
    print("\n--- Step 3: VERIFY ---")
    prompt3 = f"""Problem: {problem}
Expected formula: For N = p1^a1 * p2^a2 * p3^a3, divisors = (a1+1)(a2+1)(a3+1)

Given: 2^5 * 3^3 * 5^2
So: (5+1)(3+1)(2+1) = 6 * 4 * 3 = 72

Previous answer: {parsed2.get("final_answer", "unknown")}

Verify if the answer is correct.

Respond in JSON:
{{
  "verification": "your analysis",
  "errors_found": ["any errors in the reasoning"],
  "correct_answer": "the verified answer",
  "confidence": 0.0 to 1.0
}}"""

    start = time.time()
    raw3 = client.send_prompt(prompt3)
    dur3 = time.time() - start

    print(f"Response ({dur3:.1f}s): {raw3[:200]}...")

    try:
        parsed3 = json.loads(raw3)
    except json.JSONDecodeError:
        parsed3 = {"raw": raw3}

    step3 = ReasoningStep(
        phase="verification",
        actor="verify",
        input_context="Full reasoning",
        raw_output=raw3,
        parsed_output=parsed3,
        stated_confidence=(
            float(parsed3.get("confidence", 0))
            if isinstance(parsed3.get("confidence"), (int, float))
            else 0
        ),
        duration_seconds=dur3,
    )

    errors_found = parsed3.get("errors_found", [])
    if errors_found:
        step3.error_detected = True
        step3.error_description = "; ".join(str(e) for e in errors_found)
        step3.correction_attempted = True
        step3.correction_description = f"Corrected to: {parsed3.get('correct_answer', 'unknown')}"

    trace.add_step(step3)

    # Finalize
    trace.final_answer = parsed3.get("correct_answer") or parsed2.get("final_answer")
    trace.final_confidence = (
        float(parsed3.get("confidence", 0))
        if isinstance(parsed3.get("confidence"), (int, float))
        else 0
    )
    trace.answer_correct = str(trace.final_answer) == str(expected)
    trace.total_duration_seconds = dur1 + dur2 + dur3

    # Add research insights
    if step3.error_detected:
        trace.failure_modes.append("Initial reasoning error corrected by verification")
        trace.key_insights.append("Self-correction mechanism detected and fixed error")

    # Save outputs
    output_dir = Path("research_data")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON trace
    json_path = output_dir / f"trace_{timestamp}.json"
    trace.save(str(json_path))
    print(f"\nJSON trace saved: {json_path}")

    # Human-readable log
    log_path = output_dir / f"log_{timestamp}.txt"
    trace.save_human_readable(str(log_path))
    print(f"Human log saved: {log_path}")

    # Print summary
    print()
    print("RESEARCH TRACE SUMMARY")
    print()
    print(f"Problem: {problem}")
    print(f"Expected: {expected}")
    print(f"Final Answer: {trace.final_answer}")
    print(f"Correct: {trace.answer_correct}")
    print(f"Confidence: {trace.final_confidence:.2f}")
    print(f"Uncertainties logged: {len(trace.all_uncertainties)}")
    print(f"Errors detected: {len(trace.errors_detected)}")
    print(f"Corrections made: {len(trace.corrections_made)}")
    print(f"Duration: {trace.total_duration_seconds:.1f}s")

    if trace.key_insights:
        print("\nKey Insights:")
        for insight in trace.key_insights:
            print(f"  - {insight}")

    if trace.failure_modes:
        print("\nFailure Modes:")
        for mode in trace.failure_modes:
            print(f"  - {mode}")

    print()

    # Print human-readable trace
    print("\nFULL HUMAN-READABLE TRACE:")
    print(trace.to_human_readable())


if __name__ == "__main__":
    simple_research_run()

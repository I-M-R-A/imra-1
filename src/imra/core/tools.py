#!/usr/bin/env python3

"""IMRA-1 Tools - Computational aids for reasoning layers.

These tools help the reasoning layers perform calculations and verifications
that are difficult to do through pure LLM reasoning.
"""

from __future__ import annotations

import math
import re
from typing import Any

import sympy


class ComputationalTools:
    """Provides computational tools for reasoning layers."""

    @staticmethod
    def safe_eval(expression: str, context: dict | None = None) -> tuple[bool, Any, str]:
        """Safely evaluate a mathematical expression.

        Args:
            expression: Math expression to evaluate (e.g., "2 + 2", "sqrt(16)")
            context: Optional context variables

        Returns:
            Tuple of (success, result, error_message)
        """
        try:
            safe_namespace = {
                "abs": abs,
                "max": max,
                "min": min,
                "sum": sum,
                "sqrt": math.sqrt,
                "pow": pow,
                "log": math.log,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "pi": math.pi,
                "e": math.e,
                "factorial": math.factorial,
            }

            if context:
                safe_namespace.update(context)

            expr = expression.strip()

            if any(op in expr for op in ["x", "y", "z", "^"]):
                result = sympy.sympify(expr)
                if result.is_number:
                    result = float(result)
                return True, result, ""

            result = eval(expr, {"__builtins__": {}}, safe_namespace)
            return True, result, ""

        except ZeroDivisionError:
            return False, None, "Division by zero"
        except Exception as e:
            return False, None, f"Evaluation error: {str(e)}"

    @staticmethod
    def extract_and_compute(text: str) -> dict[str, Any]:
        """Extract mathematical expressions from text and compute them.

        Args:
            text: Text containing mathematical expressions

        Returns:
            Dict with found expressions and their results
        """
        results = {"expressions_found": [], "computations": [], "errors": []}

        patterns = [
            r"(\d+\s*[\+\-\*/]\s*\d+)",  # Simple arithmetic
            r"sqrt\((\d+)\)",  # Square roots
            r"(\d+)\^(\d+)",  # Exponents
            r"(\d+)!",  # Factorials
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                expr = match.group(0)
                results["expressions_found"].append(expr)

                success, result, error = ComputationalTools.safe_eval(expr)
                if success:
                    results["computations"].append({"expression": expr, "result": result})
                else:
                    results["errors"].append({"expression": expr, "error": error})

        return results

    @staticmethod
    def verify_calculation(claim: str, expression: str) -> tuple[bool, str]:
        """Verify if a claimed result matches actual calculation.

        Args:
            claim: Claimed result (e.g., "equals 10")
            expression: Expression to evaluate (e.g., "5 + 5")

        Returns:
            Tuple of (is_correct, explanation)
        """
        claim_numbers = re.findall(r"-?\d+\.?\d*", claim)
        if not claim_numbers:
            return False, "Could not extract number from claim"

        claimed_value = float(claim_numbers[0])

        success, actual_value, error = ComputationalTools.safe_eval(expression)

        if not success:
            return False, f"Could not evaluate expression: {error}"

        is_correct = abs(float(actual_value) - claimed_value) < 1e-6

        if is_correct:
            return True, f"Correct: {expression} = {actual_value}"
        else:
            return False, f"Incorrect: {expression} = {actual_value}, not {claimed_value}"

    @staticmethod
    def suggest_correction(expression: str, wrong_answer: Any) -> str:
        """Suggest the correct answer for a calculation.

        Args:
            expression: The expression that was miscalculated
            wrong_answer: The incorrect answer that was given

        Returns:
            Suggestion string with correct answer
        """
        success, correct_answer, error = ComputationalTools.safe_eval(expression)

        if not success:
            return f"Cannot compute {expression}: {error}"

        return f"The answer should be {correct_answer}, not {wrong_answer}"


def create_computation_prompt_addition(problem_text: str) -> str:
    """Create additional prompt text for computational problems.

    This helps the REASON layer know it can request computations.
    """
    computational_indicators = [
        "calculate",
        "compute",
        "how many",
        "find",
        "determine",
        "+",
        "-",
        "*",
        "/",
        "^",
        "=",
        "sum",
        "product",
        "difference",
        "quotient",
    ]

    needs_computation = any(
        indicator in problem_text.lower() for indicator in computational_indicators
    )

    if needs_computation:
        return """

COMPUTATIONAL HELP AVAILABLE:
If you need to perform calculations, you can request them using this format:
COMPUTE: [expression]

Example Usage:

COMPUTE: 123 * 456
COMPUTE: sqrt(144)
COMPUTE: 5!

The system will evaluate these and provide results."""

    return ""

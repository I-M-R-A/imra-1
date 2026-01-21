#!/usr/bin/env python

"""
IMRA-1 Stress Test Suite
=========================

Comprehensive testing with large numbers, edge cases, and performance benchmarks
to validate the framework's legitimacy and robustness.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field

from imra.loops import DeliberationConfig, DifferentiableDeliberationLoop
from imra.perceive import (
    NodeType,
    PerceptionRequest,
    ProgramSketch,
    SketchNode,
    SymbolicPerceiveEngine,
)
from imra.reason import OperatorRegistry, ReasonExecutor
from imra.reason.ops.differentiable import (
    GradientTape,
    diff_add,
    diff_div,
    diff_mul,
    diff_pow,
    diff_sub,
)
from imra.tracing.schemas import TraceBuffer
from imra.utils.compat import utc_now
from imra.verify import CalibrationCritic


@dataclass
class TestResult:
    """Result of a single test case."""

    # Prevent pytest from treating this dataclass as a test class (it has an __init__)
    __test__ = False

    test_id: str
    problem: str
    expected: float
    actual: float | None
    passed: bool
    error: str | None = None
    time_ms: float = 0.0
    difficulty: str = "unknown"


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    total_time_ms: float = 0.0
    results: list[TestResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_tests if self.total_tests else 0.0

    def add_result(self, result: TestResult) -> None:
        self.results.append(result)
        self.total_tests += 1
        self.total_time_ms += result.time_ms
        if result.error:
            self.errors += 1
        elif result.passed:
            self.passed += 1
        else:
            self.failed += 1


class DirectArithmeticTester:
    """
    Direct arithmetic testing - bypasses NLP parsing to test
    the core differentiable operators with exact values.
    """

    def __init__(self):
        self.registry = OperatorRegistry.default()
        self.executor = ReasonExecutor()

    def test_operation(
        self,
        op: str,
        a: float,
        b: float,
        expected: float,
        tolerance: float = 1e-6,
    ) -> tuple[bool, float, str | None]:
        """Test a single operation directly."""
        tape = GradientTape()
        try:
            result, _ = self.registry.execute(op, [a, b], tape)

            if math.isnan(expected) and math.isnan(result):
                return True, result, None
            if math.isinf(expected) and math.isinf(result):
                return expected == result, result, None

            if abs(expected) > 1e10:
                rel_diff = abs(result - expected) / abs(expected) if expected != 0 else abs(result)
                passed = rel_diff < tolerance
            else:
                passed = abs(result - expected) < tolerance

            return passed, result, None
        except Exception as e:
            return False, float("nan"), str(e)

    def test_sketch_execution(
        self,
        sketch: ProgramSketch,
        expected: float,
        variables: dict[str, float] = None,
        tolerance: float = 1e-6,
    ) -> tuple[bool, float | None, str | None]:
        """Test executing a complete program sketch."""
        trace = TraceBuffer(task_id="test", created_at=utc_now())
        try:
            result = self.executor.run(sketch, trace, variables=variables or {})
            if not result.success:
                return False, None, result.error_message

            actual = result.value
            if abs(expected) > 1e10:
                rel_diff = abs(actual - expected) / abs(expected) if expected != 0 else abs(actual)
                passed = rel_diff < tolerance
            else:
                passed = abs(actual - expected) < tolerance

            return passed, actual, None
        except Exception as e:
            return False, None, str(e)


def build_arithmetic_sketch(op: str, a: float, b: float) -> ProgramSketch:
    """Build a simple arithmetic sketch: op(a, b)."""
    nodes = [
        SketchNode(node_id="n0", node_type=NodeType.LITERAL, value=a),
        SketchNode(node_id="n1", node_type=NodeType.LITERAL, value=b),
        SketchNode(node_id="n2", node_type=NodeType.OPERATOR, value=op, children=["n0", "n1"]),
    ]
    return ProgramSketch(
        sketch_id=f"test_{op}_{a}_{b}",
        structure={"op": op},
        prior=1.0,
        nodes=nodes,
        root_node_id="n2",
    )


def run_direct_arithmetic_tests() -> BenchmarkReport:
    """Run direct arithmetic tests on the differentiable operators."""
    print()
    print("STRESS TEST 1: Direct Arithmetic Operations")
    print()

    tester = DirectArithmeticTester()
    report = BenchmarkReport()

    test_cases = [
        ("direct_add_001", "add", 5, 3, 8, "easy"),
        ("direct_sub_001", "sub", 10, 4, 6, "easy"),
        ("direct_mul_001", "mul", 7, 6, 42, "easy"),
        ("direct_div_001", "div", 24, 8, 3, "easy"),
        # Large numbers - CRITICAL FOR LEGITIMACY due to floating point issues
        ("direct_add_large_001", "add", 999999, 1, 1000000, "hard"),
        ("direct_add_large_002", "add", 9999999999, 1, 10000000000, "hard"),
        (
            "direct_add_large_003",
            "add",
            123456789012345,
            987654321098765,
            1111111110111110,
            "extreme",
        ),
        ("direct_sub_large_001", "sub", 1000000000, 999999999, 1, "hard"),
        ("direct_sub_large_002", "sub", 9223372036854775807, 1, 9223372036854775806, "extreme"),
        ("direct_mul_large_001", "mul", 99999, 99999, 9999800001, "hard"),
        ("direct_mul_large_002", "mul", 12345, 6789, 83810205, "hard"),
        ("direct_mul_large_003", "mul", 1000000, 1000000, 1000000000000, "extreme"),
        ("direct_div_large_001", "div", 1000000000000, 1000000, 1000000, "hard"),
        ("direct_div_large_002", "div", 999999999999, 999999, 1000001, "extreme"),
        ("direct_add_dec_001", "add", 3.14159265358979, 2.71828182845904, 5.85987448204883, "hard"),
        ("direct_mul_dec_001", "mul", 1.23456789, 9.87654321, 12.193263111263526, "hard"),
        ("direct_div_dec_001", "div", 1.0, 3.0, 0.3333333333333333, "hard"),
        ("direct_add_neg_001", "add", -1000000000, 1, -999999999, "hard"),
        ("direct_mul_neg_001", "mul", -99999, 99999, -9999800001, "hard"),
        ("direct_sub_neg_001", "sub", -123456789, -987654321, 864197532, "hard"),
        ("direct_edge_001", "add", 0, 0, 0, "easy"),
        ("direct_edge_002", "mul", 0, 999999999999, 0, "easy"),
        ("direct_edge_003", "mul", 1, 999999999999999, 999999999999999, "medium"),
        ("direct_edge_004", "div", 0, 999999, 0, "medium"),
        ("direct_pow_001", "pow", 2, 10, 1024, "medium"),
        ("direct_pow_002", "pow", 10, 6, 1000000, "medium"),
        ("direct_pow_003", "pow", 2, 20, 1048576, "hard"),
        ("direct_pow_004", "pow", 2, 30, 1073741824, "hard"),
        ("direct_pow_005", "pow", 10, 12, 1000000000000, "extreme"),
    ]

    print(f"\nRunning {len(test_cases)} direct arithmetic tests...\n")

    for test_id, op, a, b, expected, difficulty in test_cases:
        start = time.perf_counter()
        passed, actual, error = tester.test_operation(op, a, b, expected)
        elapsed = (time.perf_counter() - start) * 1000

        result = TestResult(
            test_id=test_id,
            problem=f"{op}({a}, {b})",
            expected=expected,
            actual=actual,
            passed=passed,
            error=error,
            time_ms=elapsed,
            difficulty=difficulty,
        )
        report.add_result(result)

        status = "✓" if passed else "✗"
        if error:
            print(
                f"  {status} [{difficulty:7}] {test_id}: {op}({a}, {b}) = {actual} (expected {expected}) ERROR: {error}"
            )
        elif not passed:
            print(
                f"  {status} [{difficulty:7}] {test_id}: {op}({a}, {b}) = {actual} (expected {expected})"
            )
        else:
            print(f"  {status} [{difficulty:7}] {test_id}: {op}({a}, {b}) = {actual}")

    return report


def run_sketch_execution_tests() -> BenchmarkReport:
    """Test the full sketch execution pipeline with large numbers."""
    print()
    print("STRESS TEST 2: Program Sketch Execution")
    print()

    tester = DirectArithmeticTester()
    report = BenchmarkReport()

    test_cases = [
        ("sketch_001", ("add", 5, 3), 8, "easy"),
        ("sketch_002", ("sub", 100, 37), 63, "easy"),
        ("sketch_003", ("mul", 12, 12), 144, "easy"),
        ("sketch_004", ("div", 144, 12), 12, "easy"),
        ("sketch_large_001", ("add", 999999999, 1), 1000000000, "hard"),
        ("sketch_large_002", ("mul", 99999, 99999), 9999800001, "hard"),
        ("sketch_large_003", ("sub", 10000000000, 1), 9999999999, "hard"),
        ("sketch_large_004", ("div", 1000000000000, 1000), 1000000000, "hard"),
        ("sketch_dec_001", ("add", 3.14159265, 2.71828182), 5.85987447, "medium"),
        ("sketch_dec_002", ("mul", 123.456, 789.012), 97408.265472, "hard"),
        ("sketch_edge_001", ("mul", 0, 9999999999), 0, "easy"),
        ("sketch_edge_002", ("add", -1000000, 1000000), 0, "medium"),
    ]

    print(f"\nRunning {len(test_cases)} sketch execution tests...\n")

    for test_id, (op, a, b), expected, difficulty in test_cases:
        sketch = build_arithmetic_sketch(op, a, b)

        start = time.perf_counter()
        passed, actual, error = tester.test_sketch_execution(sketch, expected)
        elapsed = (time.perf_counter() - start) * 1000

        result = TestResult(
            test_id=test_id,
            problem=f"Sketch: {op}({a}, {b})",
            expected=expected,
            actual=actual,
            passed=passed,
            error=error,
            time_ms=elapsed,
            difficulty=difficulty,
        )
        report.add_result(result)

        status = "✓" if passed else "✗"
        if error:
            print(f"  {status} [{difficulty:7}] {test_id}: ERROR: {error}")
        elif not passed:
            print(f"  {status} [{difficulty:7}] {test_id}: got {actual}, expected {expected}")
        else:
            print(f"  {status} [{difficulty:7}] {test_id}: {op}({a}, {b}) = {actual}")

    return report


def run_multi_step_computation_tests() -> BenchmarkReport:
    """Test multi-step computations with chained operations."""
    print()
    print("STRESS TEST 3: Multi-Step Computations")
    print()

    executor = ReasonExecutor()
    report = BenchmarkReport()

    test_cases = []

    # Test 1: (a + b) * c = (100 + 200) * 3 = 900
    nodes1 = [
        SketchNode(node_id="a", node_type=NodeType.LITERAL, value=100),
        SketchNode(node_id="b", node_type=NodeType.LITERAL, value=200),
        SketchNode(node_id="c", node_type=NodeType.LITERAL, value=3),
        SketchNode(node_id="sum", node_type=NodeType.OPERATOR, value="add", children=["a", "b"]),
        SketchNode(
            node_id="result", node_type=NodeType.OPERATOR, value="mul", children=["sum", "c"]
        ),
    ]

    test_cases.append(("multi_001", nodes1, "result", "(100 + 200) * 3", 900, "medium"))

    # Test 2: (a * b) - (c * d) = (1000 * 1000) - (999 * 999) = 1000000 - 998001 = 1999

    nodes2 = [
        SketchNode(node_id="a", node_type=NodeType.LITERAL, value=1000),
        SketchNode(node_id="b", node_type=NodeType.LITERAL, value=1000),
        SketchNode(node_id="c", node_type=NodeType.LITERAL, value=999),
        SketchNode(node_id="d", node_type=NodeType.LITERAL, value=999),
        SketchNode(node_id="prod1", node_type=NodeType.OPERATOR, value="mul", children=["a", "b"]),
        SketchNode(node_id="prod2", node_type=NodeType.OPERATOR, value="mul", children=["c", "d"]),
        SketchNode(
            node_id="result", node_type=NodeType.OPERATOR, value="sub", children=["prod1", "prod2"]
        ),
    ]

    test_cases.append(("multi_002", nodes2, "result", "(1000*1000) - (999*999)", 1999, "hard"))

    # Test 3: Large chain: ((10^6 + 10^6) * 2) / 4 = 1000000

    nodes3 = [
        SketchNode(node_id="m", node_type=NodeType.LITERAL, value=1000000),
        SketchNode(node_id="n", node_type=NodeType.LITERAL, value=1000000),
        SketchNode(node_id="two", node_type=NodeType.LITERAL, value=2),
        SketchNode(node_id="four", node_type=NodeType.LITERAL, value=4),
        SketchNode(node_id="sum", node_type=NodeType.OPERATOR, value="add", children=["m", "n"]),
        SketchNode(
            node_id="prod", node_type=NodeType.OPERATOR, value="mul", children=["sum", "two"]
        ),
        SketchNode(
            node_id="result", node_type=NodeType.OPERATOR, value="div", children=["prod", "four"]
        ),
    ]
    test_cases.append(("multi_003", nodes3, "result", "((10^6 + 10^6) * 2) / 4", 1000000, "hard"))

    # Chained addition of large numbers

    nodes4 = [
        SketchNode(node_id="a", node_type=NodeType.LITERAL, value=111111111),
        SketchNode(node_id="b", node_type=NodeType.LITERAL, value=222222222),
        SketchNode(node_id="c", node_type=NodeType.LITERAL, value=333333333),
        SketchNode(node_id="d", node_type=NodeType.LITERAL, value=333333334),
        SketchNode(node_id="s1", node_type=NodeType.OPERATOR, value="add", children=["a", "b"]),
        SketchNode(node_id="s2", node_type=NodeType.OPERATOR, value="add", children=["s1", "c"]),
        SketchNode(
            node_id="result", node_type=NodeType.OPERATOR, value="add", children=["s2", "d"]
        ),
    ]
    test_cases.append(
        (
            "multi_004",
            nodes4,
            "result",
            "111111111 + 222222222 + 333333333 + 333333334",
            1000000000,
            "extreme",
        )
    )

    # ((99999 * 99999) + 1) / 10000 = (9999800001 + 1) / 10000 = 999980.0002

    nodes5 = [
        SketchNode(node_id="x", node_type=NodeType.LITERAL, value=99999),
        SketchNode(node_id="y", node_type=NodeType.LITERAL, value=99999),
        SketchNode(node_id="one", node_type=NodeType.LITERAL, value=1),
        SketchNode(node_id="div", node_type=NodeType.LITERAL, value=10000),
        SketchNode(node_id="sq", node_type=NodeType.OPERATOR, value="mul", children=["x", "y"]),
        SketchNode(
            node_id="plus", node_type=NodeType.OPERATOR, value="add", children=["sq", "one"]
        ),
        SketchNode(
            node_id="result", node_type=NodeType.OPERATOR, value="div", children=["plus", "div"]
        ),
    ]

    test_cases.append(
        ("multi_005", nodes5, "result", "((99999^2) + 1) / 10000", 999980.0002, "extreme")
    )

    print(f"\nRunning {len(test_cases)} multi-step computation tests...\n")

    for test_id, nodes, root_id, description, expected, difficulty in test_cases:
        sketch = ProgramSketch(
            sketch_id=test_id,
            structure={},
            prior=1.0,
            nodes=nodes,
            root_node_id=root_id,
        )

        trace = TraceBuffer(task_id=test_id, created_at=utc_now())

        start = time.perf_counter()
        try:
            result = executor.run(sketch, trace)
            actual = result.value
            error = result.error_message

            if actual is not None:
                if abs(expected) > 1e10:
                    rel_diff = abs(actual - expected) / abs(expected)
                    passed = rel_diff < 1e-6
                else:
                    passed = abs(actual - expected) < 1e-4
            else:
                passed = False
        except Exception as e:
            actual = None
            error = str(e)
            passed = False

        elapsed = (time.perf_counter() - start) * 1000

        test_result = TestResult(
            test_id=test_id,
            problem=description,
            expected=expected,
            actual=actual,
            passed=passed,
            error=error,
            time_ms=elapsed,
            difficulty=difficulty,
        )
        report.add_result(test_result)

        status = "✓" if passed else "✗"
        if error:
            print(f"  {status} [{difficulty:7}] {test_id}: {description} ERROR: {error}")
        elif not passed:
            print(
                f"  {status} [{difficulty:7}] {test_id}: {description} = {actual} (expected {expected})"
            )
        else:
            print(f"  {status} [{difficulty:7}] {test_id}: {description} = {actual}")

    return report


def run_gradient_correctness_tests() -> BenchmarkReport:
    """Verify gradient computations are mathematically correct."""
    print()
    print("STRESS TEST 4: Gradient Correctness Verification")
    print()

    report = BenchmarkReport()

    test_cases = [
        # (test_id, op_fn, a, b, expected_grad_a, expected_grad_b, difficulty)
        ("grad_add_001", diff_add, 100, 200, 1.0, 1.0, "easy"),
        ("grad_sub_001", diff_sub, 100, 200, 1.0, -1.0, "easy"),
        ("grad_mul_001", diff_mul, 3, 5, 5.0, 3.0, "easy"),  # d/da(a*b)=b, d/db(a*b)=a
        ("grad_mul_002", diff_mul, 1000, 2000, 2000.0, 1000.0, "medium"),
        ("grad_div_001", diff_div, 10, 2, 0.5, -2.5, "medium"),  # d/da(a/b)=1/b, d/db(a/b)=-a/b^2
        ("grad_div_002", diff_div, 1000000, 1000, 0.001, -1.0, "hard"),
        ("grad_pow_001", diff_pow, 2, 3, 12.0, None, "medium"),  # d/da(a^b)=b*a^(b-1)=3*4=12
    ]

    print(f"\nRunning {len(test_cases)} gradient correctness tests...\n")

    for test_id, op_fn, a, b, exp_grad_a, exp_grad_b, difficulty in test_cases:
        tape = GradientTape()
        start = time.perf_counter()

        try:
            result, grads = op_fn(float(a), float(b), tape=tape)

            actual_grad_a = grads.get("da")
            passed_a = actual_grad_a is not None and abs(actual_grad_a - exp_grad_a) < 1e-6

            if exp_grad_b is not None:
                actual_grad_b = grads.get("db")
                passed_b = actual_grad_b is not None and abs(actual_grad_b - exp_grad_b) < 1e-6
            else:
                passed_b = True
                actual_grad_b = None

            passed = passed_a and passed_b
            error = None

        except Exception as e:
            passed = False
            error = str(e)
            actual_grad_a = None
            actual_grad_b = None

        elapsed = (time.perf_counter() - start) * 1000

        test_result = TestResult(
            test_id=test_id,
            problem=f"grad({op_fn.__name__})({a}, {b})",
            expected=exp_grad_a,
            actual=actual_grad_a,
            passed=passed,
            error=error,
            time_ms=elapsed,
            difficulty=difficulty,
        )
        report.add_result(test_result)

        status = "✓" if passed else "✗"
        if error:
            print(f"  {status} [{difficulty:7}] {test_id}: ERROR: {error}")
        else:
            grad_str = f"da={actual_grad_a}"
            if actual_grad_b is not None:
                grad_str += f", db={actual_grad_b}"
            print(f"  {status} [{difficulty:7}] {test_id}: {grad_str}")

    return report


def run_full_loop_stress_tests() -> BenchmarkReport:
    """Test the full PERCEIVE→REASON→VERIFY loop under stress."""
    print()
    print("STRESS TEST 5: Full Deliberation Loop")
    print()

    perceive = SymbolicPerceiveEngine(num_sketches=5)
    reason = ReasonExecutor(max_steps=100)
    verify = CalibrationCritic()

    loop = DifferentiableDeliberationLoop(
        perceive=perceive,
        reason=reason,
        verify=verify,
        config=DeliberationConfig(
            max_iterations=10,
            max_sketches_per_iteration=5,
            confidence_threshold=0.3,
        ),
    )

    report = BenchmarkReport()

    # Quick Note: Full loop can't handle arbitrary large numbers due to NLP parsing.
    # Therefore, we just focus on moderately large numbers that are still challenging.
    # These tests still stress the system without hitting parsing limits.

    test_cases = [
        ("loop_001", "What is 12345 plus 67890?", 80235, "medium"),
        ("loop_002", "Calculate 100000 minus 99999", 1, "medium"),
        ("loop_003", "What is 999 times 111?", 110889, "hard"),
        ("loop_004", "Divide 1000000 by 1000", 1000, "hard"),
        ("loop_005", "What is 99999 plus 1?", 100000, "hard"),
    ]

    print(f"\nRunning {len(test_cases)} full loop stress tests...\n")

    for test_id, problem, expected, difficulty in test_cases:
        request = PerceptionRequest(
            task_id=test_id,
            payload={},
            problem_text=problem,
        )

        start = time.perf_counter()
        result = loop.run(request)
        elapsed = (time.perf_counter() - start) * 1000

        actual = result.final_value
        if actual is not None:
            passed = abs(actual - expected) < 1
        else:
            passed = False

        test_result = TestResult(
            test_id=test_id,
            problem=problem,
            expected=expected,
            actual=actual,
            passed=passed,
            error=None if result.success else "Loop did not succeed",
            time_ms=elapsed,
            difficulty=difficulty,
        )
        report.add_result(test_result)

        status = "✓" if passed else "✗"
        print(
            f"  {status} [{difficulty:7}] {test_id}: {problem[:40]}... = {actual} (conf={result.confidence:.3f})"
        )

    return report


def run_performance_benchmark() -> dict:
    """Run performance benchmarks to measure throughput."""
    print()
    print("STRESS TEST 6: Performance Benchmark")
    print()

    registry = OperatorRegistry.default()

    iterations = 100000  # Number of operations to execute

    print(f"\nRunning {iterations:,} iterations per operation...\n")

    results = {}

    for op in ["add", "sub", "mul", "div"]:
        tape = GradientTape()
        start = time.perf_counter()

        for _i in range(iterations):
            registry.execute(op, [12345.0, 6789.0], tape)

        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed

        results[op] = {
            "iterations": iterations,
            "total_time_sec": elapsed,
            "ops_per_second": ops_per_sec,
        }

        print(
            f"  {op.upper():4}: {ops_per_sec:,.0f} ops/sec ({elapsed:.3f}s for {iterations:,} ops)"
        )

    return results


def print_final_report(reports: list[tuple[str, BenchmarkReport]]) -> None:
    """Print consolidated final report."""
    print()
    print("FINAL STRESS TEST REPORT")
    print()

    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0

    for name, report in reports:
        total_tests += report.total_tests
        total_passed += report.passed
        total_failed += report.failed
        total_errors += report.errors

        status = "✓" if report.failed == 0 and report.errors == 0 else "✗"
        print(f"\n{status} {name}:")
        print(f"    Tests: {report.total_tests}")
        print(f"    Passed: {report.passed}")
        print(f"    Failed: {report.failed}")
        print(f"    Errors: {report.errors}")
        print(f"    Pass Rate: {report.pass_rate:.1%}")
        print(f"    Time: {report.total_time_ms:.2f}ms")

    print("\n" + "")
    print("OVERALL SUMMARY:")
    print(f"    Total Tests: {total_tests}")
    print(f"    Total Passed: {total_passed}")
    print(f"    Total Failed: {total_failed}")
    print(f"    Total Errors: {total_errors}")
    overall_rate = total_passed / total_tests if total_tests else 0
    print(f"    Overall Pass Rate: {overall_rate:.1%}")

    if total_failed == 0 and total_errors == 0:
        print("\n ALL STRESS TESTS PASSED! Framework is LEGITIMATE! 🎉")
    else:
        print(f"\n {total_failed + total_errors} tests need attention")


def main():
    """Run all stress tests."""
    print()
    print("IMRA-1 STRESS TEST SUITE")
    print("Testing Framework Legitimacy with Large Numbers & Edge Cases")
    print()

    reports = []

    reports.append(("Direct Arithmetic", run_direct_arithmetic_tests()))
    reports.append(("Sketch Execution", run_sketch_execution_tests()))
    reports.append(("Multi-Step Computation", run_multi_step_computation_tests()))
    reports.append(("Gradient Correctness", run_gradient_correctness_tests()))
    reports.append(("Full Loop", run_full_loop_stress_tests()))

    run_performance_benchmark()
    print_final_report(reports)

    print()
    print("Stress Test Suite Complete")
    print()

    all_passed = all(r.failed == 0 and r.errors == 0 for _, r in reports)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

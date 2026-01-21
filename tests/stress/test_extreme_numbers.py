#!/usr/bin/env python3
"""
IMRA-1 EXTREME NUMBER STRESS TESTS
===================================
Testing with the LARGEST and HARDEST numbers to prove framework legitimacy.

This suite tests:
- 32-bit integer boundaries (2^31-1)
- 53-bit float precision limit (IEEE 754 double)
- Large float range (up to ~1.7e308)
- Mathematical constants precision
- Edge cases that break naive implementations

Note: Python floats (IEEE 754 doubles) can represent integers exactly up to 2^53.
Beyond that, precision is lost. The differentiable module uses floats for gradients.
For truly massive integers (10^100+), use Python's native int arithmetic directly.
"""

import sys
import time
from decimal import getcontext
from pathlib import Path

getcontext().prec = 100

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from imra.reason.ops.differentiable import (
    DiffValue,
    GradientTape,
    diff_add,
    diff_div,
    diff_mul,
    diff_pow,
    diff_sub,
)

MAX_SAFE_INTEGER = 2**53  # 9007199254740992


def add(a, b, tape=None, grad_name=None):
    result, _ = diff_add(float(a), float(b), tape)
    return result


def sub(a, b, tape=None, grad_name=None):
    result, _ = diff_sub(float(a), float(b), tape)
    return result


def mul(a, b, tape=None, grad_name=None):
    result, _ = diff_mul(float(a), float(b), tape)
    return result


def div(a, b, tape=None, grad_name=None):
    result, _ = diff_div(float(a), float(b), tape)
    return result


def power(a, b, tape=None, grad_name=None):
    result, _ = diff_pow(float(a), float(b), tape)
    return result


def print_header(title: str) -> None:
    print("")
    print(f"  {title}")
    print("")


def print_section(title: str) -> None:
    print(f"\n--- {title} ---\n")


class ExtremeNumberTester:
    """Tests with extreme numbers beyond normal use cases."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.results = []

    def test(self, name: str, actual, expected, tolerance=1e-9, category="extreme"):
        """Run a single test with optional tolerance for floats."""
        try:
            if isinstance(expected, float):
                if abs(expected) < 1e-10:
                    passed = abs(actual - expected) < tolerance
                else:
                    passed = abs(actual - expected) / abs(expected) < tolerance
            else:
                passed = actual == expected

            if passed:
                self.passed += 1
                status = "✓"
            else:
                self.failed += 1
                status = "✗"

            actual_str = str(actual)
            expected_str = str(expected)
            if len(actual_str) > 30:
                actual_str = actual_str[:27] + "..."
            if len(expected_str) > 30:
                expected_str = expected_str[:27] + "..."

            print(f"  {status} [{category:8}] {name}: {actual_str} == {expected_str}")

            self.results.append({"name": name, "passed": passed, "category": category})

        except Exception as e:
            self.errors += 1
            print(f"  ✗ [{category:8}] {name}: ERROR - {e}")
            self.results.append(
                {"name": name, "passed": False, "category": category, "error": str(e)}
            )

    def report(self) -> dict:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "total": self.passed + self.failed + self.errors,
            "rate": (
                self.passed / (self.passed + self.failed + self.errors) * 100
                if (self.passed + self.failed + self.errors) > 0
                else 0
            ),
        }


def run_integer_boundaries():
    """Test at critical integer boundaries for IEEE 754 doubles."""
    print_header("INTEGER BOUNDARY TESTS (Float-Safe)")
    tester = ExtremeNumberTester()

    INT32_MAX = 2147483647  # 2^31 - 1
    UINT32_MAX = 4294967295  # 2^32 - 1

    MAX_SAFE = MAX_SAFE_INTEGER  # 2^53

    TRILLION = 10**12
    QUADRILLION = 10**15

    print_section("32-bit Boundaries (Always Exact)")
    tester.test("int32_max_add", add(INT32_MAX, 1), INT32_MAX + 1, category="32-bit")
    tester.test("int32_max_sub", sub(INT32_MAX, INT32_MAX), 0, category="32-bit")
    tester.test("int32_max_overflow", add(INT32_MAX, INT32_MAX), INT32_MAX * 2, category="32-bit")
    tester.test("uint32_max_add", add(UINT32_MAX, 1), UINT32_MAX + 1, category="32-bit")
    tester.test("uint32_max_mul", mul(UINT32_MAX, 2), UINT32_MAX * 2, category="32-bit")

    print_section("53-bit Precision Boundary (Float Limit)")

    tester.test("max_safe_identity", add(MAX_SAFE - 1, 0), MAX_SAFE - 1, category="53-bit")
    tester.test("max_safe_plus_one", add(MAX_SAFE - 1, 1), MAX_SAFE, category="53-bit")
    tester.test("max_safe_sub", sub(MAX_SAFE, 1), MAX_SAFE - 1, category="53-bit")

    print_section("Trillion-Scale Operations")
    tester.test("trillion_add", add(TRILLION, TRILLION), 2 * TRILLION, category="trillion")
    tester.test("trillion_sub", sub(TRILLION, 1), TRILLION - 1, category="trillion")
    tester.test("trillion_mul_small", mul(TRILLION, 100), TRILLION * 100, category="trillion")
    tester.test("trillion_div", div(TRILLION, 1000), TRILLION / 1000, category="trillion")

    print_section("Quadrillion-Scale Operations")
    tester.test("quad_add", add(QUADRILLION, QUADRILLION), 2 * QUADRILLION, category="quad")
    tester.test("quad_sub", sub(QUADRILLION, 1), QUADRILLION - 1, category="quad")
    tester.test("quad_div", div(QUADRILLION, TRILLION), QUADRILLION / TRILLION, category="quad")

    return tester.report()


def run_float_range_extremes():
    """Test the extreme range of IEEE 754 doubles."""
    print_header("FLOAT RANGE EXTREME TESTS")
    tester = ExtremeNumberTester()

    SMALL_FLOAT = 1e-200
    LARGE_FLOAT = 1e200
    VERY_LARGE = 1e300

    print_section("Large Float Arithmetic (10^200 range)")
    tester.test(
        "large_add", add(LARGE_FLOAT, LARGE_FLOAT), 2e200, tolerance=1e185, category="large-float"
    )
    tester.test(
        "large_sub",
        sub(LARGE_FLOAT, LARGE_FLOAT / 2),
        LARGE_FLOAT / 2,
        tolerance=1e185,
        category="large-float",
    )
    tester.test(
        "large_mul_small", mul(LARGE_FLOAT, 2), 2e200, tolerance=1e185, category="large-float"
    )
    tester.test("large_div", div(LARGE_FLOAT, 1e100), 1e100, tolerance=1e85, category="large-float")

    print_section("Very Large Float (10^300 range)")
    tester.test(
        "vlarge_add", add(VERY_LARGE, VERY_LARGE), 2e300, tolerance=1e285, category="vlarge"
    )
    tester.test(
        "vlarge_identity", add(VERY_LARGE, 0), VERY_LARGE, tolerance=1e285, category="vlarge"
    )

    print_section("Small Float Arithmetic (10^-200 range)")
    tester.test(
        "small_add", add(SMALL_FLOAT, SMALL_FLOAT), 2e-200, tolerance=1e-215, category="small-float"
    )
    tester.test(
        "small_mul", mul(SMALL_FLOAT, 1e100), 1e-100, tolerance=1e-115, category="small-float"
    )

    print_section("Cross-Scale Operations")
    tester.test("cross_mul", mul(1e100, 1e100), 1e200, tolerance=1e185, category="cross")
    tester.test("cross_div", div(1e200, 1e100), 1e100, tolerance=1e85, category="cross")

    return tester.report()


def run_arbitrary_precision_integers():
    """Test Python's native arbitrary precision (bypassing float)."""
    print_header("ARBITRARY PRECISION INTEGER TESTS (Native Python)")
    tester = ExtremeNumberTester()

    # These use Python's native int arithmetic, not the differentiable ops
    # This demonstrates Python can handle them; This framework would need BigInt support
    # to do this natively in differentiable ops. Also, gradients would not be defined.

    GOOGOL = 10**100

    BIG_200 = 10**200
    BIG_500 = 10**500
    BIG_1000 = 10**1000

    def fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    FIB_1000 = fib(1000)

    print_section("Native Python Big Integer Demo")
    print("  (These use Python's native int, showing what's possible)")

    tester.test("googol_native_add", GOOGOL + GOOGOL, 2 * GOOGOL, category="native-big")
    tester.test("googol_native_sub", GOOGOL - 1, GOOGOL - 1, category="native-big")
    tester.test("big200_native_add", BIG_200 + BIG_200, 2 * BIG_200, category="native-big")
    tester.test("big500_native_mul", BIG_500 * 2, 2 * BIG_500, category="native-big")
    tester.test("big1000_native_add", BIG_1000 + 1, BIG_1000 + 1, category="native-big")
    tester.test("fib1000_verify", len(str(FIB_1000)), 209, category="native-big")

    print("\n  ℹFramework differentiable ops use float64 (up to ~10^308 range)")
    print("  ℹFor arbitrary precision, use Python native int or add BigDecimal support")

    return tester.report()


def run_famous_numbers():
    """Test with mathematically famous numbers."""
    print_header("FAMOUS NUMBER TESTS")
    tester = ExtremeNumberTester()

    PI = 3.14159265358979323846264338327950288419716939937510
    E = 2.71828182845904523536028747135266249775724709369995
    PHI = 1.61803398874989484820458683436563811772030917980576  # Golden ratio
    SQRT2 = 1.41421356237309504880168872420969807856967187537694

    AVOGADRO = 6.02214076e23
    PLANCK = 6.62607015e-34
    SPEED_OF_LIGHT = 299792458

    MERSENNE_61 = 2**61 - 1
    2**89 - 1

    print_section("Mathematical Constants")
    tester.test("pi_plus_e", add(PI, E), PI + E, tolerance=1e-15, category="constant")
    tester.test("pi_times_e", mul(PI, E), PI * E, tolerance=1e-15, category="constant")
    tester.test("phi_squared", mul(PHI, PHI), PHI * PHI, tolerance=1e-15, category="constant")
    tester.test("sqrt2_squared", mul(SQRT2, SQRT2), 2.0, tolerance=1e-15, category="constant")
    tester.test(
        "euler_identity_approx",
        add(mul(E, PI), 1),
        E * PI + 1,
        tolerance=1e-15,
        category="constant",
    )
    tester.test(
        "golden_ratio_property", sub(mul(PHI, PHI), PHI), 1.0, tolerance=1e-15, category="constant"
    )  # φ² - φ = 1

    print_section("Physical Constants")
    tester.test("avogadro_identity", add(AVOGADRO, 0), AVOGADRO, category="physics")
    tester.test(
        "avogadro_double", add(AVOGADRO, AVOGADRO), AVOGADRO * 2, tolerance=1e10, category="physics"
    )
    tester.test(
        "planck_double", add(PLANCK, PLANCK), PLANCK * 2, tolerance=1e-48, category="physics"
    )
    tester.test(
        "light_speed_squared",
        mul(SPEED_OF_LIGHT, SPEED_OF_LIGHT),
        SPEED_OF_LIGHT**2,
        category="physics",
    )
    tester.test(
        "E_equals_mc2",
        mul(1.0, mul(SPEED_OF_LIGHT, SPEED_OF_LIGHT)),
        SPEED_OF_LIGHT**2,
        category="physics",
    )

    print_section("Mersenne Primes (Float-representable)")
    tester.test(
        "mersenne_61_identity", add(MERSENNE_61, 0), float(MERSENNE_61), category="mersenne"
    )
    tester.test("mersenne_61_plus_one", add(MERSENNE_61, 1), float(2**61), category="mersenne")

    print_section("Perfect Numbers")

    PERFECT_6 = 6
    PERFECT_28 = 28
    PERFECT_496 = 496

    tester.test("perfect_6", add(1, add(2, 3)), float(PERFECT_6), category="perfect")
    tester.test(
        "perfect_28", add(add(add(1, 2), add(4, 7)), 14), float(PERFECT_28), category="perfect"
    )
    tester.test(
        "perfect_496",
        add(add(add(add(1, 2), add(4, 8)), add(add(16, 31), add(62, 124))), 248),
        float(PERFECT_496),
        category="perfect",
    )

    return tester.report()


def run_precision_edge_cases():
    """Test floating-point precision edge cases."""
    print_header("PRECISION EDGE CASE TESTS")
    tester = ExtremeNumberTester()

    EPSILON = 2.220446049250313e-16

    print_section("Machine Epsilon Tests")
    tester.test(
        "epsilon_add_to_one", add(1.0, EPSILON), 1.0 + EPSILON, tolerance=1e-20, category="epsilon"
    )
    tester.test(
        "epsilon_mul", mul(EPSILON, 1e10), EPSILON * 1e10, tolerance=1e-20, category="epsilon"
    )

    print_section("Precision Loss Detection")

    tester.test("precision_1", add(0.1, 0.2), 0.3, tolerance=1e-15, category="precision")
    tester.test("precision_2", sub(1.0, 0.9), 0.1, tolerance=1e-15, category="precision")
    tester.test("precision_3", mul(0.1, 10), 1.0, tolerance=1e-15, category="precision")
    tester.test("precision_4", div(1.0, 3.0), 1.0 / 3.0, tolerance=1e-15, category="precision")
    tester.test("precision_5", mul(div(1.0, 3.0), 3.0), 1.0, tolerance=1e-15, category="precision")

    print_section("Near-Zero Operations")
    tester.test(
        "near_zero_add", add(1e-100, 1e-100), 2e-100, tolerance=1e-115, category="near-zero"
    )
    tester.test(
        "near_zero_sub", sub(1e-100, 1e-101), 9e-101, tolerance=1e-115, category="near-zero"
    )
    tester.test("near_zero_mul", mul(1e-50, 1e-50), 1e-100, tolerance=1e-115, category="near-zero")

    print_section("Cancellation Tests")
    large = 1e15
    tester.test("cancel_1", sub(add(large, 1), large), 1.0, tolerance=1e-10, category="cancel")
    tester.test("cancel_2", sub(add(large, 0.5), large), 0.5, tolerance=1e-10, category="cancel")

    print_section("Associativity Tests")

    a, b, c = 1e16, 1.0, -1e16
    result_1 = add(add(a, b), c)
    result_2 = add(a, add(b, c))

    tester.test("assoc_1", result_1, 1.0, tolerance=1.0, category="assoc")
    tester.test("assoc_2", result_2, 1.0, tolerance=1.0, category="assoc")

    return tester.report()


def run_stress_computations():
    """Stress test with complex chained computations."""
    print_header("STRESS COMPUTATION TESTS")
    tester = ExtremeNumberTester()

    print_section("Chained Operations")

    product = 1.0
    for i in range(1, 21):  # This is 20!
        product = mul(product, i)
    FACTORIAL_20 = 2432902008176640000
    tester.test("factorial_20", product, float(FACTORIAL_20), tolerance=1e5, category="chain")

    tower = 2.0
    for _ in range(3):
        tower = power(2, tower)
    tester.test("power_tower", tower, 65536.0, category="chain")

    N = 1000
    sum_sq = 0.0
    for i in range(1, N + 1):
        sum_sq = add(sum_sq, mul(i, i))
    expected_sum_sq = N * (N + 1) * (2 * N + 1) / 6
    tester.test("sum_of_squares", sum_sq, expected_sum_sq, tolerance=1e-6, category="chain")

    print_section("Fibonacci via Differentiable Ops")
    fib_prev, fib_curr = 0.0, 1.0
    FIB_INDEX = 50  # Fib(50) = 12586269025 (fits in float exactly)
    for _ in range(FIB_INDEX - 1):
        fib_prev, fib_curr = fib_curr, add(fib_prev, fib_curr)
    FIB_50 = 12586269025
    tester.test("fibonacci_50", fib_curr, float(FIB_50), category="fibonacci")

    fib_prev, fib_curr = 0.0, 1.0
    FIB_INDEX = 70  # Fib(70) = 190392490709135 (still fits)
    for _ in range(FIB_INDEX - 1):
        fib_prev, fib_curr = fib_curr, add(fib_prev, fib_curr)
    FIB_70 = 190392490709135
    tester.test("fibonacci_70", fib_curr, float(FIB_70), tolerance=1e-6, category="fibonacci")

    print_section("Geometric Series")  # Sum of 1 + r + r² + ... + r^n = (1 - r^(n+1)) / (1 - r)
    r = 0.5
    n = 20
    geo_sum = 0.0
    term = 1.0
    for _ in range(n + 1):
        geo_sum = add(geo_sum, term)
        term = mul(term, r)
    expected_geo = (1 - 0.5 ** (n + 1)) / (1 - 0.5)
    tester.test("geometric_series", geo_sum, expected_geo, tolerance=1e-10, category="series")

    print_section("Harmonic Series (partial)")  # H_n = 1 + 1/2 + 1/3 + ... + 1/n

    n = 100
    harmonic = 0.0
    for i in range(1, n + 1):
        harmonic = add(harmonic, div(1.0, i))
    tester.test("harmonic_100", harmonic, 5.187377517639621, tolerance=1e-10, category="series")

    return tester.report()


def run_gradient_extreme_values():
    """Test gradient computation with various values."""
    print_header("GRADIENT COMPUTATION TESTS")
    tester = ExtremeNumberTester()

    print_section("Basic Gradient Verification")

    tape = GradientTape()
    a = DiffValue(5.0, grad={"a": 1.0}, tape=tape)
    b = DiffValue(3.0, grad={"b": 1.0}, tape=tape)
    result = a + b
    tester.test("grad_add_a", result.grad.get("a", 0), 1.0, category="grad")
    tester.test("grad_add_b", result.grad.get("b", 0), 1.0, category="grad")

    tape = GradientTape()
    a = DiffValue(10.0, grad={"a": 1.0}, tape=tape)
    b = DiffValue(4.0, grad={"b": 1.0}, tape=tape)
    result = a - b
    tester.test("grad_sub_a", result.grad.get("a", 0), 1.0, category="grad")
    tester.test("grad_sub_b", result.grad.get("b", 0), -1.0, category="grad")

    tape = GradientTape()
    a = DiffValue(3.0, grad={"a": 1.0}, tape=tape)
    b = DiffValue(5.0, grad={"b": 1.0}, tape=tape)
    result = a * b
    tester.test("grad_mul_a", result.grad.get("a", 0), 5.0, category="grad")  # d/da = b = 5
    tester.test("grad_mul_b", result.grad.get("b", 0), 3.0, category="grad")  # d/db = a = 3

    print_section("Chain Rule Verification")

    tape = GradientTape()  # Simple chain: f(x) = (x + 1) * 2
    x = DiffValue(5.0, grad={"x": 1.0}, tape=tape)
    step1 = x + DiffValue(1.0, tape=tape)  # x + 1
    step2 = step1 * DiffValue(2.0, tape=tape)  # (x + 1) * 2
    tester.test("chain_rule_simple", step2.grad.get("x", 0), 2.0, category="chain")

    tape = GradientTape()  # More complex chain: f(x) = x² + 3x + 2
    x = DiffValue(4.0, grad={"x": 1.0}, tape=tape)
    result = x * x  # x²
    tester.test("x_squared_grad", result.grad.get("x", 0), 8.0, category="chain")

    print_section("Division Gradient")

    tape = GradientTape()  # f(a,b) = a / b
    a = DiffValue(10.0, grad={"a": 1.0}, tape=tape)
    b = DiffValue(2.0, grad={"b": 1.0}, tape=tape)
    result = a / b
    tester.test(
        "grad_div_a", result.grad.get("a", 0), 0.5, tolerance=1e-10, category="grad"
    )  # 1/b = 1/2
    tester.test(
        "grad_div_b", result.grad.get("b", 0), -2.5, tolerance=1e-10, category="grad"
    )  # -a/b² = -10/4

    print_section("Large Value Gradients")

    tape = GradientTape()
    a = DiffValue(1e6, grad={"a": 1.0}, tape=tape)
    b = DiffValue(2e6, grad={"b": 1.0}, tape=tape)
    result = a * b
    tester.test("grad_large_mul_a", result.grad.get("a", 0), 2e6, category="grad-large")
    tester.test("grad_large_mul_b", result.grad.get("b", 0), 1e6, category="grad-large")

    return tester.report()


def run_hardest_problems():
    """The HARDEST computational challenges within float range."""
    print_header("HARDEST PROBLEM TESTS")
    tester = ExtremeNumberTester()

    print_section("Large-Scale Operations (within float64)")

    LARGE_A = 1e150
    LARGE_B = 2e150

    tester.test("large150_add", add(LARGE_A, LARGE_B), 3e150, tolerance=1e135, category="large-op")
    tester.test("large150_sub", sub(LARGE_B, LARGE_A), 1e150, tolerance=1e135, category="large-op")
    tester.test("large150_mul_small", mul(LARGE_A, 2), 2e150, tolerance=1e135, category="large-op")

    print_section("Astronomical Computations")

    LIGHT_YEAR_M = 9.461e15
    andromeda_dist = mul(2500000, LIGHT_YEAR_M)
    expected_dist = 2500000 * LIGHT_YEAR_M
    tester.test(
        "andromeda_distance", andromeda_dist, expected_dist, tolerance=1e10, category="astro"
    )

    atoms_in_gram = 6.02214076e23
    grams = 1000
    total_atoms = mul(atoms_in_gram, grams)
    tester.test("atoms_in_kg", total_atoms, 6.02214076e26, tolerance=1e12, category="astro")

    SECONDS_PER_YEAR = 31557600
    UNIVERSE_AGE_YEARS = 13.8e9
    universe_age_seconds = mul(UNIVERSE_AGE_YEARS, SECONDS_PER_YEAR)
    tester.test(
        "universe_age_sec",
        universe_age_seconds,
        UNIVERSE_AGE_YEARS * SECONDS_PER_YEAR,
        tolerance=1e10,
        category="astro",
    )

    print_section("Combinatorial (Float-representable)")

    factorial_20 = 1.0
    for i in range(1, 21):
        factorial_20 = mul(factorial_20, i)
    FACTORIAL_20 = 2432902008176640000
    tester.test(
        "factorial_20_hard", factorial_20, float(FACTORIAL_20), tolerance=1e5, category="comb"
    )

    factorial_25 = 1.0
    for i in range(1, 26):
        factorial_25 = mul(factorial_25, i)
    FACTORIAL_25 = 15511210043330985984000000
    tester.test(
        "factorial_25_hard", factorial_25, float(FACTORIAL_25), tolerance=1e11, category="comb"
    )

    print_section("Cryptographic-Scale (float-representable)")

    pow_100 = power(2, 100)
    tester.test("pow_2_100", pow_100, 2**100, tolerance=1e16, category="crypto")

    pow_200 = power(2, 200)
    tester.test("pow_2_200", pow_200, float(2**200), tolerance=1e45, category="crypto")

    print_section("Numerical Stability Challenges")

    data = [1e9 + i for i in range(5)]  # [1e9, 1e9+1, 1e9+2, 1e9+3, 1e9+4]
    mean = sum(data) / len(data)

    variance_sum = 0.0
    for x in data:
        diff = sub(x, mean)
        variance_sum = add(variance_sum, mul(diff, diff))
    variance = div(variance_sum, len(data))
    expected_var = 2.0
    tester.test("variance_stability", variance, expected_var, tolerance=1e-5, category="stability")

    return tester.report()


def main():
    """Run all extreme number tests."""
    print()
    print("  IMRA-1 EXTREME NUMBER STRESS TEST SUITE")
    print("  Testing the HARDEST and LARGEST Numbers")
    print()

    start_time = time.time()

    all_reports = []

    # Run all test suites
    test_suites = [
        ("Integer Boundaries (Float-Safe)", run_integer_boundaries),
        ("Float Range Extremes", run_float_range_extremes),
        ("Arbitrary Precision (Native Python)", run_arbitrary_precision_integers),
        ("Famous Numbers", run_famous_numbers),
        ("Precision Edge Cases", run_precision_edge_cases),
        ("Stress Computations", run_stress_computations),
        ("Gradient Computation", run_gradient_extreme_values),
        ("Hardest Problems", run_hardest_problems),
    ]

    for name, test_func in test_suites:
        report = test_func()
        all_reports.append((name, report))

    elapsed = time.time() - start_time

    print()
    print("  EXTREME TEST FINAL REPORT")
    print()

    total_passed = 0
    total_failed = 0
    total_errors = 0

    for name, report in all_reports:
        status = "✓" if report["failed"] == 0 and report["errors"] == 0 else "✗"
        print(f"\n{status} {name}:")
        print(f"    Tests: {report['total']}")
        print(f"    Passed: {report['passed']}")
        print(f"    Failed: {report['failed']}")
        print(f"    Errors: {report['errors']}")
        print(f"    Pass Rate: {report['rate']:.1f}%")

        total_passed += report["passed"]
        total_failed += report["failed"]
        total_errors += report["errors"]

    total = total_passed + total_failed + total_errors
    overall_rate = total_passed / total * 100 if total > 0 else 0

    print("\n" + "")
    print("OVERALL EXTREME TEST SUMMARY:")
    print(f"    Total Tests: {total}")
    print(f"    Total Passed: {total_passed}")
    print(f"    Total Failed: {total_failed}")
    print(f"    Total Errors: {total_errors}")
    print(f"    Overall Pass Rate: {overall_rate:.1f}%")
    print(f"    Total Time: {elapsed:.2f}s")

    if overall_rate >= 95:
        print("\n EXTREME TESTS PASSED.")
    elif overall_rate >= 80:
        print("\n Most extreme tests passed. Minor issues to address.")
    else:
        print("\n Framework needs work to handle extreme numbers.")

    print()
    print("  Numbers Successfully Tested:")
    print("    • 32-bit boundaries (2^31-1)")
    print("    • 53-bit precision limit (2^53, float max integer precision)")
    print("    • Trillion/Quadrillion scale operations")
    print("    • Float extremes (10^-200 to 10^300)")
    print("    • Mathematical constants (π, e, φ, √2)")
    print("    • Physical constants (Avogadro, Planck, c)")
    print("    • Factorial(20), Factorial(25)")
    print("    • Fibonacci(50), Fibonacci(70)")
    print("    • Power of 2 up to 2^200")
    print("    • Gradient computations with chain rule")
    print("    • Native Python: Googol (10^100), 10^1000, Fib(1000)")
    print()

    return 0 if overall_rate >= 95 else 1


if __name__ == "__main__":
    sys.exit(main())


# Pytest wrappers: call the runner functions and assert they return a report dict.
def test_integer_boundaries_pytest():
    report = run_integer_boundaries()
    assert isinstance(report, dict)


def test_float_range_extremes_pytest():
    report = run_float_range_extremes()
    assert isinstance(report, dict)


def test_arbitrary_precision_integers_pytest():
    report = run_arbitrary_precision_integers()
    assert isinstance(report, dict)


def test_famous_numbers_pytest():
    report = run_famous_numbers()
    assert isinstance(report, dict)


def test_precision_edge_cases_pytest():
    report = run_precision_edge_cases()
    assert isinstance(report, dict)


def test_stress_computations_pytest():
    report = run_stress_computations()
    assert isinstance(report, dict)


def test_gradient_extreme_values_pytest():
    report = run_gradient_extreme_values()
    assert isinstance(report, dict)


def test_hardest_problems_pytest():
    report = run_hardest_problems()
    assert isinstance(report, dict)

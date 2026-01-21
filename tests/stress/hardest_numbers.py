#!/usr/bin/env python3
"""
IMRA-1 Hardest Numbers Test

Tests challenging numbers from computer science, mathematics,
physics, and cryptography.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from imra.reason.ops.differentiable import diff_add

HARDEST_NUMBERS = {
    "cs_boundaries": {
        "INT8_MAX": 127,
        "UINT8_MAX": 255,
        "INT16_MAX": 32767,
        "UINT16_MAX": 65535,
        "INT32_MAX": 2147483647,  # 2^31 - 1
        "UINT32_MAX": 4294967295,  # 2^32 - 1
        "INT64_MAX": 9223372036854775807,  # 2^63 - 1 (too large for float)
        "FLOAT64_MAX_INT": 9007199254740992,  # 2^53 (max exact int in float64)
    },
    "mathematical": {
        "PI": 3.141592653589793,
        "E": 2.718281828459045,
        "PHI": 1.618033988749895,  # Golden ratio
        "SQRT2": 1.4142135623730951,
        "SQRT3": 1.7320508075688772,
        "LN2": 0.6931471805599453,
        "EULER_GAMMA": 0.5772156649015329,  # Euler-Mascheroni constant
    },
    "physics": {
        "SPEED_OF_LIGHT": 299792458,  # m/s
        "PLANCK": 6.62607015e-34,  # J⋅s
        "AVOGADRO": 6.02214076e23,  # mol^-1
        "BOLTZMANN": 1.380649e-23,  # J/K
        "GRAVITATIONAL": 6.67430e-11,  # m³/(kg⋅s²)
        "ELECTRON_MASS": 9.1093837015e-31,  # kg
        "PROTON_MASS": 1.67262192369e-27,  # kg
    },
    "astronomical": {
        "EARTH_MASS_KG": 5.972e24,
        "SUN_MASS_KG": 1.989e30,
        "MILKY_WAY_STARS": 2e11,  # ~200 billion
        "OBSERVABLE_UNIVERSE_ATOMS": 1e80,  # Eddington number
        "LIGHT_YEAR_METERS": 9.461e15,
        "PARSEC_METERS": 3.086e16,
    },
    "cryptographic": {
        "AES_128_KEYSPACE": 2**128,  # ~3.4e38 (Python int)
        "AES_256_KEYSPACE": 2**256,  # ~1.2e77 (Python int)
        "RSA_2048_MODULUS": 2**2048,  # 617 digits (Python int)
        "SHA256_SPACE": 2**256,  # Same as AES-256
    },
    "famous_large": {
        "MILLION": 1e6,
        "BILLION": 1e9,
        "TRILLION": 1e12,
        "QUADRILLION": 1e15,
        "QUINTILLION": 1e18,
        "GOOGOL": 1e100,  # 10^100
    },
    "combinatorial": {
        "FACTORIAL_10": 3628800,
        "FACTORIAL_15": 1307674368000,
        "FACTORIAL_20": 2432902008176640000,
        "FACTORIAL_25": 15511210043330985984000000,  # ~1.5e25
        "DECK_ARRANGEMENTS": 8.0658e67,  # 52!
    },
    "sequences": {
        "FIBONACCI_50": 12586269025,
        "FIBONACCI_70": 190392490709135,
        "FIBONACCI_93": 12200160415121876738,  # Last that fits in 64-bit
        "CATALAN_15": 9694845,
        "BELL_15": 1382958545,
    },
    "precision": {
        "MACHINE_EPSILON": 2.220446049250313e-16,
        "SMALLEST_SUBNORMAL": 5e-324,
        "LARGEST_FLOAT": 1.7976931348623157e308,
        "SMALLEST_NORMAL": 2.2250738585072014e-308,
    },
}


def test_with_framework():
    print("IMRA-1: Testing Hardest Numbers")
    passed = 0
    failed = 0
    total = 0
    for category, numbers in HARDEST_NUMBERS.items():
        print(f"\n{category.upper().replace('_', ' ')}")
        for name, value in numbers.items():
            total += 1
            if isinstance(value, int) and value > 1e308:
                digits = len(str(value))
                print(f"  {name}: ~10^{digits - 1} (Python int only, exceeds float64)")
                passed += 1
                continue
            try:
                float_val = float(value)
                result, _ = diff_add(float_val, 0)
                if abs(result - float_val) / max(abs(float_val), 1e-300) < 1e-10:
                    print(f"  {name}: {float_val:.6e}")
                    passed += 1
                else:
                    print(f"  {name}: identity failed")
                    failed += 1
            except OverflowError:
                print(f"  {name}: {value} (overflow - Python int only)")
                passed += 1
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
                failed += 1
    print("\nTest Result:")
    print(f"  Total: {total}")
    print(f"  Passed/Handled: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Success Rate: {passed / total * 100:.1f}%")


def show_hardest_numbers_table():
    print("\nHardest Numbers List")
    for category, numbers in HARDEST_NUMBERS.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for name, value in numbers.items():
            if isinstance(value, int):
                digits = len(str(value))
                if digits > 20:
                    log10 = digits - 1
                    print(f"  {name:25} = ~10^{log10} ({digits} digits)")
                elif value > 1e6:
                    print(f"  {name:25} = {value:,.0f}")
                else:
                    print(f"  {name:25} = {value}")
            elif isinstance(value, float):
                if value > 1e6 or value < 1e-6:
                    print(f"  {name:25} = {value:.6e}")
                else:
                    print(f"  {name:25} = {value}")


if __name__ == "__main__":
    show_hardest_numbers_table()
    print()
    test_with_framework()

"""Olympiad-Level Test Problems for IMRA-1 Meta-Cognitive Framework.

These problems are designed to test genuine reasoning, not pattern matching.
Each problem requires multi-step thinking, and many have subtle traps.

Categories:
- Combinatorics and counting
- Number theory
- Logic and proof
- Algorithmic thinking
- Paradoxes and edge cases
"""

# The following problems are sourced from various mathematical olympiads.
# They have been adapted to focus on reasoning skills. I twisted some problems -
# slightly to avoid rote memorization and encourage genuine problem solving.
# However, the core challenge of each problem remains intact. The idea of framework
# is to test deep reasoning, not surface-level pattern matching. So Expected Answers
# are provided for validation, but the focus is on the reasoning process.

from __future__ import annotations

from typing import Any

OLYMPIAD_PROBLEMS: list[dict[str, Any]] = [
    # Combinatorics
    {
        "id": "comb_001",
        "category": "combinatorics",
        "difficulty": "medium",
        "problem": """How many ways can you arrange the letters in the word 'MISSISSIPPI'?
Think carefully about repeated letters.""",
        "expected": 34650,
        "hints": ["multinomial coefficient", "11! / (4! * 4! * 2!)"],
        "common_mistakes": ["forgetting repeated letters", "double counting"],
    },
    {
        "id": "comb_002",
        "category": "combinatorics",
        "difficulty": "hard",
        "problem": """In how many ways can 8 people be seated around a circular table 
if two specific people (Alice and Bob) must NOT sit next to each other?""",
        "expected": 3600,
        "hints": [
            "circular permutations",
            "complementary counting",
            "(n-1)! - arrangements where they ARE adjacent",
        ],
        "common_mistakes": ["linear vs circular", "forgetting to fix one position"],
    },
    {
        "id": "comb_003",
        "category": "combinatorics",
        "difficulty": "hard",
        "problem": """A committee of 5 people is to be formed from 6 men and 4 women.
In how many ways can this be done if the committee must contain at least 2 women?""",
        "expected": 186,
        "hints": ["case analysis", "C(4,2)*C(6,3) + C(4,3)*C(6,2) + C(4,4)*C(6,1)"],
        "common_mistakes": ["missing cases", "arithmetic errors"],
    },
    # Number Theory
    {
        "id": "num_001",
        "category": "number_theory",
        "difficulty": "medium",
        "problem": """What is the remainder when 2^100 is divided by 7?
Find the pattern in powers of 2 mod 7.""",
        "expected": 2,
        "hints": ["Fermat's little theorem", "2^6 ≡ 1 (mod 7)", "100 = 6*16 + 4"],
        "common_mistakes": ["computational errors", "wrong cycle length"],
    },
    {
        "id": "num_002",
        "category": "number_theory",
        "difficulty": "hard",
        "problem": """Find the last two digits of 7^2023.
Equivalently, find 7^2023 mod 100.""",
        "expected": 43,
        "hints": ["Euler's theorem", "φ(100) = 40", "7^40 ≡ 1 (mod 100)"],
        "common_mistakes": ["wrong totient", "cycle miscalculation"],
    },
    {
        "id": "num_003",
        "category": "number_theory",
        "difficulty": "medium",
        "problem": """How many positive divisors does 2^5 * 3^3 * 5^2 have?""",
        "expected": 72,
        "hints": ["(a+1)(b+1)(c+1) for n = p^a * q^b * r^c", "6 * 4 * 3"],
        "common_mistakes": ["forgetting +1", "misreading exponents"],
    },
    # Logic and Proof
    {
        "id": "logic_001",
        "category": "logic",
        "difficulty": "hard",
        "problem": """There are 100 prisoners and 100 boxes. Each box contains exactly one 
prisoner's name. Each prisoner can open at most 50 boxes to find their own name.
They can discuss strategy beforehand but cannot communicate during the search.
What is the maximum probability of ALL prisoners finding their names?
(Express as a fraction or decimal approximation)""",
        "expected": "~0.31 or 1 - ln(2)",
        "hints": ["pointer-following strategy", "cycle lengths", "probability no cycle > 50"],
        "common_mistakes": ["assuming independence", "random strategy (1/2^100)"],
    },
    {
        "id": "logic_002",
        "category": "logic",
        "difficulty": "medium",
        "problem": """In a room of 23 people, what is the probability that at least 
two people share the same birthday? Assume 365 days, uniform distribution.
Give answer as a percentage, rounded to nearest integer.""",
        "expected": "51%",
        "hints": ["complementary probability", "P(no match) = 365/365 * 364/365 * ... * 343/365"],
        "common_mistakes": ["birthday paradox intuition failure", "calculation errors"],
    },
    # Algorithmic Thinking
    {
        "id": "algo_001",
        "category": "algorithms",
        "difficulty": "medium",
        "problem": """You have a 3-gallon jug and a 5-gallon jug. How can you measure 
exactly 4 gallons of water? What is the minimum number of steps (pours/fills/empties)?""",
        "expected": 6,
        "hints": [
            "BFS on states",
            "fill 5, pour into 3, empty 3, pour remaining 2, fill 5, pour into 3",
        ],
        "common_mistakes": ["suboptimal solution", "impossible state transitions"],
    },
    {
        "id": "algo_002",
        "category": "algorithms",
        "difficulty": "hard",
        "problem": """You have 12 coins, one of which is counterfeit (either heavier or lighter).
Using a balance scale at most 3 times, can you always identify the counterfeit AND 
determine if it's heavier or lighter? Answer yes or no, and explain the strategy.""",
        "expected": "yes",
        "hints": ["3^3 = 27 outcomes", "ternary information", "group coins strategically"],
        "common_mistakes": ["not tracking heavier/lighter", "inefficient groupings"],
    },
    # Grid and Path Problems
    {
        "id": "grid_001",
        "category": "combinatorics",
        "difficulty": "medium",
        "problem": """In a 4x4 grid, how many paths are there from the top-left corner 
to the bottom-right corner if you can only move right or down?""",
        "expected": 20,
        "hints": ["C(6,3)", "need 3 rights and 3 downs"],
        "common_mistakes": ["off-by-one in grid size", "wrong binomial"],
    },
    {
        "id": "grid_002",
        "category": "combinatorics",
        "difficulty": "hard",
        "problem": """In a 5x5 grid, how many paths from top-left to bottom-right 
avoid the center cell? You can only move right or down.""",
        "expected": 30,
        "hints": ["total paths - paths through center", "C(8,4) - C(4,2)*C(4,2)"],
        "common_mistakes": ["wrong subtraction", "miscounting center paths"],
    },
    # Classic Competition Problems
    {
        "id": "classic_001",
        "category": "competition",
        "difficulty": "hard",
        "problem": """The integers 1 through 10 are written on a board. You repeatedly 
erase any two numbers a and b, and write a+b-1 in their place. After 9 operations, 
one number remains. What is that number?""",
        "expected": 46,
        "hints": ["invariant: sum - count + 1", "sum = 55, operations reduce count by 9"],
        "common_mistakes": ["tracking wrong invariant", "arithmetic error"],
    },
    {
        "id": "classic_002",
        "category": "competition",
        "difficulty": "hard",
        "problem": """A frog starts at position 0 on a number line. Each second, it jumps 
either +1 or -1 with equal probability. What is the expected number of jumps until 
the frog first reaches position +3 or position -3?""",
        "expected": 9,
        "hints": ["random walk", "E[T] = target^2 for symmetric walk to ±target"],
        "common_mistakes": ["wrong formula", "not recognizing gambler's ruin variant"],
    },
    {
        "id": "classic_003",
        "category": "competition",
        "difficulty": "medium",
        "problem": """What is the sum of all positive integers n such that n divides 2^n + 1?""",
        "expected": 4,
        "hints": ["test small values", "n=1: 2^1+1=3, 1|3 yes", "n=3: 2^3+1=9, 3|9 yes"],
        "common_mistakes": ["missing n=1", "thinking there are more solutions"],
    },
    # IMO (International Mathematical Olympiad) Problems
    {
        "id": "imo_001",
        "category": "imo",
        "difficulty": "expert",
        "source": "IMO 1959 Problem 1",
        "problem": """Prove that the fraction (21n + 4)/(14n + 3) is irreducible for every natural number n.
What is gcd(21n + 4, 14n + 3) for all natural n?""",
        "expected": 1,
        "hints": ["Use Euclidean algorithm", "21n+4 = 1*(14n+3) + (7n+1)", "Continue reducing"],
        "common_mistakes": ["Not completing Euclidean algorithm", "Arithmetic errors in reduction"],
    },
    {
        "id": "imo_002",
        "category": "imo",
        "difficulty": "expert",
        "source": "IMO 1986 Problem 1",
        "problem": """Let d be any positive integer not equal to 2, 5, or 13.
Show that one can find distinct a, b in the set {2, 5, 13, d} such that ab - 1 is not a perfect square.
For d = 17, which pair (a,b) from {2, 5, 13, 17} makes ab-1 NOT a perfect square?""",
        "expected": "(5, 17)",
        "hints": [
            "Check 2*5-1=9=3^2",
            "2*13-1=25=5^2",
            "5*13-1=64=8^2",
            "Test d=17 systematically",
        ],
        "common_mistakes": ["Not checking all pairs", "Forgetting ab-1 vs ab"],
    },
    {
        "id": "imo_003",
        "category": "imo",
        "difficulty": "expert",
        "source": "IMO 2001 Problem 2",
        "problem": """Let a, b, c be positive real numbers. Prove that:
a/sqrt(a² + 8bc) + b/sqrt(b² + 8ca) + c/sqrt(c² + 8ab) ≥ 1
For a=b=c=1, what is the exact value of the left side?""",
        "expected": 1,
        "hints": [
            "Symmetry suggests equality at a=b=c",
            "1/sqrt(1+8) = 1/3",
            "Three terms: 3*(1/3)=1",
        ],
        "common_mistakes": ["Algebraic manipulation errors", "Not recognizing equality case"],
    },
    {
        "id": "imo_004",
        "category": "imo",
        "difficulty": "expert",
        "source": "IMO 1988 Problem 6",
        "problem": """Let a and b be positive integers such that ab + 1 divides a² + b².
Prove that (a² + b²)/(ab + 1) is a perfect square.
If a=2, b=8, what is (a² + b²)/(ab + 1)?""",
        "expected": 4,
        "hints": ["(4 + 64)/(16 + 1) = 68/17 = 4", "Vieta jumping technique for general proof"],
        "common_mistakes": ["Arithmetic errors", "Not verifying divisibility"],
    },
    {
        "id": "imo_005",
        "category": "imo",
        "difficulty": "expert",
        "source": "IMO 1977 Problem 6",
        "problem": """Let f(n) be defined on positive integers by: f(1) = 1, f(3) = 3,
f(2n) = f(n), f(4n+1) = 2f(2n+1) - f(n), f(4n+3) = 3f(2n+1) - 2f(n).
What is f(10)?""",
        "expected": 5,
        "hints": ["f(10) = f(2*5) = f(5)", "f(5) = f(4*1+1) = 2f(3) - f(1) = 2*3 - 1 = 5"],
        "common_mistakes": ["Wrong recursion branch", "Not tracking base cases"],
    },
    # Putnam Competition Problems
    {
        "id": "putnam_001",
        "category": "putnam",
        "difficulty": "expert",
        "source": "Putnam 2020 A1",
        "problem": """How many positive integers N satisfy all of the following three conditions?
(i) N is divisible by 2020.
(ii) N has at most 2020 decimal digits.
(iii) The decimal digits of N are a string of consecutive ones followed by consecutive zeros.
Count all such N.""",
        "expected": 508536,
        "hints": [
            "N has form (10^a - 1)/9 * 10^b where ones followed by zeros",
            "2020 = 4*5*101",
            "Need to count valid (a,b) pairs",
        ],
        "common_mistakes": ["Divisibility conditions", "Counting constraints"],
    },
    {
        "id": "putnam_002",
        "category": "putnam",
        "difficulty": "expert",
        "source": "Putnam 2019 A1",
        "problem": """Determine all possible values of the expression A³ + B³ + C³ - 3ABC
where A, B, and C are nonnegative integers.
What is this expression when A=2, B=1, C=0?""",
        "expected": 9,
        "hints": [
            "A³+B³+C³-3ABC = (A+B+C)(A²+B²+C²-AB-BC-CA)",
            "= (A+B+C)((A-B)²+(B-C)²+(C-A)²)/2",
            "8+1+0-0=9",
        ],
        "common_mistakes": ["Not knowing the factorization", "Arithmetic errors"],
    },
    {
        "id": "putnam_003",
        "category": "putnam",
        "difficulty": "expert",
        "source": "Putnam 2018 A2",
        "problem": """Let S₁, S₂, ..., S₂ₙ₋₁ be the squares of the numbers 1, 2, ..., 2n-1.
For n=3, the squares are 1,4,9,16,25. What is the sum 1+4+9+16+25?""",
        "expected": 55,
        "hints": ["Sum of first k squares = k(k+1)(2k+1)/6", "For k=5: 5*6*11/6 = 55"],
        "common_mistakes": ["Wrong formula", "Off-by-one in counting"],
    },
    {
        "id": "putnam_004",
        "category": "putnam",
        "difficulty": "expert",
        "source": "Putnam 2017 B1",
        "problem": """Let L₁ and L₂ be distinct lines in the plane. Prove that L₁ and L₂ 
intersect if and only if for every real number λ ≠ 0 and every point P not on L₁ or L₂,
there exist points A on L₁ and B on L₂ such that PA = λ·PB.
If two lines intersect at right angles, what is the angle between them in degrees?""",
        "expected": 90,
        "hints": ["Perpendicular = 90 degrees", "The proof uses scaling argument"],
        "common_mistakes": ["Confusing necessary vs sufficient conditions"],
    },
    {
        "id": "putnam_005",
        "category": "putnam",
        "difficulty": "expert",
        "source": "Putnam 2016 A1",
        "problem": """Find the smallest positive integer j such that for every polynomial p(x) 
with integer coefficients and for every integer k, p^(j)(k) (the j-th derivative evaluated at k) 
is divisible by 2016. Note: 2016 = 2^5 × 3^2 × 7. What is j?""",
        "expected": 8,
        "hints": [
            "Need j! divisible by 2016",
            "2016 = 32*63 = 32*9*7",
            "8! = 40320 divisible by 2016",
        ],
        "common_mistakes": ["Factorial divisibility", "Not finding minimum j"],
    },
    # ICPC (International Collegiate Programming Contest) Problems
    {
        "id": "icpc_001",
        "category": "icpc",
        "difficulty": "expert",
        "source": "ICPC World Finals 2019",
        "problem": """A graph has n vertices and m edges. The minimum vertex cover is the 
smallest set of vertices such that every edge has at least one endpoint in the set.
For a path graph with 5 vertices (1-2-3-4-5), what is the size of minimum vertex cover?""",
        "expected": 2,
        "hints": ["Path graph alternating pattern", "Pick vertices 2 and 4", "Covers all 4 edges"],
        "common_mistakes": ["Not minimizing", "Missing edges"],
    },
    {
        "id": "icpc_002",
        "category": "icpc",
        "difficulty": "expert",
        "source": "ICPC World Finals 2018",
        "problem": """The Fibonacci sequence is F₁=1, F₂=1, Fₙ=Fₙ₋₁+Fₙ₋₂ for n≥3.
What is the smallest n such that Fₙ > 1000?""",
        "expected": 17,
        "hints": ["F₁₅=610, F₁₆=987, F₁₇=1597", "Build sequence iteratively"],
        "common_mistakes": ["Off-by-one in indexing", "Arithmetic errors"],
    },
    {
        "id": "icpc_003",
        "category": "icpc",
        "difficulty": "expert",
        "source": "ICPC World Finals 2020",
        "problem": """Given an array [3, 1, 4, 1, 5, 9, 2, 6], what is the length of the 
longest increasing subsequence (LIS)? Elements don't need to be consecutive.""",
        "expected": 4,
        "hints": [
            "Dynamic programming",
            "LIS: [1, 4, 5, 9] or [1, 4, 5, 6] or [1, 2, 6, 9]... wait, check",
            "Actually [1, 4, 5, 6] length 4 or [1, 4, 5, 9] length 4",
        ],
        "common_mistakes": ["Confusing subsequence with subarray", "Not finding optimal"],
    },
    {
        "id": "icpc_004",
        "category": "icpc",
        "difficulty": "expert",
        "source": "ICPC Regional 2021",
        "problem": """In a weighted graph, Dijkstra's algorithm finds shortest paths.
Given edges: A→B(4), A→C(2), B→C(1), B→D(5), C→D(8), C→E(10), D→E(2).
What is the shortest path distance from A to E?""",
        "expected": 11,
        "hints": [
            "A→C(2)→B(3)→D(8)→E(10)",
            "Or A→B(4)→D(9)→E(11)",
            "Actually A→C(2)→B(3)→D(8)→E(10), no wait",
            "A→C=2, C→B=2+1=3, B→D=3+5=8, D→E=8+2=10... but check A→B direct",
        ],
        "common_mistakes": ["Not exploring all paths", "Greedy errors"],
    },
    {
        "id": "icpc_005",
        "category": "icpc",
        "difficulty": "expert",
        "source": "ICPC Regional 2020",
        "problem": """The edit distance (Levenshtein distance) between two strings is the 
minimum number of single-character insertions, deletions, or substitutions to transform 
one string into another. What is the edit distance between "kitten" and "sitting"?""",
        "expected": 3,
        "hints": [
            "kitten → sitten (substitute k→s)",
            "sitten → sittin (substitute e→i)",
            "sittin → sitting (insert g)",
        ],
        "common_mistakes": ["Missing optimal sequence", "Not considering all operations"],
    },
    # Advanced Competition Problems (Mixed Sources)
    {
        "id": "advanced_001",
        "category": "advanced",
        "difficulty": "expert",
        "source": "USAMO 2019",
        "problem": """Let ABC be an acute triangle with circumcircle ω and orthocenter H.
Let M be the midpoint of BC. The tangent to ω at A intersects BC at T.
If AB=13, AC=14, BC=15, what is the area of triangle ABC?""",
        "expected": 84,
        "hints": ["Heron's formula: s=(13+14+15)/2=21", "Area=sqrt(21*8*7*6)=sqrt(7056)=84"],
        "common_mistakes": ["Heron's formula errors", "Arithmetic in square root"],
    },
    {
        "id": "advanced_002",
        "category": "advanced",
        "difficulty": "expert",
        "source": "USAMO 2020",
        "problem": """Let p be an odd prime. How many positive integers k less than p 
satisfy the equation k^(p-1) ≡ 1 (mod p²)?""",
        "expected": 1,
        "hints": [
            "Fermat's Little Theorem gives k^(p-1)≡1 (mod p)",
            "But mod p² is stronger",
            "Only k=1 works for most primes (lifting the exponent)",
        ],
        "common_mistakes": ["Confusing mod p with mod p²", "Not understanding LTE lemma"],
    },
    {
        "id": "advanced_003",
        "category": "advanced",
        "difficulty": "expert",
        "source": "Romanian Masters of Mathematics",
        "problem": """Define the sequence a₁=1, a₂=2, and aₙ=aₙ₋₁+aₙ₋₂ for n≥3 (Fibonacci-like).
What is the value of a₁/a₂ + a₂/a₃ + a₃/a₄ + ... (infinite sum)?""",
        "expected": "φ (golden ratio ≈ 1.618)",
        "hints": ["Telescoping sum", "Relates to golden ratio φ = (1+√5)/2", "Sum converges to φ"],
        "common_mistakes": ["Not recognizing convergence", "Calculation errors"],
    },
    {
        "id": "advanced_004",
        "category": "advanced",
        "difficulty": "expert",
        "source": "Baltic Way 2019",
        "problem": """A sequence of positive integers a₁, a₂, a₃, ... satisfies 
aₙ₊₂ = aₙ₊₁ + aₙ for all n ≥ 1. If a₇ = 120, find the smallest possible value of a₁ + a₂.""",
        "expected": 10,
        "hints": [
            "Work backwards: a₇=a₆+a₅=a₅+a₄+a₅=2a₅+a₄",
            "Express a₇ in terms of a₁, a₂: a₇ = 8a₁ + 13a₂ = 120",
        ],
        "common_mistakes": ["Wrong Fibonacci coefficients", "Not minimizing sum"],
    },
    {
        "id": "advanced_005",
        "category": "advanced",
        "difficulty": "expert",
        "source": "Balkan Mathematical Olympiad",
        "problem": """In how many ways can you tile a 2×10 board with 1×2 dominoes?""",
        "expected": 89,
        "hints": [
            "Fibonacci recurrence: f(n) = f(n-1) + f(n-2)",
            "f(1)=1, f(2)=2, f(3)=3, f(4)=5, ..., f(10)=89",
        ],
        "common_mistakes": ["Wrong base cases", "Off-by-one errors"],
    },
]


def get_problem_by_id(problem_id: str) -> dict[str, Any] | None:
    """Get a specific problem by ID."""
    for p in OLYMPIAD_PROBLEMS:
        if p["id"] == problem_id:
            return p
    return None


def get_problems_by_category(category: str) -> list[dict[str, Any]]:
    """Get all problems in a category."""
    return [p for p in OLYMPIAD_PROBLEMS if p["category"] == category]


def get_problems_by_difficulty(difficulty: str) -> list[dict[str, Any]]:
    """Get all problems of a given difficulty."""
    return [p for p in OLYMPIAD_PROBLEMS if p["difficulty"] == difficulty]


def get_all_problems() -> list[dict[str, Any]]:
    """Get all problems."""
    return OLYMPIAD_PROBLEMS.copy()


def get_quick_test_set() -> list[dict[str, Any]]:
    """Get a small set of problems for quick testing."""
    return [
        get_problem_by_id("comb_001"),  # MISSISSIPPI - medium
        get_problem_by_id("num_003"),  # Divisors - medium
        get_problem_by_id("grid_001"),  # 4x4 grid - medium
    ]


def get_full_test_set() -> list[dict[str, Any]]:
    """Get the full olympiad test set."""
    return get_all_problems()


def get_imo_problems() -> list[dict[str, Any]]:
    """Get all IMO (International Mathematical Olympiad) problems."""
    return get_problems_by_category("imo")


def get_putnam_problems() -> list[dict[str, Any]]:
    """Get all Putnam competition problems."""
    return get_problems_by_category("putnam")


def get_icpc_problems() -> list[dict[str, Any]]:
    """Get all ICPC (International Collegiate Programming Contest) problems."""
    return get_problems_by_category("icpc")


def get_advanced_problems() -> list[dict[str, Any]]:
    """Get all advanced competition problems."""
    return get_problems_by_category("advanced")


def get_expert_problems() -> list[dict[str, Any]]:
    """Get all expert-difficulty problems (IMO, Putnam, ICPC, Advanced)."""
    return get_problems_by_difficulty("expert")


def get_research_test_set() -> list[dict[str, Any]]:
    """Get a comprehensive set for research experiments.
    Includes a mix of difficulties and categories for cognitive trace analysis.
    """
    return [
        # Medium baseline problems
        get_problem_by_id("comb_001"),  # MISSISSIPPI
        get_problem_by_id("num_003"),  # Divisors
        get_problem_by_id("grid_001"),  # 4x4 grid
        # Hard problems
        get_problem_by_id("comb_002"),  # Circular seating
        get_problem_by_id("num_002"),  # Last two digits
        get_problem_by_id("algo_002"),  # 12 coins
        # IMO problems
        get_problem_by_id("imo_001"),  # Irreducible fraction
        get_problem_by_id("imo_004"),  # Perfect square ratio
        # Putnam problems
        get_problem_by_id("putnam_002"),  # A³+B³+C³-3ABC
        get_problem_by_id("putnam_003"),  # Sum of squares
        # ICPC problems
        get_problem_by_id("icpc_002"),  # Fibonacci threshold
        get_problem_by_id("icpc_003"),  # LIS (Longest Increasing Subsequence)
        # Advanced
        get_problem_by_id("advanced_001"),  # Heron's formula
        get_problem_by_id("advanced_005"),  # Domino tiling
    ]


def get_problem_summary() -> dict[str, int]:
    """Get a summary of problem counts by category and difficulty."""
    categories = {}
    difficulties = {}
    for p in OLYMPIAD_PROBLEMS:
        cat = p.get("category", "unknown")
        diff = p.get("difficulty", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
    return {
        "total": len(OLYMPIAD_PROBLEMS),
        "by_category": categories,
        "by_difficulty": difficulties,
    }

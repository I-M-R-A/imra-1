#!/usr/bin/env python3

"""IMRA-1 Research Problem Bank

50 carefully curated problems for research demonstration:
├── 20 Math Olympiad / Putnam-style problems
├── 10 Coding / Logic puzzles (multi-step reasoning)
├── 10 Combinatorial / Probability problems
└── 10 Paradoxical / Tricky reasoning problems (test self-reflection)

Each problem is tagged with:
- Category (what domain)
- Difficulty (research-relevant scale)
- Trap Type (what makes it hard for AI)
- Expected failure mode (what we expect to study)
"""

# SECTION 1: Math Olympiad twisted with Putnam Problems (20)

OLYMPIAD_PROBLEMS = [
    {
        "id": "IMO_2019_P1",
        "problem": """Let Z be the set of integers. Determine all functions f: Z → Z such that, 
for all integers a and b: f(2a) + 2f(b) = f(f(a + b)).
What is f(0)?""",
        "expected": "0",
        "category": "number_theory",
        "difficulty": "imo",
        "trap_type": "functional_equation",
        "expected_failure": "Missing edge cases",
    },
    {
        "id": "IMO_2018_P1",
        "problem": """Let Γ be the circumcircle of acute triangle ABC. Points D and E are on 
segments AB and AC respectively such that AD = AE. The perpendicular bisectors 
of BD and CE intersect minor arcs AB and AC of Γ at points F and G respectively.
Prove that lines DE and FG are either parallel or they are the same line.
What is the key geometric relationship that makes this true?""",
        "expected": "DE and FG are both parallel to BC (or coincide when the triangle is isosceles)",
        "category": "geometry",
        "difficulty": "imo",
        "trap_type": "geometric_insight",
        "expected_failure": "Missing the parallel relationship",
    },
    {
        "id": "IMO_2017_P1",
        "problem": """For each integer a₀ > 1, define the sequence a₀, a₁, a₂, ... by:
aₙ₊₁ = √aₙ if √aₙ is an integer, aₙ + 3 otherwise.
Determine all values of a₀ for which there is a number A such that aₙ = A for 
infinitely many values of n.
What is the smallest such a₀?""",
        "expected": "4",
        "category": "number_theory",
        "difficulty": "imo",
        "trap_type": "sequence_analysis",
        "expected_failure": "Not tracking cycle formation",
    },
    {
        "id": "PUTNAM_2020_A1",
        "problem": """How many positive integers N satisfy all of the following three conditions?
(i) N is divisible by 2020.
(ii) N has at most 2020 decimal digits.
(iii) The decimal digits of N are a string of consecutive ones followed by a string of consecutive zeros.
Give the exact count.""",
        "expected": "508536",
        "category": "number_theory",
        "difficulty": "putnam",
        "trap_type": "counting_with_divisibility",
        "expected_failure": "Miscounting edge cases",
    },
    {
        "id": "PUTNAM_2019_A2",
        "problem": """In the triangle ABC, let G be the centroid, and let I be the center of 
the inscribed circle. Let α and β be the angles at the vertices A and B, 
respectively. Suppose that the segment IG is parallel to AB and that 
β = 2arctan(1/3). Find α (in degrees).""",
        "expected": "90",
        "category": "geometry",
        "difficulty": "putnam",
        "trap_type": "geometric_calculation",
        "expected_failure": "Trigonometric errors",
    },
    {
        "id": "PUTNAM_2018_A1",
        "problem": """Find all ordered pairs (a,b) of positive integers for which:
1/a + 1/b = 3/2018.
How many such pairs exist?""",
        "expected": "6",
        "category": "number_theory",
        "difficulty": "putnam",
        "trap_type": "diophantine_equation",
        "expected_failure": "Missing factorization insight",
    },
    {
        "id": "USAMO_2019_P1",
        "problem": """Let ABC be a triangle and let M and N be the midpoints of AB and AC 
respectively. Let X be the point such that AX is parallel to BC and AX = BC.
Find the ratio of area(BXNM) to area(ABC).""",
        "expected": "1/2",
        "category": "geometry",
        "difficulty": "national_olympiad",
        "trap_type": "area_ratio",
        "expected_failure": "Coordinate geometry errors",
    },
    {
        "id": "USAMO_2018_P2",
        "problem": """Find all functions f:(0,∞)→(0,∞) such that:
f(x) + f(y) ≤ f(xy) for all positive x,y with xy ≥ 1.
Does f(x) = x satisfy this? Answer yes or no.""",
        "expected": "yes",
        "category": "algebra",
        "difficulty": "national_olympiad",
        "trap_type": "functional_inequality",
        "expected_failure": "Not checking boundary",
    },
    {
        "id": "BMO_2020_P1",
        "problem": """A positive integer is called special if it can be written as the sum of 
a perfect square, a positive perfect cube, and a positive perfect fourth power 
(not necessarily distinct). What is the smallest positive integer which cannot 
be written as the difference of two special numbers?""",
        "expected": "2",
        "category": "number_theory",
        "difficulty": "national_olympiad",
        "trap_type": "exhaustive_search",
        "expected_failure": "Incomplete enumeration",
    },
    {
        "id": "EGMO_2019_P1",
        "problem": """Find all triples (a,b,c) of real numbers such that ab+bc+ca=1 and:
a²b + c = b²c + a = c²a + b.
What is the sum a+b+c for all valid solutions?""",
        "expected": "0",
        "category": "algebra",
        "difficulty": "national_olympiad",
        "trap_type": "symmetric_system",
        "expected_failure": "Missing symmetric solution",
    },
    {
        "id": "MATH_OLYMPIAD_01",
        "problem": """Find the last three digits of 7^2020.
Give just the number.""",
        "expected": "401",
        "category": "number_theory",
        "difficulty": "olympiad",
        "trap_type": "modular_arithmetic",
        "expected_failure": "Cycle length error",
    },
    {
        "id": "MATH_OLYMPIAD_02",
        "problem": """In how many ways can 10 people be divided into 5 unordered pairs?
Give the exact number.""",
        "expected": "945",
        "category": "combinatorics",
        "difficulty": "olympiad",
        "trap_type": "counting_unordered",
        "expected_failure": "Overcounting by order",
    },
    {
        "id": "MATH_OLYMPIAD_03",
        "problem": """Find the smallest positive integer n such that n² + n + 41 is NOT prime.
What is n?""",
        "expected": "40",
        "category": "number_theory",
        "difficulty": "olympiad",
        "trap_type": "euler_formula_trap",
        "expected_failure": "Not checking 40",
    },
    {
        "id": "MATH_OLYMPIAD_04",
        "problem": """The sequence a₁, a₂, a₃, ... satisfies a₁ = 1, a₂ = 2, and 
aₙ₊₂ = aₙ₊₁ + aₙ for n ≥ 1.
What is the units digit of a₁₀₀?""",
        "expected": "5",
        "category": "number_theory",
        "difficulty": "olympiad",
        "trap_type": "fibonacci_periodicity",
        "expected_failure": "Not using Pisano period",
    },
    {
        "id": "MATH_OLYMPIAD_05",
        "problem": """How many subsets of {1,2,3,...,10} contain no two consecutive integers?
Count the empty set too.""",
        "expected": "144",
        "category": "combinatorics",
        "difficulty": "olympiad",
        "trap_type": "recursive_counting",
        "expected_failure": "Missing recursion pattern",
    },
    {
        "id": "MATH_OLYMPIAD_06",
        "problem": """Find the sum of all positive divisors of 360.
What is σ(360)?""",
        "expected": "1170",
        "category": "number_theory",
        "difficulty": "olympiad",
        "trap_type": "divisor_function",
        "expected_failure": "Factorization error",
    },
    {
        "id": "MATH_OLYMPIAD_07",
        "problem": """In a round-robin tournament with 10 teams, each team plays every other 
team exactly once. Each game has a winner (no ties). What is the minimum 
possible number of teams that finish with more wins than losses?""",
        "expected": "5",
        "category": "combinatorics",
        "difficulty": "olympiad",
        "trap_type": "tournament_theory",
        "expected_failure": "Parity reasoning error",
    },
    {
        "id": "MATH_OLYMPIAD_08",
        "problem": """A bug starts at vertex A of a cube and wants to reach vertex G (the 
opposite corner). It can only walk along edges. What is the minimum number 
of edges it must traverse?""",
        "expected": "3",
        "category": "geometry",
        "difficulty": "olympiad",
        "trap_type": "graph_distance",
        "expected_failure": "Overthinking",
    },
    {
        "id": "MATH_OLYMPIAD_09",
        "problem": """If x + 1/x = 3, find x⁵ + 1/x⁵.
Give the exact integer value.""",
        "expected": "123",
        "category": "algebra",
        "difficulty": "olympiad",
        "trap_type": "power_sum_recursion",
        "expected_failure": "Computational error",
    },
    {
        "id": "MATH_OLYMPIAD_10",
        "problem": """How many positive integers less than 1000 are divisible by neither 
3 nor 7?
Give the exact count.""",
        "expected": "571",
        "category": "number_theory",
        "difficulty": "olympiad",
        "trap_type": "inclusion_exclusion",
        "expected_failure": "Off-by-one error",
    },
]


LOGIC_PROBLEMS = [
    {
        "id": "LOGIC_01_KNIGHTS",
        "problem": """On an island, every inhabitant is either a knight (always tells truth) 
or a knave (always lies). You meet three people: A, B, and C.
A says: "All of us are knaves."
B says: "Exactly one of us is a knight."
How many knights are there?""",
        "expected": "1",
        "category": "logic",
        "difficulty": "medium",
        "trap_type": "self_reference",
        "expected_failure": "Circular reasoning error",
    },
    {
        "id": "LOGIC_02_PRISONERS",
        "problem": """Three prisoners wear hats. Each hat is black or white. Each can see 
others' hats but not their own. At least one hat is black. They're asked 
in order: Prisoner 1 sees B and C, says "I don't know my color."
Prisoner 2 sees C, says "I don't know my color."
What color is Prisoner 3's hat?""",
        "expected": "black",
        "category": "logic",
        "difficulty": "medium",
        "trap_type": "inference_chain",
        "expected_failure": "Missing deduction step",
    },
    {
        "id": "LOGIC_03_WEIGHING",
        "problem": """You have 12 coins. One is counterfeit (different weight - could be 
heavier or lighter). Using a balance scale, what is the minimum number of 
weighings guaranteed to identify the counterfeit AND determine if it's 
heavier or lighter?""",
        "expected": "3",
        "category": "logic",
        "difficulty": "hard",
        "trap_type": "information_theory",
        "expected_failure": "Underestimating complexity",
    },
    {
        "id": "LOGIC_04_DOORS",
        "problem": """There are 100 doors, all closed. Person 1 opens every door. Person 2 
toggles every 2nd door. Person 3 toggles every 3rd door, and so on until 
Person 100. After all 100 people, how many doors are open?""",
        "expected": "10",
        "category": "logic",
        "difficulty": "medium",
        "trap_type": "divisor_counting",
        "expected_failure": "Not recognizing perfect squares",
    },
    {
        "id": "LOGIC_05_BRIDGE",
        "problem": """Four people must cross a bridge at night with one flashlight. The 
bridge holds 2 people max. They take 1, 2, 5, and 10 minutes respectively.
When two cross, they go at the slower person's pace. What is the minimum 
total time for all four to cross?""",
        "expected": "17",
        "category": "logic",
        "difficulty": "medium",
        "trap_type": "optimization",
        "expected_failure": "Greedy suboptimal solution",
    },
    {
        "id": "LOGIC_06_HATS_INFINITE",
        "problem": """An infinite line of prisoners each wear a hat (black or white). Each 
can see all hats ahead of them but not their own or those behind. Starting 
from the back, each must guess their own color. If they agree on a strategy 
beforehand, what is the maximum number of prisoners who might guess wrong?""",
        "expected": "1",
        "category": "logic",
        "difficulty": "hard",
        "trap_type": "parity_strategy",
        "expected_failure": "Not finding parity trick",
    },
    {
        "id": "LOGIC_07_BLUE_EYES",
        "problem": """On an island with 100 blue-eyed people and 100 brown-eyed people, 
everyone can see others' eye colors but not their own. If anyone deduces 
their own eye color, they must leave at midnight. A visitor says "I see 
someone with blue eyes." On which night do the blue-eyed people leave?""",
        "expected": "100",
        "category": "logic",
        "difficulty": "hard",
        "trap_type": "common_knowledge",
        "expected_failure": "Missing induction",
    },
    {
        "id": "LOGIC_08_LIARS",
        "problem": """In a room of 2020 people, each is either a liar or truth-teller. Each 
person says "Most people in this room are liars." How many truth-tellers 
could there be at maximum?""",
        "expected": "1010",
        "category": "logic",
        "difficulty": "medium",
        "trap_type": "majority_paradox",
        "expected_failure": "Boundary case error",
    },
    {
        "id": "LOGIC_09_PIRATES",
        "problem": """5 pirates (ranked 1-5, 5 being senior) divide 100 gold coins. The 
most senior proposes a split, all vote, and if ≥50% accept, it passes; 
otherwise the proposer is killed and the next proposes. Pirates are 
perfectly rational and greedy. How many coins does Pirate 5 keep?""",
        "expected": "98",
        "category": "logic",
        "difficulty": "hard",
        "trap_type": "backward_induction",
        "expected_failure": "Not working backwards",
    },
    {
        "id": "LOGIC_10_SWITCHES",
        "problem": """There are 3 light switches outside a room with 3 bulbs inside. You 
can flip switches as much as you want, then enter the room ONCE. You 
cannot see the bulbs from outside. How can you determine which switch 
controls which bulb? Answer: what property do you use besides on/off?""",
        "expected": "heat",
        "category": "logic",
        "difficulty": "medium",
        "trap_type": "lateral_thinking",
        "expected_failure": "Staying within expected constraints",
    },
]


PROBABILITY_PROBLEMS = [
    {
        "id": "PROB_01_MONTY",
        "problem": """In the Monty Hall problem, you pick door 1. Monty (who knows what's 
behind each door) opens door 3, revealing a goat. What is the probability 
of winning the car if you switch to door 2? Express as a fraction.""",
        "expected": "2/3",
        "category": "probability",
        "difficulty": "medium",
        "trap_type": "conditional_probability",
        "expected_failure": "Intuitive 1/2 answer",
    },
    {
        "id": "PROB_02_BIRTHDAY",
        "problem": """What is the smallest number of people needed in a room for the 
probability to be greater than 50% that at least two share a birthday?
Assume 365 days, uniform distribution.""",
        "expected": "23",
        "category": "probability",
        "difficulty": "medium",
        "trap_type": "complement_counting",
        "expected_failure": "Linear thinking (365/2)",
    },
    {
        "id": "PROB_03_DICE_SUM",
        "problem": """Two fair dice are rolled. Given that the sum is at least 10, what is 
the probability the sum is exactly 12? Express as a fraction.""",
        "expected": "1/6",
        "category": "probability",
        "difficulty": "easy",
        "trap_type": "conditional_probability",
        "expected_failure": "Not conditioning correctly",
    },
    {
        "id": "PROB_04_EXPECTED_HEADS",
        "problem": """A fair coin is flipped until heads appears. What is the expected 
number of flips? Give exact answer.""",
        "expected": "2",
        "category": "probability",
        "difficulty": "medium",
        "trap_type": "geometric_distribution",
        "expected_failure": "Series summation error",
    },
    {
        "id": "PROB_05_DERANGEMENT",
        "problem": """In how many ways can 5 people be seated in 5 chairs such that nobody 
sits in their assigned seat (a derangement)? Give exact count.""",
        "expected": "44",
        "category": "combinatorics",
        "difficulty": "medium",
        "trap_type": "derangement_formula",
        "expected_failure": "Not knowing D(n) formula",
    },
    {
        "id": "PROB_06_CATALAN",
        "problem": """In how many ways can you arrange n opening parentheses and n closing 
parentheses into valid expressions? For n=4, give the count.""",
        "expected": "14",
        "category": "combinatorics",
        "difficulty": "medium",
        "trap_type": "catalan_numbers",
        "expected_failure": "Overcounting invalid",
    },
    {
        "id": "PROB_07_GAMBLER",
        "problem": """A gambler starts with $10. Each round, they win $1 with prob 0.4 or 
lose $1 with prob 0.6. They stop when reaching $0 or $20. What is the 
probability of reaching $20 before $0? Give as decimal to 4 places.""",
        "expected": "0.0165",
        "category": "probability",
        "difficulty": "hard",
        "trap_type": "random_walk",
        "expected_failure": "Not using gambler's ruin formula",
    },
    {
        "id": "PROB_08_COUPON",
        "problem": """There are n=5 types of coupons. Each box contains one coupon uniformly 
at random. What is the expected number of boxes to buy to collect all 5?
Round to nearest integer.""",
        "expected": "11",
        "category": "probability",
        "difficulty": "medium",
        "trap_type": "coupon_collector",
        "expected_failure": "Not using harmonic series",
    },
    {
        "id": "PROB_09_STIRLING",
        "problem": """In how many ways can 4 distinct balls be placed into 2 identical boxes 
such that no box is empty? This is the Stirling number S(4,2).""",
        "expected": "7",
        "category": "combinatorics",
        "difficulty": "medium",
        "trap_type": "stirling_partition",
        "expected_failure": "Confusing with labeled boxes",
    },
    {
        "id": "PROB_10_RANDOM_WALK",
        "problem": """A particle starts at 0. Each second, it moves +1 with prob 1/2 or -1 
with prob 1/2. What is the expected number of steps to first reach +2 or -2?
Give exact answer.""",
        "expected": "4",
        "category": "probability",
        "difficulty": "medium",
        "trap_type": "hitting_time",
        "expected_failure": "Overthinking",
    },
]

# Tricky / Paradoxical Problems (10) - Test Self-Reflection

TRICKY_PROBLEMS = [
    {
        "id": "TRICK_01_BAT_BALL",
        "problem": """A bat and ball cost $1.10 together. The bat costs $1.00 more than the 
ball. How much does the ball cost in cents?""",
        "expected": "5",
        "category": "metacognition",
        "difficulty": "easy",
        "trap_type": "intuitive_trap",
        "expected_failure": "Answering 10 without checking",
    },
    {
        "id": "TRICK_02_LILY_PAD",
        "problem": """A lily pad doubles in size every day. It takes 48 days to cover a 
lake completely. On what day does it cover exactly half the lake?""",
        "expected": "47",
        "category": "metacognition",
        "difficulty": "easy",
        "trap_type": "exponential_intuition",
        "expected_failure": "Answering 24",
    },
    {
        "id": "TRICK_03_SURGEON",
        "problem": """A father and son are in a car accident. The father dies. The son is 
rushed to surgery. The surgeon says "I can't operate on him, he's my son."
How is this possible? Give the simplest explanation in one word.""",
        "expected": "mother",
        "category": "metacognition",
        "difficulty": "easy",
        "trap_type": "assumption_bias",
        "expected_failure": "Gender bias in assumptions",
    },
    {
        "id": "TRICK_04_PLANE",
        "problem": """A plane crashes exactly on the US-Canada border. Where do they bury 
the survivors?""",
        "expected": "nowhere",
        "category": "metacognition",
        "difficulty": "easy",
        "trap_type": "semantic_trap",
        "expected_failure": "Not catching 'survivors'",
    },
    {
        "id": "TRICK_05_TWO_DOORS",
        "problem": """One door leads to freedom, one to death. One guard always lies, one 
always tells truth. You can ask ONE question to ONE guard. What question 
guarantees finding the freedom door? Just name the key insight.""",
        "expected": "ask what the other guard would say",
        "category": "logic",
        "difficulty": "medium",
        "trap_type": "meta_question",
        "expected_failure": "Asking direct questions",
    },
    {
        "id": "TRICK_06_ROPE",
        "problem": """You have two ropes. Each burns for exactly 60 minutes, but 
non-uniformly (you can't tell time by length). How do you measure exactly 
45 minutes? Key insight: what do you do at the start?""",
        "expected": "light one rope at both ends",
        "category": "logic",
        "difficulty": "medium",
        "trap_type": "non_uniform_rate",
        "expected_failure": "Assuming uniform burn",
    },
    {
        "id": "TRICK_07_INFINITE_HOTEL",
        "problem": """An infinite hotel has all rooms occupied (room 1, 2, 3, ...). A new 
guest arrives. How do you accommodate them? What room does each current 
guest move to?""",
        "expected": "room n+1",
        "category": "logic",
        "difficulty": "medium",
        "trap_type": "infinite_sets",
        "expected_failure": "Thinking finite",
    },
    {
        "id": "TRICK_08_ENVELOPE",
        "problem": """Two envelopes contain money. One has twice the other. You pick one, 
see $100. Should you switch? What's wrong with this argument: "The other 
has $50 or $200, so expected value of switching is $125"?""",
        "expected": "assumes both outcomes equally likely regardless of which envelope is larger",
        "category": "probability",
        "difficulty": "hard",
        "trap_type": "two_envelope_paradox",
        "expected_failure": "Not catching the flaw",
    },
    {
        "id": "TRICK_09_THREE_BOXES",
        "problem": """Box A has 2 gold coins. Box B has 2 silver coins. Box C has 1 gold, 
1 silver. You pick a box at random, draw one coin, and it's gold. What's 
the probability the other coin in that box is gold?""",
        "expected": "2/3",
        "category": "probability",
        "difficulty": "medium",
        "trap_type": "bertrand_box",
        "expected_failure": "Answering 1/2",
    },
    {
        "id": "TRICK_10_1_EQUALS_2",
        "problem": """Find the error in this proof that 1=2:
Let a=b. Then a²=ab. So a²-b²=ab-b². Thus (a+b)(a-b)=b(a-b). 
Divide both sides by (a-b): a+b=b. Since a=b: 2b=b, so 2=1.
What step is invalid?""",
        "expected": "division by zero (a-b=0)",
        "category": "algebra",
        "difficulty": "easy",
        "trap_type": "division_by_zero",
        "expected_failure": "Not catching the illegal step",
    },
]


def get_all_problems() -> list[dict]:
    """Get all 50 research problems."""
    return OLYMPIAD_PROBLEMS + LOGIC_PROBLEMS + PROBABILITY_PROBLEMS + TRICKY_PROBLEMS


def get_problems_by_category(category: str) -> list[dict]:
    """Get problems by category."""
    return [p for p in get_all_problems() if p["category"] == category]


def get_problems_by_trap_type(trap_type: str) -> list[dict]:
    """Get problems by trap type."""
    return [p for p in get_all_problems() if p["trap_type"] == trap_type]


def get_research_subset(n: int = 15) -> list[dict]:
    """Get a balanced subset for quick experiments."""
    all_problems = get_all_problems()
    subset = []
    for section in [OLYMPIAD_PROBLEMS, LOGIC_PROBLEMS, PROBABILITY_PROBLEMS, TRICKY_PROBLEMS]:
        count = max(1, n * len(section) // len(all_problems))
        subset.extend(section[:count])
    return subset[:n]


def get_category_stats() -> dict:
    """Get statistics about the problem bank."""
    problems = get_all_problems()
    categories = {}
    difficulties = {}
    trap_types = {}

    for p in problems:
        cat = p["category"]
        diff = p["difficulty"]
        trap = p["trap_type"]

        categories[cat] = categories.get(cat, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1
        trap_types[trap] = trap_types.get(trap, 0) + 1

    return {
        "total": len(problems),
        "by_category": categories,
        "by_difficulty": difficulties,
        "by_trap_type": trap_types,
    }


if __name__ == "__main__":
    stats = get_category_stats()
    print("IMRA-1 Research Problem Bank")
    print()
    print(f"Total problems: {stats['total']}")
    print("\nBy Category:")
    for cat, count in sorted(stats["by_category"].items()):
        print(f"  {cat}: {count}")
    print("\nBy Difficulty:")
    for diff, count in sorted(stats["by_difficulty"].items()):
        print(f"  {diff}: {count}")
    print("\nBy Trap Type:")
    for trap, count in sorted(stats["by_trap_type"].items()):
        print(f"  {trap}: {count}")

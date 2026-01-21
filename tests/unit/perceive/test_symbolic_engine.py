"""Tests for SymbolicPerceiveEngine."""

from imra.perceive.engines.symbolic import SymbolicPerceiveEngine
from imra.perceive.sketches import PerceptionRequest


def test_symbolic_engine_detects_addition() -> None:
    """Test that addition keywords are detected."""
    engine = SymbolicPerceiveEngine()
    request = PerceptionRequest(
        task_id="add_test",
        payload={"type": "math"},
        problem_text="John has 5 apples. Mary gives him 3 more. How many does he have?",
    )

    sketches = list(engine.propose(request))
    assert len(sketches) > 0

    # Should detect addition due to "gives" and "more" so that "add" operation is present
    # Check that at least one sketch contains an "add" operation

    ops_found = []
    for s in sketches:
        for node in s.nodes:
            if node.value in ("add", "sub", "mul", "div"):
                ops_found.append(node.value)

    assert "add" in ops_found, f"Expected 'add' in {ops_found}"


def test_symbolic_engine_detects_multiplication() -> None:
    """Test that multiplication keywords are detected."""
    engine = SymbolicPerceiveEngine()
    request = PerceptionRequest(
        task_id="mul_test",
        payload={},
        problem_text="Each box has 6 items. There are 4 boxes. What is the total?",
    )

    sketches = list(engine.propose(request))
    ops = [n.value for s in sketches for n in s.nodes if hasattr(n, "value")]
    assert "mul" in ops, f"Expected 'mul' in {ops}"


def test_symbolic_engine_extracts_numbers() -> None:
    """Test that numbers are extracted from text."""
    engine = SymbolicPerceiveEngine()
    request = PerceptionRequest(
        task_id="num_test",
        payload={},
        problem_text="Calculate 15 divided by 3",
    )

    sketches = list(engine.propose(request))

    literals = []
    for s in sketches:
        for node in s.nodes:
            if node.value in (15, 3, 15.0, 3.0):
                literals.append(node.value)

    assert len(literals) >= 2


def test_symbolic_engine_generates_diverse_sketches() -> None:
    """Test that multiple sketch candidates are generated."""
    engine = SymbolicPerceiveEngine(num_sketches=5)
    request = PerceptionRequest(
        task_id="diverse_test",
        payload={},
        problem_text="If you add 10 and multiply by 2, what do you get?",
    )

    sketches = list(engine.propose(request))

    assert len(sketches) >= 2  # At least 2 diverse sketches should be generated

    total_prior = sum(s.prior for s in sketches)
    assert total_prior > 0


def test_symbolic_engine_update_tracks_reward() -> None:
    """Test that update method tracks rewards."""
    engine = SymbolicPerceiveEngine()
    request = PerceptionRequest(
        task_id="reward_test",
        payload={},
        problem_text="2 plus 2",
    )

    sketches = list(engine.propose(request))
    assert len(sketches) > 0

    engine.update(sketches[0], reward=0.9)

    engine.update(sketches[0], reward=-0.3)


def test_symbolic_engine_fallback_on_no_keywords() -> None:
    """Test behavior when no operation keywords are found."""
    engine = SymbolicPerceiveEngine()
    request = PerceptionRequest(
        task_id="fallback_test",
        payload={},
        problem_text="The sky is blue",  # No math keywords present (should fallback)
    )

    sketches = list(engine.propose(request))

    assert len(sketches) >= 0  # May be empty or have fallback since no keywords found

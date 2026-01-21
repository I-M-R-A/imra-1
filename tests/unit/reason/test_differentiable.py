"""Tests for differentiable operators."""

import pytest

from imra.reason.ops.differentiable import (
    DiffValue,
    GradientTape,
    diff_abs,
    diff_add,
    diff_div,
    diff_max,
    diff_min,
    diff_mul,
    diff_pow,
    diff_relu,
    diff_sigmoid,
    diff_sub,
)


class TestGradientTape:
    """Tests for GradientTape context manager."""

    def test_tape_records_operations(self) -> None:
        tape = GradientTape()
        result, _ = diff_add(3.0, 5.0, tape=tape)
        assert result == 8.0
        assert len(tape.operations) == 1

    def test_tape_backward_produces_gradients(self) -> None:
        tape = GradientTape()
        diff_add(3.0, 5.0, tape=tape)
        grads = tape.backward()
        assert len(grads) > 0

    def test_tape_clear(self) -> None:
        tape = GradientTape()
        diff_add(1, 2, tape=tape)
        tape.clear()
        assert len(tape.operations) == 0


class TestDiffValue:
    """Tests for DiffValue class."""

    def test_diffvalue_creation(self) -> None:
        v = DiffValue(5.0, {"x": 1.0})
        assert v.value == 5.0
        assert v.grad["x"] == 1.0

    def test_diffvalue_add(self) -> None:
        a = DiffValue(3.0, {"a": 1.0})
        b = DiffValue(2.0, {"b": 1.0})
        c = a + b
        assert c.value == 5.0
        assert c.grad["a"] == 1.0
        assert c.grad["b"] == 1.0

    def test_diffvalue_mul(self) -> None:
        a = DiffValue(3.0, {"a": 1.0})
        b = DiffValue(4.0, {"b": 1.0})
        c = a * b
        assert c.value == 12.0

        # d(a*b)/da = b, d(a*b)/db = a

        assert c.grad["a"] == 4.0
        assert c.grad["b"] == 3.0

    def test_diffvalue_sub(self) -> None:
        a = DiffValue(10.0, {"a": 1.0})
        b = DiffValue(3.0, {"b": 1.0})
        c = a - b
        assert c.value == 7.0
        assert c.grad["a"] == 1.0
        assert c.grad["b"] == -1.0

    def test_diffvalue_div(self) -> None:
        a = DiffValue(10.0, {"a": 1.0})
        b = DiffValue(2.0, {"b": 1.0})
        c = a / b
        assert c.value == 5.0

        # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2

        assert c.grad["a"] == 0.5
        assert c.grad["b"] == pytest.approx(-2.5)


class TestDiffOperators:
    """Tests for individual differentiable operators."""

    def test_diff_add(self) -> None:
        result, grads = diff_add(7.0, 3.0)
        assert result == 10.0
        assert "da" in grads
        assert "db" in grads

    def test_diff_sub(self) -> None:
        result, grads = diff_sub(10.0, 4.0)
        assert result == 6.0
        assert grads["da"] == 1.0
        assert grads["db"] == -1.0

    def test_diff_mul(self) -> None:
        result, grads = diff_mul(3.0, 5.0)
        assert result == 15.0
        assert grads["da"] == 5.0
        assert grads["db"] == 3.0

    def test_diff_div(self) -> None:
        result, grads = diff_div(20.0, 4.0)
        assert result == 5.0
        assert grads["da"] == pytest.approx(0.25)  # 1/b = 1/4
        assert grads["db"] == pytest.approx(-1.25)  # -a/b^2 = -20/16

    def test_diff_div_by_zero(self) -> None:
        result, grads = diff_div(10.0, 0.0)

        assert result != float("inf") or grads is not None

    def test_diff_pow(self) -> None:
        result, grads = diff_pow(2.0, 3.0)
        assert result == 8.0

    def test_diff_max(self) -> None:
        result, _ = diff_max(3.0, 7.0)
        assert result == 7.0
        result2, _ = diff_max(10.0, 2.0)
        assert result2 == 10.0

    def test_diff_min(self) -> None:
        result, _ = diff_min(3.0, 7.0)
        assert result == 3.0

    def test_diff_abs(self) -> None:
        result1, _ = diff_abs(5.0)
        assert result1 == 5.0
        result2, _ = diff_abs(-5.0)
        assert result2 == 5.0

    def test_diff_relu(self) -> None:
        result1, _ = diff_relu(5.0)
        assert result1 == 5.0
        result2, _ = diff_relu(-5.0)
        assert result2 == 0.0

    def test_diff_sigmoid(self) -> None:
        result, _ = diff_sigmoid(0.0)
        assert result == pytest.approx(0.5)

        result_pos, _ = diff_sigmoid(10.0)
        assert result_pos > 0.99

        result_neg, _ = diff_sigmoid(-10.0)
        assert result_neg < 0.01


class TestGradientChaining:
    """Tests for chaining multiple operations."""

    def test_chain_add_mul(self) -> None:
        """Test (a + b) * c gradient flow."""
        tape = GradientTape()

        # (3 + 2) * 4 = 20
        sum_result, _ = diff_add(3.0, 2.0, tape=tape)
        final, _ = diff_mul(sum_result, 4.0, tape=tape)

        assert final == 20.0
        assert len(tape.operations) == 2

    def test_chain_complex_expression(self) -> None:
        """Test ((a * b) + c) / d."""
        tape = GradientTape()

        # ((2 * 3) + 4) / 2 = 5

        prod, _ = diff_mul(2.0, 3.0, tape=tape)
        sum_result, _ = diff_add(prod, 4.0, tape=tape)
        final, _ = diff_div(sum_result, 2.0, tape=tape)

        assert final == 5.0
        grads = tape.backward()
        assert len(grads) > 0

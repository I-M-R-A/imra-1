"""Differentiable arithmetic operators with gradient tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GradientTape:
    """Records operations for backward pass."""

    operations: list[tuple[str, Any, Any, Any]] = field(default_factory=list)

    def record(self, op: str, inputs: Any, output: Any, local_grad: Any) -> None:
        self.operations.append((op, inputs, output, local_grad))

    def clear(self) -> None:
        """Clear all recorded operations."""
        self.operations.clear()

    def backward(self, seed: float = 1.0) -> dict[str, float]:
        """Compute gradients w.r.t. inputs."""
        grads: dict[str, float] = {}
        grad = seed

        for op, inputs, _output, local_grad in reversed(self.operations):
            if isinstance(local_grad, tuple):
                # Binary op: distribute gradient
                for i, (_inp, lg) in enumerate(zip(inputs, local_grad, strict=False)):
                    key = f"{op}_input_{i}"
                    grads[key] = grad * lg
            else:
                grads[f"{op}_input"] = grad * local_grad
            grad *= sum(local_grad) if isinstance(local_grad, tuple) else local_grad

        return grads


@dataclass
class DiffValue:
    """A value with gradient tracking capability using named gradients."""

    value: float
    grad: dict[str, float] = field(default_factory=dict)
    tape: GradientTape | None = None
    name: str = ""

    def __add__(self, other: DiffValue | float) -> DiffValue:
        if isinstance(other, DiffValue):
            other_val = other.value
            # Combine gradients: d(a+b)/da = 1, d(a+b)/db = 1
            combined_grad = {**self.grad, **other.grad}
        else:
            other_val = other
            combined_grad = dict(self.grad)
        result = DiffValue(self.value + other_val, grad=combined_grad, tape=self.tape)
        if self.tape:
            self.tape.record("add", (self.value, other_val), result.value, (1.0, 1.0))
        return result

    def __sub__(self, other: DiffValue | float) -> DiffValue:
        if isinstance(other, DiffValue):
            other_val = other.value
            # d(a-b)/da = 1, d(a-b)/db = -1
            combined_grad = dict(self.grad)
            for k, v in other.grad.items():
                combined_grad[k] = combined_grad.get(k, 0) - v
        else:
            other_val = other
            combined_grad = dict(self.grad)
        result = DiffValue(self.value - other_val, grad=combined_grad, tape=self.tape)
        if self.tape:
            self.tape.record("sub", (self.value, other_val), result.value, (1.0, -1.0))
        return result

    def __mul__(self, other: DiffValue | float) -> DiffValue:
        if isinstance(other, DiffValue):
            other_val = other.value
            # d(a*b)/da = b, d(a*b)/db = a
            combined_grad = {k: v * other_val for k, v in self.grad.items()}
            for k, v in other.grad.items():
                combined_grad[k] = combined_grad.get(k, 0) + v * self.value
        else:
            other_val = other
            combined_grad = {k: v * other_val for k, v in self.grad.items()}
        result = DiffValue(self.value * other_val, grad=combined_grad, tape=self.tape)
        if self.tape:
            self.tape.record("mul", (self.value, other_val), result.value, (other_val, self.value))
        return result

    def __truediv__(self, other: DiffValue | float) -> DiffValue:
        if isinstance(other, DiffValue):
            other_val = other.value if abs(other.value) >= 1e-10 else 1e-10
            # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
            combined_grad = {k: v / other_val for k, v in self.grad.items()}
            for k, v in other.grad.items():
                combined_grad[k] = combined_grad.get(k, 0) + v * (-self.value / (other_val**2))
        else:
            other_val = other if abs(other) >= 1e-10 else 1e-10
            combined_grad = {k: v / other_val for k, v in self.grad.items()}
        result = DiffValue(self.value / other_val, grad=combined_grad, tape=self.tape)
        if self.tape:
            self.tape.record(
                "div",
                (self.value, other_val),
                result.value,
                (1.0 / other_val, -self.value / (other_val**2)),
            )
        return result

    def __neg__(self) -> DiffValue:
        neg_grad = {k: -v for k, v in self.grad.items()}
        result = DiffValue(-self.value, grad=neg_grad, tape=self.tape)
        if self.tape:
            self.tape.record("neg", (self.value,), result.value, (-1.0,))
        return result

    def __repr__(self) -> str:
        return f"DiffValue({self.value:.4f}, grad={self.grad})"


def diff_add(a: float, b: float, tape: GradientTape | None = None) -> tuple[float, dict]:
    """Differentiable addition."""
    result = a + b
    grad_info = {"da": 1.0, "db": 1.0}
    if tape:
        tape.record("add", (a, b), result, (1.0, 1.0))
    return result, grad_info


def diff_sub(a: float, b: float, tape: GradientTape | None = None) -> tuple[float, dict]:
    """Differentiable subtraction."""
    result = a - b
    grad_info = {"da": 1.0, "db": -1.0}
    if tape:
        tape.record("sub", (a, b), result, (1.0, -1.0))
    return result, grad_info


def diff_mul(a: float, b: float, tape: GradientTape | None = None) -> tuple[float, dict]:
    """Differentiable multiplication."""
    result = a * b
    grad_info = {"da": b, "db": a}
    if tape:
        tape.record("mul", (a, b), result, (b, a))
    return result, grad_info


def diff_div(a: float, b: float, tape: GradientTape | None = None) -> tuple[float, dict]:
    """Differentiable division with safe denominator."""
    if abs(b) < 1e-10:
        b = 1e-10
    result = a / b
    grad_info = {"da": 1.0 / b, "db": -a / (b**2)}
    if tape:
        tape.record("div", (a, b), result, (1.0 / b, -a / (b**2)))
    return result, grad_info


def diff_pow(a: float, b: float, tape: GradientTape | None = None) -> tuple[float, dict]:
    """Differentiable power."""
    result = a**b
    grad_a = b * (a ** (b - 1)) if a > 0 else 0.0
    grad_b = (a**b) * math.log(a) if a > 0 else 0.0
    grad_info = {"da": grad_a, "db": grad_b}
    if tape:
        tape.record("pow", (a, b), result, (grad_a, grad_b))
    return result, grad_info


def diff_max(a: float, b: float, tape: GradientTape | None = None) -> tuple[float, dict]:
    """Differentiable max (subgradient)."""
    result = max(a, b)
    # Subgradient: 1 for the larger, 0 for smaller
    grad_a = 1.0 if a >= b else 0.0
    grad_b = 1.0 if b > a else 0.0
    grad_info = {"da": grad_a, "db": grad_b}
    if tape:
        tape.record("max", (a, b), result, (grad_a, grad_b))
    return result, grad_info


def diff_min(a: float, b: float, tape: GradientTape | None = None) -> tuple[float, dict]:
    """Differentiable min (subgradient)."""
    result = min(a, b)
    grad_a = 1.0 if a <= b else 0.0
    grad_b = 1.0 if b < a else 0.0
    grad_info = {"da": grad_a, "db": grad_b}
    if tape:
        tape.record("min", (a, b), result, (grad_a, grad_b))
    return result, grad_info


def diff_abs(a: float, tape: GradientTape | None = None) -> tuple[float, dict]:
    """Differentiable absolute value (subgradient)."""
    result = abs(a)
    grad_a = 1.0 if a >= 0 else -1.0
    grad_info = {"da": grad_a}
    if tape:
        tape.record("abs", (a,), result, (grad_a,))
    return result, grad_info


def diff_relu(a: float, tape: GradientTape | None = None) -> tuple[float, dict]:
    """Differentiable ReLU."""
    result = max(0.0, a)
    grad_a = 1.0 if a > 0 else 0.0
    grad_info = {"da": grad_a}
    if tape:
        tape.record("relu", (a,), result, (grad_a,))
    return result, grad_info


def diff_sigmoid(a: float, tape: GradientTape | None = None) -> tuple[float, dict]:
    """Differentiable sigmoid."""
    result = 1.0 / (1.0 + math.exp(-a))
    grad_a = result * (1.0 - result)
    grad_info = {"da": grad_a}
    if tape:
        tape.record("sigmoid", (a,), result, (grad_a,))
    return result, grad_info

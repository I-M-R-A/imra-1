"""Operator registry with differentiable support."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from imra.reason.ops import differentiable as diff_ops
from imra.reason.ops.differentiable import GradientTape


@dataclass
class Operator:
    """An operator with metadata and gradient support."""

    name: str
    arity: int
    fn: Callable[..., tuple[float, dict]]
    description: str = ""
    differentiable: bool = True


@dataclass
class OperatorRegistry:
    """Registry of differentiable operators."""

    operators: dict[str, Operator] = field(default_factory=dict)

    def register(self, op: Operator) -> None:
        self.operators[op.name] = op

    def lookup(self, name: str) -> Operator:
        if name not in self.operators:
            raise KeyError(f"Operator '{name}' not registered")
        return self.operators[name]

    def execute(
        self,
        name: str,
        args: list[float],
        tape: GradientTape | None = None,
    ) -> tuple[float, dict]:
        """Execute operator with arguments and optional gradient tape."""
        op = self.lookup(name)
        if len(args) != op.arity:
            raise ValueError(f"Operator '{name}' expects {op.arity} args, got {len(args)}")
        return op.fn(*args, tape=tape)

    def list_operators(self) -> list[str]:
        return list(self.operators.keys())

    @classmethod
    def default(cls) -> OperatorRegistry:
        """Build registry with all standard differentiable operators."""
        registry = cls()

        # Arithmetic
        registry.register(
            Operator(
                name="add",
                arity=2,
                fn=diff_ops.diff_add,
                description="Addition: a + b",
            )
        )
        registry.register(
            Operator(
                name="sub",
                arity=2,
                fn=diff_ops.diff_sub,
                description="Subtraction: a - b",
            )
        )
        registry.register(
            Operator(
                name="mul",
                arity=2,
                fn=diff_ops.diff_mul,
                description="Multiplication: a * b",
            )
        )
        registry.register(
            Operator(
                name="div",
                arity=2,
                fn=diff_ops.diff_div,
                description="Division: a / b",
            )
        )
        registry.register(
            Operator(
                name="pow",
                arity=2,
                fn=diff_ops.diff_pow,
                description="Power: a ^ b",
            )
        )

        # Comparison / selection
        registry.register(
            Operator(
                name="max",
                arity=2,
                fn=diff_ops.diff_max,
                description="Maximum: max(a, b)",
            )
        )
        registry.register(
            Operator(
                name="min",
                arity=2,
                fn=diff_ops.diff_min,
                description="Minimum: min(a, b)",
            )
        )

        # Unary
        registry.register(
            Operator(
                name="abs",
                arity=1,
                fn=diff_ops.diff_abs,
                description="Absolute value: |a|",
            )
        )
        registry.register(
            Operator(
                name="relu",
                arity=1,
                fn=diff_ops.diff_relu,
                description="ReLU: max(0, a)",
            )
        )
        registry.register(
            Operator(
                name="sigmoid",
                arity=1,
                fn=diff_ops.diff_sigmoid,
                description="Sigmoid: 1 / (1 + exp(-a))",
            )
        )

        return registry

"""Symbolic PERCEIVE engine for arithmetic and logic tasks."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable

from imra.perceive.engines.base import PerceiveEngine
from imra.perceive.sketches import NodeType, PerceptionRequest, ProgramSketch, SketchNode


class SymbolicPerceiveEngine(PerceiveEngine):
    """
    A rule-based intuition engine that parses word problems
    and proposes candidate program sketches.
    """

    ADDITION_KEYWORDS = {
        "add",
        "adds",
        "added",
        "adding",
        "plus",
        "sum",
        "total",
        "combined",
        "together",
        "more",
        "increase",
        "increased",
        "gain",
        "gained",
    }
    SUBTRACTION_KEYWORDS = {
        "subtract",
        "subtracts",
        "subtracted",
        "subtracting",
        "minus",
        "difference",
        "less",
        "fewer",
        "remove",
        "removes",
        "removed",
        "removing",
        "left",
        "remain",
        "remains",
        "remaining",
        "decrease",
        "decreased",
        "lose",
        "loses",
        "lost",
        "take",
        "takes",
        "took",
        "taken",
        "away",
        "give",
        "gives",
        "gave",
        "given",
    }

    MULTIPLICATION_KEYWORDS = {
        "multiply",
        "multiplies",
        "multiplied",
        "multiplying",
        "times",
        "product",
        "each",
        "per",
        "every",
        "double",
        "triple",
        "twice",
    }

    DIVISION_KEYWORDS = {
        "divide",
        "divides",
        "divided",
        "dividing",
        "split",
        "quotient",
        "shared",
        "ratio",
        "per",
        "half",
        "quarter",
        "portion",
    }

    def __init__(self, num_sketches: int = 3, temperature: float = 0.3) -> None:
        self.num_sketches = num_sketches
        self.temperature = temperature
        self._history: list[tuple[ProgramSketch, float]] = []

    def propose(self, request: PerceptionRequest) -> Iterable[ProgramSketch]:
        """Generate candidate sketches from problem text."""
        text = request.problem_text or request.payload.get("text", "")
        variables = request.variables or request.payload.get("variables", {})

        if not text:
            yield self._fallback_sketch(request.task_id)
            return

        numbers = self._extract_numbers(text)

        ops = self._detect_operations(text.lower())

        for i in range(self.num_sketches):
            sketch = self._build_sketch(
                task_id=request.task_id,
                sketch_idx=i,
                numbers=numbers,
                variables=variables,
                operations=ops,
            )
            yield sketch

    def update(self, sketch: ProgramSketch, reward: float) -> None:
        """Store feedback for future prior adjustment."""
        self._history.append((sketch, reward))

        # Could train a model here; for now just track history, due to time constraints.

    def _extract_numbers(self, text: str) -> list[float]:
        """Pull numeric values from problem text."""
        pattern = r"-?\d+\.?\d*"
        matches = re.findall(pattern, text)
        return [float(m) for m in matches]

    def _detect_operations(self, text: str) -> list[str]:
        """Identify likely operations from keywords."""
        ops = []
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        words = set(clean_text.split())
        if words & self.ADDITION_KEYWORDS:
            ops.append("add")
        if words & self.SUBTRACTION_KEYWORDS:
            ops.append("sub")
        if words & self.MULTIPLICATION_KEYWORDS:
            ops.append("mul")
        if words & self.DIVISION_KEYWORDS:
            ops.append("div")
        if not ops:
            ops.append("add")

        return ops

    def _build_sketch(
        self,
        task_id: str,
        sketch_idx: int,
        numbers: list[float],
        variables: dict,
        operations: list[str],
    ) -> ProgramSketch:
        """Construct a program sketch from extracted info."""
        nodes: list[SketchNode] = []
        node_counter = 0

        def make_node(ntype: NodeType, value: object, children: list[str] | None = None) -> str:
            nonlocal node_counter
            nid = f"n{node_counter}"
            node_counter += 1
            nodes.append(
                SketchNode(
                    node_id=nid,
                    node_type=ntype,
                    value=value,
                    children=children or [],
                )
            )
            return nid

        num_ids = [make_node(NodeType.LITERAL, n) for n in numbers]

        var_ids = [make_node(NodeType.VARIABLE, k) for k in variables.keys()]

        all_operands = num_ids + var_ids
        if len(all_operands) < 2:
            all_operands = [make_node(NodeType.LITERAL, 0), make_node(NodeType.LITERAL, 0)]

        op = operations[sketch_idx % len(operations)] if operations else "add"

        if sketch_idx == 0:
            root_id = self._chain_ops(op, all_operands, make_node)
        elif sketch_idx == 1 and len(all_operands) > 2:
            mid = len(all_operands) // 2
            left = self._chain_ops(op, all_operands[:mid], make_node)
            right = self._chain_ops(op, all_operands[mid:], make_node)
            root_id = make_node(NodeType.OPERATOR, op, [left, right])
        else:
            shuffled = all_operands.copy()
            random.shuffle(shuffled)
            root_id = self._chain_ops(op, shuffled, make_node)

        base_prior = 0.6 - 0.1 * sketch_idx
        prior = max(0.1, min(0.9, base_prior + self.temperature * (random.random() - 0.5)))

        return ProgramSketch(
            sketch_id=f"{task_id}-sketch-{sketch_idx}",
            structure={"ops": [op], "operand_count": len(all_operands)},
            prior=prior,
            nodes=nodes,
            root_node_id=root_id,
            alternative_count=self.num_sketches,
        )

    def _chain_ops(
        self, op: str, operand_ids: list[str], make_node: Callable[..., str]
    ) -> str:
        """Chain binary operations left-to-right."""
        if len(operand_ids) == 1:
            return operand_ids[0]

        result = operand_ids[0]
        for oid in operand_ids[1:]:
            result = make_node(NodeType.OPERATOR, op, [result, oid])
        return result

    def _fallback_sketch(self, task_id: str) -> ProgramSketch:
        """Return a minimal sketch when input is empty."""
        return ProgramSketch(
            sketch_id=f"{task_id}-fallback",
            structure={"ops": []},
            prior=0.1,
            nodes=[SketchNode(node_id="n0", node_type=NodeType.LITERAL, value=0)],
            root_node_id="n0",
        )

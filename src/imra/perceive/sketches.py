"""Data structures for perception inputs and sketch outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    """Types of nodes in a program sketch."""

    LITERAL = "literal"
    VARIABLE = "variable"
    OPERATOR = "operator"
    BRANCH = "branch"
    LOOP = "loop"
    CALL = "call"


@dataclass(slots=True)
class SketchNode:
    """Single node in a program sketch graph."""

    node_id: str
    node_type: NodeType
    value: Any = None
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "value": self.value,
            "children": self.children,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PerceptionRequest:
    """Input to the PERCEIVE module."""

    task_id: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    problem_text: str | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    expected_type: str | None = None  # "numeric", "boolean", "code", etc.


@dataclass(slots=True)
class ProgramSketch:
    """A candidate solution hypothesis from PERCEIVE."""

    sketch_id: str
    structure: dict[str, Any]
    prior: float
    annotations: dict[str, Any] = field(default_factory=dict)

    nodes: list[SketchNode] = field(default_factory=list)
    root_node_id: str | None = None

    confidence_bands: dict[str, tuple[float, float]] = field(default_factory=dict)
    alternative_count: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "sketch_id": self.sketch_id,
                "structure": self.structure,
                "prior": self.prior,
                "annotations": self.annotations,
                "nodes": [n.to_dict() for n in self.nodes],
                "root_node_id": self.root_node_id,
                "confidence_bands": self.confidence_bands,
                "alternative_count": self.alternative_count,
            },
            indent=2,
        )

    @staticmethod
    def from_expression(expr: str, sketch_id: str, prior: float = 0.5) -> ProgramSketch:
        """Parse a simple S-expression into a sketch."""
        tokens = expr.replace("(", " ( ").replace(")", " ) ").split()
        nodes: list[SketchNode] = []
        node_counter = 0

        def parse_tokens(toks: list[str], idx: int) -> tuple[str, int]:
            nonlocal node_counter
            if idx >= len(toks):
                return "", idx

            token = toks[idx]
            if token == "(":
                idx += 1
                op_name = toks[idx]
                node_id = f"n{node_counter}"
                node_counter += 1
                idx += 1

                children = []
                while idx < len(toks) and toks[idx] != ")":
                    child_id, idx = parse_tokens(toks, idx)
                    if child_id:
                        children.append(child_id)

                nodes.append(
                    SketchNode(
                        node_id=node_id,
                        node_type=NodeType.OPERATOR,
                        value=op_name.lower(),
                        children=children,
                    )
                )
                return node_id, idx + 1
            elif token == ")":
                return "", idx
            else:
                node_id = f"n{node_counter}"
                node_counter += 1
                try:
                    val = float(token) if "." in token else int(token)
                    nodes.append(SketchNode(node_id=node_id, node_type=NodeType.LITERAL, value=val))
                except ValueError:
                    nodes.append(
                        SketchNode(node_id=node_id, node_type=NodeType.VARIABLE, value=token)
                    )
                return node_id, idx + 1

        root_id, _ = parse_tokens(tokens, 0)

        return ProgramSketch(
            sketch_id=sketch_id,
            structure={
                "expression": expr,
                "ops": [n.value for n in nodes if n.node_type == NodeType.OPERATOR],
            },
            prior=prior,
            nodes=nodes,
            root_node_id=root_id,
        )

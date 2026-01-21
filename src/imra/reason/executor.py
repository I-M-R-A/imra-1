"""Differentiable executor for program sketches."""

from __future__ import annotations

from dataclasses import dataclass, field

from imra.perceive.sketches import NodeType, ProgramSketch, SketchNode
from imra.reason.ops.differentiable import GradientTape
from imra.reason.ops.registry import OperatorRegistry
from imra.tracing.schemas import TraceBuffer, TraceEvent


@dataclass(slots=True)
class ReasonResult:
    """Result of executing a program sketch."""

    value: float | int | None
    success: bool
    trace: TraceBuffer
    gradients: dict[str, float] = field(default_factory=dict)
    step_count: int = 0
    error_message: str | None = None


class ReasonExecutor:
    """
    Executes program sketches via registered differentiable operators.
    Traverses sketch graphs, evaluates nodes, and tracks gradients.
    """

    def __init__(
        self,
        registry: OperatorRegistry | None = None,
        max_steps: int = 1000,
        variables: dict[str, float] | None = None,
    ) -> None:
        self.registry = registry or OperatorRegistry.default()
        self.max_steps = max_steps
        self.variables = variables or {}

    def run(
        self,
        sketch: ProgramSketch,
        trace: TraceBuffer,
        variables: dict[str, float] | None = None,
    ) -> ReasonResult:
        """Execute sketch and return result with gradients."""
        vars_ = {**self.variables, **(variables or {})}
        tape = GradientTape()
        step_count = 0

        # Build node lookup
        node_map: dict[str, SketchNode] = {n.node_id: n for n in sketch.nodes}
        cache: dict[str, float] = {}

        def evaluate(node_id: str) -> float:
            nonlocal step_count
            if step_count >= self.max_steps:
                raise RuntimeError(f"Exceeded max steps ({self.max_steps})")

            if node_id in cache:
                return cache[node_id]

            node = node_map.get(node_id)
            if node is None:
                raise ValueError(f"Unknown node: {node_id}")

            step_count += 1

            if node.node_type == NodeType.LITERAL:
                result = float(node.value)
                trace.append(
                    TraceEvent(
                        kind="reason.step",
                        payload={"node": node_id, "type": "literal", "value": result},
                    )
                )

            elif node.node_type == NodeType.VARIABLE:
                var_name = str(node.value)
                if var_name not in vars_:
                    raise ValueError(f"Undefined variable: {var_name}")
                result = float(vars_[var_name])
                trace.append(
                    TraceEvent(
                        kind="reason.step",
                        payload={
                            "node": node_id,
                            "type": "variable",
                            "name": var_name,
                            "value": result,
                        },
                    )
                )

            elif node.node_type == NodeType.OPERATOR:
                op_name = str(node.value)
                child_values = [evaluate(cid) for cid in node.children]

                try:
                    result, grad_info = self.registry.execute(op_name, child_values, tape)
                except Exception as e:
                    raise RuntimeError(f"Operator '{op_name}' failed: {e}") from e

                trace.append(
                    TraceEvent(
                        kind="reason.step",
                        payload={
                            "node": node_id,
                            "type": "operator",
                            "op": op_name,
                            "inputs": child_values,
                            "output": result,
                            "gradients": grad_info,
                        },
                    )
                )

            elif node.node_type == NodeType.BRANCH:
                # Conditional: first child is condition, second is true branch, third is false
                if len(node.children) < 3:
                    raise ValueError("Branch node requires 3 children: condition, true, false")
                cond = evaluate(node.children[0])
                result = evaluate(node.children[1]) if cond > 0.5 else evaluate(node.children[2])
                trace.append(
                    TraceEvent(
                        kind="reason.branch",
                        payload={
                            "node": node_id,
                            "condition": cond,
                            "selected": cond > 0.5,
                            "result": result,
                        },
                    )
                )

            else:
                raise ValueError(f"Unsupported node type: {node.node_type}")

            cache[node_id] = result
            return result

        try:
            if not sketch.root_node_id or sketch.root_node_id not in node_map:
                # Fallback: try legacy structure format
                return self._run_legacy(sketch, trace, tape)

            final_value = evaluate(sketch.root_node_id)
            gradients = tape.backward()

            return ReasonResult(
                value=final_value,
                success=True,
                trace=trace,
                gradients=gradients,
                step_count=step_count,
            )

        except Exception as e:
            trace.append(
                TraceEvent(
                    kind="reason.error",
                    payload={"error": str(e)},
                )
            )
            return ReasonResult(
                value=None,
                success=False,
                trace=trace,
                step_count=step_count,
                error_message=str(e),
            )

    def _run_legacy(
        self,
        sketch: ProgramSketch,
        trace: TraceBuffer,
        tape: GradientTape,
    ) -> ReasonResult:
        """Handle old-style sketches with just ops list."""
        value: float = 0.0
        ops = sketch.structure.get("ops", [])

        for op_name in ops:
            try:
                op = self.registry.lookup(op_name)
                # Default args for legacy mode
                if op.arity == 2:
                    result, grad_info = op.fn(3.0, 5.0, tape=tape)
                elif op.arity == 1:
                    result, grad_info = op.fn(value, tape=tape)
                else:
                    result, grad_info = 0.0, {}

                trace.append(
                    TraceEvent(
                        kind="reason.step",
                        payload={"op": op_name, "output": result, "gradients": grad_info},
                    )
                )
                value = result
            except Exception as e:
                trace.append(
                    TraceEvent(
                        kind="reason.error",
                        payload={"op": op_name, "error": str(e)},
                    )
                )

        return ReasonResult(
            value=value,
            success=True,
            trace=trace,
            gradients=tape.backward() if tape.operations else {},
            step_count=len(ops),
        )

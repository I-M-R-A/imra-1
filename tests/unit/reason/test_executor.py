from imra.perceive.sketches import NodeType, ProgramSketch, SketchNode
from imra.reason.executor import ReasonExecutor
from imra.tracing.schemas import TraceBuffer
from imra.utils.compat import utc_now


def test_executor_runs_graph_sketch() -> None:
    """Test executing a proper graph-based sketch: (ADD 3 (MUL 2 4)) = 11."""
    executor = ReasonExecutor()

    nodes = [
        SketchNode(node_id="n1", node_type=NodeType.LITERAL, value=3),
        SketchNode(node_id="n2", node_type=NodeType.LITERAL, value=2),
        SketchNode(node_id="n3", node_type=NodeType.LITERAL, value=4),
        SketchNode(node_id="n4", node_type=NodeType.OPERATOR, value="mul", children=["n2", "n3"]),
        SketchNode(node_id="n5", node_type=NodeType.OPERATOR, value="add", children=["n1", "n4"]),
    ]
    sketch = ProgramSketch(
        sketch_id="test_graph",
        structure={"description": "(ADD 3 (MUL 2 4))"},
        prior=0.8,
        nodes=nodes,
        root_node_id="n5",
    )

    trace = TraceBuffer(task_id="test", created_at=utc_now())
    result = executor.run(sketch, trace)

    assert result.success
    assert result.value == 11  # 3 + (2 * 4) = 11
    assert result.step_count == 5  # 5 nodes evaluated


def test_executor_runs_legacy_ops() -> None:
    """Test backward compatibility with old ops-list structure."""
    executor = ReasonExecutor()
    sketch = ProgramSketch(sketch_id="s", structure={"ops": ["add", "mul"]}, prior=0.5)
    trace = TraceBuffer(task_id="demo", created_at=utc_now())
    result = executor.run(sketch, trace)
    assert result.success
    assert result.value == 15.0  # add(3, 5) = 8, then mul(3, 5) = 15, final value = 15

    assert len(result.trace.events) == 2


def test_executor_with_variables() -> None:
    """Test variable substitution."""
    executor = ReasonExecutor()

    nodes = [
        SketchNode(node_id="x", node_type=NodeType.VARIABLE, value="x"),
        SketchNode(node_id="y", node_type=NodeType.VARIABLE, value="y"),
        SketchNode(node_id="add_xy", node_type=NodeType.OPERATOR, value="add", children=["x", "y"]),
    ]
    sketch = ProgramSketch(
        sketch_id="var_test",
        structure={},
        prior=0.9,
        nodes=nodes,
        root_node_id="add_xy",
    )

    trace = TraceBuffer(task_id="var_test", created_at=utc_now())
    result = executor.run(sketch, trace, variables={"x": 10, "y": 7})

    assert result.success
    assert result.value == 17  # x + y = 10 + 7


def test_executor_branch_node() -> None:
    """Test conditional branch evaluation."""
    executor = ReasonExecutor()

    nodes = [
        SketchNode(node_id="cond", node_type=NodeType.LITERAL, value=1.0),
        SketchNode(node_id="then", node_type=NodeType.LITERAL, value=100),
        SketchNode(node_id="else", node_type=NodeType.LITERAL, value=0),
        SketchNode(node_id="branch", node_type=NodeType.BRANCH, children=["cond", "then", "else"]),
    ]
    sketch = ProgramSketch(
        sketch_id="branch_test",
        structure={},
        prior=0.7,
        nodes=nodes,
        root_node_id="branch",
    )

    trace = TraceBuffer(task_id="branch", created_at=utc_now())
    result = executor.run(sketch, trace)

    assert result.success
    assert result.value == 100  # condition > 0.5, so take "then" branch if true

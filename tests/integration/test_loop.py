from imra.loops.deliberation import DifferentiableDeliberationLoop
from imra.perceive.engines.stub import StubPerceiveEngine
from imra.perceive.sketches import PerceptionRequest
from imra.reason.executor import ReasonExecutor
from imra.utils.compat import utc_now
from imra.verify.critics.stub import StubVerifyCritic


def test_loop_returns_trace() -> None:
    loop = DifferentiableDeliberationLoop(
        StubPerceiveEngine(), ReasonExecutor(), StubVerifyCritic()
    )
    request = PerceptionRequest(
        task_id="demo",
        payload={},
        metadata={"created_at": utc_now()},
    )
    result = loop.run(request)
    assert result.trace.events

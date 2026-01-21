"""High-level interface bridging legacy entry points with the new IMRA package."""

from __future__ import annotations

from imra.loops.deliberation import DifferentiableDeliberationLoop
from imra.perceive.engines.stub import StubPerceiveEngine
from imra.perceive.sketches import PerceptionRequest
from imra.reason.executor import ReasonExecutor
from imra.utils.compat import utc_now
from imra.verify.critics.stub import StubVerifyCritic


def run_demo(task_id: str = "demo") -> str:
    loop = DifferentiableDeliberationLoop(
        StubPerceiveEngine(), ReasonExecutor(), StubVerifyCritic()
    )
    request = PerceptionRequest(
        task_id=task_id,
        payload={},
        metadata={"created_at": utc_now()},
    )
    result = loop.run(request)
    return result.trace.to_markdown()

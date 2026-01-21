from imra.tracing.schemas import TraceBuffer, TraceEvent
from imra.utils.compat import utc_now
from imra.verify.critics.stub import StubVerifyCritic


def test_stub_confidence_range() -> None:
    critic = StubVerifyCritic()
    trace = TraceBuffer(task_id="demo", created_at=utc_now())
    trace.append(TraceEvent(kind="reason.step", payload={"output": 3}))
    score = critic.score(trace)
    assert 0.0 <= score <= 1.0

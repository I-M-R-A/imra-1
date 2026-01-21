from imra.perceive.engines.stub import StubPerceiveEngine
from imra.perceive.sketches import PerceptionRequest


def test_stub_emits_sketch() -> None:
    engine = StubPerceiveEngine()
    request = PerceptionRequest(task_id="t1", payload={})
    sketches = list(engine.propose(request))
    assert len(sketches) == 1
    assert sketches[0].sketch_id == "t1-sketch"

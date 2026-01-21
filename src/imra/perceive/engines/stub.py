"""Reference PerceiveEngine implementation for smoke tests."""

from __future__ import annotations

from collections.abc import Iterable

from imra.perceive.engines.base import PerceiveEngine
from imra.perceive.sketches import PerceptionRequest, ProgramSketch


class StubPerceiveEngine(PerceiveEngine):
    def propose(self, request: PerceptionRequest) -> Iterable[ProgramSketch]:
        yield ProgramSketch(
            sketch_id=f"{request.task_id}-sketch",
            structure={"ops": ["add", "mul"]},
            prior=0.5,
            annotations={"source": "stub"},
        )

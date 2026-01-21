"""Base classes for intuition engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from imra.perceive.sketches import PerceptionRequest, ProgramSketch


class PerceiveEngine(ABC):
    """Interface for generating program sketches from raw problems."""

    @abstractmethod
    def propose(self, request: PerceptionRequest) -> Iterable[ProgramSketch]:
        """Return a stream of candidate sketches ordered by internal priors."""

    def update(self, sketch: ProgramSketch, reward: float) -> None:  # noqa: B027
        """Optional post-hoc learning signal."""

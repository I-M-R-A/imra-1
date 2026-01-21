"""IMRA-1 Research Module.

This module provides the research instrumentation for studying AI cognition.
The primary outputs are CognitiveTraces - detailed records of AI reasoning
that can be analyzed to understand how AI thinks, fails, and self-corrects.

Key Components:
- CognitiveTrace: The research data structure
- ResearchOrchestrator: Produces research-grade traces
- Analysis tools for studying traces
"""

from .cognitive_trace import (
    CognitivePhase,
    CognitiveTrace,
    ReasoningStep,
    UncertaintyMarker,
    UncertaintyType,
)
from .orchestrator import ResearchOrchestrator

__all__ = [
    "CognitiveTrace",
    "ReasoningStep",
    "UncertaintyMarker",
    "UncertaintyType",
    "CognitivePhase",
    "ResearchOrchestrator",
]

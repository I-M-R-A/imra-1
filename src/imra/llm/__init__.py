"""IMRA-1 Multi-LLM Integration Layer.

Provides:
- OllamaClient: Low-level Ollama API wrapper (unlimited timeout for quality)
- MultiLLMOrchestrator: Basic orchestration (legacy)
- MetaCognitiveOrchestrator: Full meta-cognitive deliberation framework
- Meta prompts for PERCEIVE/REASON/VERIFY roles
"""

from .meta_orchestrator import DeliberationTrace, MetaCognitiveOrchestrator, run_olympiad_test
from .meta_prompts import (
    get_perceive_prompt,
    get_reason_prompt,
    get_reflection_prompt,
    get_verify_consensus_prompt,
    get_verify_final_prompt,
)
from .ollama_client import OllamaClient
from .orchestrator import MultiLLMOrchestrator

__all__ = [
    "OllamaClient",
    "MultiLLMOrchestrator",
    "MetaCognitiveOrchestrator",
    "DeliberationTrace",
    "run_olympiad_test",
    "get_perceive_prompt",
    "get_reason_prompt",
    "get_verify_consensus_prompt",
    "get_verify_final_prompt",
    "get_reflection_prompt",
]

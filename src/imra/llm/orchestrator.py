"""Multi-LLM orchestrator for IMRA-1.

Provides a minimal orchestrator that assigns roles to LLMs (PERCEIVE1,
PERCEIVE2, REASON, VERIFY), requests structured JSON sketches, executes a
simple execution step (delegated to REASON LLM or local execution), and
collects a trace for verification and distillation.

This is scaffolding for research-grade multi-LLM integration. It returns
structured traces suitable for storage and later analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .ollama_client import OllamaClient


@dataclass
class Sketch:
    author: str
    sketch: dict[str, Any]
    confidence: float = 0.0


@dataclass
class TraceEvent:
    phase: str
    actor: str
    payload: dict[str, Any]


@dataclass
class DeliberationTrace:
    events: list[TraceEvent] = field(default_factory=list)

    def add(self, phase: str, actor: str, payload: dict[str, Any]) -> None:
        self.events.append(TraceEvent(phase=phase, actor=actor, payload=payload))

    def to_dict(self) -> dict[str, Any]:
        return {"events": [vars(e) for e in self.events]}


class MultiLLMOrchestrator:
    """Orchestrates multiple Ollama LLMs under IMRA-1 roles.

    Roles are mapped to model names (or the same model multiple times).
    The orchestrator assumes each LLM returns structured JSON when asked
    for a sketch (or the orchestrator will try to parse JSON out of text).
    """

    def __init__(self, models: dict[str, str]) -> None:
        # models: role -> model_name
        self.models = models
        self.clients = {role: OllamaClient(model=name) for role, name in models.items()}

    def _call(self, role: str, prompt: str) -> str:
        client = self.clients[role]
        return client.send_prompt(prompt)

    def _parse_json_or_text(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            # Try to extract a JSON substring
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except Exception:
                    pass
            # fallback: return raw text
            return {"raw": text}

    def propose_sketch(self, role: str, problem: str) -> Sketch:
        prompt = json.dumps({"phase": "sketch_proposal", "problem": problem}, indent=2)
        raw = self._call(role, prompt)
        parsed = self._parse_json_or_text(raw)
        confidence = float(parsed.get("confidence", 0.0)) if isinstance(parsed, dict) else 0.0
        return Sketch(author=role, sketch=parsed, confidence=confidence)

    def execute_sketch(self, role: str, sketch: Sketch) -> dict[str, Any]:
        prompt = json.dumps({"phase": "execute", "sketch": sketch.sketch}, indent=2)
        raw = self._call(role, prompt)
        parsed = self._parse_json_or_text(raw)
        return parsed

    def verify_trace(self, role: str, trace: DeliberationTrace) -> dict[str, Any]:
        prompt = json.dumps({"phase": "verify", "trace": trace.to_dict()}, indent=2)
        raw = self._call(role, prompt)
        parsed = self._parse_json_or_text(raw)
        return parsed

    def run_deliberation(self, problem: str, max_iterations: int = 3) -> dict[str, Any]:
        trace = DeliberationTrace()

        # Sketch phase: PERCEIVE1 and PERCEIVE2
        s1 = self.propose_sketch("perceive1", problem)
        trace.add("sketch_proposal", s1.author, {"sketch": s1.sketch, "confidence": s1.confidence})
        s2 = self.propose_sketch("perceive2", problem)
        trace.add("sketch_proposal", s2.author, {"sketch": s2.sketch, "confidence": s2.confidence})

        # Consensus detection by VERIFY
        verify_result = self.verify_trace("verify", trace)
        trace.add("consensus_check", "verify", verify_result)

        # Execution by REASON
        exec1 = self.execute_sketch("reason", s1)
        trace.add("execution", "reason", {"input_sketch": s1.sketch, "result": exec1})
        exec2 = self.execute_sketch("reason", s2)
        trace.add("execution", "reason", {"input_sketch": s2.sketch, "result": exec2})

        # Final verify
        final_verify = self.verify_trace("verify", trace)
        trace.add("final_verify", "verify", final_verify)

        return {"trace": trace.to_dict(), "final": final_verify}

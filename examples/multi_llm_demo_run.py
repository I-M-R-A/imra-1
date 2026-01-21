"""Runner: attempt real Ollama orchestrator run, fall back to simulated mock.

This script will try to run `MultiLLMOrchestrator` with detected models.
If Ollama calls fail, it will replace `OllamaClient` with a mock that
returns deterministic JSON for each role, simulating the multi-LLM loop.
"""

from __future__ import annotations

import json
import os
import traceback

from imra.llm import MultiLLMOrchestrator, OllamaClient

MODELS = {  # Define models for each role, using env vars or defaults
    "perceive1": os.getenv("IMRA_LLM_PERCEIVE1", "qwen2.5-coder:3b"),
    "perceive2": os.getenv("IMRA_LLM_PERCEIVE2", "qwen2.5-coder:7b"),
    "reason": os.getenv("IMRA_LLM_REASON", "codellama:latest"),
    "verify": os.getenv("IMRA_LLM_VERIFY", "nomic-embed-text:latest"),
}

PROBLEM = os.getenv(
    "IMRA_DEMO_PROBLEM", "How many unique paths from top-left to bottom-right in a 3x3 grid?"
)


class MockOllamaClient(OllamaClient):
    def __init__(self, model: str = "mock") -> None:
        self.model = model
        self.base_url = "mock"

    def send_prompt(self, prompt: str, params=None) -> str:
        lower = prompt.lower()
        if "sketch_proposal" in lower:
            sketch = {
                "confidence": 0.85,
                "plan": ["use combinatorics: binomial"],
                "operations": ["C(6,3)"],
            }
            return json.dumps(sketch)
        if "execute" in lower:
            # C(6,3) = 20

            result = {"value": 20, "steps": ["compute C(6,3) = 20"]}
            return json.dumps(result)
        if "verify" in lower or "consensus" in lower:
            verdict = {"accepted": True, "confidence": 0.92, "notes": []}
            return json.dumps(verdict)
        return json.dumps({"raw": prompt})


def run() -> None:
    print("Attempting real Ollama orchestrator run...")
    orch = MultiLLMOrchestrator(models=MODELS)
    try:
        result = orch.run_deliberation(PROBLEM)
        print(json.dumps(result, indent=2))
        return
    except Exception:
        print("Real orchestrator run failed; falling back to simulation.")
        print(traceback.format_exc())

    print("Running simulated multi-LLM loop using MockOllamaClient...")
    orch = MultiLLMOrchestrator(models=MODELS)
    for role in orch.clients:
        orch.clients[role] = MockOllamaClient(model=f"mock-{role}")
    result = orch.run_deliberation(PROBLEM)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()

"""Demo: multi-LLM orchestrator usage.

Requires `requests` and an Ollama instance reachable via OLLAMA_URL.
Set up model mappings via the `models` dict.
"""

import os

from imra.llm import MultiLLMOrchestrator

MODELS = {
    "perceive1": os.getenv("IMRA_LLM_PERCEIVE1", "ollama-perceive1"),
    "perceive2": os.getenv("IMRA_LLM_PERCEIVE2", "ollama-perceive2"),
    "reason": os.getenv("IMRA_LLM_REASON", "ollama-reason"),
    "verify": os.getenv("IMRA_LLM_VERIFY", "ollama-verify"),
}

PROBLEM = "What is the number of unique paths from top-left to bottom-right in a 3x3 grid?"

if __name__ == "__main__":
    orch = MultiLLMOrchestrator(models=MODELS)
    result = orch.run_deliberation(PROBLEM)
    import json

    print(json.dumps(result, indent=2))

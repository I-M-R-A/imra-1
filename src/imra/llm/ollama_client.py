"""Ollama client wrapper for IMRA-1 meta-cognitive framework.

Configurable via OLLAMA_URL env var (default http://localhost:11434).
Sends prompt text and returns model output as text. This is a thin wrapper
so the orchestrator can call different LLM endpoints uniformly.

For research-grade testing, timeouts are disabled by default to allow
models sufficient time for deep reasoning on olympiad-level problems.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import requests
except Exception:
    requests = None

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

OLLAMA_TIMEOUT = (
    None
    if os.getenv("OLLAMA_TIMEOUT", "").lower() in ("", "none", "unlimited")
    else int(os.getenv("OLLAMA_TIMEOUT", "0")) or None
)


class OllamaClient:
    def __init__(
        self,
        model: str = "llama2",
        base_url: str | None = None,
        timeout: int | None = OLLAMA_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url or OLLAMA_URL
        self.timeout = timeout

    def _check_requests(self) -> bool:
        return requests is not None

    def send_prompt(self, prompt: str, params: dict[str, Any] | None = None) -> str:
        """Send prompt to Ollama and return raw text response.

        For meta-cognitive research, we allow unlimited time for the model
        to reason through complex problems. The timeout can be configured
        via OLLAMA_TIMEOUT env var or constructor parameter.
        """
        params = params or {}
        payload = {"prompt": prompt, **params}
        headers = {"Content-Type": "application/json"}
        http_timeout = self.timeout if self.timeout else 600  # 10min for HTTP calls

        if self._check_requests():
            endpoints = [
                f"{self.base_url}/api/models/{self.model}/completions",
                f"{self.base_url}/api/models/{self.model}/generate",
                f"{self.base_url}/v1/complete",
                f"{self.base_url}/completions",
            ]
            last_err = None
            for url in endpoints:
                try:
                    r = requests.post(
                        url, headers=headers, data=json.dumps(payload), timeout=http_timeout
                    )
                    if r.status_code == 200:
                        try:
                            body = r.json()
                        except Exception:
                            return r.text
                        if isinstance(body, dict):
                            if "choices" in body and body["choices"]:
                                text = body["choices"][0].get("text") or body["choices"][0].get(
                                    "message", {}
                                ).get("content")
                                return text if text is not None else json.dumps(body)
                            if "completion" in body:
                                return body["completion"]
                            if "result" in body:
                                return body["result"]
                            if "object" in body and body.get("object") == "list" and "data" in body:
                                last_err = (r.status_code, "model list returned")
                                continue
                            return json.dumps(body)
                        return str(body)
                    else:
                        last_err = (r.status_code, r.text)
                except Exception as e:
                    last_err = e

        try:
            from subprocess import PIPE, Popen

            cmd = [
                "ollama",
                "run",
                self.model,
                "--format",
                "json",
                "--hidethinking",
                "--think=false",
                "--nowordwrap",
            ]
            p = Popen(cmd, stdout=PIPE, stderr=PIPE, stdin=PIPE, text=True)
            # timeout=None means wait indefinitely for complex reasoning
            out, err = p.communicate(input=prompt, timeout=self.timeout)
            if p.returncode == 0 and out:
                return out
            combined = (out or "") + "\n" + (err or "")
            raise RuntimeError(f"Ollama CLI failed: {p.returncode} {combined.strip()}")
        except Exception as cli_err:
            raise RuntimeError(
                f"Failed to call Ollama via HTTP and CLI; last HTTP error: {last_err}, CLI error: {cli_err}"
            ) from cli_err

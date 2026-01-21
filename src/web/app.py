"""FastAPI surface for IMRA-1 demos."""

from __future__ import annotations

from fastapi import FastAPI

from core.interface import run_demo

app = FastAPI(title="IMRA-1 API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
def run(task_id: str = "demo") -> dict[str, str]:
    trace = run_demo(task_id=task_id)
    return {"task_id": task_id, "trace": trace}

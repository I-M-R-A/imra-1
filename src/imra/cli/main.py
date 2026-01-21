"""Typer-powered CLI bootstrap."""

from __future__ import annotations

from pathlib import Path

import typer

from imra.config import ImraConfig
from imra.loops.deliberation import DifferentiableDeliberationLoop
from imra.perceive.engines.stub import StubPerceiveEngine
from imra.perceive.sketches import PerceptionRequest
from imra.reason.executor import ReasonExecutor
from imra.tracing.schemas import TraceBuffer, TraceEvent
from imra.utils import configure_logging
from imra.utils.compat import utc_now
from imra.verify.critics.stub import StubVerifyCritic

app = typer.Typer(help="IMRA-1 research CLI")


def bootstrap_app() -> typer.Typer:
    return app


@app.command()
def run(config_path: str = typer.Option("configs/base.yaml", help="YAML config")) -> None:
    cfg = ImraConfig.from_yaml(config_path)
    configure_logging(trace_dir=cfg.runtime.trace_dir)
    perceive = StubPerceiveEngine()
    reason = ReasonExecutor()
    verify = StubVerifyCritic()
    loop = DifferentiableDeliberationLoop(perceive, reason, verify)
    request = PerceptionRequest(
        task_id="demo",
        payload={},
        metadata={"created_at": utc_now()},
    )
    result = loop.run(request)
    typer.echo(result.trace.to_markdown())


@app.command()
def trace(output: str = typer.Option("trace.md")) -> None:
    buffer = TraceBuffer(task_id="demo", created_at=utc_now())
    buffer.append(TraceEvent(kind="perceive.sketch", payload={"id": "demo-sketch"}))
    buffer.append(TraceEvent(kind="reason.step", payload={"op": "add", "output": 13}))
    buffer.append(TraceEvent(kind="verify.score", payload={"confidence": 0.8}))
    markdown = buffer.to_markdown()
    Path(output).write_text(markdown, encoding="utf-8")
    typer.echo(f"Trace saved to {output}")


if __name__ == "__main__":
    app()

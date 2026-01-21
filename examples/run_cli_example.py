"""Demonstration CLI for emitting a mock reasoning trace."""

from __future__ import annotations

import argparse
from pathlib import Path

from imra.utils.compat import utc_now

try:
    from rich.console import Console as RichConsole
    from rich.panel import Panel as RichPanel
except ImportError:  # pragma: no cover
    RichConsole = None
    RichPanel = None

from imra.cli.main import bootstrap_app
from imra.tracing.formatters.markdown import MarkdownFormatter
from imra.tracing.schemas import TraceBuffer, TraceEvent


class PlainConsole:
    def print(self, *objects: object, **kwargs: object) -> None:  # noqa: ARG002
        print(*objects)


class PlainPanel:
    @staticmethod
    def fit(text: str, subtitle: str | None = None) -> str:
        return f"{text}{f' ({subtitle})' if subtitle else ''}"


ConsoleClass = RichConsole if RichConsole is not None else PlainConsole
PanelClass = RichPanel if RichPanel is not None else PlainPanel

console = ConsoleClass()


def generate_mock_trace() -> TraceBuffer:
    buffer = TraceBuffer(task_id="demo", created_at=utc_now())
    buffer.append(
        TraceEvent(kind="perceive.sketch", payload={"prior": 0.62, "sketch": "(ADD X (MUL Y Z))"})
    )
    buffer.append(
        TraceEvent(kind="reason.step", payload={"op": "mul", "lhs": 5, "rhs": 2, "output": 10})
    )
    buffer.append(
        TraceEvent(kind="reason.step", payload={"op": "add", "lhs": 3, "rhs": 10, "output": 13})
    )
    buffer.append(TraceEvent(kind="verify.score", payload={"confidence": 0.81, "coherence": 0.94}))
    buffer.append(TraceEvent(kind="loop.directive", payload={"action": "accept"}))
    return buffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-trace", action="store_true", help="render a built-in arithmetic trace"
    )
    parser.add_argument(
        "--save", type=Path, help="optional path to save Markdown trace", default=None
    )
    args = parser.parse_args()

    app = bootstrap_app()

    console.print(PanelClass.fit("IMRA-1 CLI", subtitle=str(app)))

    if args.demo_trace:
        trace = generate_mock_trace()
        md = MarkdownFormatter().format(trace)
        console.print(md)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            args.save.write_text(md, encoding="utf-8")
            console.print(f"Trace saved to {args.save}")


if __name__ == "__main__":
    main()

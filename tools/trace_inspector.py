"""Simple utility to pretty-print trace JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def load_trace(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def main(path: str) -> None:
    table = Table(title=f"Trace: {path}")
    table.add_column("idx", justify="right")
    table.add_column("kind")
    table.add_column("payload")
    for idx, event in enumerate(load_trace(Path(path))):
        table.add_row(str(idx), event.get("kind", "?"), json.dumps(event.get("payload", {})))
    console.print(table)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        console.print("Usage: python trace_inspector.py <trace.jsonl>")
        raise SystemExit(1)
    main(sys.argv[1])

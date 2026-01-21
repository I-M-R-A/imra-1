"""Utilities for assembling research-ready documentation bundles."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
OUTPUT_DIR = Path("build/docs")

console = Console()


def collect_markdown() -> list[Path]:
    return sorted(DOCS_DIR.rglob("*.md"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table = Table(title="IMRA-1 Documentation Bundle")
    table.add_column("Document")
    table.add_column("Bytes", justify="right")
    for path in collect_markdown():
        target = OUTPUT_DIR / path.relative_to(DOCS_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.read_text(), encoding="utf-8")
        table.add_row(str(path.relative_to(DOCS_DIR)), str(path.stat().st_size))
    console.print(table)


if __name__ == "__main__":
    main()

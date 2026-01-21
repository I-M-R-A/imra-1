"""Simple registry for benchmark datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dataset:
    name: str
    path: Path
    description: str


class DatasetRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Dataset] = {}

    def register(self, dataset: Dataset) -> None:
        self._items[dataset.name] = dataset

    def get(self, name: str) -> Dataset:
        if name not in self._items:
            raise KeyError(f"Dataset '{name}' not registered")
        return self._items[name]

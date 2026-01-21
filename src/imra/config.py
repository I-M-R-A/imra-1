"""Configuration helpers for IMRA modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ModuleConfig:
    perceive: str
    reason: str
    verify: str


@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 0
    device: str = "cpu"
    precision: str = "fp32"
    trace_dir: Path = Path(".imra/traces")


@dataclass(slots=True)
class ImraConfig:
    runtime: RuntimeConfig
    modules: ModuleConfig

    @staticmethod
    def from_yaml(path: Path | str) -> ImraConfig:
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        runtime = RuntimeConfig(
            seed=data.get("runtime", {}).get("seed", 0),
            device=data.get("runtime", {}).get("device", "cpu"),
            precision=data.get("runtime", {}).get("precision", "fp32"),
            trace_dir=Path(data.get("logging", {}).get("trace_dir", ".imra/traces")),
        )
        modules = ModuleConfig(
            perceive=data.get("modules", {}).get(
                "perceive", "imra.perceive.engines.stub:StubPerceiveEngine"
            ),
            reason=data.get("modules", {}).get("reason", "imra.reason.executor:ReasonExecutor"),
            verify=data.get("modules", {}).get(
                "verify", "imra.verify.critics.stub:StubVerifyCritic"
            ),
        )
        return ImraConfig(runtime=runtime, modules=modules)

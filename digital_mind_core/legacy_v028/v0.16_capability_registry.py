
"""
v0.16_capability_registry.py

Discovers read-only providers from a controlled local provider directory.

A provider module may expose:
    PROVIDERS = [
        {
            "name": "provider-name",
            "capability": "vision.perceive",
            "callable": perceive,
            "read_only": True,
            "priority": 100,
        }
    ]

Only providers explicitly marked read_only=True are eligible for AutoConnect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import importlib.util
import sys


@dataclass(slots=True)
class Provider:
    name: str
    capability: str
    fn: Callable[..., Any]
    read_only: bool
    priority: int
    source: str


class CapabilityRegistry:
    def __init__(self, provider_dirs=()):
        self.provider_dirs = [Path(p).resolve() for p in provider_dirs]
        self.providers: Dict[str, List[Provider]] = {}
        self.discovery_errors: List[dict] = []

    def register(self, provider: Provider) -> None:
        if not provider.read_only:
            return
        self.providers.setdefault(provider.capability, []).append(provider)
        self.providers[provider.capability].sort(
            key=lambda p: (-p.priority, p.name)
        )

    def discover(self) -> int:
        before = sum(len(v) for v in self.providers.values())

        for root in self.provider_dirs:
            if not root.exists() or not root.is_dir():
                continue

            for path in sorted(root.glob("provider_*.py")):
                try:
                    module_name = (
                        "kernel_provider_"
                        + path.stem
                        + "_"
                        + str(abs(hash(str(path))))[:8]
                    )
                    spec = importlib.util.spec_from_file_location(module_name, path)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"cannot load {path}")
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)

                    defs = getattr(mod, "PROVIDERS", [])
                    for raw in defs:
                        if not isinstance(raw, dict):
                            continue
                        fn = raw.get("callable")
                        if not callable(fn):
                            continue
                        self.register(
                            Provider(
                                name=str(raw.get("name", path.stem)),
                                capability=str(raw["capability"]),
                                fn=fn,
                                read_only=bool(raw.get("read_only", False)),
                                priority=int(raw.get("priority", 0)),
                                source=str(path),
                            )
                        )
                except Exception as exc:
                    self.discovery_errors.append({
                        "path": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    })

        after = sum(len(v) for v in self.providers.values())
        return after - before

    def resolve(self, capability: str) -> Optional[Callable[..., Any]]:
        items = self.providers.get(capability, [])
        return items[0].fn if items else None

    def snapshot(self) -> dict:
        return {
            cap: [
                {
                    "name": p.name,
                    "priority": p.priority,
                    "read_only": p.read_only,
                    "source": p.source,
                }
                for p in providers
            ]
            for cap, providers in sorted(self.providers.items())
        }

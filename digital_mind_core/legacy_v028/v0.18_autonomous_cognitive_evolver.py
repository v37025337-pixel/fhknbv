
"""
v0.18_autonomous_cognitive_evolver.py

Autonomous, benchmark-gated cognitive development loop.

This module does not use a fixed order for logic/thinking/self/workspace/
intelligence. Each cycle:
    benchmark -> choose largest measured deficit -> generate candidate
    -> isolated compile/regression -> re-benchmark -> promote or reject

The candidate mechanisms come from the generic mutation library in
v0.18_cognitive_lab.py. The selector chooses which one to integrate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import importlib.util
import json
import sys
import time


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _current_tree_from_here() -> Path:
    return Path(__file__).resolve().parent


def evolve(
    *,
    max_cycles: int = 8,
    target_score: float = 0.999,
    workspace_root: Optional[str] = None,
) -> dict:
    current = _current_tree_from_here()
    workspace = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else current.parent
    )

    lab = _load(
        "cognitive_lab_runtime",
        current / "v0.18_cognitive_lab.py",
    )
    ctl = _load(
        "cognitive_rewrite_runtime",
        current / "v0.16_self_rewrite_controller.py",
    )

    controller = ctl.SelfRewriteController(str(workspace))
    journal: List[dict] = []

    for cycle in range(1, max_cycles + 1):
        before = lab.run_benchmark(controller.current)

        if all(float(before["scores"][d]) >= target_score for d in lab.DIMENSIONS):
            journal.append({
                "cycle": cycle,
                "status": "complete",
                "reason": "all target dimensions reached threshold",
                "benchmark": before,
            })
            break

        selection = lab.choose_deficit(before, controller.current)
        if selection is None:
            journal.append({
                "cycle": cycle,
                "status": "stopped",
                "reason": "no remaining candidate dimension",
                "benchmark": before,
            })
            break

        selected = selection["dimension"]
        enabled = list(before.get("enabled", []))
        if selected not in enabled:
            enabled.append(selected)

        core_source = lab.build_core_source(enabled)
        manifest = {
            "enabled": enabled,
            "last_selected": selected,
            "selection": selection,
            "cycle": cycle,
            "parent_mean_score": before["mean_score"],
            "parent_revision": lab.tree_revision(controller.current),
            "operational_notice": (
                "conscious_integration is an operational global-access/"
                "conflict-monitoring mechanism, not evidence of subjective consciousness"
            ),
        }

        proposal_id = (
            f"cognitive_{cycle:02d}_{selected}_"
            + hashlib.sha256(
                f"{manifest['parent_revision']}|{selected}|{cycle}".encode()
            ).hexdigest()[:8]
        )

        proposal = ctl.RewriteProposal(
            proposal_id=proposal_id,
            goal=f"develop:{selected}",
            rationale=(
                f"selected from measured deficit={selection['deficit']:.6f}; "
                f"current score={selection['score']:.6f}"
            ),
            changes=[
                ctl.FileChange("v0.18_cognitive_core.py", core_source),
                ctl.FileChange(
                    "cognitive_manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2),
                ),
            ],
            tests=[
                "test_baseline.py",
                "test_provider_autodiscovery.py",
                "test_data_world.py",
                "test_cognitive_core.py",
            ],
        )

        gate = controller.stage_and_evaluate(proposal)
        event: Dict[str, Any] = {
            "cycle": cycle,
            "selected": selected,
            "selection": selection,
            "before": before,
            "gate_stage": gate.stage,
            "gate_accepted": gate.accepted,
            "gate_reason": gate.reason,
        }

        if not gate.accepted:
            event["status"] = "rejected"
            journal.append(event)
            break

        candidate = Path(gate.candidate_path)
        after = lab.run_benchmark(candidate)
        event["candidate"] = after

        selected_improved = (
            float(after["scores"][selected])
            > float(before["scores"][selected]) + 1e-12
        )
        mean_improved = (
            float(after["mean_score"])
            > float(before["mean_score"]) + 1e-12
        )
        no_regression = all(
            float(after["scores"][d]) + 1e-12 >= float(before["scores"][d])
            for d in lab.DIMENSIONS
        )

        event["selected_improved"] = selected_improved
        event["mean_improved"] = mean_improved
        event["no_regression"] = no_regression

        if not (selected_improved and mean_improved and no_regression):
            event["status"] = "rejected"
            event["reason"] = "benchmark improvement gate failed"
            journal.append(event)
            break

        promoted = controller.promote(gate)
        event["status"] = "promoted"
        event["revision"] = promoted.promoted_revision
        event["after"] = lab.run_benchmark(controller.current)
        journal.append(event)

        # Reload lab after atomic tree replacement to ensure the next cycle
        # uses the active revision's code, not a stale module object.
        lab = _load(
            f"cognitive_lab_runtime_{cycle}",
            controller.current / "v0.18_cognitive_lab.py",
        )

    final = lab.run_benchmark(controller.current)
    report = {
        "cycles": journal,
        "final": final,
        "final_revision": lab.tree_revision(controller.current),
        "notice": (
            "These scores measure operational mechanisms only. "
            "They do not establish phenomenal consciousness or general intelligence."
        ),
    }

    (controller.current / "cognitive_evolution_journal.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = evolve()
    print(json.dumps(result, ensure_ascii=False, indent=2))

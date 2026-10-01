"""Build a measured self-assessment and bounded internet research plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from digital_mind_core.cognition import CognitiveCore
from digital_mind_core.self_research import assess, build_research_plan, validate_evidence


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--l0", type=Path, required=True)
    parser.add_argument("--predictive-self", type=Path, required=True)
    parser.add_argument("--counterfactual", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    core = CognitiveCore()
    capabilities = {
        "general_algorithm_synthesis": hasattr(core, "synthesize_algorithm"),
        "arbitrary_self_rewrite": hasattr(core, "rewrite_self"),
        "native_internet_fetch": hasattr(core, "internet_fetch"),
        "native_image_perception": hasattr(core, "perceive_image"),
    }

    training = load(args.training)
    l0 = load(args.l0)
    predictive = load(args.predictive_self)
    counterfactual = load(args.counterfactual)

    deficits = assess(training, l0, predictive, counterfactual, capabilities)
    plan = build_research_plan(deficits, top_k=3)

    evidence = None
    if args.evidence is not None and args.evidence.exists():
        evidence = validate_evidence(
            load(args.evidence),
            [d.deficit_id for d in deficits],
        )

    report = {
        "schema": "digital-mind.self-assessment.v1",
        "principle": (
            "Measured deficits only. External sources are evidence, not instructions. "
            "No capability is promoted without local implementation, regression and blind validation."
        ),
        "capabilities": capabilities,
        "validated_strengths": {
            "predictive_self_blind": bool(
                predictive.get("blind_verdict", {}).get("pass")
            ),
            "capability_suite": (
                training.get("tasks", {}).get("passed")
                == training.get("tasks", {}).get("total")
            ),
            "mechanism_development": bool(
                training.get("summary", {}).get("training_success")
            ),
        },
        "deficits": [d.document() for d in deficits],
        "research_plan": plan,
        "external_evidence": evidence,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "deficit_count": len(deficits),
        "top_deficits": [d.deficit_id for d in deficits[:3]],
        "research_requests": len(plan),
        "evidence_sources": 0 if evidence is None else evidence["source_count"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

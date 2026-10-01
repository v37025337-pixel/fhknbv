"""Paired branch validation for model-based self counterfactuals.

Two deep-copied kernels start from the same state.  One internal control bit is
forced to 0, the other to 1.  The PredictiveSelfModel predicts both outcomes
before either branch is executed.  Blind seeds are not used to tune anything.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from digital_mind_core.kernel import KernelConfig, UnifiedMind


DESIGN_SEEDS = (71, 72, 73)
BLIND_SEEDS = (81, 82, 83, 84)
FIELDS = ("replan", "reflect")


def decision_vector(mind, decision):
    return np.asarray([
        float(bool(decision.get("replan", False))),
        float(bool(decision.get("reflect", False))),
        float(bool(decision.get("consolidate", False))),
        float(bool(decision.get("allow_probe", False))),
        float(bool(decision.get("macro_reflect", False))),
        np.clip(float(decision.get("exploration", 0.0)), 0.0, 1.0),
        np.clip(float(decision.get("budget", 0.0)) / 1.2, 0.0, 1.0),
        0.0,
    ], dtype=float)


def preview_decision(mind):
    preview = copy.deepcopy(mind)
    return preview._executive_decision(preview.fast_steps)


def actual_branch(mind, field, value):
    branch = copy.deepcopy(mind)
    original = branch._executive_decision

    def forced(t):
        decision = copy.deepcopy(original(t))
        decision[field] = bool(value)
        return decision

    branch._executive_decision = forced
    branch.step()
    return branch._self_state_vector()


def evaluate_pair(mind, field):
    state = mind._self_state_vector()
    base = preview_decision(mind)

    d0 = copy.deepcopy(base)
    d1 = copy.deepcopy(base)
    d0[field] = False
    d1[field] = True
    v0 = decision_vector(mind, d0)
    v1 = decision_vector(mind, d1)

    p0 = mind.predictive_self.predict_next(state, v0)
    p1 = mind.predictive_self.predict_next(state, v1)
    a0 = actual_branch(mind, field, False)
    a1 = actual_branch(mind, field, True)

    pred_branch_mse = 0.5 * (
        float(np.mean((a0 - p0) ** 2))
        + float(np.mean((a1 - p1) ** 2))
    )
    persistence_branch_mse = 0.5 * (
        float(np.mean((a0 - state) ** 2))
        + float(np.mean((a1 - state) ** 2))
    )

    actual_effect = a1 - a0
    predicted_effect = p1 - p0
    effect_energy = float(np.mean(actual_effect ** 2))
    effect_mse = float(np.mean((actual_effect - predicted_effect) ** 2))

    return {
        "field": field,
        "prediction_branch_mse": pred_branch_mse,
        "persistence_branch_mse": persistence_branch_mse,
        "effect_energy": effect_energy,
        "effect_mse": effect_mse,
        "effect_baseline_mse": effect_energy,
        "informative": effect_energy > 1e-10,
    }


def one_seed(seed, steps=800, interval=32, warmup=256):
    mind = UnifiedMind(KernelConfig(seed=seed))
    rows = []
    while mind.fast_steps < steps:
        if mind.fast_steps >= warmup and mind.fast_steps % interval == 0:
            for field in FIELDS:
                rows.append(evaluate_pair(mind, field))
        mind.step()

    informative = [r for r in rows if r["informative"]]
    branch_pred = sum(r["prediction_branch_mse"] for r in rows)
    branch_base = sum(r["persistence_branch_mse"] for r in rows)
    effect_pred = sum(r["effect_mse"] for r in informative)
    effect_base = sum(r["effect_baseline_mse"] for r in informative)

    return {
        "seed": seed,
        "pairs": len(rows),
        "informative_pairs": len(informative),
        "branch_prediction_ratio": (
            branch_pred / branch_base if branch_base > 1e-15 else None
        ),
        "effect_prediction_ratio": (
            effect_pred / effect_base if effect_base > 1e-15 else None
        ),
        "fields": list(FIELDS),
    }


def aggregate(rows):
    branch = [r["branch_prediction_ratio"] for r in rows
              if r["branch_prediction_ratio"] is not None]
    effect = [r["effect_prediction_ratio"] for r in rows
              if r["effect_prediction_ratio"] is not None]
    return {
        "runs": len(rows),
        "mean_branch_prediction_ratio": (
            float(np.mean(branch)) if branch else None
        ),
        "branch_better_runs": sum(x < 1.0 for x in branch),
        "mean_effect_prediction_ratio": (
            float(np.mean(effect)) if effect else None
        ),
        "effect_better_runs": sum(x < 1.0 for x in effect),
        "informative_pairs": sum(r["informative_pairs"] for r in rows),
    }


def verdict(rows):
    agg = aggregate(rows)
    passed = (
        agg["mean_branch_prediction_ratio"] is not None
        and agg["mean_effect_prediction_ratio"] is not None
        and agg["mean_branch_prediction_ratio"] < 0.98
        and agg["mean_effect_prediction_ratio"] < 0.95
        and agg["branch_better_runs"] >= 3
        and agg["effect_better_runs"] >= 3
        and agg["informative_pairs"] >= 20
    )
    return {
        "pass": passed,
        "reason": (
            "PAIRED_SELF_COUNTERFACTUAL_VALIDATED"
            if passed else "PAIRED_SELF_COUNTERFACTUAL_NOT_VALIDATED"
        ),
        **agg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()

    design = [one_seed(seed, args.steps) for seed in DESIGN_SEEDS]
    blind = [one_seed(seed, args.steps) for seed in BLIND_SEEDS]

    report = {
        "schema": "digital-mind.self-counterfactual-benchmark.v1",
        "claim": "paired simulator intervention only; no phenomenal consciousness claim",
        "intervention_fields": list(FIELDS),
        "design_seeds": list(DESIGN_SEEDS),
        "blind_seeds": list(BLIND_SEEDS),
        "design_runs": design,
        "design_verdict": verdict(design),
        "blind_runs": blind,
        "blind_verdict": verdict(blind),
        "runtime_influence_enabled": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "design": report["design_verdict"],
        "blind": report["blind_verdict"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

"""Blind validation for the operational Predictive Digital Self."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from digital_mind_core.kernel import KernelConfig, UnifiedMind


DESIGN_SEEDS = (51, 52, 53, 54)
BLIND_SEEDS = (61, 62, 63, 64)


def one(seed, steps):
    mind = UnifiedMind(KernelConfig(seed=seed))
    mind.run(steps)
    report = mind.report()["predictive_self"]
    prediction = report["prediction_mse"]
    baseline = report["persistence_baseline_mse"]
    improvement = None if prediction is None or baseline is None else baseline - prediction
    ratio = None if baseline in (None, 0.0) or prediction is None else prediction / baseline
    return {
        "seed": seed,
        "steps": steps,
        "prediction_mse": prediction,
        "persistence_baseline_mse": baseline,
        "absolute_improvement": improvement,
        "mse_ratio": ratio,
        "evaluation_count": report["evaluation_count"],
        "mode": report["mode"],
    }


def aggregate(rows):
    valid = [
        x for x in rows
        if x["absolute_improvement"] is not None
        and x["persistence_baseline_mse"] is not None
    ]
    return {
        "runs": len(rows),
        "valid_runs": len(valid),
        "positive_runs": sum(x["absolute_improvement"] > 0 for x in valid),
        "mean_prediction_mse": (
            sum(x["prediction_mse"] for x in valid) / len(valid)
            if valid else None
        ),
        "mean_persistence_mse": (
            sum(x["persistence_baseline_mse"] for x in valid) / len(valid)
            if valid else None
        ),
        "mean_ratio": (
            sum(x["mse_ratio"] for x in valid) / len(valid)
            if valid else None
        ),
    }


def verdict(rows):
    agg = aggregate(rows)
    passed = (
        agg["valid_runs"] == len(rows)
        and agg["positive_runs"] >= 3
        and agg["mean_ratio"] is not None
        and agg["mean_ratio"] < 0.98
        and all(x["evaluation_count"] >= 256 for x in rows)
    )
    return {
        "pass": passed,
        "reason": (
            "BLIND_SELF_PREDICTION_BEATS_PERSISTENCE"
            if passed else "BLIND_SELF_PREDICTION_NOT_VALIDATED"
        ),
        **agg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    design = [one(seed, args.steps) for seed in DESIGN_SEEDS]
    design_verdict = verdict(design)

    # Blind seeds are always reported, but never used to tune the model here.
    blind = [one(seed, args.steps) for seed in BLIND_SEEDS]
    blind_verdict = verdict(blind)

    report = {
        "schema": "digital-mind.predictive-self-benchmark.v1",
        "claim": "operational next-self-state prediction only; no phenomenal or causal-self claim",
        "steps_per_seed": args.steps,
        "design_seeds": list(DESIGN_SEEDS),
        "blind_seeds": list(BLIND_SEEDS),
        "design_runs": design,
        "design_verdict": design_verdict,
        "blind_runs": blind,
        "blind_verdict": blind_verdict,
        "runtime_influence_enabled": False,
        "counterfactual_status": "MODEL_BASED_SHADOW_ONLY",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "design_pass": design_verdict["pass"],
        "blind_pass": blind_verdict["pass"],
        "blind_mean_ratio": blind_verdict["mean_ratio"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

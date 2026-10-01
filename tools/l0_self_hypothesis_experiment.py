"""Blind test of the kernel's first self-selected L0 research hypothesis.

The production model is never modified. A family of bounded shadow routers is
evaluated on identical transitions. Hyperparameters are selected using design
seeds only and frozen before untouched blind seeds are examined.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from digital_mind_core.kernel import KernelConfig, UnifiedMind
from digital_mind_core.l0_research_candidate import DriftAwareCorrectionRouter


DESIGN_SEEDS = (111, 112, 113, 114)
BLIND_SEEDS = (121, 122, 123, 124)


def validate_hypothesis(document):
    if document.get("candidate_mode") != "SHADOW_ONLY":
        raise ValueError("hypothesis candidate must be shadow-only")
    if document.get("internet_evidence_is_executable") is not False:
        raise ValueError("internet evidence must not be executable")
    deficit = document.get("deficit", {})
    if deficit.get("deficit_id") != "l0_correction_stability":
        raise ValueError("first experiment only supports the selected L0 stability deficit")
    mechanisms = " ".join(
        item.get("mechanism", "")
        for item in document.get("selected_mechanisms", [])
        if isinstance(item, dict)
    ).lower()
    required = {
        "drift": "drift" in mechanisms,
        "retire_or_downweight": ("retire" in mechanisms or "downweight" in mechanisms),
        "condition_authority": ("condition" in mechanisms or "authority" in mechanisms),
    }
    if not all(required.values()):
        raise ValueError(f"selected evidence does not support candidate structure: {required}")
    return required


def configs():
    for threshold, decay, warmup, margin in itertools.product(
        (0.10, 0.50),
        (0.95, 0.99),
        (8, 16),
        (0.0, 0.005),
    ):
        yield {
            "drift_threshold": threshold,
            "decay": decay,
            "warmup": warmup,
            "margin_ratio": margin,
        }


def key(config):
    return (
        config["drift_threshold"],
        config["decay"],
        config["warmup"],
        config["margin_ratio"],
    )


def run_seed(seed, steps, candidate_configs, evaluation_start=128):
    mind = UnifiedMind(KernelConfig(seed=seed))
    routers = {
        key(config): DriftAwareCorrectionRouter(**config)
        for config in candidate_configs
    }
    metrics = {
        k: {
            "candidate_loss": 0.0,
            "runtime_loss": 0.0,
            "count": 0,
            "active_steps": 0,
        }
        for k in routers
    }

    for _ in range(steps):
        drift_score = float(mind.model.fast_gate)
        shadow_weight = float(mind.model.shadow_weight)
        authority = {
            k: router.select(
                drift_score=drift_score,
                shadow_weight=shadow_weight,
            )
            for k, router in routers.items()
        }

        mind.step()

        baseline_error = mind.model.last_baseline_error
        correction = mind.model.last_correction
        runtime_error = mind.model.last_runtime_error
        if baseline_error is None or correction is None or runtime_error is None:
            raise RuntimeError("kernel did not expose transition observability")

        for k, router in routers.items():
            router.observe(
                baseline_error=baseline_error,
                correction=correction,
                shadow_weight=shadow_weight,
                drift_score=drift_score,
            )
            if mind.fast_steps <= evaluation_start:
                continue
            candidate_error = baseline_error - authority[k] * correction
            row = metrics[k]
            row["candidate_loss"] += float(np.mean(candidate_error ** 2))
            row["runtime_loss"] += float(np.mean(runtime_error ** 2))
            row["count"] += 1
            row["active_steps"] += int(authority[k] > 1e-12)

    rows = {}
    for config in candidate_configs:
        k = key(config)
        row = metrics[k]
        count = max(1, row["count"])
        candidate_mse = row["candidate_loss"] / count
        runtime_mse = row["runtime_loss"] / count
        rows[str(k)] = {
            "config": dict(config),
            "candidate_mse": candidate_mse,
            "runtime_mse": runtime_mse,
            "improvement": runtime_mse - candidate_mse,
            "active_steps": row["active_steps"],
            "evaluated_steps": row["count"],
            "router": routers[k].report(),
        }
    return {
        "seed": seed,
        "steps": steps,
        "rows": rows,
        "production_final_l0_weight": float(mind.model.l0_weight),
        "production_final_shadow_weight": float(mind.model.shadow_weight),
    }


def aggregate(seed_runs, candidate_configs):
    result = {}
    for config in candidate_configs:
        name = str(key(config))
        per_seed = [run["rows"][name] for run in seed_runs]
        gains = [row["improvement"] for row in per_seed]
        result[name] = {
            "config": dict(config),
            "mean_improvement": float(np.mean(gains)),
            "positive_seed_count": sum(g > 0 for g in gains),
            "worst_seed_improvement": float(min(gains)),
            "seed_improvements": gains,
            "active_steps": sum(row["active_steps"] for row in per_seed),
            "mean_candidate_mse": float(np.mean([r["candidate_mse"] for r in per_seed])),
            "mean_runtime_mse": float(np.mean([r["runtime_mse"] for r in per_seed])),
        }
    return result


def choose_candidate(design):
    valid = []
    for name, row in design.items():
        if (
            row["active_steps"] >= 32
            and row["mean_improvement"] > 1e-6
            and row["positive_seed_count"] >= 3
            and row["worst_seed_improvement"] >= -5e-5
        ):
            config = row["config"]
            complexity = (
                0.2 * (config["warmup"] / 16.0)
                + 0.1 * (config["decay"] > 0.95)
                + 0.1 * (config["margin_ratio"] > 0.0)
            )
            valid.append((
                row["mean_improvement"],
                -complexity,
                name,
            ))
    return max(valid)[2] if valid else None


def blind_verdict(selected, blind):
    if selected is None:
        return {
            "pass": False,
            "reason": "NO_DESIGN_CONFIGURATION_PASSED",
            "selected": None,
        }
    row = blind[selected]
    passed = (
        row["active_steps"] >= 32
        and row["mean_improvement"] > 1e-6
        and row["positive_seed_count"] >= 3
        and row["worst_seed_improvement"] >= -5e-5
    )
    return {
        "pass": passed,
        "reason": (
            "SELF_SELECTED_L0_HYPOTHESIS_BLIND_PASS"
            if passed else "SELF_SELECTED_L0_HYPOTHESIS_BLIND_FAIL"
        ),
        "selected": selected,
        **row,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=640)
    args = parser.parse_args()

    hypothesis = json.loads(args.hypothesis.read_text(encoding="utf-8"))
    evidence_requirements = validate_hypothesis(hypothesis)
    candidate_configs = list(configs())

    design_runs = [
        run_seed(seed, args.steps, candidate_configs)
        for seed in DESIGN_SEEDS
    ]
    design = aggregate(design_runs, candidate_configs)
    selected = choose_candidate(design)

    # Blind runs happen only after candidate identity has been frozen.
    blind_runs = [
        run_seed(seed, args.steps, candidate_configs)
        for seed in BLIND_SEEDS
    ]
    blind = aggregate(blind_runs, candidate_configs)
    verdict = blind_verdict(selected, blind)

    report = {
        "schema": "digital-mind.self-selected-l0-hypothesis-experiment.v1",
        "hypothesis": hypothesis,
        "evidence_requirements": evidence_requirements,
        "production_runtime_modified": False,
        "design_seeds": list(DESIGN_SEEDS),
        "blind_seeds": list(BLIND_SEEDS),
        "steps_per_seed": args.steps,
        "candidate_count": len(candidate_configs),
        "design_runs": design_runs,
        "design_aggregate": design,
        "selected_on_design_only": selected,
        "blind_runs": blind_runs,
        "blind_aggregate": blind,
        "promotion_verdict": verdict,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected": selected,
        "blind_pass": verdict["pass"],
        "reason": verdict["reason"],
        "mean_improvement": verdict.get("mean_improvement"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

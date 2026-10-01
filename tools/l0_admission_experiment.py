"""Causal/temporal experiment for L0 admission gates.

No runtime gate is modified here. Candidate gates see only PAST shadow-benefit
observations. Their value is measured on the NEXT transition. Candidate choice
uses design seeds; blind seeds are never used to choose the candidate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from digital_mind_core.kernel import KernelConfig, UnifiedMind


DESIGN_SEEDS = (11, 12, 13, 14)
BLIND_SEEDS = (21, 22, 23, 24)


def blocks(history, width=16, count=4):
    need = width * count
    if len(history) < need:
        return None
    tail = history[-need:]
    return [
        sum(tail[i * width:(i + 1) * width]) / width
        for i in range(count)
    ]


def current_lcb64(history):
    b = blocks(history, 16, 4)
    if b is None:
        return False
    mean = statistics.mean(b)
    if len(set(b)) == 1:
        return mean > 0
    stdev = statistics.stdev(b)
    return mean - 2.0 * stdev / (len(b) ** 0.5) > 0


def split48_16(history):
    if len(history) < 64:
        return False
    tail = history[-64:]
    return statistics.mean(tail[:48]) > 0 and statistics.mean(tail[48:]) > 0


def two_block32(history):
    if len(history) < 32:
        return False
    tail = history[-32:]
    return statistics.mean(tail[:16]) > 0 and statistics.mean(tail[16:]) > 0


def majority4(history):
    b = blocks(history, 16, 4)
    if b is None:
        return False
    return statistics.mean(b) > 0 and sum(x > 0 for x in b) >= 3


GATES = {
    "current_lcb64": current_lcb64,
    "split48_16": split48_16,
    "two_block32": two_block32,
    "majority4": majority4,
}


def one_seed(seed, steps):
    mind = UnifiedMind(KernelConfig(seed=seed))
    history = []
    rows = {name: {"gain": 0.0, "active": 0, "negative": 0}
            for name in GATES}
    shadow_nonzero = 0

    for _ in range(steps):
        decisions = {name: fn(history) for name, fn in GATES.items()}
        shadow_weight = mind.model.shadow_weight
        if shadow_weight > 1e-12:
            shadow_nonzero += 1
        mind.step()
        benefit = (
            float(mind.model.shadow_benefits[-1])
            if mind.model.shadow_benefits else 0.0
        )
        for name, active in decisions.items():
            if active and shadow_weight > 1e-12:
                rows[name]["active"] += 1
                rows[name]["gain"] += benefit
                rows[name]["negative"] += int(benefit < 0)
        history.append(benefit)

    for name in rows:
        active = rows[name]["active"]
        rows[name]["mean_gain_when_active"] = (
            rows[name]["gain"] / active if active else 0.0
        )
    return {
        "seed": seed,
        "steps": steps,
        "shadow_nonzero_steps": shadow_nonzero,
        "final_shadow_weight": float(mind.model.shadow_weight),
        "current_runtime_l0_weight": float(mind.model.l0_weight),
        "gates": rows,
    }


def aggregate(runs):
    result = {}
    for name in GATES:
        seed_gains = [r["gates"][name]["gain"] for r in runs]
        active = sum(r["gates"][name]["active"] for r in runs)
        gain = sum(seed_gains)
        result[name] = {
            "total_gain": gain,
            "active_steps": active,
            "mean_gain_per_active_step": gain / active if active else 0.0,
            "positive_seed_count": sum(x > 0 for x in seed_gains),
            "nonnegative_seed_count": sum(x >= 0 for x in seed_gains),
            "worst_seed_gain": min(seed_gains),
            "seed_gains": seed_gains,
        }
    return result


def choose(design):
    baseline = design["current_lcb64"]
    candidates = []
    complexity = {
        "current_lcb64": 3,
        "split48_16": 2,
        "two_block32": 2,
        "majority4": 2,
    }
    for name, row in design.items():
        if name == "current_lcb64":
            continue
        improvement = row["total_gain"] - baseline["total_gain"]
        if (
            row["active_steps"] >= 32
            and improvement > 1e-4
            and row["positive_seed_count"] >= 3
        ):
            candidates.append((
                row["total_gain"],
                -complexity[name],
                name,
            ))
    if not candidates:
        return None
    return max(candidates)[2]


def blind_verdict(selected, blind, design):
    if selected is None:
        return {
            "pass": False,
            "reason": "NO_DESIGN_CANDIDATE_BEAT_CURRENT_GATE",
        }
    row = blind[selected]
    base = blind["current_lcb64"]
    improvement = row["total_gain"] - base["total_gain"]
    passed = (
        row["active_steps"] >= 32
        and improvement > 1e-4
        and row["positive_seed_count"] >= 3
        and row["worst_seed_gain"] >= -1e-4
    )
    return {
        "pass": passed,
        "selected": selected,
        "blind_total_gain": row["total_gain"],
        "blind_baseline_gain": base["total_gain"],
        "blind_improvement": improvement,
        "positive_seed_count": row["positive_seed_count"],
        "worst_seed_gain": row["worst_seed_gain"],
        "active_steps": row["active_steps"],
        "reason": "BLIND_GAIN_GATE_PASSED" if passed
                  else "BLIND_GAIN_GATE_FAILED",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    design_runs = [one_seed(seed, args.steps) for seed in DESIGN_SEEDS]
    design = aggregate(design_runs)
    selected = choose(design)

    # Only after selection is frozen do blind seeds run.
    blind_runs = [one_seed(seed, args.steps) for seed in BLIND_SEEDS]
    blind = aggregate(blind_runs)
    verdict = blind_verdict(selected, blind, design)

    report = {
        "schema": "digital-mind.l0-admission-experiment.v1",
        "principle": "past-only gate decision; next-transition benefit; design/blind separation",
        "steps_per_seed": args.steps,
        "design_seeds": list(DESIGN_SEEDS),
        "blind_seeds": list(BLIND_SEEDS),
        "design_runs": design_runs,
        "design_aggregate": design,
        "selected_on_design_only": selected,
        "blind_runs": blind_runs,
        "blind_aggregate": blind,
        "promotion_verdict": verdict,
        "runtime_changed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected": selected,
        "promotion_pass": verdict["pass"],
        "reason": verdict["reason"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

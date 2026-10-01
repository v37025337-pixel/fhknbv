"""Train the current kernel, then evaluate capabilities on independent tasks.

This is deliberately not a marketing demo: failures are recorded in the JSON
report instead of being hidden.  Phase order is fixed:
  1. closed-loop learning + mechanism development
  2. independent task suite
  3. negative capability probes / limitations
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import traceback

import numpy as np

from digital_mind_core.cognition import CognitiveCore
from digital_mind_core.github_sidecar import GitHubRepositorySidecar
from digital_mind_core.kernel import (
    Intervention,
    KernelConfig,
    UnifiedMind,
    planning_benchmark,
)
from digital_mind_core.persistence import load_replay, save_replay, state_fingerprint
from digital_mind_core.tool_module import MindModule


OPTIONS = {
    "max_numeric_cost": 3,
    "max_boolean_cost": 3,
    "max_numeric_candidates": 256,
    "max_boolean_candidates": 64,
}


def safe_number(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def run_case(name, fn):
    try:
        details = fn()
        if isinstance(details, bool):
            passed, details = details, {}
        else:
            details = details or {}
            passed = bool(details.pop("pass", True))
        return {"name": name, "pass": passed, "details": details}
    except Exception as exc:
        return {
            "name": name,
            "pass": False,
            "details": {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            },
        }


def train_one(seed, steps):
    mind = UnifiedMind(KernelConfig(seed=seed))
    mind.run(steps)
    report = mind.report()
    evolution = report["mechanism_evolution"]
    interventions = report.get("interventions", [])
    prediction = report.get("prediction", {})
    l0 = report.get("l0", {})
    questions = report.get("questions", {})
    return mind, {
        "seed": seed,
        "steps": steps,
        "evolution_active": bool(evolution.get("active")),
        "evolution_attempts": int(evolution.get("attempts", 0)),
        "evolution_calls": int(evolution.get("l0_calls", 0)),
        "attempt_limit": evolution.get("attempt_limit"),
        "model_predicted_cost_saved": safe_number(
            evolution.get("model_predicted_cost_saved", 0.0)
        ),
        "prediction_mean_loss": safe_number(prediction.get("mean_loss")),
        "l0_mean_innovation_weight": safe_number(
            l0.get("mean_innovation_weight")
        ),
        "l0_final_weight": safe_number(mind.model.l0_weight),
        "l0_shadow_weight": safe_number(mind.model.shadow_weight),
        "l0_model_version": int(
            mind.model.runtime.models["world_innovation"].version
        ),
        "l0_tail_baseline_loss": safe_number(
            np.mean(mind.model.baseline_losses[-64:])
            if mind.model.baseline_losses else None
        ),
        "l0_tail_ensemble_loss": safe_number(
            np.mean(mind.model.ensemble_losses[-64:])
            if mind.model.ensemble_losses else None
        ),
        "questions_created": int(questions.get("created", 0)),
        "questions_predictively_explained": int(
            questions.get("predictively_explained", 0)
        ),
        "supported_interventions": sum(
            item.get("status") == "supported" for item in interventions
        ),
        "cognitive_cycles": int(report.get("cognition", {}).get("cycles", 0)),
        "plan_ticks": int(report.get("plan_ticks", getattr(mind, "plan_ticks", 0))),
        "mechanisms": sorted(mind.cognition.mechanism_info().keys()),
    }


def phase_training(steps=192, seeds=(1, 2, 3)):
    runs = []
    minds = []
    for seed in seeds:
        mind, metrics = train_one(seed, steps)
        minds.append(mind)
        runs.append(metrics)

    # Development challenge after environmental learning: create a new
    # independently admitted scalar mechanism from examples + held-out cases.
    core = minds[0].cognition
    development = core.synthesize_mechanism(
        {
            "name": "development_minimum",
            "args": ["a", "b"],
            "constants": [0, 1],
            "examples": [
                {"inputs": {"a": 1.0, "b": 3.0}, "output": 1.0},
                {"inputs": {"a": 4.0, "b": 2.0}, "output": 2.0},
                {"inputs": {"a": -2.0, "b": 7.0}, "output": -2.0},
                {"inputs": {"a": 8.0, "b": -1.0}, "output": -1.0},
            ],
        },
        validation_examples=[
            {"inputs": {"a": 5.0, "b": 3.0}, "output": 3.0},
            {"inputs": {"a": 0.0, "b": 4.0}, "output": 0.0},
        ],
        holdout_examples=[
            {"inputs": {"a": 100.0, "b": -9.0}, "output": -9.0},
            {"inputs": {"a": -7.0, "b": -2.0}, "output": -7.0},
        ],
        max_rounds=4,
        engine_options=OPTIONS,
    )
    hidden_value = None
    if development.get("status") == "FOUND":
        hidden_value = core.run_mechanism(
            "development_minimum", a=11.0, b=-13.0
        )

    return minds, {
        "runs": runs,
        "all_closed_loop_mechanisms_active": all(
            row["evolution_active"] for row in runs
        ),
        "development": {
            "status": development.get("status"),
            "admission": development.get("admission"),
            "execution": development.get("execution"),
            "cost": development.get("cost"),
            "hidden_input": {"a": 11.0, "b": -13.0},
            "hidden_output": hidden_value,
            "hidden_pass": hidden_value == -13.0,
        },
    }


def capability_cases(trained_mind):
    cases = []

    def logic():
        result = trained_mind.process({
            "facts": [("edge", "A", "B"), ("edge", "B", "C")],
            "rules": [
                ([("edge", "$x", "$y"), ("edge", "$y", "$z")],
                 ("reachable", "$x", "$z"))
            ],
            "queries": [("reachable", "A", "C")],
        })
        value = result["inferences"][0]["value"]
        return {"pass": value is True, "inference": value}
    cases.append(run_case("symbolic_multihop_logic", logic))

    def path_planning():
        result = trained_mind.process({
            "graph": {
                "A": [("B", 1.0), ("C", 9.0)],
                "B": [("C", 1.0)],
                "C": [],
            },
            "start": "A",
            "goal": "C",
        })
        plan = result.get("plan")
        return {"pass": plan == ["A", "B", "C"], "plan": plan}
    cases.append(run_case("graph_path_planning", path_planning))

    def generated_mechanism_hidden():
        value = trained_mind.cognition.run_mechanism(
            "development_minimum", a=42.0, b=-4.0
        )
        return {"pass": value == -4.0, "value": value}
    cases.append(run_case("generated_mechanism_transfer", generated_mechanism_hidden))

    def mechanism_roundtrip():
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mechanisms.json"
            trained_mind.cognition.save_mechanisms(path)
            restored = CognitiveCore()
            restored.load_mechanisms(path)
            value = restored.run_mechanism(
                "development_minimum", a=-21.0, b=7.0
            )
            return {
                "pass": value == -21.0,
                "value": value,
                "restored_admission": restored.mechanism_info(
                    "development_minimum"
                ).get("admission"),
            }
    cases.append(run_case("mechanism_persistence_roundtrip", mechanism_roundtrip))

    def structured_data():
        mind = UnifiedMind(KernelConfig(
            seed=9, l0_learning=False, questions=False
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "states.csv"
            path.write_text(
                "Rank,State,Population\n"
                "1,Alabama,4849377\n"
                "2,Alaska,736732\n",
                encoding="utf-8",
            )
            connection = mind.connect(path)
            result = mind.process({
                "persist_facts": True,
                "queries": [
                    ("dataset_field", "connection_1", "Population", "int"),
                    ("dataset_rows", "connection_1", 2),
                ],
            })
            values = [q["value"] for q in result["inferences"]]
            return {
                "pass": connection.attached and all(values),
                "domain": connection.domain,
                "query_values": values,
            }
    cases.append(run_case("structured_data_to_logic", structured_data))

    def delayed_reward_planner():
        result = planning_benchmark(16)
        better = result["8"]["score"] > result["1"]["score"]
        same_steps = result["8"]["steps"] == result["1"]["steps"]
        return {
            "pass": better and same_steps,
            "horizon_1_score": result["1"]["score"],
            "horizon_8_score": result["8"]["score"],
            "steps_equal": same_steps,
        }
    cases.append(run_case("delayed_reward_planning", delayed_reward_planner))

    def causal_fixed_look():
        exp = Intervention(1, 0, np.ones(2), 0)
        for i in range(48):
            sign = -1 if i % 2 else 1
            exp.observe(sign, 2 * sign + .01 * (i % 3), 48)
        return {
            "pass": exp.status == "supported"
                    and exp.confidence_interval[0] > 0,
            "status": exp.status,
            "effect": exp.effect,
            "confidence_interval": exp.confidence_interval,
        }
    cases.append(run_case("randomized_intervention_fixed_analysis", causal_fixed_look))

    def replay_continuity():
        mind = UnifiedMind(KernelConfig(
            seed=4, l0_learning=False, questions=False
        ))
        mind.run(96)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            save_replay(mind, path)
            restored = load_replay(path)
            same_before = state_fingerprint(restored) == state_fingerprint(mind)
            mind.run(8)
            restored.run(8)
            same_after = state_fingerprint(restored) == state_fingerprint(mind)
            return {
                "pass": same_before and same_after,
                "same_before": same_before,
                "same_after_8_more_steps": same_after,
            }
    cases.append(run_case("deterministic_replay_continuity", replay_continuity))

    def module_interface():
        module = MindModule(KernelConfig(
            seed=2, l0_learning=False, questions=False
        ))
        status = module.execute({"op": "status"})
        result = module.execute({
            "op": "reason",
            "task": {
                "facts": [("p", "a")],
                "rules": [([("p", "$x")], ("q", "$x"))],
                "queries": [("q", "a")],
            },
            "include_state": False,
        })
        return {
            "pass": result["inferences"][0]["value"] is True
                    and isinstance(status, dict),
            "inference": result["inferences"][0]["value"],
        }
    cases.append(run_case("stateful_json_module_interface", module_interface))

    def github_binding():
        sha = os.environ.get("GITHUB_SHA") or "0" * 40
        sidecar = GitHubRepositorySidecar(bound_head=sha)
        result = sidecar.verify(
            lambda repo, branch: {
                "sha": sha,
                "commit": {"message": "CI current head"},
                "html_url": f"https://github.com/{repo}/commit/{sha}",
            }
        )
        return {
            "pass": result["status"] == "SYNCED",
            "status": result["status"],
            "head": result["remote_head"],
        }
    cases.append(run_case("github_sidecar_binding", github_binding))

    return cases


def negative_capability_probes():
    probes = []

    core = CognitiveCore()

    def probe(name, supported, evidence):
        probes.append({
            "capability": name,
            "supported": bool(supported),
            "evidence": evidence,
        })

    probe(
        "general_algorithm_synthesis",
        hasattr(core, "synthesize_algorithm"),
        "CognitiveCore.synthesize_algorithm attribute",
    )
    probe(
        "arbitrary_self_rewrite",
        hasattr(core, "rewrite_self"),
        "CognitiveCore.rewrite_self attribute",
    )
    probe(
        "native_unrestricted_internet",
        hasattr(core, "internet_fetch"),
        "CognitiveCore.internet_fetch attribute",
    )
    probe(
        "native_image_perception",
        hasattr(core, "perceive_image"),
        "CognitiveCore.perceive_image attribute",
    )

    try:
        result = core.synthesize_mechanism({
            "name": "reverse_text",
            "args": ["text"],
            "examples": [
                {"inputs": {"text": "abc"}, "output": "cba"},
                {"inputs": {"text": "xy"}, "output": "yx"},
            ],
        })
        string_supported = result.get("status") == "FOUND"
        evidence = (
            f"status={result.get('status')}; "
            f"explanation={result.get('explanation')}"
        )
    except Exception as exc:
        string_supported = False
        evidence = f"{type(exc).__name__}: {exc}"
    probe(
        "string_program_synthesis",
        string_supported,
        evidence,
    )

    return probes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=192)
    args = parser.parse_args()

    minds, training = phase_training(steps=args.steps)
    tasks = capability_cases(minds[0])
    limitations = negative_capability_probes()

    passed = sum(item["pass"] for item in tasks)
    report = {
        "schema": "digital-mind.training-capability-report.v1",
        "phase_order": [
            "learning_and_development",
            "independent_tasks",
            "negative_capability_probes",
        ],
        "training": training,
        "tasks": {
            "passed": passed,
            "total": len(tasks),
            "results": tasks,
        },
        "negative_capability_probes": limitations,
        "summary": {
            "training_success": (
                training["all_closed_loop_mechanisms_active"]
                and training["development"]["hidden_pass"]
            ),
            "task_pass_rate": passed / len(tasks) if tasks else 0.0,
            "supported_negative_probe_count": sum(
                item["supported"] for item in limitations
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True))
    if not report["summary"]["training_success"] or passed != len(tasks):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

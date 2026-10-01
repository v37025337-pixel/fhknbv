"""Autonomous development target selection.

The manager chooses what to work on from measured deficits and its own failed
attempt history.  The host may satisfy capability requests, but does not choose
the target or research query.
"""
from __future__ import annotations

import copy
import re


def _tokens(text):
    return set(re.findall(r"[a-z0-9_]+", str(text).lower()))


def choose_target(state):
    deficits = state.get("deficits") or []
    if not deficits:
        raise ValueError("no measured deficits")
    ranked = []
    for item in deficits:
        severity = float(item["severity"])
        failed = int(item.get("failed_attempts", 0))
        passed = int(item.get("passed_attempts", 0))
        # Repeated failure lowers immediate priority enough to explore another
        # major deficit, but never erases the deficit.
        score = severity - 0.10 * failed - 0.20 * passed
        ranked.append((score, severity, item["deficit_id"], copy.deepcopy(item)))
    ranked.sort(reverse=True)
    score, severity, _, target = ranked[0]
    target["development_score"] = score
    return target, [
        {"deficit_id": row[3]["deficit_id"], "score": row[0],
         "severity": row[1], "failed_attempts": row[3].get("failed_attempts",0)}
        for row in ranked
    ]


def build_research_request(state, target):
    previous = [
        item for item in state.get("attempt_history", [])
        if item.get("deficit_id") == target["deficit_id"]
    ]
    avoided = []
    for attempt in previous:
        avoided.extend(attempt.get("mechanisms", []))

    templates = {
        "self_counterfactual_identifiability": (
            "sequential causal inference adaptive agents endogenous policy "
            "randomized interventions propensity logging doubly robust "
            "counterfactual effect uncertainty calibration"
        ),
        "l0_correction_stability": (
            "online residual correction concept drift real versus virtual drift "
            "delayed feedback prequential calibration residual normalization "
            "future holdout stability"
        ),
        "algorithm_genesis": (
            "typed program synthesis sequences lists map filter fold CEGIS "
            "bounded DSL heldout generalization resource limits"
        ),
        "string_program_synthesis": (
            "typed string transformation synthesis from examples FlashFill "
            "bounded DSL CEGIS hidden examples"
        ),
    }
    query = templates[target["deficit_id"]]
    if avoided:
        # Search for a genuinely different mechanism, not parameter retuning.
        query += " alternatives excluding " + " ".join(
            sorted(_tokens(" ".join(avoided)))
        )
    return {
        "capability":"google_public_search",
        "provider_preference":"google",
        "query":query,
        "purpose":target["question"],
        "avoid_repeating_mechanisms":avoided,
        "result_limit":6,
    }


def step1(state):
    target, ranking = choose_target(state)
    request = build_research_request(state, target)
    return {
        "schema":"digital-mind.autodev-step1.v1",
        "action":"RESEARCH",
        "selected_target":target,
        "ranking":ranking,
        "capability_request":request,
        "chosen_by":"measured severity minus own failed/passed attempt penalties",
        "host_chose_target":False,
    }


def step2(step1_document, evidence_document):
    target = step1_document.get("selected_target") or {}
    deficit_id = target.get("deficit_id")
    if evidence_document.get("target_deficit") != deficit_id:
        raise ValueError("evidence target does not match autonomously selected deficit")
    query = (step1_document.get("capability_request") or {}).get("query", "")
    query_tokens = _tokens(query + " " + target.get("question", ""))

    candidates = []
    for source in evidence_document.get("sources", []):
        source_id = source.get("source_id")
        for mechanism in source.get("mechanisms", []):
            tokens = _tokens(mechanism)
            overlap = len(tokens & query_tokens)
            causal_bonus = sum(
                word in tokens for word in (
                    "propensity","randomized","randomized","positivity",
                    "counterfactual","uncertainty","causal","adaptive"
                )
            )
            score = 2.0 * overlap + 0.75 * causal_bonus
            candidates.append({
                "mechanism": mechanism,
                "source_id": source_id,
                "source_title": source.get("title"),
                "url": source.get("url"),
                "score": score,
            })
    candidates.sort(key=lambda x: (-x["score"], x["mechanism"]))

    selected = []
    used_sources = set()
    for item in candidates:
        if item["source_id"] not in used_sources:
            selected.append(copy.deepcopy(item))
            used_sources.add(item["source_id"])
        if len(selected) >= 3:
            break
    if len(selected) < 3:
        for item in candidates:
            if item not in selected:
                selected.append(copy.deepcopy(item))
            if len(selected) >= 3:
                break
    if not selected:
        raise ValueError("no research mechanisms available")

    return {
        "schema":"digital-mind.autodev-step2.v1",
        "action":"HYPOTHESIS",
        "selected_target":copy.deepcopy(target),
        "selected_mechanisms":selected,
        "hypothesis":(
            "For " + deficit_id + ", combine " +
            "; ".join(item["mechanism"] for item in selected) +
            " and test the resulting counterfactual estimator against the current "
            "observational PredictiveSelf counterfactual baseline."
        ),
        "required_evaluation":{
            "design_then_blind":True,
            "production_influence_before_pass":False,
            "effect_uncertainty_required":True,
        },
        "host_chose_mechanisms":False,
    }

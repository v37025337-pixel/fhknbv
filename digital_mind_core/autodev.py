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

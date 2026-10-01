"""Bounded hypothesis genesis from measured deficits and external evidence."""
from __future__ import annotations

from collections import Counter
import copy
import re


_STOP = {
    "the","a","an","and","or","of","to","in","on","by","for","with","from",
    "is","are","be","before","after","using","without","under","every","only",
}


def _tokens(text):
    return {
        token for token in re.findall(r"[a-z0-9_]+", str(text).lower())
        if len(token) > 2 and token not in _STOP
    }


def choose_primary_deficit(self_assessment):
    deficits = self_assessment.get("deficits")
    if not isinstance(deficits, list) or not deficits:
        raise ValueError("self assessment has no measured deficits")
    valid = [
        item for item in deficits
        if isinstance(item, dict)
        and isinstance(item.get("deficit_id"), str)
        and isinstance(item.get("severity"), (int, float))
    ]
    if not valid:
        raise ValueError("self assessment has no valid deficits")
    return copy.deepcopy(max(
        valid,
        key=lambda item: (float(item["severity"]), item["deficit_id"]),
    ))


def synthesize_hypothesis(self_assessment, *, mechanism_count=3):
    primary = choose_primary_deficit(self_assessment)
    deficit_id = primary["deficit_id"]

    evidence_doc = self_assessment.get("external_evidence") or {}
    sources = evidence_doc.get("sources") or []
    relevant = [
        item for item in sources
        if isinstance(item, dict) and item.get("deficit_id") == deficit_id
    ]
    if not relevant:
        raise ValueError("no external evidence for primary deficit")

    plan = next(
        (item for item in self_assessment.get("research_plan", [])
         if isinstance(item, dict) and item.get("deficit_id") == deficit_id),
        {},
    )
    target_text = " ".join([
        primary.get("question", ""),
        primary.get("success_test", ""),
        plan.get("objective", ""),
    ])
    target_tokens = _tokens(target_text)

    candidates = []
    frequency = Counter()
    for source in relevant:
        for mechanism in source.get("candidate_mechanisms", []):
            if isinstance(mechanism, str):
                frequency[mechanism] += 1

    for source in relevant:
        for mechanism in source.get("candidate_mechanisms", []):
            if not isinstance(mechanism, str):
                continue
            overlap = len(_tokens(mechanism) & target_tokens)
            bounded_bonus = sum(
                term in mechanism.lower()
                for term in ("bounded", "pre-update", "prequential", "downweight", "estimate")
            )
            score = 2.0 * overlap + 1.5 * frequency[mechanism] + 0.5 * bounded_bonus
            candidates.append({
                "mechanism": mechanism,
                "score": float(score),
                "source_id": source.get("source_id"),
                "source_title": source.get("title"),
                "url": source.get("url"),
            })

    candidates.sort(
        key=lambda item: (-item["score"], item["mechanism"], item["source_id"] or "")
    )

    selected = []
    seen_sources = set()
    for item in candidates:
        # Prefer source diversity before taking multiple ideas from one paper.
        if item["source_id"] not in seen_sources or len(selected) >= len(relevant):
            selected.append(item)
            seen_sources.add(item["source_id"])
        if len(selected) >= int(mechanism_count):
            break
    if len(selected) < int(mechanism_count):
        for item in candidates:
            if item not in selected:
                selected.append(item)
            if len(selected) >= int(mechanism_count):
                break

    mechanisms = [item["mechanism"] for item in selected]
    hypothesis = (
        f"For measured deficit '{deficit_id}', combining "
        + "; ".join(mechanisms)
        + " should improve future validated performance while preserving the "
          "existing conservative admission boundary."
    )

    return {
        "schema": "digital-mind.self-hypothesis.v1",
        "selected_by": "highest measured severity, then evidence-ranked bounded mechanisms",
        "deficit": primary,
        "selected_mechanisms": selected,
        "hypothesis": hypothesis,
        "baseline": "current runtime unchanged",
        "candidate_mode": "SHADOW_ONLY",
        "acceptance_test": primary.get("success_test"),
        "rejection_rule": (
            "Reject if candidate fails design or untouched blind seeds, "
            "causes any regression failure, or needs current-target leakage."
        ),
        "internet_evidence_is_executable": False,
    }

"""
v0.15 self-development selector.

This module is intentionally a selector, not a self-modifying executor.
It scores development candidates from observed benchmark deficits.
External mutation/deployment remains permission-gated.
"""

from collections import defaultdict

def choose(deficits, candidate_meta):
    scored = []
    for key, d in deficits.items():
        meta = candidate_meta[key]
        blocked = d.get("blocked", 0)
        misrouted = d.get("misrouted", 0)
        families = len(set(d.get("families", [])))
        score = (
            2.0 * blocked
            + 1.25 * misrouted
            + 0.9 * families
            + 1.8 * meta["autonomy_gain"]
            + 1.5 * meta["upstream_blocker"]
            - 0.8 * meta["complexity"]
            - 0.7 * meta["risk"]
        )
        scored.append((score, key))
    return max(scored)[1] if scored else None

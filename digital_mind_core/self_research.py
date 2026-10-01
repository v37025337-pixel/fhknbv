"""Measured self-audit and host-mediated research planning.

The kernel may diagnose deficits from its own reports and formulate research
requests. External sources are evidence, not executable instructions. Any code
change still requires local synthesis/implementation plus tests and blind
validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from urllib.parse import urlparse


@dataclass(frozen=True)
class Deficit:
    deficit_id: str
    severity: float
    evidence: tuple[str, ...]
    question: str
    success_test: str

    def document(self):
        return asdict(self)


RESEARCH_TEMPLATES = {
    "l0_correction_stability": {
        "query": (
            "online residual model correction under concept drift using "
            "prequential validation adaptive gating mixture of experts "
            "and delayed future evaluation"
        ),
        "objective": (
            "Find rigorous mechanisms that improve a residual correction "
            "without using the current target to decide current authority. "
            "Prioritize online validation, drift adaptation, expert weighting, "
            "and stability under non-stationarity."
        ),
    },
    "self_counterfactual_identifiability": {
        "query": (
            "sequential causal inference counterfactual prediction for adaptive "
            "agents endogenous policies off-policy evaluation paired interventions"
        ),
        "objective": (
            "Find methods that distinguish predictive self-correlation from "
            "identified effects of changing an internal decision. Prioritize "
            "sequential ignorability, overlap, sensitivity analysis, randomized "
            "or paired interventions, and uncertainty."
        ),
    },
    "algorithm_genesis": {
        "query": (
            "program synthesis from examples typed DSL grammar induction CEGIS "
            "component based synthesis algorithms loops lists trees"
        ),
        "objective": (
            "Find bounded, testable ways to expand scalar expression synthesis "
            "to richer algorithms while preserving typing, sandboxing, "
            "counterexample refinement, and held-out generalization."
        ),
    },
    "string_program_synthesis": {
        "query": (
            "program synthesis from examples string transformations typed DSL "
            "FlashFill PROSE counterexample guided synthesis"
        ),
        "objective": (
            "Find constrained techniques for learning string transformations "
            "from examples without arbitrary code execution."
        ),
    },
}


def _dig(document, *path, default=None):
    value = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def assess(training, l0_admission, predictive_self, self_counterfactual,
           capabilities):
    deficits = []

    runs = _dig(training, "training", "runs", default=[]) or []
    final_weights = [
        row.get("l0_final_weight")
        for row in runs if isinstance(row, dict)
    ]
    if runs and all(value == 0.0 for value in final_weights):
        deficits.append(Deficit(
            "l0_correction_stability",
            1.0,
            (
                "L0 trained on every step but final authority remained zero "
                "across all measured training runs.",
                "Blind admission experiment found no softer gate that beat "
                "the current conservative gate.",
            ),
            "Why is the learned residual correction unstable on future "
            "transitions despite nonzero shadow authority?",
            "A new correction mechanism must reduce prequential MSE on "
            "unseen seeds without weakening the existing admission gate.",
        ))

    cf_design = _dig(self_counterfactual, "design_verdict", "pass")
    cf_blind = _dig(self_counterfactual, "blind_verdict", "pass")
    if cf_design is False or cf_blind is False or cf_design != cf_blind:
        deficits.append(Deficit(
            "self_counterfactual_identifiability",
            0.95,
            (
                f"paired self-counterfactual design_pass={cf_design}",
                f"paired self-counterfactual blind_pass={cf_blind}",
                "Predictive self-state model can outperform persistence while "
                "interventional effect estimates are less stable.",
            ),
            "Which self-counterfactual effects are identifiable under an "
            "adaptive endogenous policy, and with what uncertainty?",
            "Paired interventions must pass both design and untouched blind "
            "seeds with calibrated effect uncertainty before affecting policy.",
        ))

    if not bool(capabilities.get("general_algorithm_synthesis")):
        deficits.append(Deficit(
            "algorithm_genesis",
            0.85,
            (
                "Current CognitiveCore has no general algorithm-synthesis entry point.",
                "Existing mechanism synthesis is a bounded scalar-expression DSL.",
            ),
            "How can the bounded synthesis grammar expand to typed algorithms "
            "over sequences and structured data without arbitrary execution?",
            "At least three unseen algorithm families must pass holdout/CEGIS "
            "and full regression with bounded resource use.",
        ))

    negative = _dig(training, "negative_capability_probes", default=[]) or []
    string_probe = next(
        (x for x in negative
         if isinstance(x, dict)
         and x.get("capability") == "string_program_synthesis"),
        None,
    )
    if string_probe is not None and not string_probe.get("supported", False):
        deficits.append(Deficit(
            "string_program_synthesis",
            0.65,
            ("String reversal synthesis was not supported in the capability probe.",),
            "What restricted typed string DSL can be synthesized from examples "
            "without adding arbitrary Python execution?",
            "String transforms must generalize on hidden examples and execute "
            "only through a bounded interpreter.",
        ))

    deficits.sort(key=lambda d: (-d.severity, d.deficit_id))
    return deficits


def build_research_plan(deficits, top_k=3):
    requests = []
    for deficit in list(deficits)[:int(top_k)]:
        template = RESEARCH_TEMPLATES.get(deficit.deficit_id)
        if template is None:
            continue
        requests.append({
            "deficit_id": deficit.deficit_id,
            "severity": deficit.severity,
            "question": deficit.question,
            "query": template["query"],
            "objective": template["objective"],
            "success_test": deficit.success_test,
        })
    return requests


def validate_evidence(document, allowed_deficits):
    if not isinstance(document, dict):
        raise ValueError("evidence document must be an object")
    sources = document.get("sources")
    if not isinstance(sources, list):
        raise ValueError("sources must be a list")
    if len(sources) > 64:
        raise ValueError("too many external sources")
    cleaned = []
    allowed = set(allowed_deficits)
    seen = set()
    for item in sources:
        if not isinstance(item, dict):
            raise ValueError("invalid source entry")
        source_id = item.get("source_id")
        deficit_id = item.get("deficit_id")
        url = item.get("url")
        title = item.get("title")
        summary = item.get("summary")
        mechanisms = item.get("candidate_mechanisms", [])
        if (
            not isinstance(source_id, str)
            or source_id in seen
            or deficit_id not in allowed
            or not isinstance(url, str)
            or not isinstance(title, str)
            or not isinstance(summary, str)
            or not isinstance(mechanisms, list)
            or not all(isinstance(x, str) for x in mechanisms)
        ):
            raise ValueError("invalid evidence fields")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("external evidence URL must be HTTPS")
        if len(summary) > 2000 or len(mechanisms) > 16:
            raise ValueError("external evidence entry exceeds bounds")
        seen.add(source_id)
        cleaned.append(copy.deepcopy(item))
    return {"sources": cleaned, "source_count": len(cleaned)}

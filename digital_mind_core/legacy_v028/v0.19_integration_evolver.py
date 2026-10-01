
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import product
from typing import List
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class ArchitectureSpec:
    state_mode: str
    arbitration: str
    reflection: bool
    trace_mode: str

    @property
    def architecture_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:10]


def generate_architecture_space() -> List[ArchitectureSpec]:
    return [
        ArchitectureSpec(*values)
        for values in product(
            ("state", "blackboard"),
            ("attention", "adaptive"),
            (False, True),
            ("compact", "full"),
        )
    ]


_PATCH_TEMPLATE = r'''

# ---- v0.19 generated integration architecture ----
_V019_ORIGINAL_INIT = CognitiveCore.__init__

def _v019_init(self):
    _V019_ORIGINAL_INIT(self)
    shared = {
        "cycle": 0,
        "observations": [],
        "inferences": [],
        "plan": [],
        "decision": None,
        "last_frame": None,
        "trace": [],
        "context": None,
    }
    self.state = shared
    self.blackboard = shared
    self.memory = []
    self.integration_architecture = __SPEC__

CognitiveCore.__init__ = _v019_init


def _v019_trace_event(self, stage, payload):
    event = {
        "cycle": self.state["cycle"],
        "stage": stage,
        "payload": payload,
    }
    self.state["trace"].append(event)
    if __FULL_TRACE__:
        self.memory.append(event)
    else:
        self.memory = self.memory[-31:] + [event]
    return event


def _v019_process(self, task):
    task = dict(task or {})
    self.state["cycle"] += 1
    cycle = self.state["cycle"]
    self.workspace.reset()

    observations = list(task.get("observations", []))
    self.state["observations"] = observations
    self.state["context"] = task.get("context", "default")
    _v019_trace_event(self, "observation", {
        "count": len(observations),
        "items": observations if __FULL_TRACE__ else None,
    })

    self.logic.reset()
    for item in task.get("facts", []):
        if len(item) >= 1:
            self.logic.add_fact(item[0], *item[1:])
    for rule in task.get("rules", []):
        premises, conclusion = rule
        self.logic.add_rule(premises, conclusion)
    self.logic.infer()

    inference_queries = []
    for q in task.get("queries", []):
        inference_queries.append({
            "query": q,
            "value": self.logic.query(q[0], *q[1:]),
        })
    self.state["inferences"] = inference_queries
    _v019_trace_event(self, "inference", inference_queries)

    graph = task.get("graph", {})
    start = task.get("start")
    goal = task.get("goal")
    if start is not None and goal is not None:
        plan, cost = self.thinking.plan_path(
            start, goal, graph,
            blocked_edges=set(map(tuple, task.get("blocked_edges", []))),
        )
    else:
        plan, cost = [], float("inf")
    self.state["plan"] = plan
    _v019_trace_event(self, "planning", {"path": plan, "cost": cost})

    for proposal in task.get("proposals", []):
        self.workspace.publish(
            proposal.get("source", "external"),
            topic=proposal.get("topic", "action"),
            value=proposal.get("value", proposal.get("action")),
            confidence=float(proposal.get("confidence", 0.5)),
            urgency=float(proposal.get("urgency", 0.5)),
            action=proposal.get("action"),
        )

    if len(plan) >= 2:
        self.workspace.publish(
            "thinking",
            topic="action",
            value=plan[1],
            confidence=0.75,
            urgency=0.55,
            action=plan[1],
        )

    for iq in inference_queries:
        mapping = task.get("logic_actions", {})
        key = repr(tuple(iq["query"]))
        action = mapping.get(key)
        if iq["value"] and action is not None:
            self.workspace.publish(
                "logic",
                topic="action",
                value=action,
                confidence=0.8,
                urgency=0.6,
                action=action,
            )

    frame = self.workspace.integrate()
    self.state["last_frame"] = frame
    _v019_trace_event(self, "workspace", {
        "attention": frame.get("attention"),
        "conflicts": frame.get("conflicts", []),
    })

    signals = [s for s in frame.get("signals", []) if s.get("action") is not None]
    actions = []
    for s in signals:
        if s["action"] not in actions:
            actions.append(s["action"])
    for action in task.get("candidate_actions", []):
        if action not in actions:
            actions.append(action)

    context = self.state["context"]
    feedback = task.get("feedback", {})
    for action, reward in feedback.items():
        self.intelligence.update_policy(context, action, float(reward))

    chosen = None
    reason = "no candidate action"

    if actions:
        if __ADAPTIVE__ and feedback:
            chosen = self.intelligence.choose_action(context, actions)
            reason = "adaptive policy after feedback"
        elif frame.get("attention") is not None and frame["attention"].get("action") in actions:
            chosen = frame["attention"]["action"]
            reason = "workspace attention"
        elif __ADAPTIVE__:
            chosen = self.intelligence.choose_action(context, actions)
            reason = "adaptive policy"
        else:
            chosen = actions[0]
            reason = "first viable coordinated action"

    challenge = None
    if __REFLECTION__ and chosen is not None:
        challenge = self.workspace.challenge(chosen)
        _v019_trace_event(self, "challenge", challenge)
        if challenge is not None:
            chosen_signal = next((s for s in signals if s.get("action") == chosen), None)
            chosen_strength = (
                chosen_signal["confidence"] * chosen_signal["urgency"]
                if chosen_signal else 0.0
            )
            alt_strength = challenge["confidence"] * challenge["urgency"]
            if alt_strength > chosen_strength:
                chosen = challenge["action"]
                reason = "self-challenge found stronger alternative"

    evidence = [
        "cycle:%s" % cycle,
        "inferences:%s" % sum(1 for x in inference_queries if x["value"]),
        "plan_length:%s" % len(plan),
        "conflicts:%s" % len(frame.get("conflicts", [])),
    ]
    self.self_model.record_decision(
        action=chosen,
        reason=reason,
        evidence=evidence,
    )
    self.state["decision"] = chosen
    _v019_trace_event(self, "decision", {
        "action": chosen,
        "reason": reason,
        "evidence": evidence,
    })

    return {
        "cycle": cycle,
        "action": chosen,
        "reason": reason,
        "plan": plan,
        "inferences": inference_queries,
        "conflicts": frame.get("conflicts", []),
        "challenge": challenge,
        "state": dict(self.state),
    }


def _v019_cycle(self, task):
    return _v019_process(self, task)


def _v019_reason_trace(self, limit=None):
    trace = list(self.state.get("trace", []))
    return trace if limit is None else trace[-int(limit):]


def _v019_explain(self):
    return {
        "decision": self.state.get("decision"),
        "last_decision": self.self_model.last_decision(),
        "cycle": self.state.get("cycle"),
        "trace": _v019_reason_trace(self),
        "architecture": dict(self.integration_architecture),
    }


CognitiveCore.process = _v019_process
CognitiveCore.cycle = _v019_cycle
CognitiveCore.reason_trace = _v019_reason_trace
CognitiveCore.decision_trace = _v019_reason_trace
CognitiveCore.explain = _v019_explain
# ---- end v0.19 generated integration architecture ----
'''


def integration_patch(spec: ArchitectureSpec) -> str:
    return (
        _PATCH_TEMPLATE
        .replace("__SPEC__", repr(asdict(spec)))
        .replace("__FULL_TRACE__", "True" if spec.trace_mode == "full" else "False")
        .replace("__REFLECTION__", "True" if spec.reflection else "False")
        .replace("__ADAPTIVE__", "True" if spec.arbitration == "adaptive" else "False")
    )


def build_candidate_source(base_source: str, spec: ArchitectureSpec) -> str:
    return base_source.rstrip() + "\n" + integration_patch(spec) + "\n"


def _load_core_from_source(source: str, candidate_id: str):
    tmp = Path(tempfile.mkdtemp(prefix=f"v019_{candidate_id}_"))
    p = tmp / "v0.18_cognitive_core.py"
    p.write_text(source, encoding="utf-8")
    mod = _load(f"candidate_core_{candidate_id}", p)
    return mod.CognitiveCore(), tmp


def selection_benchmark(core) -> dict:
    checks = []

    r1 = core.process({
        "observations": ["door_locked", "goal=exit"],
        "facts": [("edge", "A", "B"), ("edge", "B", "C")],
        "rules": [
            ([("edge", "$x", "$y"), ("edge", "$y", "$z")],
             ("reachable", "$x", "$z"))
        ],
        "queries": [("reachable", "A", "C")],
        "graph": {"A": [("B", 1), ("C", 5)], "B": [("C", 1)], "C": []},
        "start": "A",
        "goal": "C",
        "proposals": [
            {"source": "sensor", "topic": "action", "value": "wait",
             "action": "wait", "confidence": .4, "urgency": .3}
        ],
    })
    stages = [x["stage"] for x in core.reason_trace()]
    checks.append(r1["plan"] == ["A", "B", "C"])
    checks.append(any(x["value"] for x in r1["inferences"]))
    checks.append(all(x in stages for x in ("observation", "inference", "planning", "decision")))

    before = core.state["cycle"]
    r2 = core.cycle({
        "observations": ["second_cycle"],
        "proposals": [
            {"source": "planner", "topic": "action", "value": "left",
             "action": "left", "confidence": .8, "urgency": .6},
            {"source": "critic", "topic": "action", "value": "right",
             "action": "right", "confidence": .7, "urgency": .5},
        ],
    })
    checks.append(core.state["cycle"] == before + 1)
    checks.append(core.blackboard is core.state)
    checks.append(core.self_model.last_decision()["action"] == r2["action"])

    context = "adaptive-test"
    core.process({
        "context": context,
        "candidate_actions": ["left", "right"],
        "feedback": {"left": 0.1, "right": 1.0},
    })
    r3 = core.process({
        "context": context,
        "candidate_actions": ["left", "right"],
    })
    if core.integration_architecture["arbitration"] == "adaptive":
        checks.append(r3["action"] == "right")
    else:
        checks.append(r3["action"] in {"left", "right"})

    exp = core.explain()
    checks.append(exp["decision"] == core.state["decision"])
    checks.append(len(exp["trace"]) >= 1)

    if core.integration_architecture["reflection"]:
        core.process({
            "proposals": [
                {"source":"p1","topic":"action","value":"left","action":"left","confidence":.7,"urgency":.5},
                {"source":"p2","topic":"action","value":"right","action":"right","confidence":.95,"urgency":.9},
            ]
        })
        checks.append(any(x["stage"] == "challenge" for x in core.reason_trace()))

    passed = sum(bool(x) for x in checks)
    return {
        "passed": passed,
        "total": len(checks),
        "accuracy": passed / len(checks),
        "checks": checks,
    }


def existing_cognitive_regression(candidate_tree: Path) -> dict:
    lab = _load(
        "v019_regression_lab_" + hashlib.sha256(str(candidate_tree).encode()).hexdigest()[:8],
        candidate_tree / "v0.18_cognitive_lab.py",
    )
    return lab.run_benchmark(candidate_tree)


def candidate_complexity(spec: ArchitectureSpec) -> float:
    return (
        1.0
        + 0.25 * int(spec.reflection)
        + 0.20 * int(spec.trace_mode == "full")
        + 0.15 * int(spec.arbitration == "adaptive")
        + 0.05 * int(spec.state_mode == "blackboard")
    )


def search(current_tree: str | Path) -> dict:
    current_tree = Path(current_tree).resolve()
    base_source = (current_tree / "v0.18_cognitive_core.py").read_text(encoding="utf-8")
    outcomes = []

    for spec in generate_architecture_space():
        source = build_candidate_source(base_source, spec)
        core, tmp = _load_core_from_source(source, spec.architecture_id)

        cand_tree = Path(tempfile.mkdtemp(prefix=f"v019_tree_{spec.architecture_id}_"))
        shutil.copytree(current_tree, cand_tree, dirs_exist_ok=True)
        (cand_tree / "v0.18_cognitive_core.py").write_text(source, encoding="utf-8")

        integration = selection_benchmark(core)
        regression = existing_cognitive_regression(cand_tree)
        no_regression = all(float(v) >= 0.999 for v in regression["scores"].values())
        complexity = candidate_complexity(spec)

        outcomes.append({
            "architecture": asdict(spec),
            "architecture_id": spec.architecture_id,
            "integration": integration,
            "regression_scores": regression["scores"],
            "no_regression": no_regression,
            "complexity": complexity,
            "source": source,
        })

        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(cand_tree, ignore_errors=True)

    valid = [x for x in outcomes if x["no_regression"]]
    if not valid:
        return {"winner": None, "candidates": outcomes}

    winner = max(
        valid,
        key=lambda x: (
            x["integration"]["accuracy"],
            -x["complexity"],
            x["architecture_id"],
        ),
    )
    return {"winner": winner, "candidates": outcomes}


def promote_best(workspace_root: str | Path) -> dict:
    workspace_root = Path(workspace_root).resolve()
    current = workspace_root / "current"
    result = search(current)
    winner = result["winner"]
    if winner is None:
        return {"promoted": False, "reason": "no regression-safe candidate"}

    ctl = _load("v019_rewrite_controller", current / "v0.16_self_rewrite_controller.py")
    controller = ctl.SelfRewriteController(str(workspace_root))

    integration_test = '''
from pathlib import Path
import importlib.util, sys
root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("integration_test_core", root/"v0.18_cognitive_core.py")
m = importlib.util.module_from_spec(spec)
sys.modules["integration_test_core"] = m
spec.loader.exec_module(m)
core = m.CognitiveCore()
assert callable(getattr(core, "process", None))
assert callable(getattr(core, "reason_trace", None))
assert hasattr(core, "state") and hasattr(core, "blackboard")
r = core.process({
    "facts": [("edge","A","B"),("edge","B","C")],
    "rules": [([("edge","$x","$y"),("edge","$y","$z")],("reachable","$x","$z"))],
    "queries": [("reachable","A","C")],
    "graph": {"A":[("B",1)],"B":[("C",1)],"C":[]},
    "start": "A",
    "goal": "C",
})
assert r["plan"] == ["A","B","C"]
assert any(x["value"] for x in r["inferences"])
stages = [x["stage"] for x in core.reason_trace()]
for stage in ("observation","inference","planning","decision"):
    assert stage in stages
assert core.self_model.last_decision()["action"] == r["action"]
print("INTEGRATION_PASS")
'''

    proposal = ctl.RewriteProposal(
        proposal_id="v019_integration_" + winner["architecture_id"],
        goal="blind-deficit:cross-module-integration",
        rationale=(
            "Automatically selected from generated architectures; "
            f"integration_accuracy={winner['integration']['accuracy']:.3f}"
        ),
        changes=[
            ctl.FileChange("v0.18_cognitive_core.py", winner["source"]),
            ctl.FileChange(
                "v0.19_integration_manifest.json",
                json.dumps({
                    "winner": {k:v for k,v in winner.items() if k != "source"},
                    "candidate_count": len(result["candidates"]),
                }, ensure_ascii=False, indent=2),
            ),
            ctl.FileChange("test_v019_integration.py", integration_test),
        ],
        tests=[
            "test_baseline.py",
            "test_provider_autodiscovery.py",
            "test_data_world.py",
            "test_cognitive_core.py",
            "test_v019_integration.py",
        ],
    )

    gate = controller.stage_and_evaluate(proposal)
    if not gate.accepted:
        return {
            "promoted": False,
            "reason": gate.reason,
            "gate_stage": gate.stage,
            "winner": {k:v for k,v in winner.items() if k != "source"},
        }

    promoted = controller.promote(gate)
    return {
        "promoted": True,
        "revision": promoted.promoted_revision,
        "winner": {k:v for k,v in winner.items() if k != "source"},
        "candidates": [
            {k:v for k,v in c.items() if k != "source"}
            for c in result["candidates"]
        ],
        "gate_tests": {
            k: {"pass": v["pass"], "returncode": v["returncode"]}
            for k, v in gate.test_results.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(promote_best(Path(__file__).resolve().parent.parent), ensure_ascii=False, indent=2))

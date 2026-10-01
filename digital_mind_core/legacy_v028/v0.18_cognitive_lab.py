
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import hashlib
import importlib.util
import json
import sys

DIMENSIONS = (
    "logic",
    "thinking",
    "self_model",
    "conscious_integration",
    "intelligence",
)

def _load_core(tree: Path):
    p = tree / "v0.18_cognitive_core.py"
    if not p.exists():
        return None
    name = "cognitive_core_" + hashlib.sha256(str(tree).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.CognitiveCore()

def _manifest(tree: Path) -> dict:
    p = tree / "cognitive_manifest.json"
    if not p.exists():
        return {"enabled": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"enabled": []}

def _proxy_scores(tree: Path) -> Dict[str, float]:
    names = {p.name for p in tree.iterdir() if p.is_file()}
    return {
        "logic": 0.0,
        "thinking": 0.25 if "v0.10_social_planner.py" in names else 0.0,
        "self_model": 0.50 if {
            "v0.15_self_development_selector.py",
            "v0.16_self_rewrite_controller.py",
        }.issubset(names) else 0.0,
        "conscious_integration": 0.25 if "v0.13_unified_runtime.py" in names else 0.0,
        "intelligence": 0.50 if {
            "v0.14_autoconnect_runtime.py",
            "v0.17_data_world.py",
        }.issubset(names) else 0.25 if "v0.14_autoconnect_runtime.py" in names else 0.0,
    }

def _logic_score(core):
    if core is None or not hasattr(core, "logic"):
        return 0.0, ["no general logic mechanism"]
    L = core.logic
    checks = []
    L.reset()
    L.add_fact("edge", "a", "b")
    L.add_fact("edge", "b", "c")
    L.add_rule([("edge", "$x", "$y"), ("edge", "$y", "$z")], ("edge", "$x", "$z"))
    L.infer()
    checks.append(L.query("edge", "a", "c"))
    L.reset()
    L.add_fact("parent", "ana", "bob")
    L.add_fact("parent", "bob", "cy")
    L.add_rule([("parent", "$x", "$y"), ("parent", "$y", "$z")], ("grandparent", "$x", "$z"))
    L.infer()
    checks.append(L.query("grandparent", "ana", "cy"))
    L.reset()
    L.add_fact("safe", "door", truth=True)
    L.add_fact("safe", "door", truth=False)
    checks.append(len(L.contradictions()) == 1)
    L.reset()
    L.add_fact("red", "x")
    checks.append(not L.query("blue", "x"))
    return sum(bool(x) for x in checks)/len(checks), checks

def _thinking_score(core):
    if core is None or not hasattr(core, "thinking"):
        return 0.0, ["no general deliberation mechanism"]
    T = core.thinking
    checks = []
    edges = {
        "A": [("B", 1.0), ("C", 5.0)],
        "B": [("C", 1.0), ("D", 4.0)],
        "C": [("D", 1.0)],
        "D": [],
    }
    path, cost = T.plan_path("A", "D", edges)
    checks.append(path == ["A","B","C","D"] and abs(cost-3.0)<1e-9)
    path2, cost2 = T.plan_path("A", "D", edges, blocked_edges={("B","C")})
    checks.append(path2 == ["A","B","D"] and abs(cost2-5.0)<1e-9)
    hypotheses = {"h1","h2","h3","h4"}
    probes = {
        "weak": {"yes":{"h1","h2","h3"}, "no":{"h4"}},
        "balanced": {"yes":{"h1","h2"}, "no":{"h3","h4"}},
    }
    checks.append(T.choose_probe(hypotheses, probes) == "balanced")
    cf = T.counterfactual(start="A", goal="D", edges=edges, remove_edges={("B","C")})
    checks.append(cf["changed"] is True and cf["path"] == ["A","B","D"])
    return sum(bool(x) for x in checks)/len(checks), checks

def _self_score(core):
    if core is None or not hasattr(core, "self_model"):
        return 0.0, ["no explicit operational self-model"]
    S = core.self_model
    checks = []
    S.reset()
    S.predict("taskA", .8); S.observe("taskA", True)
    S.predict("taskA", .8); S.observe("taskA", True)
    checks.append(S.estimate("taskA") > .5)
    checks.append(S.claim_supported("unknown_capability", threshold=.6) is False)
    S.record_decision(action="use_adapter", reason="source is structured", evidence=["artifact:1","rule:route"])
    last = S.last_decision()
    checks.append(last["action"]=="use_adapter" and "structured" in last["reason"] and len(last["evidence"])==2)
    S.record_failure("vision", origin="external_provider", detail="provider missing")
    checks.append(S.last_failure()["boundary"]=="environment")
    return sum(bool(x) for x in checks)/len(checks), checks

def _workspace_score(core):
    if core is None or not hasattr(core, "workspace"):
        return 0.0, ["no global-access/conflict-monitoring workspace"]
    W = core.workspace
    checks = []
    W.reset()
    W.publish("logic", topic="door", value="open", confidence=.9, urgency=.4, action="enter")
    W.publish("sensor", topic="door", value="closed", confidence=.95, urgency=.9, action="wait")
    frame1 = W.integrate()
    checks.append(any(c["topic"]=="door" for c in frame1["conflicts"]))
    checks.append(frame1["attention"]["source"]=="sensor")
    W.publish("planner", topic="route", value="left", confidence=.8, urgency=.5, action="left")
    W.publish("critic", topic="route", value="right", confidence=.7, urgency=.45, action="right")
    challenge = W.challenge("left")
    checks.append(
        challenge is not None
        and challenge.get("action") is not None
        and challenge["action"] != "left"
    )
    frame2 = W.integrate()
    checks.append(frame2["previous_frame_id"]==frame1["frame_id"])
    return sum(bool(x) for x in checks)/len(checks), checks

def _intelligence_score(core):
    if core is None or not hasattr(core, "intelligence"):
        return 0.0, ["no general transfer/adaptation mechanism"]
    I = core.intelligence
    checks = []
    model = I.fit_affine([(1,5),(2,7),(4,11)])
    checks.append(model is not None and abs(model(10)-23)<1e-9)
    mapped = I.transfer_relations([("a","parent","b"),("b","parent","c")], {"a":"x","b":"y","c":"z"})
    checks.append(("x","parent","y") in mapped and ("y","parent","z") in mapped)
    I.reset_policy()
    for _ in range(6):
        I.update_policy("ctx","left",.2); I.update_policy("ctx","right",.9)
    checks.append(I.choose_action("ctx",["left","right"])=="right")
    for _ in range(20):
        I.update_policy("ctx","left",1.0); I.update_policy("ctx","right",0.0)
    checks.append(I.choose_action("ctx",["left","right"])=="left")
    return sum(bool(x) for x in checks)/len(checks), checks

def run_benchmark(tree):
    tree = Path(tree).resolve()
    core = _load_core(tree)
    proxy = _proxy_scores(tree)
    functional = {}
    details = {}
    functional["logic"], details["logic"] = _logic_score(core)
    functional["thinking"], details["thinking"] = _thinking_score(core)
    functional["self_model"], details["self_model"] = _self_score(core)
    functional["conscious_integration"], details["conscious_integration"] = _workspace_score(core)
    functional["intelligence"], details["intelligence"] = _intelligence_score(core)
    scores = {d:max(proxy[d], functional[d]) for d in DIMENSIONS}
    return {
        "scores": scores,
        "proxy_scores": proxy,
        "functional_scores": functional,
        "details": details,
        "mean_score": sum(scores.values())/len(scores),
        "enabled": list(_manifest(tree).get("enabled", [])),
    }

def tree_revision(tree):
    tree = Path(tree)
    h = hashlib.sha256()
    for p in sorted(tree.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(tree).as_posix().encode())
        h.update(b"\\0"); h.update(p.read_bytes()); h.update(b"\\0")
    return h.hexdigest()

def choose_deficit(benchmark, tree):
    enabled = set(benchmark.get("enabled", []))
    revision = tree_revision(tree)
    candidates = []
    for dim in DIMENSIONS:
        if dim in enabled:
            continue
        score = float(benchmark["scores"][dim])
        deficit = 1.0-score
        digest = hashlib.sha256(f"{revision}|{dim}".encode()).digest()
        tie = int.from_bytes(digest[:4], "big") / 2**32
        candidates.append({
            "dimension":dim, "score":score, "deficit":deficit,
            "tie_break":tie, "priority":deficit + tie*1e-6,
        })
    return max(candidates, key=lambda x:x["priority"]) if candidates else None

LOGIC_BLOCK = '\nclass RelationalLogic:\n    def __init__(self):\n        self.reset()\n\n    def reset(self):\n        self._facts = set()\n        self._rules = []\n\n    def add_fact(self, predicate, *args, truth=True):\n        self._facts.add((str(predicate), tuple(args), bool(truth)))\n\n    def add_rule(self, premises, conclusion):\n        self._rules.append((list(premises), tuple(conclusion)))\n\n    @staticmethod\n    def _bind(pattern, fact, env):\n        pred, fargs, truth = fact\n        if truth is not True or pattern[0] != pred or len(pattern)-1 != len(fargs):\n            return None\n        out = dict(env)\n        for token, value in zip(pattern[1:], fargs):\n            if isinstance(token, str) and token.startswith("$"):\n                old = out.get(token)\n                if old is not None and old != value:\n                    return None\n                out[token] = value\n            elif token != value:\n                return None\n        return out\n\n    def _matches(self, premises, idx=0, env=None):\n        env = {} if env is None else env\n        if idx >= len(premises):\n            yield env\n            return\n        pat = premises[idx]\n        for fact in list(self._facts):\n            bound = self._bind(pat, fact, env)\n            if bound is not None:\n                yield from self._matches(premises, idx+1, bound)\n\n    @staticmethod\n    def _subst(atom, env):\n        return tuple(env.get(x, x) if isinstance(x, str) and x.startswith("$") else x for x in atom)\n\n    def infer(self, max_rounds=32):\n        for _ in range(max_rounds):\n            before = len(self._facts)\n            for premises, conclusion in self._rules:\n                for env in self._matches(premises):\n                    atom = self._subst(conclusion, env)\n                    self.add_fact(atom[0], *atom[1:], truth=True)\n            if len(self._facts) == before:\n                break\n\n    def query(self, predicate, *args, truth=True):\n        return (str(predicate), tuple(args), bool(truth)) in self._facts\n\n    def contradictions(self):\n        pos = {(p, a) for p, a, t in self._facts if t}\n        neg = {(p, a) for p, a, t in self._facts if not t}\n        return sorted(pos & neg, key=repr)\n'
THINKING_BLOCK = '\nclass DeliberationField:\n    def plan_path(self, start, goal, edges, blocked_edges=None):\n        import heapq\n        blocked = set(blocked_edges or ())\n        q = [(0.0, start, [start])]\n        best = {start: 0.0}\n        while q:\n            cost, node, path = heapq.heappop(q)\n            if node == goal:\n                return path, cost\n            if cost > best.get(node, float("inf")):\n                continue\n            for nxt, weight in edges.get(node, []):\n                if (node, nxt) in blocked:\n                    continue\n                nc = cost + float(weight)\n                if nc < best.get(nxt, float("inf")):\n                    best[nxt] = nc\n                    heapq.heappush(q, (nc, nxt, path + [nxt]))\n        return [], float("inf")\n\n    @staticmethod\n    def _entropy(n):\n        if n <= 1:\n            return 0.0\n        import math\n        return math.log2(n)\n\n    def choose_probe(self, hypotheses, probes):\n        n = len(hypotheses)\n        if n <= 1:\n            return next(iter(probes), None)\n        best = None\n        best_expected = float("inf")\n        for name, outcomes in probes.items():\n            expected = 0.0\n            for subset in outcomes.values():\n                k = len(set(subset) & set(hypotheses))\n                if k:\n                    expected += (k / n) * self._entropy(k)\n            if expected < best_expected:\n                best_expected = expected\n                best = name\n        return best\n\n    def counterfactual(self, *, start, goal, edges, remove_edges):\n        original, _ = self.plan_path(start, goal, edges)\n        changed, cost = self.plan_path(start, goal, edges, blocked_edges=remove_edges)\n        return {\n            "original_path": original,\n            "path": changed,\n            "cost": cost,\n            "changed": changed != original,\n        }\n'
SELF_BLOCK = '\nclass OperationalSelfModel:\n    def __init__(self):\n        self.reset()\n\n    def reset(self):\n        self._stats = {}\n        self._predictions = {}\n        self._decisions = []\n        self._failures = []\n\n    def predict(self, capability, probability):\n        self._predictions.setdefault(capability, []).append(float(probability))\n\n    def observe(self, capability, success):\n        s = self._stats.setdefault(capability, {"success": 0, "total": 0})\n        s["total"] += 1\n        s["success"] += int(bool(success))\n\n    def estimate(self, capability):\n        s = self._stats.get(capability)\n        if not s:\n            return 0.5\n        return (s["success"] + 1) / (s["total"] + 2)\n\n    def claim_supported(self, capability, threshold=0.7):\n        s = self._stats.get(capability)\n        if not s or s["total"] < 2:\n            return False\n        return self.estimate(capability) >= threshold\n\n    def record_decision(self, *, action, reason, evidence):\n        self._decisions.append({\n            "action": action,\n            "reason": str(reason),\n            "evidence": list(evidence),\n        })\n\n    def last_decision(self):\n        return self._decisions[-1] if self._decisions else None\n\n    def record_failure(self, capability, *, origin, detail):\n        external = {\n            "external_provider", "network", "permission", "remote_service",\n            "missing_input",\n        }\n        self._failures.append({\n            "capability": capability,\n            "origin": origin,\n            "detail": detail,\n            "boundary": "environment" if origin in external else "self",\n        })\n\n    def last_failure(self):\n        return self._failures[-1] if self._failures else None\n'
WORKSPACE_BLOCK = '\nclass CognitiveWorkspace:\n    def __init__(self):\n        self.reset()\n\n    def reset(self):\n        self._signals = []\n        self._frame_counter = 0\n        self._last_frame_id = None\n\n    def publish(self, source, *, topic, value, confidence, urgency, action=None):\n        self._signals.append({\n            "source": source,\n            "topic": topic,\n            "value": value,\n            "confidence": float(confidence),\n            "urgency": float(urgency),\n            "action": action,\n        })\n\n    def _conflicts(self):\n        by_topic = {}\n        for s in self._signals:\n            by_topic.setdefault(s["topic"], []).append(s)\n        conflicts = []\n        for topic, items in by_topic.items():\n            values = {repr(x["value"]) for x in items}\n            if len(values) > 1:\n                conflicts.append({\n                    "topic": topic,\n                    "sources": [x["source"] for x in items],\n                    "values": [x["value"] for x in items],\n                })\n        return conflicts\n\n    def integrate(self):\n        self._frame_counter += 1\n        frame_id = self._frame_counter\n        attention = None\n        if self._signals:\n            attention = max(\n                self._signals,\n                key=lambda x: (x["urgency"] * x["confidence"], x["urgency"]),\n            )\n        frame = {\n            "frame_id": frame_id,\n            "previous_frame_id": self._last_frame_id,\n            "attention": attention,\n            "conflicts": self._conflicts(),\n            "signals": list(self._signals),\n        }\n        self._last_frame_id = frame_id\n        return frame\n\n    def challenge(self, chosen_action):\n        alternatives = [\n            s for s in self._signals\n            if s.get("action") is not None and s.get("action") != chosen_action\n        ]\n        if not alternatives:\n            return None\n        return max(\n            alternatives,\n            key=lambda x: x["confidence"] * x["urgency"],\n        )\n'
INTELLIGENCE_BLOCK = '\nclass AdaptiveTransferEngine:\n    def __init__(self):\n        self.reset_policy()\n\n    def fit_affine(self, examples):\n        points = [(float(x), float(y)) for x, y in examples]\n        if len(points) < 2:\n            return None\n        pair = None\n        for i in range(len(points)):\n            for j in range(i+1, len(points)):\n                if points[i][0] != points[j][0]:\n                    pair = (points[i], points[j])\n                    break\n            if pair:\n                break\n        if pair is None:\n            return None\n        (x1, y1), (x2, y2) = pair\n        a = (y2-y1)/(x2-x1)\n        b = y1-a*x1\n        if any(abs((a*x+b)-y) > 1e-9 for x, y in points):\n            return None\n        return lambda x: a*float(x)+b\n\n    def transfer_relations(self, relations, mapping):\n        return [\n            (mapping.get(left, left), rel, mapping.get(right, right))\n            for left, rel, right in relations\n        ]\n\n    def reset_policy(self):\n        self._policy = {}\n\n    def update_policy(self, context, action, reward):\n        key = (context, action)\n        s = self._policy.setdefault(key, {"n": 0, "mean": 0.0})\n        s["n"] += 1\n        alpha = 0.35\n        if s["n"] == 1:\n            s["mean"] = float(reward)\n        else:\n            s["mean"] = (1-alpha)*s["mean"] + alpha*float(reward)\n\n    def choose_action(self, context, actions):\n        ranked = []\n        for action in actions:\n            s = self._policy.get((context, action))\n            score = s["mean"] if s else 0.0\n            ranked.append((score, repr(action), action))\n        return max(ranked)[2]\n'

BLOCKS = {
    "logic": LOGIC_BLOCK,
    "thinking": THINKING_BLOCK,
    "self_model": SELF_BLOCK,
    "conscious_integration": WORKSPACE_BLOCK,
    "intelligence": INTELLIGENCE_BLOCK,
}

def build_core_source(enabled):
    enabled = [x for x in DIMENSIONS if x in set(enabled)]
    blocks = [BLOCKS[x] for x in enabled]
    init_lines = []
    if "logic" in enabled:
        init_lines.append("self.logic = RelationalLogic()")
    if "thinking" in enabled:
        init_lines.append("self.thinking = DeliberationField()")
    if "self_model" in enabled:
        init_lines.append("self.self_model = OperationalSelfModel()")
    if "conscious_integration" in enabled:
        init_lines.append("self.workspace = CognitiveWorkspace()")
    if "intelligence" in enabled:
        init_lines.append("self.intelligence = AdaptiveTransferEngine()")
    body = "\n        ".join(init_lines) if init_lines else "pass"
    return (
        "# Generated cognitive core. Operational mechanisms only.\n\n"
        + "\n\n".join(blocks)
        + "\n\nclass CognitiveCore:\n"
        + "    def __init__(self):\n"
        + "        " + body + "\n"
    )

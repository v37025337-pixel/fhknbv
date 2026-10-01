# Generated cognitive core. Operational mechanisms only.


class RelationalLogic:
    def __init__(self):
        self.reset()

    def reset(self):
        self._facts = set()
        self._rules = []

    def add_fact(self, predicate, *args, truth=True):
        self._facts.add((str(predicate), tuple(args), bool(truth)))

    def add_rule(self, premises, conclusion):
        self._rules.append((list(premises), tuple(conclusion)))

    @staticmethod
    def _bind(pattern, fact, env):
        pred, fargs, truth = fact
        if truth is not True or pattern[0] != pred or len(pattern)-1 != len(fargs):
            return None
        out = dict(env)
        for token, value in zip(pattern[1:], fargs):
            if isinstance(token, str) and token.startswith("$"):
                if token in out and out[token] != value:
                    return None
                out[token] = value
            elif token != value:
                return None
        return out

    def _matches(self, premises, idx=0, env=None):
        env = {} if env is None else env
        if idx >= len(premises):
            yield env
            return
        pat = premises[idx]
        for fact in list(self._facts):
            bound = self._bind(pat, fact, env)
            if bound is not None:
                yield from self._matches(premises, idx+1, bound)

    @staticmethod
    def _subst(atom, env):
        return tuple(env.get(x, x) if isinstance(x, str) and x.startswith("$") else x for x in atom)

    def infer(self, max_rounds=32):
        for _ in range(max_rounds):
            before = len(self._facts)
            for premises, conclusion in self._rules:
                for env in self._matches(premises):
                    atom = self._subst(conclusion, env)
                    self.add_fact(atom[0], *atom[1:], truth=True)
            if len(self._facts) == before:
                break

    def query(self, predicate, *args, truth=True):
        return (str(predicate), tuple(args), bool(truth)) in self._facts

    def contradictions(self):
        pos = {(p, a) for p, a, t in self._facts if t}
        neg = {(p, a) for p, a, t in self._facts if not t}
        return sorted(pos & neg, key=repr)



class DeliberationField:
    def plan_path(self, start, goal, edges, blocked_edges=None):
        import heapq
        blocked = set(blocked_edges or ())
        q = [(0.0, start, [start])]
        best = {start: 0.0}
        while q:
            cost, node, path = heapq.heappop(q)
            if node == goal:
                return path, cost
            if cost > best.get(node, float("inf")):
                continue
            for nxt, weight in edges.get(node, []):
                if (node, nxt) in blocked:
                    continue
                nc = cost + float(weight)
                if nc < best.get(nxt, float("inf")):
                    best[nxt] = nc
                    heapq.heappush(q, (nc, nxt, path + [nxt]))
        return [], float("inf")

    @staticmethod
    def _entropy(n):
        if n <= 1:
            return 0.0
        import math
        return math.log2(n)

    def choose_probe(self, hypotheses, probes):
        n = len(hypotheses)
        if n <= 1:
            return next(iter(probes), None)
        best = None
        best_expected = float("inf")
        for name, outcomes in probes.items():
            expected = 0.0
            for subset in outcomes.values():
                k = len(set(subset) & set(hypotheses))
                if k:
                    expected += (k / n) * self._entropy(k)
            if expected < best_expected:
                best_expected = expected
                best = name
        return best

    def counterfactual(self, *, start, goal, edges, remove_edges):
        original, _ = self.plan_path(start, goal, edges)
        changed, cost = self.plan_path(start, goal, edges, blocked_edges=remove_edges)
        return {
            "original_path": original,
            "path": changed,
            "cost": cost,
            "changed": changed != original,
        }



class OperationalSelfModel:
    def __init__(self):
        self.reset()

    def reset(self):
        self._stats = {}
        self._predictions = {}
        self._decisions = []
        self._failures = []

    def predict(self, capability, probability):
        self._predictions.setdefault(capability, []).append(float(probability))

    def observe(self, capability, success):
        s = self._stats.setdefault(capability, {"success": 0, "total": 0})
        s["total"] += 1
        s["success"] += int(bool(success))

    def estimate(self, capability):
        s = self._stats.get(capability)
        if not s:
            return 0.5
        return (s["success"] + 1) / (s["total"] + 2)

    def claim_supported(self, capability, threshold=0.7):
        s = self._stats.get(capability)
        if not s or s["total"] < 2:
            return False
        return self.estimate(capability) >= threshold

    def record_decision(self, *, action, reason, evidence):
        self._decisions.append({
            "action": action,
            "reason": str(reason),
            "evidence": list(evidence),
        })

    def last_decision(self):
        return self._decisions[-1] if self._decisions else None

    def record_failure(self, capability, *, origin, detail):
        external = {
            "external_provider", "network", "permission", "remote_service",
            "missing_input",
        }
        self._failures.append({
            "capability": capability,
            "origin": origin,
            "detail": detail,
            "boundary": "environment" if origin in external else "self",
        })

    def last_failure(self):
        return self._failures[-1] if self._failures else None



class CognitiveWorkspace:
    def __init__(self):
        self.reset()

    def reset(self):
        self._signals = []
        self._frame_counter = 0
        self._last_frame_id = None

    def publish(self, source, *, topic, value, confidence, urgency, action=None):
        self._signals.append({
            "source": source,
            "topic": topic,
            "value": value,
            "confidence": float(confidence),
            "urgency": float(urgency),
            "action": action,
        })

    def _conflicts(self):
        by_topic = {}
        for s in self._signals:
            by_topic.setdefault(s["topic"], []).append(s)
        conflicts = []
        for topic, items in by_topic.items():
            values = {repr(x["value"]) for x in items}
            if len(values) > 1:
                conflicts.append({
                    "topic": topic,
                    "sources": [x["source"] for x in items],
                    "values": [x["value"] for x in items],
                })
        return conflicts

    def integrate(self):
        self._frame_counter += 1
        frame_id = self._frame_counter
        attention = None
        if self._signals:
            attention = max(
                self._signals,
                key=lambda x: (x["urgency"] * x["confidence"], x["urgency"]),
            )
        frame = {
            "frame_id": frame_id,
            "previous_frame_id": self._last_frame_id,
            "attention": attention,
            "conflicts": self._conflicts(),
            "signals": list(self._signals),
        }
        self._last_frame_id = frame_id
        return frame

    def challenge(self, chosen_action):
        alternatives = [
            s for s in self._signals
            if s.get("action") is not None and s.get("action") != chosen_action
        ]
        if not alternatives:
            return None
        return max(
            alternatives,
            key=lambda x: x["confidence"] * x["urgency"],
        )



class AdaptiveTransferEngine:
    def __init__(self):
        self.reset_policy()

    def fit_affine(self, examples):
        points = [(float(x), float(y)) for x, y in examples]
        if len(points) < 2:
            return None
        pair = None
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                if points[i][0] != points[j][0]:
                    pair = (points[i], points[j])
                    break
            if pair:
                break
        if pair is None:
            return None
        (x1, y1), (x2, y2) = pair
        a = (y2-y1)/(x2-x1)
        b = y1-a*x1
        if any(abs((a*x+b)-y) > 1e-9 for x, y in points):
            return None
        return lambda x: a*float(x)+b

    def transfer_relations(self, relations, mapping):
        return [
            (mapping.get(left, left), rel, mapping.get(right, right))
            for left, rel, right in relations
        ]

    def reset_policy(self):
        self._policy = {}

    def update_policy(self, context, action, reward):
        key = (context, action)
        s = self._policy.setdefault(key, {"n": 0, "mean": 0.0})
        s["n"] += 1
        alpha = 0.35
        if s["n"] == 1:
            s["mean"] = float(reward)
        else:
            s["mean"] = (1-alpha)*s["mean"] + alpha*float(reward)

    def choose_action(self, context, actions):
        ranked = []
        for action in actions:
            s = self._policy.get((context, action))
            score = s["mean"] if s else 0.0
            ranked.append((score, repr(action), action))
        return max(ranked)[2]


class CognitiveCore:
    def __init__(self):
        self.logic = RelationalLogic()
        self.thinking = DeliberationField()
        self.self_model = OperationalSelfModel()
        self.workspace = CognitiveWorkspace()
        self.intelligence = AdaptiveTransferEngine()


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
    self.integration_architecture = {'state_mode': 'state', 'arbitration': 'adaptive', 'reflection': True, 'trace_mode': 'full'}

CognitiveCore.__init__ = _v019_init


def _v019_trace_event(self, stage, payload):
    event = {
        "cycle": self.state["cycle"],
        "stage": stage,
        "payload": payload,
    }
    self.state["trace"].append(event)
    if True:
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
        "items": observations if True else None,
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
        if True and feedback:
            chosen = self.intelligence.choose_action(context, actions)
            reason = "adaptive policy after feedback"
        elif frame.get("attention") is not None and frame["attention"].get("action") in actions:
            chosen = frame["attention"]["action"]
            reason = "workspace attention"
        elif True:
            chosen = self.intelligence.choose_action(context, actions)
            reason = "adaptive policy"
        else:
            chosen = actions[0]
            reason = "first viable coordinated action"

    challenge = None
    if True and chosen is not None:
        challenge = self.workspace.challenge(chosen)
        _v019_trace_event(self, "challenge", challenge)
        if challenge is not None:
            chosen_signal = next((s for s in signals if s.get("action") == chosen), None)
            chosen_strength = (
                chosen_signal["confidence"] * chosen_signal["urgency"]
                if chosen_signal else 0.0
            )
            alt_strength = challenge["confidence"] * challenge["urgency"]

            chosen_policy = self.intelligence._policy.get((context, chosen), {}).get("mean", 0.0)
            alt_policy = self.intelligence._policy.get((context, challenge["action"]), {}).get("mean", 0.0)

            chosen_combined = 0.60 * float(chosen_policy) + 0.40 * float(chosen_strength)
            alt_combined = 0.60 * float(alt_policy) + 0.40 * float(alt_strength)

            if alt_combined > chosen_combined:
                chosen = challenge["action"]
                reason = "self-challenge won combined learned-and-current evidence"

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
    return _v023_copy.deepcopy(trace if limit is None else trace[-int(limit):])


def _v019_explain(self):
    return _v023_copy.deepcopy({
        "decision": self.state.get("decision"),
        "last_decision": self.self_model.last_decision(),
        "cycle": self.state.get("cycle"),
        "trace": _v019_reason_trace(self),
        "architecture": dict(self.integration_architecture),
    })


CognitiveCore.process = _v019_process
CognitiveCore.cycle = _v019_cycle
CognitiveCore.reason_trace = _v019_reason_trace
CognitiveCore.decision_trace = _v019_reason_trace
CognitiveCore.explain = _v019_explain
# ---- end v0.19 generated integration architecture ----

# v0.21 memory integration: facts_context
_V021_PREV_PROCESS = CognitiveCore.process
def _v021_process(self, task):
    task = dict(task or {})
    self.state.setdefault("persistent_facts", [])
    self.state.setdefault("persistent_context", None)
    if task.get("persist_facts"):
        merged = list(self.state["persistent_facts"]) + list(task.get("facts", []))
        task["facts"] = merged
    if task.get("preserve_context"):
        if "context" not in task and self.state.get("persistent_context") is not None:
            task["context"] = self.state["persistent_context"]
    result = _V021_PREV_PROCESS(self, task)
    if task.get("persist_facts"):
        seen=[]
        for fact in task.get("facts", []):
            t=tuple(fact)
            if t not in seen: seen.append(t)
        self.state["persistent_facts"] = seen
    if task.get("preserve_context") and task.get("context") is not None:
        self.state["persistent_context"] = task["context"]
        self.state["context"] = task["context"]
    return result
CognitiveCore.process = _v021_process
CognitiveCore.cycle = _v021_process

# v0.22 bounded cognitive trace limit=1000
_V022_PREV_PROCESS = CognitiveCore.process
def _v022_process(self, task):
    result = _V022_PREV_PROCESS(self, task)
    if len(self.state.get("trace", [])) > 1000:
        self.state["trace"] = self.state["trace"][-1000:]
        self.blackboard = self.state
    if len(self.memory) > 1000:
        self.memory = self.memory[-1000:]
    # Refresh returned snapshot after trimming.
    result["state"] = dict(self.state)
    return result
CognitiveCore.process = _v022_process
CognitiveCore.cycle = _v022_process

import copy as _v023_copy
_V023_PREV_PROCESS = CognitiveCore.process
def _v023_process(self, task):
    result = _V023_PREV_PROCESS(self, task)
    result["state"] = _v023_copy.deepcopy(self.state)
    return result
CognitiveCore.process = _v023_process
CognitiveCore.cycle = _v023_process

# v0.24 self-model feedback loop: explicit outcome
_V024_PREV_PROCESS = CognitiveCore.process
def _v024_process(self, task):
    task = dict(task or {})
    result = _V024_PREV_PROCESS(self, task)
    outcome = task.get("outcome")
    if isinstance(outcome, dict):
        capability = outcome.get("capability")
        success = outcome.get("success")
        if capability is not None and success is not None:
            self.self_model.observe(str(capability), bool(success))
    return result
CognitiveCore.process = _v024_process
CognitiveCore.cycle = _v024_process

# v0.25 batch self-model outcomes
_V025_PREV_PROCESS = CognitiveCore.process
def _v025_process(self, task):
    task = dict(task or {})
    result = _V025_PREV_PROCESS(self, task)
    outcomes = task.get("outcomes", [])
    if isinstance(outcomes, list):
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            capability = outcome.get("capability")
            success = outcome.get("success")
            if capability is not None and success is not None:
                self.self_model.observe(str(capability), bool(success))
    return result
CognitiveCore.process = _v025_process
CognitiveCore.cycle = _v025_process

# v0.26 persistent rules
_V026_PREV_PROCESS = CognitiveCore.process
def _v026_process(self, task):
    task = _v023_copy.deepcopy(dict(task or {}))
    self.state.setdefault("persistent_rules", [])
    if task.get("persist_rules"):
        task["rules"] = _v023_copy.deepcopy(self.state["persistent_rules"]) + list(task.get("rules", []))
    result = _V026_PREV_PROCESS(self, task)
    if task.get("persist_rules"):
        self.state["persistent_rules"] = _v023_copy.deepcopy(task.get("rules", []))
    result["state"] = _v023_copy.deepcopy(self.state)
    return result
CognitiveCore.process = _v026_process
CognitiveCore.cycle = _v026_process

# v0.27 persistent rule deduplication
_V027_PREV_PROCESS = CognitiveCore.process
def _v027_process(self, task):
    task = dict(task or {})
    result = _V027_PREV_PROCESS(self, task)
    if task.get("persist_rules"):
        deduped = []
        for rule in self.state.get("persistent_rules", []):
            if rule not in deduped:
                deduped.append(rule)
        self.state["persistent_rules"] = deduped
        result["state"]["persistent_rules"] = _v023_copy.deepcopy(deduped)
    for key in result:
        if key != 'state':
            result[key] = _v023_copy.deepcopy(result[key])
    return result
CognitiveCore.process = _v027_process
CognitiveCore.cycle = _v027_process


# v0.28 mechanism genesis integration
import importlib.util as _v028_importlib_util
import sys as _v028_sys
from pathlib import Path as _v028_Path

def _v028_load_genesis_module():
    name = "kernel_v028_mechanism_genesis"
    if name in _v028_sys.modules:
        return _v028_sys.modules[name]
    path = _v028_Path(__file__).resolve().with_name("v0.28_mechanism_genesis.py")
    spec = _v028_importlib_util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    mod = _v028_importlib_util.module_from_spec(spec)
    _v028_sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_V028_PREV_INIT = CognitiveCore.__init__

def _v028_init(self):
    _V028_PREV_INIT(self)
    self.mechanism_registry = {}
    self._mechanism_objects = {}

CognitiveCore.__init__ = _v028_init


def _v028_synthesize_mechanism(
    self,
    spec,
    *,
    validation_examples=None,
    max_rounds=8,
    engine_options=None,
):
    mod = _v028_load_genesis_module()
    engine = mod.MechanismGenesis(**dict(engine_options or {}))

    if validation_examples:
        outcome = mod.cegis_synthesize(
            engine,
            dict(spec),
            list(validation_examples),
            max_rounds=max_rounds,
        )
        result = outcome["result"]
        history = outcome["history"]
        counterexamples_added = outcome["counterexamples_added"]
        status = outcome["status"]
    else:
        result = engine.synthesize(dict(spec))
        history = []
        counterexamples_added = 0
        status = result.status

    summary = mod.result_dict(result)
    summary["status"] = status
    summary["history"] = history
    summary["counterexamples_added"] = counterexamples_added

    if status == "FOUND":
        self._mechanism_objects[result.name] = result
        self.mechanism_registry[result.name] = summary
        self.self_model.observe("mechanism_synthesis", True)
        self.self_model.record_decision(
            action="register_mechanism:" + result.name,
            reason="mechanism satisfied synthesis specification",
            evidence=[
                "examples:%s/%s" % (result.examples_passed, result.examples_total),
                "cost:%s" % result.cost,
                "counterexamples_added:%s" % counterexamples_added,
            ],
        )
    else:
        self.self_model.observe("mechanism_synthesis", False)

    return summary


def _v028_run_mechanism(self, name, **inputs):
    if name not in self._mechanism_objects:
        raise KeyError("unknown synthesized mechanism: %s" % name)
    mod = _v028_load_genesis_module()
    return mod.execute_result(self._mechanism_objects[name], inputs)


def _v028_mechanism_info(self, name=None):
    if name is None:
        return dict(self.mechanism_registry)
    return dict(self.mechanism_registry[name])

CognitiveCore.synthesize_mechanism = _v028_synthesize_mechanism
CognitiveCore.run_mechanism = _v028_run_mechanism
CognitiveCore.mechanism_info = _v028_mechanism_info


# v0.28 persistent generated-mechanism loader
_V028P_PREV_INIT = CognitiveCore.__init__
_V028P_PREV_RUN = CognitiveCore.run_mechanism
_V028P_PREV_INFO = CognitiveCore.mechanism_info

def _v028p_load_generated():
    name = "kernel_v028_generated_mechanisms"
    if name in _v028_sys.modules:
        return _v028_sys.modules[name]
    path = _v028_Path(__file__).resolve().with_name("v0.28_generated_mechanisms.py")
    if not path.exists():
        return None
    spec = _v028_importlib_util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    mod = _v028_importlib_util.module_from_spec(spec)
    _v028_sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def _v028p_init(self):
    _V028P_PREV_INIT(self)
    self._persisted_mechanisms = {}
    mod = _v028p_load_generated()
    if mod is not None:
        for name, item in getattr(mod, "GENERATED_MECHANISMS", {}).items():
            if isinstance(item, dict) and callable(item.get("function")):
                self._persisted_mechanisms[name] = dict(item)

CognitiveCore.__init__ = _v028p_init

def _v028p_run(self, name, **inputs):
    if name in self._mechanism_objects:
        return _V028P_PREV_RUN(self, name, **inputs)
    if name in self._persisted_mechanisms:
        return self._persisted_mechanisms[name]["function"](**inputs)
    raise KeyError("unknown synthesized mechanism: %s" % name)

def _v028p_info(self, name=None):
    if name is None:
        merged = _V028P_PREV_INFO(self)
        for key, value in self._persisted_mechanisms.items():
            merged[key] = {
                "status": "PERSISTED",
                "source_sha256": value.get("source_sha256"),
                "cost": value.get("cost"),
                "counterexamples_added": value.get("counterexamples_added"),
            }
        return merged
    if name in self.mechanism_registry:
        return _V028P_PREV_INFO(self, name)
    if name in self._persisted_mechanisms:
        value = self._persisted_mechanisms[name]
        return {
            "status": "PERSISTED",
            "source_sha256": value.get("source_sha256"),
            "cost": value.get("cost"),
            "counterexamples_added": value.get("counterexamples_added"),
        }
    raise KeyError(name)

CognitiveCore.run_mechanism = _v028p_run
CognitiveCore.mechanism_info = _v028p_info

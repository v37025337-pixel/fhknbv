from __future__ import annotations

"""L0 Language v0.5 — ARC: Activated Relational Convergence.

ARC is a native execution algorithm for L0.  It is intentionally *not* a
renaming of ``for``/``while`` or a conventional instruction scheduler.

A program is a set of pure transformations over named state cells.  The engine:

1. derives causal dependencies from read/write sets;
2. collapses cycles into strongly connected components (SCCs);
3. evaluates independent transformations against the same immutable snapshot;
4. resolves all proposals for each cell deterministically;
5. commits a whole micro-round atomically;
6. repeats cyclic SCCs until an exact/tolerance fixed point is reached;
7. publishes effects only after the transaction is stable.

The model deliberately refuses to guess in three cases:

* two different ``set`` proposals for one cell -> ``ARCConflict``;
* a cyclic component that oscillates -> ``ARCOscillation``;
* a cyclic component that does not converge within its proof/runtime budget ->
  ``ARCUnprovenConvergence``.

Resource budgets are transactional: no effects are flushed on failure.

This implementation is a research prototype.  It demonstrates deterministic
order-independent scheduling within the stated pure/merge semantics.  It does
not prove termination for arbitrary programs (which is impossible in general).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence
import copy
import hashlib
import json
import math
import numbers
from types import MappingProxyType


# ---------------------------------------------------------------------------
# Exceptions / status
# ---------------------------------------------------------------------------


class ARCError(RuntimeError):
    pass


class ARCConflict(ARCError):
    pass


class ARCOscillation(ARCError):
    pass


class ARCUnprovenConvergence(ARCError):
    pass


class ARCResourcePressure(ARCError):
    pass


class ARCInvalidProgram(ARCError, ValueError):
    pass


class ARCIncompleteResult(ARCError):
    """The graph is quiescent but a requested value was not produced."""


def validate_finite_value(value: Any, where: str = "state") -> None:
    """Keep exact integers; reject non-finite real values at every ARC boundary."""
    if isinstance(value, numbers.Integral):
        return
    if isinstance(value, numbers.Real):
        if not math.isfinite(value):
            raise ARCInvalidProgram(f"{where} contains non-finite value {value!r}")
    elif isinstance(value, Mapping):
        for k, v in value.items():
            validate_finite_value(k, where + " key")
            validate_finite_value(v, f"{where}[{k!r}]")
    elif isinstance(value, (list, tuple, set, frozenset)):
        for i, v in enumerate(value):
            validate_finite_value(v, f"{where}[{i}]")
    elif hasattr(value, "dtype") and hasattr(value, "flat"):
        for v in value.flat:
            validate_finite_value(v, where)


# ---------------------------------------------------------------------------
# IR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """A candidate state change produced from one immutable snapshot."""

    rule: str
    ordinal: int
    target: str
    value: Any
    mode: str = "set"  # set | delta | union | min | max

    @property
    def proposal_id(self) -> tuple[str, int, str]:
        return (self.rule, self.ordinal, self.target)


@dataclass(frozen=True)
class EffectIntent:
    """An effect is data until the whole ARC transaction stabilizes."""

    rule: str
    ordinal: int
    kind: str
    payload: Any

    @property
    def intent_id(self) -> tuple[str, int, str]:
        return (self.rule, self.ordinal, self.kind)


@dataclass(frozen=True)
class RuleResult:
    proposals: tuple[Proposal, ...] = ()
    effects: tuple[EffectIntent, ...] = ()


RuleFn = Callable[[Mapping[str, Any]], RuleResult]
GuardFn = Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class ARCRule:
    name: str
    reads: frozenset[str]
    writes: frozenset[str]
    evaluate: RuleFn
    guard: GuardFn | None = None
    # Explicit semantic dependencies are allowed for imported imperative code.
    # Native L0 should prefer data dependencies and only use ``after`` when the
    # order itself is semantically meaningful.
    after: frozenset[str] = frozenset()
    demand_tags: frozenset[str] = frozenset()

    def enabled(self, snapshot: Mapping[str, Any]) -> bool:
        return True if self.guard is None else bool(self.guard(snapshot))


@dataclass(frozen=True)
class ARCProgram:
    rules: tuple[ARCRule, ...]
    # Optional cell-specific resolution law.  If absent, the proposal's mode is
    # used and all proposals for that cell must agree on the mode.
    merges: Mapping[str, str] = field(default_factory=dict)
    observables: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        names = [r.name for r in self.rules]
        if len(set(names)) != len(names):
            raise ARCInvalidProgram("rule names must be unique")
        known = set(names)
        for r in self.rules:
            missing = set(r.after) - known
            if missing:
                raise ARCInvalidProgram(f"{r.name}: unknown explicit dependencies {sorted(missing)}")


@dataclass(frozen=True)
class ARCConfig:
    max_component_rounds: int = 10_000
    max_rule_evaluations: int = 1_000_000
    max_proposals: int = 1_000_000
    float_tolerance: float = 0.0
    detect_oscillation: bool = True


@dataclass
class ARCStats:
    rule_evaluations: int = 0
    proposals: int = 0
    component_rounds: int = 0
    components_executed: int = 0
    rules_skipped_by_demand: int = 0
    effects_staged: int = 0
    effects_committed: int = 0
    candidate_checks: int = 0


@dataclass(frozen=True)
class ARCResult:
    state: Mapping[str, Any]
    effects: tuple[EffectIntent, ...]
    stats: ARCStats
    status: str = "STABLE_EXACT"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _dependency_graph(program: ARCProgram) -> dict[str, set[str]]:
    """Return predecessor -> successors.

    A -> B if B reads a cell written by A, or B explicitly declares ``after A``.
    Write/write conflicts do *not* impose an arbitrary order; they are resolved
    atomically or rejected by the resolver.
    """

    graph = {r.name: set() for r in program.rules}
    by_name = {r.name: r for r in program.rules}
    for a in program.rules:
        for b in program.rules:
            if a.name == b.name:
                continue
            if a.writes & b.reads:
                graph[a.name].add(b.name)
        for pred in a.after:
            graph[pred].add(a.name)
    return graph


def _tarjan(graph: Mapping[str, set[str]]) -> list[tuple[str, ...]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    out: list[tuple[str, ...]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in sorted(graph[v]):
            if w not in indices:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], indices[w])

        if low[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            out.append(tuple(sorted(comp)))

    for v in sorted(graph):
        if v not in indices:
            strongconnect(v)
    return out


def _component_order(graph: Mapping[str, set[str]], components: Sequence[tuple[str, ...]]) -> list[tuple[str, ...]]:
    c_of: dict[str, int] = {}
    for i, comp in enumerate(components):
        for n in comp:
            c_of[n] = i

    succ = {i: set() for i in range(len(components))}
    indeg = {i: 0 for i in range(len(components))}
    for a, outs in graph.items():
        ca = c_of[a]
        for b in outs:
            cb = c_of[b]
            if ca != cb and cb not in succ[ca]:
                succ[ca].add(cb)
                indeg[cb] += 1

    ready = sorted(i for i, d in indeg.items() if d == 0)
    order: list[tuple[str, ...]] = []
    while ready:
        i = ready.pop(0)
        order.append(components[i])
        for j in sorted(succ[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
                ready.sort()
    if len(order) != len(components):
        raise ARCInvalidProgram("component graph unexpectedly cyclic")
    return order


def _self_cyclic(rule: ARCRule) -> bool:
    return bool(rule.reads & rule.writes)


def _cyclic_nodes_indexed(rules: Mapping[str, ARCRule]) -> set[str]:
    """Find causal cycles in O(V+E) after indexed edge construction.

    Edges are created only where a writer feeds a reader of the same cell, plus
    explicit ``after`` edges.  Kosaraju is implemented iteratively so deep
    causal chains do not consume Python recursion depth.
    """
    readers: dict[str, set[str]] = {}
    writers: dict[str, set[str]] = {}
    for r in rules.values():
        for c in r.reads:
            readers.setdefault(c, set()).add(r.name)
        for c in r.writes:
            writers.setdefault(c, set()).add(r.name)

    g = {n: set() for n in rules}
    rg = {n: set() for n in rules}
    for cell in set(readers) & set(writers):
        for a in writers[cell]:
            for b in readers[cell]:
                g[a].add(b)
                rg[b].add(a)
    for r in rules.values():
        for pred in r.after:
            if pred in rules:
                g[pred].add(r.name)
                rg[r.name].add(pred)

    seen: set[str] = set()
    order: list[str] = []
    for start in sorted(g):
        if start in seen:
            continue
        stack: list[tuple[str, bool]] = [(start, False)]
        while stack:
            v, expanded = stack.pop()
            if expanded:
                order.append(v)
                continue
            if v in seen:
                continue
            seen.add(v)
            stack.append((v, True))
            for w in sorted(g[v], reverse=True):
                if w not in seen:
                    stack.append((w, False))

    seen.clear()
    cyclic: set[str] = set()
    for start in reversed(order):
        if start in seen:
            continue
        comp: list[str] = []
        stack = [start]
        seen.add(start)
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in rg[v]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        if len(comp) > 1:
            cyclic.update(comp)
        elif comp and comp[0] in g[comp[0]]:
            cyclic.add(comp[0])
    return cyclic


# ---------------------------------------------------------------------------
# Demand slicing
# ---------------------------------------------------------------------------


def _demand_slice(program: ARCProgram, requested: set[str] | None) -> set[str]:
    """Reverse causal slice using indexes, not repeated full rule scans."""
    if not requested:
        return {r.name for r in program.rules}

    by_name = {r.name: r for r in program.rules}
    writers: dict[str, set[str]] = {}
    tagged: set[str] = set()
    for r in program.rules:
        for cell in r.writes:
            writers.setdefault(cell, set()).add(r.name)
        if r.demand_tags & requested:
            tagged.add(r.name)

    needed_rules: set[str] = set()
    seen_cells: set[str] = set()
    cell_stack = list(requested)
    rule_stack = list(tagged)

    while cell_stack or rule_stack:
        while cell_stack:
            cell = cell_stack.pop()
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            rule_stack.extend(sorted(writers.get(cell, ())))

        if not rule_stack:
            break
        name = rule_stack.pop()
        if name in needed_rules:
            continue
        needed_rules.add(name)
        r = by_name[name]
        for cell in r.reads:
            if cell not in seen_cells:
                cell_stack.append(cell)
        for pred in r.after:
            if pred not in needed_rules:
                rule_stack.append(pred)

    return needed_rules


# ---------------------------------------------------------------------------
# Deterministic resolver
# ---------------------------------------------------------------------------


def _eq(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
        if a == b:
            return True
        if tol == 0 or isinstance(a, numbers.Integral) and isinstance(b, numbers.Integral):
            return False
        # as_integer_ratio preserves mixed int/float distinctions above 2**53.
        from fractions import Fraction
        return abs(Fraction(a) - Fraction(b)) <= Fraction(tol)
    if isinstance(a, (tuple, list)) and isinstance(b, type(a)):
        return len(a) == len(b) and all(_eq(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_eq(a[k], b[k], tol) for k in a)
    return bool(a == b)


def _resolve_cell(old: Any, proposals: Sequence[Proposal], mode: str, tol: float) -> Any:
    # Deduplicate by proposal identity.  Duplicate delivery of one proposal in a
    # micro-round cannot double-apply a delta.
    uniq: dict[tuple[str, int, str], Proposal] = {}
    for p in proposals:
        prev = uniq.get(p.proposal_id)
        if prev is not None and (prev.mode != p.mode or not _eq(prev.value, p.value, tol)):
            raise ARCConflict(f"proposal id collision with different content: {p.proposal_id}")
        uniq[p.proposal_id] = p
    ps = [uniq[k] for k in sorted(uniq)]

    if mode == "set":
        first = ps[0].value
        for p in ps[1:]:
            if not _eq(first, p.value, tol):
                raise ARCConflict(
                    f"conflicting set proposals for {p.target}: {first!r} vs {p.value!r}"
                )
        return copy.deepcopy(first)

    if mode == "delta":
        vals = [p.value for p in ps]
        if isinstance(old, numbers.Integral) and all(isinstance(v, numbers.Integral) for v in vals):
            return int(old) + sum(int(v) for v in vals)
        if all(isinstance(v, (int, float)) for v in vals) and isinstance(old, (int, float)):
            # math.fsum makes the result independent of proposal enumeration for
            # ordinary IEEE floats to a much stronger degree than repeated +.
            return old + math.fsum(float(v) for v in vals)
        total = old
        for p in ps:
            total = total + p.value
        return total

    if mode == "union":
        base = set() if old is None else set(old)
        for p in ps:
            base |= set(p.value)
        return base

    if mode == "min":
        vals = [old] + [p.value for p in ps] if old is not None else [p.value for p in ps]
        return min(vals)

    if mode == "max":
        vals = [old] + [p.value for p in ps] if old is not None else [p.value for p in ps]
        return max(vals)

    raise ARCInvalidProgram(f"unknown merge mode {mode!r}")


def _commit(state: dict[str, Any], snapshot: Mapping[str, Any], proposals: Sequence[Proposal], merges: Mapping[str, str], tol: float) -> set[str]:
    """Resolve against one snapshot, then mutate only changed cells atomically."""
    grouped: dict[str, list[Proposal]] = {}
    for p in proposals:
        grouped.setdefault(p.target, []).append(p)

    resolved: dict[str, Any] = {}
    changed: set[str] = set()
    # Compute *all* values before mutating state so targets cannot observe one
    # another's same-wave commits.
    for target in sorted(grouped):
        ps = grouped[target]
        modes = {p.mode for p in ps}
        mode = merges.get(target)
        if mode is None:
            if len(modes) != 1:
                raise ARCConflict(f"mixed proposal modes for {target}: {sorted(modes)}")
            mode = next(iter(modes))
        old = snapshot.get(target)
        value = _resolve_cell(old, ps, mode, tol)
        validate_finite_value(value, f"proposal[{target!r}]")
        resolved[target] = value
        if target not in snapshot or not _eq(old, value, tol):
            changed.add(target)

    for target in sorted(changed):
        state[target] = resolved[target]
    return changed



# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _stable_hash(state: Mapping[str, Any], keys: Iterable[str]) -> str:
    def norm(v: Any) -> Any:
        if isinstance(v, set):
            return [norm(x) for x in sorted(v, key=repr)]
        if isinstance(v, dict):
            return {str(k): norm(v[k]) for k in sorted(v, key=str)}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        return v

    payload = {k: norm(state.get(k)) for k in sorted(set(keys))}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def execute_arc(
    program: ARCProgram,
    initial_state: Mapping[str, Any],
    *,
    requested: Iterable[str] | None = None,
    config: ARCConfig = ARCConfig(),
    rule_order: Sequence[str] | None = None,
) -> ARCResult:
    """Execute one transactional ARC evaluation as indexed causal waves.

    No global scheduler scan is required after compilation.  The runtime keeps
    a reverse dependency index ``cell -> readers``.  A committed change therefore
    activates only transformations that actually read that cell.  Explicit
    ``after`` edges use a second successor index.

    ``rule_order`` is only a test hook that perturbs enumeration order.
    """

    state = copy.deepcopy(dict(initial_state))
    validate_finite_value(state)
    if not math.isfinite(config.float_tolerance) or config.float_tolerance < 0:
        raise ARCInvalidProgram("float_tolerance must be finite and non-negative")
    stats = ARCStats()
    requested_set = set(requested or ())
    active_names = _demand_slice(program, requested_set if requested_set else None)
    stats.rules_skipped_by_demand = len(program.rules) - len(active_names)
    rules = {r.name: r for r in program.rules if r.name in active_names}

    cyclic_rules = _cyclic_nodes_indexed(rules) if config.detect_oscillation else set()

    if rule_order is not None:
        rank = {name: i for i, name in enumerate(rule_order)}
    else:
        rank = {}

    readers: dict[str, set[str]] = {}
    seeds: set[str] = set()
    after_successors: dict[str, set[str]] = {}
    for r in rules.values():
        if not r.reads:
            seeds.add(r.name)
        for cell in r.reads:
            readers.setdefault(cell, set()).add(r.name)
        for pred in r.after:
            if pred in rules:
                after_successors.setdefault(pred, set()).add(r.name)

    staged_effects: dict[tuple[str, int, str], EffectIntent] = {}
    effect_ids_by_rule: dict[str, set[tuple[str, int, str]]] = {}
    stable_status = "STABLE_EXACT" if config.float_tolerance == 0.0 else "STABLE_TOLERANCE"
    completed_order: set[str] = set()
    all_cells = set(state)
    for r in rules.values():
        all_cells |= set(r.reads) | set(r.writes)

    tags = set().union(*(r.demand_tags for r in program.rules)) if program.rules else set()
    expected_cells = requested_set - tags

    def finish():
        missing = expected_cells - state.keys()
        if missing:
            raise ARCIncompleteResult(f"requested values were not produced: {sorted(missing)}")
        effects = tuple(staged_effects[k] for k in sorted(staged_effects))
        stats.effects_committed = len(effects)
        return ARCResult(state=state, effects=effects, stats=stats, status=stable_status)

    # Initial observations seed the first dependency frontier.
    candidates: set[str] = set(seeds)
    for cell in state:
        candidates |= readers.get(cell, set())

    seen: dict[str, int] = {}

    for wave in range(config.max_component_rounds):
        stats.component_rounds += 1
        if not candidates:
            return finish()

        if config.detect_oscillation and cyclic_rules and (candidates & cyclic_rules):
            h = _stable_hash(state, all_cells)
            if h in seen:
                raise ARCOscillation(
                    f"global state repeated from wave {seen[h]} at wave {wave}"
                )
            seen[h] = wave

        eligible: list[str] = []
        blocked: set[str] = set()
        for name in candidates:
            stats.candidate_checks += 1
            r = rules[name]
            if not r.reads.issubset(state.keys()):
                blocked.add(name)
                continue
            if not r.after.issubset(completed_order):
                blocked.add(name)
                continue
            eligible.append(name)

        if not eligible:
            # A candidate with missing data will be reintroduced when one of its
            # read cells is committed. A candidate waiting only on ``after`` will
            # be reintroduced by that predecessor's completion. If neither can
            # happen, there is no executable causal work left.
            return finish()

        snapshot = MappingProxyType(state)
        props: list[Proposal] = []
        effs: list[EffectIntent] = []
        names_sorted = sorted(eligible, key=lambda n: (rank.get(n, 10**9), n))
        for name in names_sorted:
            r = rules[name]
            # Reaching this point means the causal action has been considered in
            # this semantic stage.  ``after`` therefore becomes satisfied even
            # if the activation guard is false.
            completed_order.add(name)
            # An effect is an intention for the final state, not an event log.
            for intent_id in effect_ids_by_rule.pop(name, ()):
                staged_effects.pop(intent_id, None)
            if not r.enabled(snapshot):
                continue
            stats.rule_evaluations += 1
            if stats.rule_evaluations > config.max_rule_evaluations:
                raise ARCResourcePressure("rule evaluation budget exceeded")
            rr = r.evaluate(snapshot)
            for p in rr.proposals:
                if p.rule != r.name:
                    raise ARCInvalidProgram(f"proposal claims rule {p.rule!r}, expected {r.name!r}")
                if p.target not in r.writes:
                    raise ARCInvalidProgram(f"{r.name} proposed undeclared write to {p.target}")
                props.append(p)
            for effect in rr.effects:
                if effect.rule != name:
                    raise ARCInvalidProgram("effect claims a different rule")
                validate_finite_value(effect.payload, f"effect[{name!r}]")
            effs.extend(rr.effects)
            stats.proposals += len(rr.proposals)
            stats.effects_staged += len(rr.effects)
            if stats.proposals > config.max_proposals:
                raise ARCResourcePressure("proposal budget exceeded")

        for e in effs:
            staged_effects[e.intent_id] = e
            effect_ids_by_rule.setdefault(e.rule, set()).add(e.intent_id)

        if props:
            changed = _commit(state, snapshot, props, program.merges, config.float_tolerance)
        else:
            changed = set()

        # Build the *next* frontier without scanning unrelated rules.
        next_candidates: set[str] = set()
        for cell in changed:
            next_candidates |= readers.get(cell, set())
        for name in eligible:
            next_candidates |= after_successors.get(name, set())

        # A blocked rule is only retained if one of its explicit order
        # dependencies may already have become satisfied in this wave. Missing
        # data dependencies will naturally re-add it through the reader index.
        for name in blocked:
            r = rules[name]
            if r.after and r.after.issubset(completed_order) and r.reads.issubset(state.keys()):
                next_candidates.add(name)

        if not changed and not next_candidates:
            return finish()

        candidates = next_candidates

    raise ARCUnprovenConvergence(
        f"ARC did not reach a fixed point within {config.max_component_rounds} causal waves"
    )


# ---------------------------------------------------------------------------
# Convenience constructors used by tests and future frontends
# ---------------------------------------------------------------------------


def proposal_rule(
    name: str,
    *,
    reads: Iterable[str],
    writes: Iterable[str],
    fn: Callable[[Mapping[str, Any]], Mapping[str, Any] | Sequence[Proposal]],
    guard: GuardFn | None = None,
    mode: str = "set",
    after: Iterable[str] = (),
    demand_tags: Iterable[str] = (),
) -> ARCRule:
    reads_f = frozenset(reads)
    writes_f = frozenset(writes)

    def evaluate(snapshot: Mapping[str, Any]) -> RuleResult:
        raw = fn(snapshot)
        if isinstance(raw, Mapping):
            ps = tuple(
                Proposal(name, i, target, raw[target], mode)
                for i, target in enumerate(sorted(raw))
            )
        else:
            ps = tuple(raw)
        return RuleResult(ps)

    return ARCRule(
        name=name,
        reads=reads_f,
        writes=writes_f,
        evaluate=evaluate,
        guard=guard,
        after=frozenset(after),
        demand_tags=frozenset(demand_tags),
    )


def effect_rule(
    name: str,
    *,
    reads: Iterable[str],
    kind: str,
    payload: Callable[[Mapping[str, Any]], Any],
    guard: GuardFn | None = None,
    after: Iterable[str] = (),
    demand_tags: Iterable[str] = (),
) -> ARCRule:
    def evaluate(snapshot: Mapping[str, Any]) -> RuleResult:
        return RuleResult(effects=(EffectIntent(name, 0, kind, payload(snapshot)),))

    return ARCRule(
        name=name,
        reads=frozenset(reads),
        writes=frozenset(),
        evaluate=evaluate,
        guard=guard,
        after=frozenset(after),
        demand_tags=frozenset(demand_tags),
    )


def spread_rule(
    name: str,
    *,
    source: str,
    target: str,
    transform: Callable[[Any], Any],
    collection: str = "set",
    demand_tags: Iterable[str] = (),
) -> ARCRule:
    """Relational sequence traversal without a hidden loop primitive.

    Each member of ``source`` creates an independent proposal in the same
    snapshot.  The target is resolved by a declared collection law.  ``set``
    uses union; ``list`` creates a deterministic tuple indexed by source order.
    """

    def evaluate(snapshot: Mapping[str, Any]) -> RuleResult:
        xs = snapshot[source]
        if collection == "set":
            vals = {transform(x) for x in xs}
            return RuleResult((Proposal(name, 0, target, vals, "union"),))
        if collection == "list":
            vals = tuple(transform(x) for x in xs)
            return RuleResult((Proposal(name, 0, target, vals, "set"),))
        raise ARCInvalidProgram(f"unknown spread collection mode {collection!r}")

    return ARCRule(
        name=name,
        reads=frozenset({source}),
        writes=frozenset({target}),
        evaluate=evaluate,
        demand_tags=frozenset(demand_tags),
    )


__all__ = [
    "ARCError", "ARCConflict", "ARCOscillation", "ARCUnprovenConvergence",
    "ARCResourcePressure", "ARCInvalidProgram", "ARCIncompleteResult", "validate_finite_value", "Proposal", "EffectIntent",
    "RuleResult", "ARCRule", "ARCProgram", "ARCConfig", "ARCStats", "ARCResult",
    "execute_arc", "proposal_rule", "effect_rule", "spread_rule",
]

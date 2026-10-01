from __future__ import annotations

"""L0 Language v0.4 — native activation/settling control semantics.

This layer intentionally removes foreign control-flow concepts from the L0 core.
Python/JS/C/C++/Rust may contain if/while/function syntax, but after the v0.3
frontends they are lowered again into a smaller L0-native model:

    state/values + transformations + activation + settling + effects

Fundamental control primitives
------------------------------
flow      target <- expression
active    [guard] { ... }       # execute region once if its activation is true
settle    settle [guard] { ... }# re-apply while activation remains true
store     memory[index] <- value
effect    yield/call/emit/...

A branch is not retained as a primitive. Its condition is snapshotted into a
hidden activation cell and the two alternatives become complementary active
regions. A while-loop is not retained as a primitive; it becomes a settling
region. The evaluator's iteration budget is an execution-safety boundary, not
part of program semantics.

This module does not change the numerical L0 v0.24 kernel.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .semantic import SExpr, SFunction, SOp, SemanticModule, compile_semantic


class ControlLoweringError(ValueError):
    pass


class ControlExecutionError(RuntimeError):
    pass


class ControlDivergenceError(ControlExecutionError):
    pass


@dataclass(frozen=True)
class CAction:
    kind: str
    target: str | None = None
    operator: str | None = None
    args: tuple[SExpr, ...] = ()
    guard: SExpr | None = None
    body: tuple["CAction", ...] = ()
    resolved: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> Any:
        return (
            self.kind,
            self.target,
            self.operator,
            tuple(a.signature() for a in self.args),
            self.guard.signature() if self.guard else None,
            tuple(x.signature() for x in self.body),
            bool(self.resolved),
        )


@dataclass(frozen=True)
class CRelation:
    name: str
    params: tuple[str, ...]
    body: tuple[CAction, ...]
    resolved: bool = True
    defaults: tuple[SExpr, ...] = ()

    def signature(self) -> Any:
        return (self.name, self.params, tuple(x.signature() for x in self.body), self.resolved,
                tuple(e.signature() for e in self.defaults))


@dataclass(frozen=True)
class ControlModule:
    name: str
    source_language: str
    relations: tuple[CRelation, ...] = ()
    top_level: tuple[CAction, ...] = ()
    unresolved_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> Any:
        return (
            tuple(r.signature() for r in self.relations),
            tuple(a.signature() for a in self.top_level),
        )

    def fingerprint(self) -> str:
        raw = json.dumps(self.signature(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _LowerContext:
    guard_counter: int = 0
    reserved: set[str] = field(default_factory=set)

    def fresh_guard(self) -> str:
        while True:
            name = f"__l0_activation_{self.guard_counter}"
            self.guard_counter += 1
            if name not in self.reserved:
                break
        self.reserved.add(name)
        return name


def _not(e: SExpr) -> SExpr:
    return SExpr.unary("not", e, resolved=e.resolved)


def _lower_ops(ops: Iterable[SOp], ctx: _LowerContext) -> tuple[CAction, ...]:
    out: list[CAction] = []
    for op in ops:
        if op.kind == "assign":
            if len(op.args) != 1 or op.target is None:
                out.append(CAction("unresolved", operator="malformed_assign", resolved=False))
            else:
                out.append(CAction("flow", target=op.target, args=(op.args[0],), resolved=op.resolved))
            continue

        if op.kind == "declare":
            out.append(CAction("declare", target=op.target, resolved=op.resolved))
            continue

        if op.kind == "memory_store":
            if len(op.args) != 3:
                out.append(CAction("unresolved", operator="malformed_memory_store", resolved=False))
            else:
                out.append(CAction("store", args=op.args, resolved=op.resolved))
            continue

        if op.kind == "branch":
            if op.condition is None:
                out.append(CAction("unresolved", operator="branch_without_condition", resolved=False))
                continue
            # Snapshot activation before either alternative mutates state. This
            # gives mutually exclusive regions without retaining branch/else as
            # fundamental control primitives.
            g = ctx.fresh_guard()
            out.append(CAction("flow", target=g, args=(op.condition,), resolved=op.resolved and op.condition.resolved,
                               meta={"internal": "activation_snapshot"}))
            then_body = _lower_ops(op.body, ctx)
            out.append(CAction("active", guard=SExpr.name(g), body=then_body,
                               resolved=op.resolved and all(x.resolved for x in then_body)))
            if op.else_body:
                else_body = _lower_ops(op.else_body, ctx)
                out.append(CAction("active", guard=_not(SExpr.name(g)), body=else_body,
                                   resolved=op.resolved and all(x.resolved for x in else_body)))
            continue

        if op.kind == "loop" and op.operator == "while":
            if op.condition is None:
                out.append(CAction("unresolved", operator="while_without_condition", resolved=False))
                continue
            body = _lower_ops(op.body, ctx)
            out.append(CAction("settle", guard=op.condition, body=body,
                               resolved=op.resolved and op.condition.resolved and all(x.resolved for x in body),
                               meta={"lowered_from": "while"}))
            continue

        if op.kind == "loop":
            # for/for_each need iterator semantics; do not silently pretend they
            # are equivalent to settling under a predicate.
            body = _lower_ops(op.body, ctx)
            out.append(CAction("unresolved", target=op.target, operator=f"loop:{op.operator}",
                               args=op.args, guard=op.condition, body=body, resolved=False,
                               meta={"reason": "iterator semantics not yet derived"}))
            continue

        if op.kind == "return":
            out.append(CAction("effect", operator="yield", args=op.args, resolved=op.resolved))
            continue

        if op.kind == "call":
            out.append(CAction("effect", operator="call", args=op.args, resolved=op.resolved,
                               meta={"callee": op.operator}))
            continue

        if op.kind == "expr":
            out.append(CAction("effect", operator="evaluate", args=op.args, resolved=op.resolved))
            continue

        if op.kind == "relation":
            out.append(CAction("relation", target=op.target, operator=op.operator, args=op.args,
                               resolved=op.resolved, meta=dict(op.meta)))
            continue

        if op.kind == "event":
            body = _lower_ops(op.body, ctx)
            out.append(CAction("effect_region", operator=f"event:{op.operator}", args=op.args,
                               body=body, resolved=op.resolved and all(x.resolved for x in body),
                               meta=dict(op.meta)))
            continue

        if op.kind == "unresolved":
            out.append(CAction("unresolved", target=op.target, operator=op.operator, args=op.args,
                               guard=op.condition, body=_lower_ops(op.body, ctx), resolved=False,
                               meta=dict(op.meta)))
            continue

        out.append(CAction("unresolved", target=op.target, operator=f"semantic:{op.kind}", args=op.args,
                           guard=op.condition, body=_lower_ops(op.body, ctx), resolved=False,
                           meta={"source_kind": op.kind, **dict(op.meta)}))
    return tuple(out)


def _count_unresolved(actions: Iterable[CAction]) -> int:
    n = 0
    for a in actions:
        n += 0 if a.resolved else 1
        n += _count_unresolved(a.body)
    return n


def lower_semantic_to_control(module: SemanticModule) -> ControlModule:
    reserved: set[str] = set()
    def collect(value):
        if isinstance(value, str):reserved.add(value)
        elif isinstance(value, dict):
            for k,v in value.items():collect(k);collect(v)
        elif isinstance(value, (tuple,list)):
            for v in value:collect(v)
    collect(module.to_dict())
    ctx = _LowerContext(reserved=reserved)
    rels: list[CRelation] = []
    for f in module.functions:
        body = _lower_ops(f.body, ctx)
        rels.append(CRelation(f.name, f.params, body, resolved=f.resolved and all(x.resolved for x in body), defaults=f.defaults))
    top = _lower_ops(module.top_level, ctx)
    unresolved = sum((0 if r.resolved else 1) + _count_unresolved(r.body) for r in rels) + _count_unresolved(top)
    return ControlModule(
        name=module.name,
        source_language=module.source_language,
        relations=tuple(rels),
        top_level=top,
        unresolved_count=unresolved,
        metadata={
            "frontend": module.frontend,
            "control_semantics": "activation-settling-v0.4",
            "foreign_unresolved": module.unresolved_count,
        },
    )


def compile_control(source: str, language: str, name: str = "module") -> ControlModule:
    return lower_semantic_to_control(compile_semantic(source, language, name))


# ---------------------------------------------------------------------------
# Reference evaluator for the native control semantics
# ---------------------------------------------------------------------------


def _c_div(a, b):
    if b == 0:raise ZeroDivisionError('C division by zero')
    if isinstance(a,int) and isinstance(b,int):
        q=abs(a)//abs(b)
        return -q if (a<0)!=(b<0) else q
    return a/b


def _c_mod(a, b):
    if not isinstance(a,int) or not isinstance(b,int):
        raise ControlExecutionError('C remainder requires integer operands')
    return a-_c_div(a,b)*b


_BIN = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "floordiv": lambda a, b: a // b,
    "mod": lambda a, b: a % b,
    "c_div": _c_div,
    "c_mod": _c_mod,
    "pow": lambda a, b: a ** b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "and": lambda a, b: bool(a) and bool(b),
    "or": lambda a, b: bool(a) or bool(b),
    "bit_and": lambda a, b: a & b,
    "bit_or": lambda a, b: a | b,
    "bit_xor": lambda a, b: a ^ b,
}
_UN = {
    "c_int": int,
    "c_float": float,
    "not": lambda a: not bool(a),
    "neg": lambda a: -a,
    "pos": lambda a: +a,
    "bit_not": lambda a: ~a,
}


@dataclass(frozen=True)
class _Yield:
    value: Any


def _eval_expr(e: SExpr, env: dict[str, Any], module: ControlModule, budget: int) -> Any:
    if not e.resolved:
        raise ControlExecutionError(f"cannot evaluate unresolved expression: {e.signature()}")
    if e.kind == "name":
        if e.value not in env:
            raise ControlExecutionError(f"unknown state value: {e.value}")
        return env[e.value]
    if e.kind == "const":
        return e.value
    if e.kind == "binary":
        if e.operator in {"and","or","value_and","value_or"}:
            left = _eval_expr(e.args[0], env, module, budget)
            take_right = bool(left) if e.operator in {"and","value_and"} else not bool(left)
            value = _eval_expr(e.args[1], env, module, budget) if take_right else left
            return value if e.operator.startswith('value_') else bool(value)
        fn = _BIN.get(str(e.operator))
        if fn is None:
            raise ControlExecutionError(f"unsupported binary operator: {e.operator}")
        return fn(_eval_expr(e.args[0], env, module, budget), _eval_expr(e.args[1], env, module, budget))
    if e.kind == "unary":
        fn = _UN.get(str(e.operator))
        if fn is None:
            raise ControlExecutionError(f"unsupported unary operator: {e.operator}")
        return fn(_eval_expr(e.args[0], env, module, budget))
    if e.kind == "memory_load":
        base = _eval_expr(e.args[0], env, module, budget)
        idx = _eval_expr(e.args[1], env, module, budget)
        return base[idx]
    if e.kind == "member":
        # Member values are preserved structurally but not assigned Python-like
        # reflection semantics in the L0 reference evaluator.
        raise ControlExecutionError(f"member access requires adapter semantics: {e.value}")
    if e.kind == "call":
        vals = [_eval_expr(a, env, module, budget) for a in e.args]
        # First prefer L0 relations in the module.
        rel = next((r for r in module.relations if r.name == e.operator), None)
        if rel is not None:
            return execute_relation(module, rel.name, vals, max_settle_steps=budget)
        builtins = {
            "abs": abs, "min": min, "max": max, "len": len,
            "float": float,
            "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
        }
        fn = builtins.get(str(e.operator))
        if fn is None:
            raise ControlExecutionError(f"unknown relation call: {e.operator}")
        return fn(*vals)
    raise ControlExecutionError(f"unsupported expression kind: {e.kind}")


def _execute_actions(actions: Iterable[CAction], env: dict[str, Any], module: ControlModule,
                     max_settle_steps: int) -> _Yield | None:
    for action in actions:
        if not action.resolved:
            raise ControlExecutionError(f"cannot execute unresolved action: {action.signature()}")

        if action.kind == "declare":
            if action.target is not None and action.target not in env:
                env[action.target] = None
            continue

        if action.kind == "flow":
            if action.target is None or len(action.args) != 1:
                raise ControlExecutionError("malformed flow")
            env[action.target] = _eval_expr(action.args[0], env, module, max_settle_steps)
            continue

        if action.kind == "store":
            if len(action.args) != 3:
                raise ControlExecutionError("malformed store")
            base = _eval_expr(action.args[0], env, module, max_settle_steps)
            idx = _eval_expr(action.args[1], env, module, max_settle_steps)
            val = _eval_expr(action.args[2], env, module, max_settle_steps)
            base[idx] = val
            continue

        if action.kind == "active":
            if action.guard is None:
                raise ControlExecutionError("active region has no guard")
            if bool(_eval_expr(action.guard, env, module, max_settle_steps)):
                sig = _execute_actions(action.body, env, module, max_settle_steps)
                if sig is not None:
                    return sig
            continue

        if action.kind == "settle":
            if action.guard is None:
                raise ControlExecutionError("settle region has no guard")
            steps = 0
            while bool(_eval_expr(action.guard, env, module, max_settle_steps)):
                if steps >= max_settle_steps:
                    raise ControlDivergenceError(f"settling did not converge within {max_settle_steps} microsteps")
                sig = _execute_actions(action.body, env, module, max_settle_steps)
                if sig is not None:
                    return sig
                steps += 1
            continue

        if action.kind == "effect" and action.operator == "yield":
            val = None if not action.args else _eval_expr(action.args[0], env, module, max_settle_steps)
            return _Yield(val)

        if action.kind == "effect" and action.operator == "call":
            callee = action.meta.get("callee")
            vals = [_eval_expr(a, env, module, max_settle_steps) for a in action.args]
            if callee:
                rel = next((r for r in module.relations if r.name == callee), None)
                if rel is not None:
                    execute_relation(module, rel.name, vals, max_settle_steps=max_settle_steps)
                    continue
            raise ControlExecutionError(f"side-effect call requires adapter: {callee}")

        if action.kind in {"relation", "effect_region"}:
            # These are interface declarations/events. They are preserved but not
            # auto-fired by the pure reference relation evaluator.
            continue

        raise ControlExecutionError(f"unsupported action kind: {action.kind}")
    return None


def execute_relation(module: ControlModule, relation_name: str, args: Iterable[Any], *,
                     max_settle_steps: int = 10_000) -> Any:
    rel = next((r for r in module.relations if r.name == relation_name), None)
    if rel is None:
        raise ControlExecutionError(f"unknown relation: {relation_name}")
    if not rel.resolved:
        raise ControlExecutionError(f"relation contains unsupported semantics: {relation_name}")
    vals = list(args)
    minimum = len(rel.params)-len(rel.defaults)
    if not minimum <= len(vals) <= len(rel.params):
        raise ControlExecutionError(f"{relation_name} expects {minimum}..{len(rel.params)} args, got {len(vals)}")
    for default in rel.defaults[len(vals)-minimum:]:
        vals.append(_eval_expr(default, {}, module, max_settle_steps))
    env = dict(zip(rel.params, vals))
    sig = _execute_actions(rel.body, env, module, max_settle_steps)
    return None if sig is None else sig.value


# ---------------------------------------------------------------------------
# L0-native textual projection (debug/specification surface, not parser yet)
# ---------------------------------------------------------------------------


def _expr_text(e: SExpr) -> str:
    if e.kind == "name":
        return str(e.value)
    if e.kind == "const":
        if e.value is None:return 'null'
        if isinstance(e.value,bool):return repr(e.value).lower()
        if isinstance(e.value,(int,float)) and e.value<0:return f'neg({repr(-e.value)})'
        if isinstance(e.value,(list,tuple,dict,set)):
            # Debug projection must not silently discard an unrepresentable default.
            raise ControlLoweringError('container literals require the structured Control IR interface')
        return repr(e.value)
    if e.kind == "binary":
        return f"{e.operator}({_expr_text(e.args[0])}, {_expr_text(e.args[1])})"
    if e.kind == "unary":
        return f"{e.operator}({_expr_text(e.args[0])})"
    if e.kind == "call":
        return f"{e.operator}({', '.join(_expr_text(a) for a in e.args)})"
    if e.kind == "memory_load":
        return f"load({_expr_text(e.args[0])}, {_expr_text(e.args[1])})"
    return f"?{e.operator or e.kind}"


def _actions_text(actions: Iterable[CAction], indent: int = 2) -> list[str]:
    p = " " * indent
    lines: list[str] = []
    for a in actions:
        if a.kind == "flow":
            lines.append(f"{p}{a.target} <- {_expr_text(a.args[0])}")
        elif a.kind == "declare":
            lines.append(f"{p}state {a.target}")
        elif a.kind == "store":
            lines.append(f"{p}store({_expr_text(a.args[0])}, {_expr_text(a.args[1])}) <- {_expr_text(a.args[2])}")
        elif a.kind == "active":
            lines.append(f"{p}[{_expr_text(a.guard)}] {{")
            lines.extend(_actions_text(a.body, indent + 2))
            lines.append(f"{p}}}")
        elif a.kind == "settle":
            lines.append(f"{p}settle [{_expr_text(a.guard)}] {{")
            lines.extend(_actions_text(a.body, indent + 2))
            lines.append(f"{p}}}")
        elif a.kind == "effect" and a.operator == "yield":
            lines.append(f"{p}yield" + (" " + _expr_text(a.args[0]) if a.args else ""))
        elif a.kind == "relation":
            lines.append(f"{p}({', '.join(_expr_text(x) for x in a.args)}) -[{a.operator}]-> {a.target}")
        elif a.kind == "effect_region":
            lines.append(f"{p}@{a.operator} {{")
            lines.extend(_actions_text(a.body, indent + 2))
            lines.append(f"{p}}}")
        else:
            lines.append(f"{p}? {a.kind}:{a.operator or ''}")
    return lines


def render_control(module: ControlModule) -> str:
    lines = [f"module {module.name}"]
    for r in module.relations:
        lines.append("")
        start=len(r.params)-len(r.defaults)
        params=[p+(f' <- {_expr_text(r.defaults[i-start])}' if i>=start else '') for i,p in enumerate(r.params)]
        lines.append(f"relation {r.name}({', '.join(params)}) {{")
        lines.extend(_actions_text(r.body, 2))
        lines.append("}")
    if module.top_level:
        lines.append("")
        lines.append("surface {")
        lines.extend(_actions_text(module.top_level, 2))
        lines.append("}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="L0 v0.4 control-semantics compiler")
    ap.add_argument("source", type=Path)
    ap.add_argument("--language", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    src = args.source.read_text(encoding="utf-8")
    mod = compile_control(src, args.language, args.source.stem)
    if args.json:
        print(json.dumps(mod.to_dict(), indent=2, default=str))
    else:
        print(render_control(mod), end="")
        print("fingerprint", mod.fingerprint())
        print("unresolved", mod.unresolved_count)

from __future__ import annotations

"""L0 ARC v0.6 — native syntax compiler.

This module is the source-language frontend for ARC (Activated Relational
Convergence).  It parses a native `.l0` language into the v0.5 ARC runtime.

Design rules
------------
* No user-authored scheduler order.  The causal frontier is derived from
  declared read/write dependencies.
* Conflicting writes require an explicit merge law or equal `set` proposals.
* Effects are staged intents and are committed only after convergence.
* `spread` is a relational collection transform, not a hidden integer loop.
* Cyclic execution is bounded by an explicit convergence contract.
* Expressions use a small deterministic operation registry; arbitrary Python
  execution is intentionally not part of the language frontend.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import hashlib
import json
import math

from .arc import (
    ARCConfig,
    ARCInvalidProgram,
    ARCProgram,
    ARCRule,
    EffectIntent,
    Proposal,
    RuleResult,
    execute_arc,
    validate_finite_value,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class L0ARCParseError(ValueError):
    pass


class L0ARCSemanticError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int
    col: int


_KEYWORDS = {
    "program", "semantics", "arc", "input", "cell", "observe", "merge", "by", "relation", "when",
    "after", "propose", "effect", "stage", "spread", "item", "collection",
    "transform", "convergence", "tolerance", "waves", "evaluations",
    "proposals", "oscillation", "detect", "ignore", "set", "delta",
    "union", "min", "max", "list", "true", "false", "null",
}


class Lexer:
    def __init__(self, text: str):
        self.text = text

    def tokens(self) -> list[Token]:
        out: list[Token] = []
        i = 0
        line = 1
        col = 1
        n = len(self.text)
        while i < n:
            ch = self.text[i]
            if ch in " \t\r":
                i += 1
                col += 1
                continue
            if ch == "\n":
                out.append(Token("NEWLINE", "\n", line, col))
                i += 1
                line += 1
                col = 1
                continue
            if ch == "#":
                while i < n and self.text[i] != "\n":
                    i += 1
                    col += 1
                continue
            if self.text.startswith("->", i):
                out.append(Token("ARROW", "->", line, col))
                i += 2
                col += 2
                continue
            if ch in "{}(),=:[]":
                out.append(Token(ch, ch, line, col))
                i += 1
                col += 1
                continue
            if ch in "'\"":
                quote = ch
                start_line, start_col = line, col
                i += 1
                col += 1
                buf: list[str] = []
                while i < n and self.text[i] != quote:
                    if self.text[i] == "\\":
                        i += 1
                        col += 1
                        if i >= n:
                            raise L0ARCParseError(f"unterminated string at {start_line}:{start_col}")
                        esc = self.text[i]
                        mapping = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"'}
                        buf.append(mapping.get(esc, esc))
                        i += 1
                        col += 1
                    else:
                        if self.text[i] == "\n":
                            raise L0ARCParseError(f"newline in string at {line}:{col}")
                        buf.append(self.text[i])
                        i += 1
                        col += 1
                if i >= n:
                    raise L0ARCParseError(f"unterminated string at {start_line}:{start_col}")
                i += 1
                col += 1
                out.append(Token("STRING", "".join(buf), start_line, start_col))
                continue
            if ch.isdigit() or (ch == "." and i + 1 < n and self.text[i + 1].isdigit()):
                start_i, start_col = i, col
                saw_dot = False
                while i < n:
                    c = self.text[i]
                    if c.isdigit():
                        i += 1; col += 1
                    elif c == "." and not saw_dot:
                        saw_dot = True; i += 1; col += 1
                    elif c in "eE":
                        i += 1; col += 1
                        if i < n and self.text[i] in "+-":
                            i += 1; col += 1
                    else:
                        break
                raw = self.text[start_i:i]
                try:
                    number = float(raw)
                    if any(c in raw for c in ".eE") and not math.isfinite(number):
                        raise ValueError("non-finite number")
                except ValueError as exc:
                    raise L0ARCParseError(f"invalid number {raw!r} at {line}:{start_col}") from exc
                out.append(Token("NUMBER", raw, line, start_col))
                continue
            if ch.isalpha() or ch == "_":
                start_i, start_col = i, col
                i += 1; col += 1
                while i < n and (self.text[i].isalnum() or self.text[i] == "_"):
                    i += 1; col += 1
                raw = self.text[start_i:i]
                kind = raw if raw in _KEYWORDS else "IDENT"
                out.append(Token(kind, raw, line, start_col))
                continue
            raise L0ARCParseError(f"unexpected character {ch!r} at {line}:{col}")
        out.append(Token("EOF", "", line, col))
        return out


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expr:
    kind: str  # name | number | string | bool | null | call
    value: Any
    args: tuple["Expr", ...] = ()


@dataclass(frozen=True)
class InputDecl:
    name: str
    default: Expr | None = None


@dataclass(frozen=True)
class CellDecl:
    name: str
    initial: Expr | None = None


@dataclass(frozen=True)
class ProposalDecl:
    target: str
    mode: str
    expr: Expr


@dataclass(frozen=True)
class RelationDecl:
    name: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    guard: Expr | None
    after: tuple[str, ...]
    proposals: tuple[ProposalDecl, ...]


@dataclass(frozen=True)
class EffectDecl:
    name: str
    reads: tuple[str, ...]
    guard: Expr | None
    after: tuple[str, ...]
    kind: str
    payload: Expr


@dataclass(frozen=True)
class SpreadDecl:
    name: str
    source: str
    target: str
    item: str
    collection: str
    transform: Expr


@dataclass(frozen=True)
class ConvergenceDecl:
    tolerance: float = 0.0
    waves: int = 10_000
    evaluations: int = 1_000_000
    proposals: int = 1_000_000
    oscillation: str = "detect"


@dataclass(frozen=True)
class ProgramAST:
    name: str
    inputs: tuple[InputDecl, ...]
    cells: tuple[CellDecl, ...]
    observes: tuple[str, ...]
    merges: tuple[tuple[str, str], ...]
    relations: tuple[RelationDecl, ...]
    effects: tuple[EffectDecl, ...]
    spreads: tuple[SpreadDecl, ...]
    convergence: ConvergenceDecl


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Parser:
    def __init__(self, text: str):
        self.ts = Lexer(text).tokens()
        self.i = 0

    def cur(self) -> Token:
        return self.ts[self.i]

    def accept(self, kind: str) -> Token | None:
        if self.cur().kind == kind:
            t = self.cur(); self.i += 1; return t
        return None

    def expect(self, kind: str) -> Token:
        t = self.cur()
        if t.kind != kind:
            raise L0ARCParseError(f"expected {kind}, got {t.kind} {t.value!r} at {t.line}:{t.col}")
        self.i += 1
        return t

    def skip_newlines(self) -> None:
        while self.accept("NEWLINE"):
            pass

    def end_line(self) -> None:
        if self.cur().kind not in {"NEWLINE", "}", "EOF"}:
            t = self.cur()
            raise L0ARCParseError(f"expected end of line at {t.line}:{t.col}, got {t.value!r}")
        self.skip_newlines()

    def ident(self) -> str:
        return self.expect("IDENT").value

    def name_list_paren(self) -> tuple[str, ...]:
        self.expect("(")
        vals: list[str] = []
        if self.cur().kind != ")":
            vals.append(self.ident())
            while self.accept(","):
                vals.append(self.ident())
        self.expect(")")
        return tuple(vals)

    def ident_list_to_eol(self) -> tuple[str, ...]:
        vals = [self.ident()]
        while self.accept(","):
            vals.append(self.ident())
        return tuple(vals)

    def expr(self) -> Expr:
        t = self.cur()
        if t.kind == "NUMBER":
            self.i += 1
            val = float(t.value) if any(c in t.value for c in ".eE") else int(t.value)
            return Expr("number", val)
        if t.kind == "STRING":
            self.i += 1
            return Expr("string", t.value)
        if t.kind == "true":
            self.i += 1; return Expr("bool", True)
        if t.kind == "false":
            self.i += 1; return Expr("bool", False)
        if t.kind == "null":
            self.i += 1; return Expr("null", None)
        if t.kind in {"IDENT", "set", "min", "max"}:
            name = t.value; self.i += 1
            if self.accept("("):
                args: list[Expr] = []
                if self.cur().kind != ")":
                    args.append(self.expr())
                    while self.accept(","):
                        args.append(self.expr())
                self.expect(")")
                return Expr("call", name, tuple(args))
            return Expr("name", name)
        raise L0ARCParseError(f"expected expression at {t.line}:{t.col}, got {t.value!r}")

    def relation(self) -> RelationDecl:
        self.expect("relation")
        name = self.ident()
        reads = self.name_list_paren()
        self.expect("ARROW")
        writes = self.name_list_paren()
        self.expect("{")
        self.skip_newlines()
        guard = None
        after: tuple[str, ...] = ()
        props: list[ProposalDecl] = []
        while self.cur().kind != "}":
            if self.accept("when"):
                if guard is not None:
                    raise L0ARCParseError(f"relation {name} has duplicate when directive")
                guard = self.expr(); self.end_line(); continue
            if self.accept("after"):
                if after:
                    raise L0ARCParseError(f"relation {name} has duplicate after directive")
                after = self.ident_list_to_eol(); self.end_line(); continue
            if self.accept("propose"):
                target = self.ident()
                mode_t = self.cur()
                if mode_t.kind not in {"set", "delta", "union", "min", "max"}:
                    raise L0ARCParseError(f"expected proposal mode at {mode_t.line}:{mode_t.col}")
                self.i += 1
                ex = self.expr()
                props.append(ProposalDecl(target, mode_t.value, ex))
                self.end_line(); continue
            t = self.cur()
            raise L0ARCParseError(f"unexpected token in relation {name}: {t.value!r} at {t.line}:{t.col}")
        self.expect("}")
        self.skip_newlines()
        return RelationDecl(name, reads, writes, guard, after, tuple(props))

    def effect(self) -> EffectDecl:
        self.expect("effect")
        name = self.ident()
        reads = self.name_list_paren()
        self.expect("{")
        self.skip_newlines()
        guard = None
        after: tuple[str, ...] = ()
        kind = None
        payload = None
        while self.cur().kind != "}":
            if self.accept("when"):
                if guard is not None:
                    raise L0ARCParseError(f"effect {name} has duplicate when directive")
                guard = self.expr(); self.end_line(); continue
            if self.accept("after"):
                if after:
                    raise L0ARCParseError(f"effect {name} has duplicate after directive")
                after = self.ident_list_to_eol(); self.end_line(); continue
            if self.accept("stage"):
                if kind is not None:
                    raise L0ARCParseError(f"effect {name} has duplicate stage directive")
                kind = self.ident()
                payload = self.expr()
                self.end_line(); continue
            t = self.cur()
            raise L0ARCParseError(f"unexpected token in effect {name}: {t.value!r} at {t.line}:{t.col}")
        self.expect("}")
        self.skip_newlines()
        if kind is None or payload is None:
            raise L0ARCParseError(f"effect {name} requires exactly one stage directive")
        return EffectDecl(name, reads, guard, after, kind, payload)

    def spread(self) -> SpreadDecl:
        self.expect("spread")
        name = self.ident()
        source = self.ident()
        self.expect("ARROW")
        target = self.ident()
        self.expect("{")
        self.skip_newlines()
        item = None
        collection = None
        transform = None
        while self.cur().kind != "}":
            if self.accept("item"):
                if item is not None:
                    raise L0ARCParseError(f"spread {name} has duplicate item directive")
                item = self.ident(); self.end_line(); continue
            if self.accept("collection"):
                if collection is not None:
                    raise L0ARCParseError(f"spread {name} has duplicate collection directive")
                t = self.cur()
                if t.kind not in {"set", "list"}:
                    raise L0ARCParseError(f"spread collection must be set or list at {t.line}:{t.col}")
                self.i += 1; collection = t.value; self.end_line(); continue
            if self.accept("transform"):
                if transform is not None:
                    raise L0ARCParseError(f"spread {name} has duplicate transform directive")
                transform = self.expr(); self.end_line(); continue
            t = self.cur()
            raise L0ARCParseError(f"unexpected token in spread {name}: {t.value!r} at {t.line}:{t.col}")
        self.expect("}")
        self.skip_newlines()
        if item is None or collection is None or transform is None:
            raise L0ARCParseError(f"spread {name} requires item, collection, transform")
        return SpreadDecl(name, source, target, item, collection, transform)

    def convergence(self) -> ConvergenceDecl:
        self.expect("convergence")
        self.expect("{")
        self.skip_newlines()
        vals: dict[str, Any] = {}
        while self.cur().kind != "}":
            t = self.cur()
            if t.kind in {"tolerance", "waves", "evaluations", "proposals"}:
                self.i += 1
                num = self.expect("NUMBER")
                value = float(num.value) if t.kind == "tolerance" else int(float(num.value))
                if t.value in vals:
                    raise L0ARCParseError(f"duplicate convergence option {t.value!r}")
                vals[t.value] = value
                self.end_line(); continue
            if self.accept("oscillation"):
                m = self.cur()
                if m.kind not in {"detect", "ignore"}:
                    raise L0ARCParseError(f"oscillation must be detect or ignore at {m.line}:{m.col}")
                self.i += 1
                if "oscillation" in vals:
                    raise L0ARCParseError("duplicate convergence option 'oscillation'")
                vals["oscillation"] = m.value
                self.end_line(); continue
            raise L0ARCParseError(f"unexpected convergence option {t.value!r} at {t.line}:{t.col}")
        self.expect("}")
        self.skip_newlines()
        return ConvergenceDecl(**vals)

    def parse(self) -> ProgramAST:
        self.skip_newlines()
        self.expect("program")
        name = self.ident()
        self.end_line()
        self.expect("semantics")
        self.expect("arc")
        self.end_line()
        inputs: list[InputDecl] = []
        cells: list[CellDecl] = []
        observes: list[str] = []
        merges: list[tuple[str, str]] = []
        relations: list[RelationDecl] = []
        effects: list[EffectDecl] = []
        spreads: list[SpreadDecl] = []
        conv: ConvergenceDecl | None = None

        while self.cur().kind != "EOF":
            self.skip_newlines()
            if self.cur().kind == "EOF":
                break
            if self.accept("input"):
                c = self.ident()
                default = self.expr() if self.accept("=") else None
                inputs.append(InputDecl(c, default)); self.end_line(); continue
            if self.accept("cell"):
                c = self.ident()
                init = self.expr() if self.accept("=") else None
                cells.append(CellDecl(c, init)); self.end_line(); continue
            if self.accept("observe"):
                observes.append(self.ident()); self.end_line(); continue
            if self.accept("merge"):
                c = self.ident(); self.expect("by")
                m = self.cur()
                if m.kind not in {"set", "delta", "union", "min", "max"}:
                    raise L0ARCParseError(f"unknown merge law {m.value!r} at {m.line}:{m.col}")
                self.i += 1
                merges.append((c, m.value)); self.end_line(); continue
            if self.cur().kind == "relation":
                relations.append(self.relation()); continue
            if self.cur().kind == "effect":
                effects.append(self.effect()); continue
            if self.cur().kind == "spread":
                spreads.append(self.spread()); continue
            if self.cur().kind == "convergence":
                if conv is not None:
                    raise L0ARCParseError("only one convergence block is allowed")
                conv = self.convergence(); continue
            t = self.cur()
            raise L0ARCParseError(f"unexpected top-level token {t.value!r} at {t.line}:{t.col}")

        return ProgramAST(
            name=name,
            inputs=tuple(inputs),
            cells=tuple(cells),
            observes=tuple(observes),
            merges=tuple(merges),
            relations=tuple(relations),
            effects=tuple(effects),
            spreads=tuple(spreads),
            convergence=conv or ConvergenceDecl(),
        )


def parse_l0_arc(text: str) -> ProgramAST:
    return Parser(text).parse()


# ---------------------------------------------------------------------------
# Deterministic expression semantics
# ---------------------------------------------------------------------------


def _safe_div(a, b):
    if b == 0:
        raise ZeroDivisionError("L0 div by zero")
    return a / b


def _op_and(*xs): return all(bool(x) for x in xs)
def _op_or(*xs): return any(bool(x) for x in xs)
def _op_not(x): return not bool(x)
def _op_pair(a, b): return (a, b)
def _op_tuple(*xs): return tuple(xs)
def _op_set(*xs): return set(xs)
def _op_len(x): return len(x)
def _op_contains(x, y): return y in x




def _vec(v):
    if isinstance(v,(str,bytes)):
        raise L0ARCSemanticError("vector operation requires a numeric sequence")
    try: return tuple(float(x) for x in v)
    except Exception as exc: raise L0ARCSemanticError("vector operation requires a numeric sequence") from exc

def _op_at(v,i):
    vv=_vec(v); ii=int(i)
    if float(i)!=ii: raise L0ARCSemanticError("at index must be an integer")
    try: return vv[ii]
    except IndexError as exc: raise L0ARCSemanticError("at index out of range") from exc

def _op_dot(a,b):
    aa,bb=_vec(a),_vec(b)
    if len(aa)!=len(bb): raise L0ARCSemanticError("dot requires equal vector lengths")
    return sum(x*y for x,y in zip(aa,bb))

def _op_vadd(a,b):
    aa,bb=_vec(a),_vec(b)
    if len(aa)!=len(bb): raise L0ARCSemanticError("vadd requires equal vector lengths")
    return tuple(x+y for x,y in zip(aa,bb))

def _op_vsub(a,b):
    aa,bb=_vec(a),_vec(b)
    if len(aa)!=len(bb): raise L0ARCSemanticError("vsub requires equal vector lengths")
    return tuple(x-y for x,y in zip(aa,bb))

def _op_vscale(v,s):
    vv=_vec(v); ss=float(s); return tuple(ss*x for x in vv)

def _op_vnorm(v):
    vv=_vec(v); return math.sqrt(sum(x*x for x in vv))

CORE_OPS: dict[str, Callable[..., Any]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": _safe_div,
    "pow": lambda a, b: a ** b,
    "neg": lambda a: -a,
    "abs": abs,
    "min": min,
    "max": max,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "and": _op_and,
    "or": _op_or,
    "not": _op_not,
    "pair": _op_pair,
    "tuple": _op_tuple,
    "set": _op_set,
    "len": _op_len,
    "contains": _op_contains,
    "at": _op_at,
    "dot": _op_dot,
    "vadd": _op_vadd,
    "vsub": _op_vsub,
    "vscale": _op_vscale,
    "vnorm": _op_vnorm,
}


def expr_names(expr: Expr) -> set[str]:
    if expr.kind == "name":
        return {str(expr.value)}
    out: set[str] = set()
    for a in expr.args:
        out |= expr_names(a)
    return out


def eval_expr(expr: Expr, env: Mapping[str, Any], ops: Mapping[str, Callable[..., Any]] = CORE_OPS) -> Any:
    if expr.kind in {"number", "string", "bool", "null"}:
        return expr.value
    if expr.kind == "name":
        if expr.value not in env:
            raise L0ARCSemanticError(f"unknown value {expr.value!r}")
        return env[expr.value]
    if expr.kind == "call":
        if expr.value not in ops:
            raise L0ARCSemanticError(f"unknown operation {expr.value!r}")
        return ops[expr.value](*(eval_expr(a, env, ops) for a in expr.args))
    raise L0ARCSemanticError(f"unknown expression kind {expr.kind!r}")


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledL0ARC:
    ast: ProgramAST
    program: ARCProgram
    config: ARCConfig
    initial_state: Mapping[str, Any]
    required_inputs: frozenset[str]
    causal_edges: tuple[tuple[str, str, str], ...]
    fingerprint: str

    def execute(self, state: Mapping[str, Any] | None = None, *, requested: Iterable[str] | None = None):
        base = dict(self.initial_state)
        if state:
            declared = {i.name for i in self.ast.inputs} | {c.name for c in self.ast.cells}
            unknown = set(state) - declared
            if unknown:
                raise L0ARCSemanticError(f"unknown supplied state cells {sorted(unknown)}")
            base.update(state)
        missing = self.required_inputs - set(base)
        if missing:
            raise L0ARCSemanticError(f"missing required inputs {sorted(missing)}")
        req = set(requested) if requested is not None else set(self.ast.observes)
        declared = {i.name for i in self.ast.inputs} | {c.name for c in self.ast.cells}
        if req - declared:
            raise L0ARCSemanticError(f"unknown requested cells {sorted(req-declared)}")
        if self.ast.effects:
            req.add("__effects__")
        return execute_arc(self.program, base, requested=(req or None), config=self.config)


def _validate_expr(expr: Expr, allowed_names: set[str], where: str) -> None:
    unknown = expr_names(expr) - allowed_names
    if unknown:
        raise L0ARCSemanticError(f"{where}: expression reads undeclared/non-readable names {sorted(unknown)}")
    if expr.kind == "call" and expr.value not in CORE_OPS:
        raise L0ARCSemanticError(f"{where}: unknown operation {expr.value!r}")
    for a in expr.args:
        _validate_expr(a, allowed_names, where)


def _causal_edges(rules: Sequence[ARCRule]) -> tuple[tuple[str, str, str], ...]:
    edges: set[tuple[str, str, str]] = set()
    for a in rules:
        for b in rules:
            if a.name == b.name:
                continue
            for cell in sorted(a.writes & b.reads):
                edges.add((a.name, b.name, cell))
        for pred in a.after:
            edges.add((pred, a.name, "after"))
    return tuple(sorted(edges))


def compile_l0_arc(text_or_ast: str | ProgramAST) -> CompiledL0ARC:
    ast = parse_l0_arc(text_or_ast) if isinstance(text_or_ast, str) else text_or_ast

    input_names = [c.name for c in ast.inputs]
    cell_names = [c.name for c in ast.cells]
    state_names = input_names + cell_names
    if len(state_names) != len(set(state_names)):
        raise L0ARCSemanticError("input/cell names must be globally unique")
    inputs = set(input_names)
    cells = set(state_names)
    if not cells:
        raise L0ARCSemanticError("program must declare at least one input or cell")

    merge_map = dict(ast.merges)
    if len(merge_map) != len(ast.merges):
        raise L0ARCSemanticError("a cell may declare only one merge law")
    for c in merge_map:
        if c not in cells:
            raise L0ARCSemanticError(f"merge references undeclared cell {c!r}")
    for c in ast.observes:
        if c not in cells:
            raise L0ARCSemanticError(f"observe references undeclared cell {c!r}")

    initial: dict[str, Any] = {}
    required_inputs: set[str] = set()
    for inp in ast.inputs:
        if inp.default is None:
            required_inputs.add(inp.name)
        else:
            _validate_expr(inp.default, set(), f"input {inp.name}")
            initial[inp.name] = eval_expr(inp.default, {})
    for c in ast.cells:
        if c.initial is not None:
            _validate_expr(c.initial, set(), f"cell {c.name}")
            initial[c.name] = eval_expr(c.initial, {})
    try:
        validate_finite_value(initial, "initial state")
    except ARCInvalidProgram as exc:
        raise L0ARCSemanticError(str(exc)) from exc

    rules: list[ARCRule] = []
    declared_rule_names = [r.name for r in ast.relations] + [e.name for e in ast.effects] + [s.name for s in ast.spreads]
    if len(declared_rule_names) != len(set(declared_rule_names)):
        raise L0ARCSemanticError("relation/effect/spread names must be globally unique")
    rule_name_set = set(declared_rule_names)

    for rd in ast.relations:
        if len(rd.reads) != len(set(rd.reads)) or len(rd.writes) != len(set(rd.writes)):
            raise L0ARCSemanticError(f"relation {rd.name}: duplicate read/write cell")
        reads = set(rd.reads); writes = set(rd.writes)
        unknown_cells = (reads | writes) - cells
        if unknown_cells:
            raise L0ARCSemanticError(f"relation {rd.name}: undeclared cells {sorted(unknown_cells)}")
        if not rd.proposals:
            raise L0ARCSemanticError(f"relation {rd.name}: at least one proposal is required")
        ptargets = {p.target for p in rd.proposals}
        if ptargets != writes:
            raise L0ARCSemanticError(
                f"relation {rd.name}: writes {sorted(writes)} must equal proposal targets {sorted(ptargets)}"
            )
        if len(ptargets) != len(rd.proposals):
            raise L0ARCSemanticError(f"relation {rd.name}: one proposal per target per evaluation")
        if set(rd.after) - rule_name_set:
            raise L0ARCSemanticError(f"relation {rd.name}: unknown after dependencies {sorted(set(rd.after)-rule_name_set)}")
        if rd.name in rd.after:
            raise L0ARCSemanticError(f"relation {rd.name}: cannot be after itself")
        allowed = set(reads)
        if rd.guard is not None:
            _validate_expr(rd.guard, allowed, f"relation {rd.name} guard")
        for pd in rd.proposals:
            _validate_expr(pd.expr, allowed, f"relation {rd.name} proposal {pd.target}")
            explicit = merge_map.get(pd.target)
            if explicit is not None and explicit != pd.mode:
                raise L0ARCSemanticError(
                    f"relation {rd.name}: proposal mode {pd.mode!r} conflicts with merge {pd.target} by {explicit!r}"
                )

        def make_eval(name: str, proposals: tuple[ProposalDecl, ...]):
            def evaluate(snapshot: Mapping[str, Any]) -> RuleResult:
                ps = tuple(
                    Proposal(name, i, pd.target, eval_expr(pd.expr, snapshot), pd.mode)
                    for i, pd in enumerate(proposals)
                )
                return RuleResult(ps)
            return evaluate

        guard_fn = None
        if rd.guard is not None:
            guard_expr = rd.guard
            guard_fn = lambda s, ex=guard_expr: bool(eval_expr(ex, s))

        rules.append(ARCRule(
            name=rd.name,
            reads=frozenset(reads),
            writes=frozenset(writes),
            evaluate=make_eval(rd.name, rd.proposals),
            guard=guard_fn,
            after=frozenset(rd.after),
        ))

    for ed in ast.effects:
        if len(ed.reads) != len(set(ed.reads)):
            raise L0ARCSemanticError(f"effect {ed.name}: duplicate read cell")
        reads = set(ed.reads)
        if reads - cells:
            raise L0ARCSemanticError(f"effect {ed.name}: undeclared cells {sorted(reads-cells)}")
        if set(ed.after) - rule_name_set:
            raise L0ARCSemanticError(f"effect {ed.name}: unknown after dependencies {sorted(set(ed.after)-rule_name_set)}")
        if ed.name in ed.after:
            raise L0ARCSemanticError(f"effect {ed.name}: cannot be after itself")
        if ed.guard is not None:
            _validate_expr(ed.guard, reads, f"effect {ed.name} guard")
        _validate_expr(ed.payload, reads, f"effect {ed.name} payload")
        payload_expr = ed.payload
        kind = ed.kind
        name = ed.name
        def evaluate(snapshot, *, _name=name, _kind=kind, _payload=payload_expr):
            return RuleResult(effects=(EffectIntent(_name, 0, _kind, eval_expr(_payload, snapshot)),))
        guard_fn = None if ed.guard is None else (lambda s, ex=ed.guard: bool(eval_expr(ex, s)))
        rules.append(ARCRule(
            name=name,
            reads=frozenset(reads),
            writes=frozenset(),
            evaluate=evaluate,
            guard=guard_fn,
            after=frozenset(ed.after),
            demand_tags=frozenset({"__effects__"}),
        ))

    for sd in ast.spreads:
        if sd.source not in cells or sd.target not in cells:
            raise L0ARCSemanticError(f"spread {sd.name}: source/target must be declared cells")
        _validate_expr(sd.transform, {sd.item}, f"spread {sd.name} transform")
        if sd.collection == "set":
            existing = merge_map.get(sd.target)
            if existing is not None and existing != "union":
                raise L0ARCSemanticError(f"spread {sd.name}: set target {sd.target} requires union merge")
            merge_map.setdefault(sd.target, "union")
        expr = sd.transform
        item_name = sd.item
        source = sd.source
        target = sd.target
        name = sd.name
        collection = sd.collection
        def evaluate(snapshot, *, _expr=expr, _item=item_name, _source=source, _target=target, _name=name, _collection=collection):
            xs = snapshot[_source]
            if _collection == "set":
                vals = {eval_expr(_expr, {_item: x}) for x in xs}
                return RuleResult((Proposal(_name, 0, _target, vals, "union"),))
            vals = tuple(eval_expr(_expr, {_item: x}) for x in xs)
            return RuleResult((Proposal(_name, 0, _target, vals, "set"),))
        rules.append(ARCRule(
            name=name,
            reads=frozenset({source}),
            writes=frozenset({target}),
            evaluate=evaluate,
        ))

    # Static causal startability: every rule must be structurally reachable
    # from external inputs, initialized internal cells, or zero-read seed rules.
    # This catches explicit `after` cycles and unseeded data cycles before runtime.
    available = set(inputs) | set(initial)
    completed: set[str] = set()
    remaining = {r.name: r for r in rules}
    progressed = True
    while progressed:
        progressed = False
        for name in sorted(list(remaining)):
            r = remaining[name]
            if r.reads.issubset(available) and r.after.issubset(completed):
                completed.add(name)
                available |= set(r.writes)
                del remaining[name]
                progressed = True
    if remaining:
        detail = {
            name: {
                "missing_cells": sorted(set(r.reads) - available),
                "waiting_after": sorted(set(r.after) - completed),
            }
            for name, r in sorted(remaining.items())
        }
        raise L0ARCSemanticError(f"causal deadlock/unseeded component: {detail}")
    missing_observables = set(ast.observes) - available
    if missing_observables:
        raise L0ARCSemanticError(f"observables cannot be produced: {sorted(missing_observables)}")

    config = ARCConfig(
        max_component_rounds=ast.convergence.waves,
        max_rule_evaluations=ast.convergence.evaluations,
        max_proposals=ast.convergence.proposals,
        float_tolerance=ast.convergence.tolerance,
        detect_oscillation=(ast.convergence.oscillation == "detect"),
    )
    if config.max_component_rounds <= 0 or config.max_rule_evaluations <= 0 or config.max_proposals <= 0:
        raise L0ARCSemanticError("convergence budgets must be positive")
    if not math.isfinite(config.float_tolerance) or config.float_tolerance < 0:
        raise L0ARCSemanticError("convergence tolerance must be finite and non-negative")

    program = ARCProgram(tuple(rules), merges=merge_map, observables=frozenset(ast.observes))
    edges = _causal_edges(rules)
    def expr_sem(e: Expr | None):
        if e is None:
            return None
        return {
            "kind": e.kind,
            "value": e.value,
            "args": [expr_sem(a) for a in e.args],
        }

    canonical = {
        "program": ast.name,
        "semantics": "arc",
        "inputs": sorted(
            ({"name": i.name, "default": expr_sem(i.default)} for i in ast.inputs),
            key=lambda x: x["name"],
        ),
        "cells": sorted(
            ({"name": c.name, "initial": expr_sem(c.initial)} for c in ast.cells),
            key=lambda x: x["name"],
        ),
        "observes": sorted(ast.observes),
        "merges": sorted(merge_map.items()),
        "relations": sorted(
            ({
                "name": r.name,
                "reads": sorted(r.reads),
                "writes": sorted(r.writes),
                "guard": expr_sem(r.guard),
                "after": sorted(r.after),
                "proposals": sorted(
                    ({"target": p.target, "mode": p.mode, "expr": expr_sem(p.expr)} for p in r.proposals),
                    key=lambda x: x["target"],
                ),
            } for r in ast.relations),
            key=lambda x: x["name"],
        ),
        "effects": sorted(
            ({
                "name": e.name,
                "reads": sorted(e.reads),
                "guard": expr_sem(e.guard),
                "after": sorted(e.after),
                "kind": e.kind,
                "payload": expr_sem(e.payload),
            } for e in ast.effects),
            key=lambda x: x["name"],
        ),
        "spreads": sorted(
            ({
                "name": sp.name,
                "source": sp.source,
                "target": sp.target,
                "item": sp.item,
                "collection": sp.collection,
                "transform": expr_sem(sp.transform),
            } for sp in ast.spreads),
            key=lambda x: x["name"],
        ),
        "edges": edges,
        "config": {
            "waves": config.max_component_rounds,
            "evaluations": config.max_rule_evaluations,
            "proposals": config.max_proposals,
            "tolerance": config.float_tolerance,
            "oscillation": config.detect_oscillation,
        },
    }
    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CompiledL0ARC(ast, program, config, initial, frozenset(required_inputs), edges, fingerprint)


def compile_file(path: str | Path) -> CompiledL0ARC:
    p = Path(path)
    return compile_l0_arc(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Canonical rendering (for auditability / stable source form)
# ---------------------------------------------------------------------------


def render_expr(e: Expr) -> str:
    if e.kind == "name": return str(e.value)
    if e.kind == "number": return repr(e.value)
    if e.kind == "string": return json.dumps(e.value)
    if e.kind == "bool": return "true" if e.value else "false"
    if e.kind == "null": return "null"
    if e.kind == "call": return f"{e.value}(" + ", ".join(render_expr(a) for a in e.args) + ")"
    raise ValueError(e.kind)


def render_canonical(ast: ProgramAST) -> str:
    lines = [f"program {ast.name}", "semantics arc", ""]
    for i in ast.inputs:
        lines.append(f"input {i.name}" + (f" = {render_expr(i.default)}" if i.default is not None else ""))
    if ast.inputs: lines.append("")
    for c in ast.cells:
        lines.append(f"cell {c.name}" + (f" = {render_expr(c.initial)}" if c.initial is not None else ""))
    if ast.cells: lines.append("")
    for c, m in ast.merges:
        lines.append(f"merge {c} by {m}")
    for o in ast.observes:
        lines.append(f"observe {o}")
    if ast.merges or ast.observes: lines.append("")
    for r in ast.relations:
        lines.append(f"relation {r.name} (" + ", ".join(r.reads) + ") -> (" + ", ".join(r.writes) + ") {")
        if r.guard is not None: lines.append(f"    when {render_expr(r.guard)}")
        if r.after: lines.append("    after " + ", ".join(r.after))
        for p in r.proposals:
            lines.append(f"    propose {p.target} {p.mode} {render_expr(p.expr)}")
        lines.append("}")
        lines.append("")
    for e in ast.effects:
        lines.append(f"effect {e.name} (" + ", ".join(e.reads) + ") {")
        if e.guard is not None: lines.append(f"    when {render_expr(e.guard)}")
        if e.after: lines.append("    after " + ", ".join(e.after))
        lines.append(f"    stage {e.kind} {render_expr(e.payload)}")
        lines.append("}")
        lines.append("")
    for s in ast.spreads:
        lines.append(f"spread {s.name} {s.source} -> {s.target} {{")
        lines.append(f"    item {s.item}")
        lines.append(f"    collection {s.collection}")
        lines.append(f"    transform {render_expr(s.transform)}")
        lines.append("}")
        lines.append("")
    c = ast.convergence
    lines.extend([
        "convergence {",
        f"    tolerance {c.tolerance}",
        f"    waves {c.waves}",
        f"    evaluations {c.evaluations}",
        f"    proposals {c.proposals}",
        f"    oscillation {c.oscillation}",
        "}",
        "",
    ])
    return "\n".join(lines)


__all__ = [
    "L0ARCParseError", "L0ARCSemanticError", "Token", "Expr", "InputDecl", "CellDecl",
    "ProposalDecl", "RelationDecl", "EffectDecl", "SpreadDecl",
    "ConvergenceDecl", "ProgramAST", "Lexer", "Parser", "parse_l0_arc",
    "CORE_OPS", "expr_names", "eval_expr", "CompiledL0ARC", "compile_l0_arc",
    "compile_file", "render_expr", "render_canonical",
]

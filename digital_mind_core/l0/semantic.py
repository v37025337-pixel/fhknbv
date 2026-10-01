from __future__ import annotations

"""L0 Language v0.3 — cross-language semantic IR.

The native L0 grammar remains independent. Foreign source languages are parsed
with language-specific frontends and lowered into one semantic IR. The IR is
structural and conservative: unsupported constructs are retained as unresolved
nodes instead of being assigned invented semantics.

Supported frontends in this build:
- L0 v0.2 frontend (native parser/AST/IR)
- Python via CPython ``ast``
- JavaScript via the TypeScript compiler API (invoked through Node)
- C via pycparser
- C++ via Clang JSON AST
- Rust via a small conservative bootstrap parser for a core subset

The numerical L0 v0.24 kernel is intentionally not changed by this module.
"""

import argparse
import ast as pyast
import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .language import (
    AssignAction,
    CallExpr,
    EmitAction,
    L0IR,
    LearnAction,
    compile_l0_to_ir,
)


class FrontendError(ValueError):
    pass


class FrontendDependencyError(FrontendError):
    """An optional native parser/toolchain is unavailable."""


# ---------------------------------------------------------------------------
# Semantic IR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SExpr:
    kind: str
    value: Any = None
    operator: str | None = None
    args: tuple["SExpr", ...] = ()
    resolved: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def name(name: str) -> "SExpr":
        return SExpr("name", value=name)

    @staticmethod
    def const(value: Any) -> "SExpr":
        return SExpr("const", value=value)

    @staticmethod
    def binary(op: str, left: "SExpr", right: "SExpr", *, resolved: bool = True) -> "SExpr":
        return SExpr("binary", operator=op, args=(left, right), resolved=resolved)

    @staticmethod
    def unary(op: str, arg: "SExpr", *, resolved: bool = True) -> "SExpr":
        return SExpr("unary", operator=op, args=(arg,), resolved=resolved)

    @staticmethod
    def call(name: str | None, args: Iterable["SExpr"], *, resolved: bool = True) -> "SExpr":
        if name is None:
            return SExpr("call", operator="unknown", args=tuple(args), resolved=False)
        return SExpr("call", operator=name, args=tuple(args), resolved=resolved)

    @staticmethod
    def memory(base: "SExpr", index: "SExpr", *, resolved: bool = True) -> "SExpr":
        return SExpr("memory_load", args=(base, index), resolved=resolved)

    @staticmethod
    def unknown(label: str, **meta: Any) -> "SExpr":
        return SExpr("unresolved", operator=label, resolved=False, meta=meta)

    def signature(self) -> Any:
        if self.kind in ("name", "const"):
            return (self.kind, self.value)
        return (
            self.kind,
            self.operator,
            tuple(a.signature() for a in self.args),
            bool(self.resolved),
        )


@dataclass(frozen=True)
class SOp:
    kind: str
    target: str | None = None
    operator: str | None = None
    args: tuple[SExpr, ...] = ()
    condition: SExpr | None = None
    body: tuple["SOp", ...] = ()
    else_body: tuple["SOp", ...] = ()
    resolved: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> Any:
        return (
            self.kind,
            self.target,
            self.operator,
            tuple(a.signature() for a in self.args),
            self.condition.signature() if self.condition else None,
            tuple(x.signature() for x in self.body),
            tuple(x.signature() for x in self.else_body),
            bool(self.resolved),
        )


@dataclass(frozen=True)
class SFunction:
    name: str
    params: tuple[str, ...]
    body: tuple[SOp, ...]
    resolved: bool = True
    defaults: tuple[SExpr, ...] = ()

    def signature(self) -> Any:
        return (self.name, self.params, tuple(op.signature() for op in self.body), self.resolved,
                tuple(e.signature() for e in self.defaults))


@dataclass(frozen=True)
class SemanticModule:
    name: str
    source_language: str
    functions: tuple[SFunction, ...] = ()
    top_level: tuple[SOp, ...] = ()
    unresolved_count: int = 0
    frontend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> Any:
        return (
            tuple(f.signature() for f in self.functions),
            tuple(o.signature() for o in self.top_level),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _count_unresolved_expr(e: SExpr | None) -> int:
    if e is None:
        return 0
    return (0 if e.resolved else 1) + sum(_count_unresolved_expr(a) for a in e.args)


def _count_unresolved_ops(ops: Iterable[SOp]) -> int:
    n = 0
    for op in ops:
        n += 0 if op.resolved else 1
        n += sum(_count_unresolved_expr(a) for a in op.args)
        n += _count_unresolved_expr(op.condition)
        n += _count_unresolved_ops(op.body)
        n += _count_unresolved_ops(op.else_body)
    return n


def _module(name: str, language: str, functions: list[SFunction], top: list[SOp], frontend: str, **meta: Any) -> SemanticModule:
    unresolved = sum(_count_unresolved_ops(f.body) + (0 if f.resolved else 1) for f in functions)
    unresolved += _count_unresolved_ops(top)
    return SemanticModule(name, language, tuple(functions), tuple(top), unresolved, frontend, meta)


# ---------------------------------------------------------------------------
# Native L0 -> semantic IR
# ---------------------------------------------------------------------------


def _native_expr(x: Any) -> SExpr:
    if isinstance(x, (int, float, bool)):
        return SExpr.const(x)
    if isinstance(x, str):
        return SExpr.name(x)
    if isinstance(x, CallExpr):
        return SExpr.call(x.name, (_native_expr(a) for a in x.args))
    return SExpr.unknown(type(x).__name__)


def l0_ir_to_semantic(ir: L0IR) -> SemanticModule:
    top: list[SOp] = []
    for r in ir.relations:
        top.append(
            SOp(
                "relation",
                target=r.target,
                operator=r.name,
                args=tuple(SExpr.name(s) for s in r.sources),
                resolved=r.resolved,
                meta={"properties": r.properties, "origin": r.origin},
            )
        )

    for ev in ir.events:
        body: list[SOp] = []
        for a in ev.actions:
            if isinstance(a, LearnAction):
                body.append(SOp("call", operator="learn", args=(SExpr.name(a.relation), SExpr.name(a.source))))
            elif isinstance(a, EmitAction):
                body.append(SOp("call", operator="emit", args=(SExpr.name(a.name),)))
            elif isinstance(a, AssignAction):
                body.append(SOp("assign", target=a.target, args=(_native_expr(a.expression),)))
            else:
                body.append(SOp("unresolved", operator=type(a).__name__, resolved=False))
        top.append(
            SOp(
                "event",
                operator=ev.name,
                args=tuple(SExpr.name(x) for x in ev.inputs),
                body=tuple(body),
                meta={"observed_target": ev.observed_target},
            )
        )
    return _module(ir.name, "l0", [], top, "native-l0-v0.2")


def l0_to_semantic(source: str, name: str = "l0_program") -> SemanticModule:
    ir = compile_l0_to_ir(source)
    return l0_ir_to_semantic(ir)


# ---------------------------------------------------------------------------
# Python frontend
# ---------------------------------------------------------------------------


_PY_BINOPS: dict[type[Any], str] = {
    pyast.Add: "add",
    pyast.Sub: "sub",
    pyast.Mult: "mul",
    pyast.Div: "div",
    pyast.FloorDiv: "floordiv",
    pyast.Mod: "mod",
    pyast.Pow: "pow",
    pyast.MatMult: "matmul",
}
_PY_CMPOPS: dict[type[Any], str] = {
    pyast.Eq: "eq",
    pyast.NotEq: "ne",
    pyast.Lt: "lt",
    pyast.LtE: "le",
    pyast.Gt: "gt",
    pyast.GtE: "ge",
    pyast.Is: "is",
    pyast.IsNot: "is_not",
    pyast.In: "in",
    pyast.NotIn: "not_in",
}
_PY_BOOLOPS = {pyast.And: "value_and", pyast.Or: "value_or"}
_PY_UNARY = {pyast.USub: "neg", pyast.UAdd: "pos", pyast.Not: "not", pyast.Invert: "bit_not"}


def _py_call_name(node: pyast.AST) -> str | None:
    if isinstance(node, pyast.Name):
        return node.id
    if isinstance(node, pyast.Attribute):
        parts: list[str] = []
        cur: pyast.AST = node
        while isinstance(cur, pyast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, pyast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
    return None


def _py_expr(node: pyast.AST) -> SExpr:
    if isinstance(node, pyast.Name):
        return SExpr.name(node.id)
    if isinstance(node, pyast.Constant):
        return SExpr.const(node.value)
    if isinstance(node, pyast.BinOp):
        op = _PY_BINOPS.get(type(node.op))
        return SExpr.binary(op or type(node.op).__name__, _py_expr(node.left), _py_expr(node.right), resolved=op is not None)
    if isinstance(node, pyast.UnaryOp):
        op = _PY_UNARY.get(type(node.op))
        return SExpr.unary(op or type(node.op).__name__, _py_expr(node.operand), resolved=op is not None)
    if isinstance(node, pyast.BoolOp):
        op = _PY_BOOLOPS.get(type(node.op))
        vals = [_py_expr(v) for v in node.values]
        if not vals:
            return SExpr.unknown("empty_boolop")
        cur = vals[0]
        for v in vals[1:]:
            cur = SExpr.binary(op or type(node.op).__name__, cur, v, resolved=op is not None)
        return cur
    if isinstance(node, pyast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        op = _PY_CMPOPS.get(type(node.ops[0]))
        return SExpr.binary(op or type(node.ops[0]).__name__, _py_expr(node.left), _py_expr(node.comparators[0]), resolved=op is not None)
    if isinstance(node, pyast.Call):
        name = _py_call_name(node.func)
        return SExpr.call(name, (_py_expr(a) for a in node.args), resolved=name is not None and not node.keywords)
    if isinstance(node, pyast.Subscript):
        return SExpr.memory(_py_expr(node.value), _py_expr(node.slice))
    if isinstance(node, pyast.Attribute):
        name = _py_call_name(node)
        if name is not None:
            return SExpr("member", value=name)
    return SExpr.unknown(type(node).__name__, python_ast=type(node).__name__)


def _py_target(node: pyast.AST) -> tuple[str | None, SExpr | None]:
    if isinstance(node, pyast.Name):
        return node.id, None
    if isinstance(node, pyast.Subscript):
        return None, SExpr.memory(_py_expr(node.value), _py_expr(node.slice))
    return None, SExpr.unknown(type(node).__name__)


def _py_stmt(node: pyast.stmt) -> list[SOp]:
    if isinstance(node, (pyast.FunctionDef, pyast.AsyncFunctionDef)):
        return [SOp("unresolved", operator=type(node).__name__, resolved=False)]
    if isinstance(node, pyast.Assign) and len(node.targets) == 1:
        target, mem = _py_target(node.targets[0])
        value = _py_expr(node.value)
        if target is not None:
            return [SOp("assign", target=target, args=(value,), resolved=value.resolved)]
        if mem is not None and mem.kind == "memory_load":
            return [SOp("memory_store", args=(mem.args[0], mem.args[1], value), resolved=mem.resolved and value.resolved)]
        return [SOp("unresolved", operator="AssignTarget", args=(value,), resolved=False)]
    if isinstance(node, pyast.AnnAssign) and node.value is not None:
        target, mem = _py_target(node.target)
        value = _py_expr(node.value)
        if target is not None:
            return [SOp("assign", target=target, args=(value,), resolved=value.resolved)]
        if mem is not None and mem.kind == "memory_load":
            return [SOp("memory_store", args=(mem.args[0], mem.args[1], value), resolved=mem.resolved and value.resolved)]
    if isinstance(node, pyast.AugAssign):
        target, mem = _py_target(node.target)
        op = _PY_BINOPS.get(type(node.op))
        if target is not None:
            rhs = SExpr.binary(op or type(node.op).__name__, SExpr.name(target), _py_expr(node.value), resolved=op is not None)
            return [SOp("assign", target=target, args=(rhs,), resolved=rhs.resolved)]
        return [SOp("unresolved", operator="AugAssignTarget", resolved=False)]
    if isinstance(node, pyast.Return):
        return [SOp("return", args=(() if node.value is None else (_py_expr(node.value),)))]
    if isinstance(node, pyast.Expr):
        e = _py_expr(node.value)
        if e.kind == "call":
            return [SOp("call", operator=e.operator, args=e.args, resolved=e.resolved)]
        return [SOp("expr", args=(e,), resolved=e.resolved)]
    if isinstance(node, pyast.If):
        cond = _py_expr(node.test)
        body = tuple(op for s in node.body for op in _py_stmt(s))
        alt = tuple(op for s in node.orelse for op in _py_stmt(s))
        return [SOp("branch", condition=cond, body=body, else_body=alt, resolved=cond.resolved)]
    if isinstance(node, pyast.While):
        cond = _py_expr(node.test)
        body = tuple(op for s in node.body for op in _py_stmt(s))
        return [SOp("loop", operator="while", condition=cond, body=body, resolved=cond.resolved and not node.orelse)]
    if isinstance(node, pyast.For):
        if isinstance(node.target, pyast.Name):
            seq = _py_expr(node.iter)
            body = tuple(op for s in node.body for op in _py_stmt(s))
            return [SOp("loop", target=node.target.id, operator="for_each", args=(seq,), body=body, resolved=seq.resolved and not node.orelse)]
        return [SOp("unresolved", operator="ForTarget", resolved=False)]
    if isinstance(node, (pyast.Pass,)):
        return []
    return [SOp("unresolved", operator=type(node).__name__, resolved=False, meta={"python_ast": type(node).__name__})]


def python_to_semantic(source: str, name: str = "python_module") -> SemanticModule:
    tree = pyast.parse(source)
    funcs: list[SFunction] = []
    top: list[SOp] = []
    for stmt in tree.body:
        if isinstance(stmt, (pyast.FunctionDef, pyast.AsyncFunctionDef)):
            params = tuple(a.arg for a in (*stmt.args.posonlyargs, *stmt.args.args))
            body = tuple(op for s in stmt.body for op in _py_stmt(s))
            resolved = not isinstance(stmt, pyast.AsyncFunctionDef) and not stmt.args.vararg and not stmt.args.kwarg and not stmt.decorator_list and not stmt.args.kwonlyargs
            defaults = []
            for expr in stmt.args.defaults:
                try:
                    # Definition-time literals are retained, including mutable
                    # defaults. Dynamic/global defaults require an adapter.
                    defaults.append(SExpr.const(pyast.literal_eval(expr)))
                except (ValueError, TypeError, SyntaxError):
                    defaults.append(SExpr.unknown('definition_time_default'))
                    resolved = False
            funcs.append(SFunction(stmt.name, params, body, resolved=bool(resolved), defaults=tuple(defaults)))
        else:
            top.extend(_py_stmt(stmt))
    return _module(name, "python", funcs, top, "cpython-ast")


# ---------------------------------------------------------------------------
# JavaScript frontend via TypeScript compiler API
# ---------------------------------------------------------------------------


_JS_BRIDGE = r'''const ts = require('typescript');
let src=''; process.stdin.setEncoding('utf8'); process.stdin.on('data',c=>src+=c);
process.stdin.on('end',()=>{
  const sf=ts.createSourceFile('input.js',src,ts.ScriptTarget.Latest,true,ts.ScriptKind.JS);
  function expr(n){
    if(!n) return {k:'unknown',t:'Missing'};
    if(ts.isIdentifier(n)) return {k:'name',v:n.text};
    if(ts.isNumericLiteral(n)) return {k:'const',v:Number(n.text)};
    if(ts.isStringLiteral(n)) return {k:'const',v:n.text};
    if(n.kind===ts.SyntaxKind.TrueKeyword) return {k:'const',v:true};
    if(n.kind===ts.SyntaxKind.FalseKeyword) return {k:'const',v:false};
    if(ts.isParenthesizedExpression(n)) return expr(n.expression);
    if(ts.isPrefixUnaryExpression(n)) return {k:'unary',op:ts.tokenToString(n.operator)||ts.SyntaxKind[n.operator],a:expr(n.operand)};
    if(ts.isBinaryExpression(n)) return {k:'binary',op:ts.tokenToString(n.operatorToken.kind)||ts.SyntaxKind[n.operatorToken.kind],l:expr(n.left),r:expr(n.right)};
    if(ts.isCallExpression(n)) return {k:'call',name:n.expression.getText(sf),args:n.arguments.map(expr)};
    if(ts.isElementAccessExpression(n)) return {k:'index',b:expr(n.expression),i:expr(n.argumentExpression)};
    if(ts.isPropertyAccessExpression(n)) return {k:'member',v:n.getText(sf)};
    return {k:'unknown',t:ts.SyntaxKind[n.kind]};
  }
  function stmt(n){
    if(ts.isVariableStatement(n)) return {k:'vars',ds:n.declarationList.declarations.map(d=>({name:d.name.getText(sf),init:d.initializer?expr(d.initializer):null}))};
    if(ts.isExpressionStatement(n)) return {k:'exprstmt',e:expr(n.expression)};
    if(ts.isReturnStatement(n)) return {k:'return',e:n.expression?expr(n.expression):null};
    if(ts.isIfStatement(n)) return {k:'if',c:expr(n.expression),b:block(n.thenStatement),a:n.elseStatement?block(n.elseStatement):[]};
    if(ts.isWhileStatement(n)) return {k:'while',c:expr(n.expression),b:block(n.statement)};
    if(ts.isForStatement(n)) return {k:'for',init:n.initializer?n.initializer.getText(sf):null,c:n.condition?expr(n.condition):null,inc:n.incrementor?n.incrementor.getText(sf):null,b:block(n.statement)};
    if(ts.isBlock(n)) return {k:'block',b:n.statements.map(stmt)};
    return {k:'unknown',t:ts.SyntaxKind[n.kind]};
  }
  function block(n){ return ts.isBlock(n)?n.statements.map(stmt):[stmt(n)]; }
  const out={functions:[],top:[]};
  for(const n of sf.statements){
    if(ts.isFunctionDeclaration(n) && n.name){ out.functions.push({name:n.name.text,params:n.parameters.map(p=>p.name.getText(sf)),body:n.body?n.body.statements.map(stmt):[],async:!!n.modifiers?.some(m=>m.kind===ts.SyntaxKind.AsyncKeyword)}); }
    else out.top.push(stmt(n));
  }
  process.stdout.write(JSON.stringify(out));
});'''

_JS_OPS = {
    "+": "add", "-": "sub", "*": "mul", "/": "div", "%": "mod", "**": "pow",
    "<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "===": "eq",
    "!=": "ne", "!==": "ne", "&&": "and", "||": "or",
}
_JS_UNARY = {"!": "not", "-": "neg", "+": "pos", "~": "bit_not"}


def _js_raw(source: str) -> dict[str, Any]:
    try:
        p = subprocess.run(
            ["node", "-e", _JS_BRIDGE], input=source, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    except FileNotFoundError as exc:
        raise FrontendDependencyError('JavaScript frontend requires Node.js on PATH and the typescript package resolvable by Node') from exc
    if p.returncode != 0:
        if "Cannot find module 'typescript'" in p.stderr:
            raise FrontendDependencyError('JavaScript frontend requires the typescript package resolvable by Node (local node_modules or NODE_PATH)')
        raise FrontendError(f"JavaScript frontend failed: {p.stderr.strip()}")
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        raise FrontendError(f"JavaScript frontend returned invalid JSON: {exc}") from exc


def _js_expr(x: dict[str, Any] | None) -> SExpr:
    if not x:
        return SExpr.unknown("Missing")
    k = x.get("k")
    if k == "name":
        return SExpr.name(x["v"])
    if k == "const":
        return SExpr.const(x.get("v"))
    if k == "member":
        return SExpr("member", value=x.get("v"))
    if k == "index":
        return SExpr.memory(_js_expr(x.get("b")), _js_expr(x.get("i")))
    if k == "call":
        return SExpr.call(x.get("name"), (_js_expr(a) for a in x.get("args", [])), resolved=bool(x.get("name")))
    if k == "unary":
        op = _JS_UNARY.get(x.get("op"))
        return SExpr.unary(op or str(x.get("op")), _js_expr(x.get("a")), resolved=op is not None)
    if k == "binary":
        rawop = x.get("op")
        if rawop == "=":
            return SExpr("assignment_expr", operator="assign", args=(_js_expr(x.get("l")), _js_expr(x.get("r"))))
        op = _JS_OPS.get(rawop)
        return SExpr.binary(op or str(rawop), _js_expr(x.get("l")), _js_expr(x.get("r")), resolved=op is not None)
    return SExpr.unknown(str(x.get("t", k)))


def _js_stmt(x: dict[str, Any]) -> list[SOp]:
    k = x.get("k")
    if k == "vars":
        out: list[SOp] = []
        for d in x.get("ds", []):
            if d.get("init") is None:
                out.append(SOp("declare", target=d.get("name")))
            else:
                e = _js_expr(d.get("init"))
                out.append(SOp("assign", target=d.get("name"), args=(e,), resolved=e.resolved))
        return out
    if k == "exprstmt":
        e = _js_expr(x.get("e"))
        if e.kind == "assignment_expr" and len(e.args) == 2:
            left, right = e.args
            if left.kind == "name":
                return [SOp("assign", target=str(left.value), args=(right,), resolved=right.resolved)]
            if left.kind == "memory_load":
                return [SOp("memory_store", args=(left.args[0], left.args[1], right), resolved=left.resolved and right.resolved)]
            return [SOp("unresolved", operator="assignment_target", args=(left, right), resolved=False)]
        if e.kind == "call":
            return [SOp("call", operator=e.operator, args=e.args, resolved=e.resolved)]
        return [SOp("expr", args=(e,), resolved=e.resolved)]
    if k == "return":
        return [SOp("return", args=(() if x.get("e") is None else (_js_expr(x.get("e")),)))]
    if k == "if":
        c = _js_expr(x.get("c"))
        b = tuple(op for s in x.get("b", []) for op in _js_stmt(s))
        a = tuple(op for s in x.get("a", []) for op in _js_stmt(s))
        return [SOp("branch", condition=c, body=b, else_body=a, resolved=c.resolved)]
    if k == "while":
        c = _js_expr(x.get("c"))
        b = tuple(op for s in x.get("b", []) for op in _js_stmt(s))
        return [SOp("loop", operator="while", condition=c, body=b, resolved=c.resolved)]
    if k == "for":
        c = _js_expr(x.get("c")) if x.get("c") is not None else None
        b = tuple(op for s in x.get("b", []) for op in _js_stmt(s))
        # textual initializer/increment are preserved; not guessed semantically yet
        return [SOp("loop", operator="for", condition=c, body=b, resolved=False, meta={"init": x.get("init"), "increment": x.get("inc")})]
    if k == "block":
        return [op for s in x.get("b", []) for op in _js_stmt(s)]
    return [SOp("unresolved", operator=str(x.get("t", k)), resolved=False)]


def javascript_to_semantic(source: str, name: str = "javascript_module") -> SemanticModule:
    raw = _js_raw(source)
    funcs: list[SFunction] = []
    for f in raw.get("functions", []):
        body = tuple(op for s in f.get("body", []) for op in _js_stmt(s))
        funcs.append(SFunction(f["name"], tuple(f.get("params", [])), body, resolved=not bool(f.get("async"))))
    top = [op for s in raw.get("top", []) for op in _js_stmt(s)]
    return _module(name, "javascript", funcs, top, "typescript-compiler-api")


# ---------------------------------------------------------------------------
# C frontend via pycparser
# ---------------------------------------------------------------------------


def _c_imports():
    try:
        from pycparser import c_ast, c_parser
    except Exception as exc:  # pragma: no cover - environment fallback
        raise FrontendError(f"pycparser unavailable: {exc}") from exc
    return c_ast, c_parser


_C_BIN = {
    "+": "add", "-": "sub", "*": "mul", "/": "div", "%": "mod",
    "<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne",
    "&&": "and", "||": "or", "&": "bit_and", "|": "bit_or", "^": "bit_xor",
}
_C_UNARY = {"!": "not", "-": "neg", "+": "pos", "~": "bit_not"}


def _c_type(node):
    """Numeric category needed for the supported scalar C expressions."""
    kind=type(node).__name__
    if kind in {'TypeDecl','Typename'}:return _c_type(node.type)
    if kind=='IdentifierType':
        words=set(node.names)
        if words & {'float','double'}:return 'float'
        if words & {'char','short','int','long','signed','unsigned','_Bool'}:return 'int'
    return None


def _c_cast(expr, kind):
    return SExpr.unary('c_'+kind,expr,resolved=expr.resolved) if kind else expr


def _c_operator(op):
    return {'/':'c_div','%':'c_mod'}.get(op,_C_BIN.get(op))


def _c_expr(n: Any, c_ast: Any, types=None) -> SExpr:
    types = {} if types is None else types
    if n is None:
        return SExpr.unknown("Missing")
    if isinstance(n, c_ast.ID):
        return _c_cast(SExpr.name(n.name), types.get(n.name))
    if isinstance(n, c_ast.Constant):
        if n.type in ("int", "long", "unsigned", "unsigned int"):
            try:
                return SExpr.const(int(n.value, 0))
            except Exception:
                return SExpr.const(n.value)
        if n.type in ("float", "double"):
            try:
                return SExpr.const(float(n.value.rstrip("fFlL")))
            except Exception:
                return SExpr.const(n.value)
        return SExpr.const(n.value.strip('"'))
    if isinstance(n, c_ast.BinaryOp):
        op = _c_operator(n.op)
        return SExpr.binary(op or n.op, _c_expr(n.left, c_ast, types), _c_expr(n.right, c_ast, types), resolved=op is not None)
    if isinstance(n, c_ast.UnaryOp):
        op = _C_UNARY.get(n.op)
        return SExpr.unary(op or n.op, _c_expr(n.expr, c_ast, types), resolved=op is not None)
    if isinstance(n, c_ast.FuncCall):
        fname = n.name.name if isinstance(n.name, c_ast.ID) else None
        args = [] if n.args is None else [_c_expr(a, c_ast, types) for a in n.args.exprs]
        return SExpr.call(fname, args, resolved=fname is not None)
    if isinstance(n, c_ast.ArrayRef):
        return SExpr.memory(_c_expr(n.name, c_ast, types), _c_expr(n.subscript, c_ast, types))
    if isinstance(n, c_ast.Cast):
        return _c_cast(_c_expr(n.expr, c_ast, types), _c_type(n.to_type.type))
    return SExpr.unknown(type(n).__name__, c_ast=type(n).__name__)


def _c_stmt(n: Any, c_ast: Any, types=None) -> list[SOp]:
    types = {} if types is None else types
    if n is None:
        return []
    if isinstance(n, c_ast.Compound):
        return [op for s in (n.block_items or []) for op in _c_stmt(s, c_ast, types)]
    if isinstance(n, c_ast.Decl) and not isinstance(n.type, c_ast.FuncDecl):
        if n.init is None:
            return [SOp("declare", target=n.name)]
        e = _c_cast(_c_expr(n.init, c_ast, types), _c_type(n.type))
        return [SOp("assign", target=n.name, args=(e,), resolved=e.resolved)]
    if isinstance(n, c_ast.Assignment):
        rhs = _c_expr(n.rvalue, c_ast, types)
        if isinstance(n.lvalue, c_ast.ID):
            if n.op == "=":
                return [SOp("assign", target=n.lvalue.name, args=(_c_cast(rhs, types.get(n.lvalue.name)),), resolved=rhs.resolved)]
            bop = _c_operator(n.op[:-1]) if n.op.endswith("=") else None
            if bop:
                e = SExpr.binary(bop, _c_cast(SExpr.name(n.lvalue.name), types.get(n.lvalue.name)), rhs)
                return [SOp("assign", target=n.lvalue.name, args=(_c_cast(e, types.get(n.lvalue.name)),))]
        if isinstance(n.lvalue, c_ast.ArrayRef) and n.op == "=":
            mem = _c_expr(n.lvalue, c_ast, types)
            return [SOp("memory_store", args=(mem.args[0], mem.args[1], rhs), resolved=mem.resolved and rhs.resolved)]
        return [SOp("unresolved", operator=f"assignment:{n.op}", resolved=False)]
    if isinstance(n, c_ast.Return):
        return [SOp("return", args=(() if n.expr is None else (_c_cast(_c_expr(n.expr, c_ast, types), types.get(None)),)))]
    if isinstance(n, c_ast.FuncCall):
        e = _c_expr(n, c_ast, types)
        return [SOp("call", operator=e.operator, args=e.args, resolved=e.resolved)]
    if isinstance(n, c_ast.If):
        c = _c_expr(n.cond, c_ast, types)
        b = tuple(_c_stmt(n.iftrue, c_ast, types))
        a = tuple(_c_stmt(n.iffalse, c_ast, types)) if n.iffalse is not None else ()
        return [SOp("branch", condition=c, body=b, else_body=a, resolved=c.resolved)]
    if isinstance(n, c_ast.While):
        c = _c_expr(n.cond, c_ast, types)
        return [SOp("loop", operator="while", condition=c, body=tuple(_c_stmt(n.stmt, c_ast, types)), resolved=c.resolved)]
    if isinstance(n, c_ast.For):
        c = _c_expr(n.cond, c_ast, types) if n.cond is not None else None
        return [SOp("loop", operator="for", condition=c, body=tuple(_c_stmt(n.stmt, c_ast, types)), resolved=False, meta={"init": type(n.init).__name__ if n.init else None, "next": type(n.next).__name__ if n.next else None})]
    if isinstance(n, c_ast.EmptyStatement):
        return []
    return [SOp("unresolved", operator=type(n).__name__, resolved=False, meta={"c_ast": type(n).__name__})]


def c_to_semantic(source: str, name: str = "c_module") -> SemanticModule:
    c_ast, c_parser = _c_imports()
    try:
        tree = c_parser.CParser().parse(source)
    except Exception as exc:
        raise FrontendError(f"C parse failed: {exc}") from exc
    funcs: list[SFunction] = []
    top: list[SOp] = []
    for ext in tree.ext:
        if isinstance(ext, c_ast.FuncDef):
            decl = ext.decl
            params: list[str] = []
            args = getattr(decl.type, "args", None)
            if args is not None:
                params = [p.name for p in args.params if getattr(p, "name", None)]
            types = {None:_c_type(decl.type.type)}
            shadowed = set()
            class NumericTypes(c_ast.NodeVisitor):
                def visit_Decl(self, node):
                    if node.name:
                        if node.name in types:shadowed.add(node.name)
                        types[node.name] = _c_type(node.type)
                    self.generic_visit(node)
            if args is not None:NumericTypes().visit(args)
            NumericTypes().visit(ext.body)
            # Control IR has one environment per relation. Do not silently give
            # a shadowed C binding the type/storage of another lexical scope.
            funcs.append(SFunction(decl.name, tuple(params), tuple(_c_stmt(ext.body, c_ast, types)), resolved=not shadowed))
        elif isinstance(ext, c_ast.Decl) and isinstance(ext.type, c_ast.FuncDecl):
            # A prototype is a declaration, not an executable unresolved statement.
            continue
        else:
            top.extend(_c_stmt(ext, c_ast))
    return _module(name, "c", funcs, top, "pycparser")


# ---------------------------------------------------------------------------
# C++ frontend via clang JSON AST
# ---------------------------------------------------------------------------


def _clang_ast(source: str) -> dict[str, Any]:
    try:
        p = subprocess.run(
            ["clang", "-x", "c++", "-std=c++17", "-fsyntax-only", "-Xclang", "-ast-dump=json", "-"],
            input=source, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
    except FileNotFoundError as exc:
        raise FrontendDependencyError('C++ frontend requires clang with C++17 and JSON AST support on PATH') from exc
    if p.returncode != 0:
        raise FrontendError(f"C++ parse failed: {p.stderr.strip()}")
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        raise FrontendError(f"clang returned invalid JSON: {exc}") from exc


def _clang_children(n: dict[str, Any]) -> list[dict[str, Any]]:
    return [x for x in n.get("inner", []) if isinstance(x, dict)]


def _clang_unwrap(n: dict[str, Any] | None) -> dict[str, Any] | None:
    while n and n.get("kind") in {
        "ImplicitCastExpr", "ParenExpr", "CStyleCastExpr", "CXXStaticCastExpr",
        "ExprWithCleanups", "MaterializeTemporaryExpr", "CXXBindTemporaryExpr",
    }:
        ch = _clang_children(n)
        n = ch[0] if ch else None
    return n


def _cpp_expr(n: dict[str, Any] | None) -> SExpr:
    n = _clang_unwrap(n)
    if not n:
        return SExpr.unknown("Missing")
    k = n.get("kind")
    ch = _clang_children(n)
    if k == "DeclRefExpr":
        ref = n.get("referencedDecl", {})
        return SExpr.name(ref.get("name") or n.get("name") or "unknown")
    if k == "IntegerLiteral":
        try:
            return SExpr.const(int(n.get("value")))
        except Exception:
            return SExpr.const(n.get("value"))
    if k == "FloatingLiteral":
        try:
            return SExpr.const(float(n.get("value")))
        except Exception:
            return SExpr.const(n.get("value"))
    if k == "CXXBoolLiteralExpr":
        return SExpr.const(bool(n.get("value")))
    if k == "StringLiteral":
        return SExpr.const(n.get("value"))
    if k in ("BinaryOperator", "CompoundAssignOperator") and len(ch) >= 2:
        raw = n.get("opcode")
        if raw == "=":
            return SExpr("assignment_expr", operator="assign", args=(_cpp_expr(ch[0]), _cpp_expr(ch[1])))
        if raw and raw.endswith("=") and raw != "==":
            bop = _C_BIN.get(raw[:-1])
            return SExpr("compound_assignment_expr", operator=bop or raw, args=(_cpp_expr(ch[0]), _cpp_expr(ch[1])), resolved=bop is not None)
        op = _C_BIN.get(raw)
        return SExpr.binary(op or str(raw), _cpp_expr(ch[0]), _cpp_expr(ch[1]), resolved=op is not None)
    if k == "UnaryOperator" and ch:
        raw = n.get("opcode")
        op = _C_UNARY.get(raw)
        return SExpr.unary(op or str(raw), _cpp_expr(ch[0]), resolved=op is not None)
    if k in ("CallExpr", "CXXMemberCallExpr"):
        if not ch:
            return SExpr.call(None, [], resolved=False)
        callee = _clang_unwrap(ch[0])
        name = None
        if callee and callee.get("kind") == "DeclRefExpr":
            name = callee.get("referencedDecl", {}).get("name")
        elif callee and callee.get("kind") == "MemberExpr":
            name = callee.get("name")
        return SExpr.call(name, (_cpp_expr(x) for x in ch[1:]), resolved=name is not None)
    if k == "ArraySubscriptExpr" and len(ch) >= 2:
        return SExpr.memory(_cpp_expr(ch[0]), _cpp_expr(ch[1]))
    if k == "MemberExpr":
        return SExpr("member", value=n.get("name"))
    return SExpr.unknown(str(k), clang_kind=k)


def _cpp_stmt(n: dict[str, Any]) -> list[SOp]:
    k = n.get("kind")
    ch = _clang_children(n)
    if k == "CompoundStmt":
        return [op for x in ch for op in _cpp_stmt(x)]
    if k == "DeclStmt":
        out: list[SOp] = []
        for d in ch:
            if d.get("kind") != "VarDecl":
                out.append(SOp("unresolved", operator=str(d.get("kind")), resolved=False))
                continue
            init_nodes = _clang_children(d)
            if init_nodes:
                e = _cpp_expr(init_nodes[-1])
                out.append(SOp("assign", target=d.get("name"), args=(e,), resolved=e.resolved))
            else:
                out.append(SOp("declare", target=d.get("name")))
        return out
    if k in ("BinaryOperator", "CompoundAssignOperator"):
        e = _cpp_expr(n)
        if e.kind == "assignment_expr" and len(e.args) == 2:
            left, right = e.args
            if left.kind == "name":
                return [SOp("assign", target=str(left.value), args=(right,), resolved=right.resolved)]
            if left.kind == "memory_load":
                return [SOp("memory_store", args=(left.args[0], left.args[1], right), resolved=left.resolved and right.resolved)]
        if e.kind == "compound_assignment_expr" and len(e.args) == 2:
            left, right = e.args
            if left.kind == "name":
                rhs = SExpr.binary(e.operator or "unknown", left, right, resolved=e.resolved)
                return [SOp("assign", target=str(left.value), args=(rhs,), resolved=rhs.resolved)]
        return [SOp("expr", args=(e,), resolved=e.resolved)]
    if k == "ReturnStmt":
        return [SOp("return", args=(() if not ch else (_cpp_expr(ch[0]),)))]
    if k == "IfStmt":
        # clang children: condition, then, optional else
        if not ch:
            return [SOp("unresolved", operator="IfStmt", resolved=False)]
        cond = _cpp_expr(ch[0])
        body = tuple(_cpp_stmt(ch[1])) if len(ch) > 1 else ()
        alt = tuple(_cpp_stmt(ch[2])) if len(ch) > 2 else ()
        return [SOp("branch", condition=cond, body=body, else_body=alt, resolved=cond.resolved)]
    if k == "WhileStmt":
        if len(ch) < 2:
            return [SOp("unresolved", operator="WhileStmt", resolved=False)]
        cond = _cpp_expr(ch[0])
        return [SOp("loop", operator="while", condition=cond, body=tuple(_cpp_stmt(ch[1])), resolved=cond.resolved)]
    if k == "ForStmt":
        # Exact clang child layout varies. Preserve body + known condition but keep unresolved.
        body_node = ch[-1] if ch else None
        body = tuple(_cpp_stmt(body_node)) if body_node else ()
        cond = None
        for cand in ch[:-1]:
            if cand.get("kind") in ("BinaryOperator", "CXXOperatorCallExpr"):
                e = _cpp_expr(cand)
                if e.kind == "binary":
                    cond = e
                    break
        return [SOp("loop", operator="for", condition=cond, body=body, resolved=False)]
    if k in ("CallExpr", "CXXMemberCallExpr"):
        e = _cpp_expr(n)
        return [SOp("call", operator=e.operator, args=e.args, resolved=e.resolved)]
    if k == "NullStmt":
        return []
    # expression statements in clang appear directly as expression nodes
    if k and k.endswith("Expr"):
        e = _cpp_expr(n)
        return [SOp("expr", args=(e,), resolved=e.resolved)]
    return [SOp("unresolved", operator=str(k), resolved=False, meta={"clang_kind": k})]


def _walk_functions(n: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if n.get("kind") == "FunctionDecl" and n.get("name"):
        yield n
    for ch in _clang_children(n):
        yield from _walk_functions(ch)


def cpp_to_semantic(source: str, name: str = "cpp_module") -> SemanticModule:
    tree = _clang_ast(source)
    funcs: list[SFunction] = []
    for fn in _walk_functions(tree):
        ch = _clang_children(fn)
        params = tuple(x.get("name") for x in ch if x.get("kind") == "ParmVarDecl" and x.get("name"))
        body_node = next((x for x in ch if x.get("kind") == "CompoundStmt"), None)
        if body_node is None:
            continue
        funcs.append(SFunction(fn["name"], params, tuple(_cpp_stmt(body_node))))
    return _module(name, "cpp", funcs, [], "clang-json-ast")


# ---------------------------------------------------------------------------
# Rust bootstrap frontend (conservative core subset)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RTok:
    k: str
    v: str
    pos: int


_R_MULTI = ["==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "->", "::"]


def _rust_lex(src: str) -> list[_RTok]:
    out: list[_RTok] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1; continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        matched = False
        for op in _R_MULTI:
            if src.startswith(op, i):
                out.append(_RTok("op", op, i)); i += len(op); matched = True; break
        if matched: continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            out.append(_RTok("id", src[i:j], i)); i = j; continue
        if c.isdigit():
            j = i + 1
            dot = False
            while j < n and (src[j].isdigit() or (src[j] == "." and not dot)):
                if src[j] == ".": dot = True
                j += 1
            out.append(_RTok("num", src[i:j], i)); i = j; continue
        if c == '"':
            j = i + 1
            while j < n:
                if src[j] == "\\": j += 2; continue
                if src[j] == '"': j += 1; break
                j += 1
            out.append(_RTok("str", src[i:j], i)); i = j; continue
        if c in "(){}[],:;=+-*/%<>!&|":
            out.append(_RTok("p", c, i)); i += 1; continue
        out.append(_RTok("unknown", c, i)); i += 1
    out.append(_RTok("eof", "", n))
    return out


_R_BIN_PREC = {
    "||": (1, "or"), "&&": (2, "and"), "==": (3, "eq"), "!=": (3, "ne"),
    "<": (4, "lt"), "<=": (4, "le"), ">": (4, "gt"), ">=": (4, "ge"),
    "+": (5, "add"), "-": (5, "sub"), "*": (6, "mul"), "/": (6, "div"), "%": (6, "mod"),
}


class _RustParser:
    def __init__(self, src: str):
        self.t = _rust_lex(src)
        self.i = 0

    @property
    def cur(self) -> _RTok:
        return self.t[self.i]

    def pop(self, v: str | None = None) -> _RTok:
        t = self.cur
        if v is not None and t.v != v:
            raise FrontendError(f"Rust subset: expected {v!r} at {t.pos}, got {t.v!r}")
        self.i += 1
        return t

    def accept(self, v: str) -> bool:
        if self.cur.v == v:
            self.i += 1; return True
        return False

    def ident(self) -> str:
        if self.cur.k != "id":
            raise FrontendError(f"Rust subset: expected identifier at {self.cur.pos}")
        return self.pop().v

    def skip_type_until(self, stop: set[str]) -> None:
        depth = 0
        while self.cur.k != "eof":
            if depth == 0 and self.cur.v in stop:
                return
            if self.cur.v in ("<", "(", "["): depth += 1
            elif self.cur.v in (">", ")", "]") and depth > 0: depth -= 1
            self.i += 1

    def expr(self, minp: int = 0) -> SExpr:
        t = self.cur
        if t.v in ("!", "-", "+"):
            self.pop(); op = {"!":"not", "-":"neg", "+":"pos"}[t.v]
            left = SExpr.unary(op, self.expr(7))
        elif t.k == "num":
            self.pop(); left = SExpr.const(float(t.v) if "." in t.v else int(t.v))
        elif t.k == "str":
            self.pop(); left = SExpr.const(t.v[1:-1])
        elif t.k == "id" and t.v in ("true", "false"):
            self.pop(); left = SExpr.const(t.v == "true")
        elif t.k == "id":
            name = self.ident()
            if self.accept("("):
                args: list[SExpr] = []
                if not self.accept(")"):
                    args.append(self.expr())
                    while self.accept(","):
                        args.append(self.expr())
                    self.pop(")")
                left = SExpr.call(name, args)
            else:
                left = SExpr.name(name)
        elif self.accept("("):
            left = self.expr(); self.pop(")")
        else:
            self.pop()
            left = SExpr.unknown(f"rust_token:{t.v}")

        while True:
            if self.cur.v == "[":
                self.pop("["); idx = self.expr(); self.pop("]"); left = SExpr.memory(left, idx); continue
            item = _R_BIN_PREC.get(self.cur.v)
            if item is None or item[0] < minp:
                break
            raw = self.pop().v
            prec, op = item
            right = self.expr(prec + 1)
            left = SExpr.binary(op, left, right)
        return left

    def block(self) -> tuple[SOp, ...]:
        self.pop("{")
        out: list[SOp] = []
        while self.cur.k != "eof" and self.cur.v != "}":
            out.extend(self.stmt())
        self.pop("}")
        return tuple(out)

    def stmt(self) -> list[SOp]:
        if self.cur.v == "let":
            self.pop("let"); self.accept("mut"); name = self.ident()
            if self.accept(":"):
                self.skip_type_until({"=", ";"})
            if self.accept("="):
                e = self.expr(); self.accept(";")
                return [SOp("assign", target=name, args=(e,), resolved=e.resolved)]
            self.accept(";")
            return [SOp("declare", target=name)]
        if self.cur.v == "return":
            self.pop("return")
            if self.accept(";"):
                return [SOp("return")]
            e = self.expr(); self.accept(";")
            return [SOp("return", args=(e,), resolved=e.resolved)]
        if self.cur.v == "if":
            self.pop("if"); c = self.expr(); b = self.block(); a: tuple[SOp,...] = ()
            if self.accept("else"):
                if self.cur.v == "if":
                    nested = self.stmt(); a = tuple(nested)
                else:
                    a = self.block()
            return [SOp("branch", condition=c, body=b, else_body=a, resolved=c.resolved)]
        if self.cur.v == "while":
            self.pop("while"); c = self.expr(); b = self.block()
            return [SOp("loop", operator="while", condition=c, body=b, resolved=c.resolved)]

        # assignment or expression statement
        start = self.i
        left = self.expr()
        if self.cur.v in ("=", "+=", "-=", "*=", "/="):
            op = self.pop().v
            rhs = self.expr(); self.accept(";")
            if left.kind == "name":
                target = str(left.value)
                if op != "=":
                    bop = {"+=":"add", "-=":"sub", "*=":"mul", "/=":"div"}[op]
                    rhs = SExpr.binary(bop, SExpr.name(target), rhs)
                return [SOp("assign", target=target, args=(rhs,), resolved=rhs.resolved)]
            if left.kind == "memory_load" and op == "=":
                return [SOp("memory_store", args=(left.args[0], left.args[1], rhs), resolved=left.resolved and rhs.resolved)]
            return [SOp("unresolved", operator=f"rust_assign:{op}", resolved=False)]
        self.accept(";")
        if left.kind == "call":
            return [SOp("call", operator=left.operator, args=left.args, resolved=left.resolved)]
        if self.i == start:  # safety
            self.i += 1
        return [SOp("expr", args=(left,), resolved=left.resolved)]

    def function(self) -> SFunction:
        self.pop("fn"); name = self.ident(); self.pop("(")
        params: list[str] = []
        if not self.accept(")"):
            while True:
                p = self.ident(); params.append(p)
                if self.accept(":"):
                    self.skip_type_until({",", ")"})
                if self.accept(")"): break
                self.pop(",")
        if self.accept("->"):
            self.skip_type_until({"{"})
        body = self.block()
        return SFunction(name, tuple(params), body)

    def module(self, name: str) -> SemanticModule:
        funcs: list[SFunction] = []
        top: list[SOp] = []
        while self.cur.k != "eof":
            if self.cur.v in ("pub", "unsafe", "async"):
                # preserve modifier uncertainty; parse the function when possible
                modifier = self.pop().v
                if self.cur.v == "fn":
                    f = self.function()
                    funcs.append(SFunction(f.name, f.params, f.body, resolved=False))
                else:
                    top.append(SOp("unresolved", operator=f"rust_modifier:{modifier}", resolved=False))
            elif self.cur.v == "fn":
                funcs.append(self.function())
            else:
                top.extend(self.stmt())
        return _module(name, "rust", funcs, top, "bootstrap-rust-subset")


def rust_to_semantic(source: str, name: str = "rust_module") -> SemanticModule:
    return _RustParser(source).module(name)


# ---------------------------------------------------------------------------
# Unified entrypoint
# ---------------------------------------------------------------------------


FrontendFn = Any

_FRONTENDS: dict[str, FrontendFn] = {
    "l0": l0_to_semantic,
    "python": python_to_semantic,
    "javascript": javascript_to_semantic,
    "c": c_to_semantic,
    "cpp": cpp_to_semantic,
    "rust": rust_to_semantic,
}
_FRONTEND_ALIASES: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "c++": "cpp",
}


def register_frontend(language: str, fn: FrontendFn, aliases: Iterable[str] = ()) -> None:
    """Register another language adapter without changing the semantic IR."""
    key = language.lower()
    if not key:
        raise FrontendError("frontend language name cannot be empty")
    _FRONTENDS[key] = fn
    for alias in aliases:
        a = alias.lower()
        if not a:
            raise FrontendError("frontend alias cannot be empty")
        _FRONTEND_ALIASES[a] = key


def available_frontends() -> tuple[str, ...]:
    return tuple(sorted(_FRONTENDS))


def compile_semantic(source: str, language: str, name: str = "module") -> SemanticModule:
    raw = language.lower()
    lang = _FRONTEND_ALIASES.get(raw, raw)
    fn = _FRONTENDS.get(lang)
    if fn is None:
        raise FrontendError(
            f"unsupported language frontend: {language!r}; "
            f"available={available_frontends()}"
        )
    return fn(source, name)


def main() -> None:
    ap = argparse.ArgumentParser(description="L0 Language v0.3 cross-language semantic IR")
    ap.add_argument("language", choices=["l0", "python", "javascript", "js", "c", "cpp", "c++", "rust"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--signature", action="store_true")
    args = ap.parse_args()
    src = args.path.read_text(encoding="utf-8")
    mod = compile_semantic(src, args.language, args.path.stem)
    if args.signature:
        print(json.dumps(mod.signature(), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(mod.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

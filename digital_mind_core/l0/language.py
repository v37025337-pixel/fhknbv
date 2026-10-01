from __future__ import annotations

"""L0 Language v0.2: lexer, parser, AST, semantic IR, Python frontend.

This module is intentionally independent from the numerical L0 kernel.  It fixes
source-language syntax first and translates native L0 or Python into one common
IR.  Kernel/backend validation belongs after this layer.
"""

import argparse
import ast as pyast
import json
import re
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterable


class L0SyntaxError(ValueError):
    pass


class L0SemanticError(ValueError):
    pass


class TokenKind(Enum):
    IDENT = auto()
    NUMBER = auto()
    STRING = auto()
    NEWLINE = auto()
    REL_START = auto()   # -[
    REL_END = auto()     # ]->
    ARROW = auto()       # ->
    ASSIGN = auto()      # <-
    COLON = auto()
    DOT = auto()
    COMMA = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    SEMICOLON = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    value: str
    line: int
    column: int


_SINGLE = {
    ":": TokenKind.COLON,
    ".": TokenKind.DOT,
    ",": TokenKind.COMMA,
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    "[": TokenKind.LBRACKET,
    "]": TokenKind.RBRACKET,
    "{": TokenKind.LBRACE,
    "}": TokenKind.RBRACE,
    ";": TokenKind.SEMICOLON,
}


def lex(source: str) -> list[Token]:
    out: list[Token] = []
    i = 0
    line = 1
    col = 1
    n = len(source)

    def emit(kind: TokenKind, value: str, ln: int, cl: int) -> None:
        out.append(Token(kind, value, ln, cl))

    while i < n:
        ch = source[i]

        if ch in " \t\r":
            i += 1
            col += 1
            continue

        if ch == "\n":
            emit(TokenKind.NEWLINE, "\n", line, col)
            i += 1
            line += 1
            col = 1
            continue

        if ch == "#":
            while i < n and source[i] != "\n":
                i += 1
                col += 1
            continue

        if ch in {'"', "'"}:
            start=i;quote=ch;i+=1;escaped=False
            while i<n:
                if source[i]=='\n':raise L0SyntaxError(f'newline in string at {line}:{col}')
                if not escaped and source[i]==quote:break
                escaped=not escaped if source[i]=='\\' else False
                i+=1
            if i>=n:raise L0SyntaxError(f'unterminated string at {line}:{col}')
            i+=1;raw=source[start:i]
            try:pyast.literal_eval(raw)
            except (ValueError,SyntaxError) as exc:raise L0SyntaxError(f'invalid string at {line}:{col}') from exc
            emit(TokenKind.STRING,raw,line,col);col+=len(raw)
            continue

        # longest structural operators first
        if source.startswith("]->", i):
            emit(TokenKind.REL_END, "]->", line, col)
            i += 3
            col += 3
            continue
        if source.startswith("-[", i):
            emit(TokenKind.REL_START, "-[", line, col)
            i += 2
            col += 2
            continue
        if source.startswith("->", i):
            emit(TokenKind.ARROW, "->", line, col)
            i += 2
            col += 2
            continue
        if source.startswith("<-", i):
            emit(TokenKind.ASSIGN, "<-", line, col)
            i += 2
            col += 2
            continue

        if ch in _SINGLE:
            emit(_SINGLE[ch], ch, line, col)
            i += 1
            col += 1
            continue

        if ch.isalpha() or ch == "_":
            ln, cl = line, col
            j = i + 1
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            value = source[i:j]
            emit(TokenKind.IDENT, value, ln, cl)
            col += j - i
            i = j
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            match=re.match(r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',source[i:])
            value=match.group(0)
            emit(TokenKind.NUMBER,value,line,col)
            i+=len(value);col+=len(value)
            continue

        raise L0SyntaxError(f"unexpected character {ch!r} at {line}:{col}")

    out.append(Token(TokenKind.EOF, "", line, col))
    return out


# ------------------------------- AST ---------------------------------------

@dataclass(frozen=True)
class TypeSpec:
    kind: str
    dimension: int | str | None = None


@dataclass(frozen=True)
class EntityDecl:
    role: str
    name: str
    type_spec: TypeSpec | None = None


@dataclass(frozen=True)
class RelationLink:
    sources: tuple[str, ...]
    name: str
    target: str


@dataclass(frozen=True)
class RelationDecl:
    name: str
    properties: dict[str, tuple[Any, ...]]


@dataclass(frozen=True)
class CallExpr:
    name: str
    args: tuple[Any, ...]


@dataclass(frozen=True)
class LearnAction:
    relation: str
    source: str


@dataclass(frozen=True)
class AssignAction:
    target: str
    expression: Any


@dataclass(frozen=True)
class EmitAction:
    name: str


@dataclass(frozen=True)
class EventDecl:
    name: str
    inputs: tuple[str, ...]
    observed_target: str | None
    actions: tuple[Any, ...]


@dataclass(frozen=True)
class ShowStmt:
    path: tuple[str, ...]


@dataclass(frozen=True)
class ProgramAST:
    name: str
    entities: tuple[EntityDecl, ...]
    relation_links: tuple[RelationLink, ...]
    relation_decls: tuple[RelationDecl, ...]
    events: tuple[EventDecl, ...]
    shows: tuple[ShowStmt, ...]


# ------------------------------ parser -------------------------------------

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.i = 0

    @property
    def t(self) -> Token:
        return self.tokens[self.i]

    def advance(self) -> Token:
        t = self.t
        if t.kind is not TokenKind.EOF:
            self.i += 1
        return t

    def accept(self, kind: TokenKind, value: str | None = None) -> Token | None:
        t = self.t
        if t.kind is kind and (value is None or t.value == value):
            self.i += 1
            return t
        return None

    def expect(self, kind: TokenKind, value: str | None = None) -> Token:
        t = self.t
        if t.kind is not kind or (value is not None and t.value != value):
            want = kind.name if value is None else repr(value)
            raise L0SyntaxError(
                f"expected {want} at {t.line}:{t.column}, got {t.kind.name} {t.value!r}"
            )
        self.i += 1
        return t

    def accept_word(self, word: str) -> Token | None:
        return self.accept(TokenKind.IDENT, word)

    def expect_word(self, word: str) -> Token:
        return self.expect(TokenKind.IDENT, word)

    def skip_separators(self) -> None:
        while self.t.kind in (TokenKind.NEWLINE, TokenKind.SEMICOLON):
            self.advance()

    def terminator(self, *, allow_before_rbrace: bool = False) -> None:
        if self.t.kind in (TokenKind.NEWLINE, TokenKind.SEMICOLON):
            self.skip_separators()
            return
        if allow_before_rbrace and self.t.kind is TokenKind.RBRACE:
            return
        t = self.t
        raise L0SyntaxError(f"expected end of statement at {t.line}:{t.column}")

    def ident(self) -> str:
        return self.expect(TokenKind.IDENT).value

    def parse(self) -> ProgramAST:
        self.skip_separators()
        self.expect_word("program")
        name = self.ident()
        self.terminator()

        entities: list[EntityDecl] = []
        links: list[RelationLink] = []
        rels: list[RelationDecl] = []
        events: list[EventDecl] = []
        shows: list[ShowStmt] = []

        while self.t.kind is not TokenKind.EOF:
            self.skip_separators()
            if self.t.kind is TokenKind.EOF:
                break
            if self.t.kind is TokenKind.LPAREN:
                links.append(self.parse_relation_link())
                continue
            if self.t.kind is not TokenKind.IDENT:
                t = self.t
                raise L0SyntaxError(f"unexpected token {t.value!r} at {t.line}:{t.column}")
            word = self.t.value
            if word == "input":
                entities.append(self.parse_entity("input"))
            elif word == "target":
                entities.append(self.parse_entity("target"))
            elif word == "relation":
                rels.append(self.parse_relation_decl())
            elif word == "on":
                events.append(self.parse_event())
            elif word == "show":
                shows.append(self.parse_show())
            else:
                t = self.t
                raise L0SyntaxError(f"unknown statement {word!r} at {t.line}:{t.column}")

        return ProgramAST(name, tuple(entities), tuple(links), tuple(rels), tuple(events), tuple(shows))

    def parse_entity(self, role: str) -> EntityDecl:
        self.expect_word(role)
        name = self.ident()
        typ = None
        if self.accept(TokenKind.COLON):
            typ = self.parse_type()
        self.terminator()
        return EntityDecl(role, name, typ)

    def parse_type(self) -> TypeSpec:
        kind = self.ident()
        if kind == "scalar":
            return TypeSpec("scalar", None)
        if kind not in ("vector", "space"):
            t = self.tokens[self.i - 1]
            raise L0SyntaxError(f"unknown type {kind!r} at {t.line}:{t.column}")
        self.expect(TokenKind.LBRACKET)
        if self.t.kind is TokenKind.NUMBER:
            raw = self.advance().value
            if "." in raw:
                raise L0SyntaxError("dimension must be an integer")
            dim: int | str = int(raw)
            if dim < 1:
                raise L0SyntaxError("dimension must be >= 1")
        else:
            dim = self.ident()
            if dim != "adaptive":
                raise L0SyntaxError("dimension must be an integer or adaptive")
        self.expect(TokenKind.RBRACKET)
        return TypeSpec(kind, dim)

    def parse_ident_list(self) -> tuple[str, ...]:
        vals = [self.ident()]
        while self.accept(TokenKind.COMMA):
            vals.append(self.ident())
        return tuple(vals)

    def parse_relation_link(self) -> RelationLink:
        self.expect(TokenKind.LPAREN)
        sources = self.parse_ident_list()
        self.expect(TokenKind.RPAREN)
        self.expect(TokenKind.REL_START)
        name = self.ident()
        self.expect(TokenKind.REL_END)
        target = self.ident()
        self.terminator()
        return RelationLink(sources, name, target)

    def parse_relation_decl(self) -> RelationDecl:
        self.expect_word("relation")
        name = self.ident()
        self.expect(TokenKind.LBRACE)
        self.skip_separators()
        props: dict[str, tuple[Any, ...]] = {}
        while self.t.kind is not TokenKind.RBRACE:
            if self.t.kind is TokenKind.EOF:
                raise L0SyntaxError(f"unterminated relation {name!r}")
            key = self.ident()
            values: list[Any] = []
            while self.t.kind not in (TokenKind.NEWLINE, TokenKind.SEMICOLON, TokenKind.RBRACE):
                values.append(self.parse_property_atom())
            if not values:
                t = self.t
                raise L0SyntaxError(f"relation property {key!r} requires a value at {t.line}:{t.column}")
            if key in props:
                raise L0SyntaxError(f"duplicate relation property {key!r}")
            props[key] = tuple(values)
            self.terminator(allow_before_rbrace=True)
        self.expect(TokenKind.RBRACE)
        if self.t.kind in (TokenKind.NEWLINE, TokenKind.SEMICOLON):
            self.skip_separators()
        return RelationDecl(name, props)

    def parse_property_atom(self) -> Any:
        if self.t.kind is TokenKind.NUMBER:
            raw = self.advance().value
            return float(raw) if "." in raw else int(raw)
        if self.t.kind is TokenKind.IDENT:
            raw = self.advance().value
            if raw == "true":
                return True
            if raw == "false":
                return False
            return raw
        t = self.t
        raise L0SyntaxError(f"invalid property atom {t.value!r} at {t.line}:{t.column}")

    def parse_event(self) -> EventDecl:
        self.expect_word("on")
        name = self.ident()
        self.expect(TokenKind.LPAREN)
        inputs: tuple[str, ...] = ()
        observed = None
        if self.t.kind is not TokenKind.RPAREN:
            inputs = self.parse_ident_list()
            if self.accept(TokenKind.ARROW):
                observed = self.ident()
        self.expect(TokenKind.RPAREN)
        self.expect(TokenKind.LBRACE)
        self.skip_separators()
        actions: list[Any] = []
        while self.t.kind is not TokenKind.RBRACE:
            if self.t.kind is TokenKind.EOF:
                raise L0SyntaxError(f"unterminated event {name!r}")
            if self.t.kind in (TokenKind.NEWLINE, TokenKind.SEMICOLON):
                self.skip_separators()
                continue
            actions.append(self.parse_action())
        self.expect(TokenKind.RBRACE)
        if self.t.kind in (TokenKind.NEWLINE, TokenKind.SEMICOLON):
            self.skip_separators()
        return EventDecl(name, inputs, observed, tuple(actions))

    def parse_action(self) -> Any:
        if self.t.kind is not TokenKind.IDENT:
            t = self.t
            raise L0SyntaxError(f"expected action at {t.line}:{t.column}")
        if self.t.value == "learn":
            self.advance()
            relation = self.ident()
            self.expect_word("from")
            source = self.ident()
            self.terminator(allow_before_rbrace=True)
            return LearnAction(relation, source)
        if self.t.value == "emit":
            self.advance()
            name = self.ident()
            self.terminator(allow_before_rbrace=True)
            return EmitAction(name)

        target = self.ident()
        self.expect(TokenKind.ASSIGN)
        expr = self.parse_expression()
        self.terminator(allow_before_rbrace=True)
        return AssignAction(target, expr)

    def parse_expression(self) -> Any:
        if self.t.kind is TokenKind.NUMBER:
            raw = self.advance().value
            return float(raw) if "." in raw else int(raw)
        if self.t.kind is not TokenKind.IDENT:
            t = self.t
            raise L0SyntaxError(f"expected expression at {t.line}:{t.column}")
        name = self.advance().value
        if name == "true":
            return True
        if name == "false":
            return False
        if self.accept(TokenKind.LPAREN):
            args: list[Any] = []
            if self.t.kind is not TokenKind.RPAREN:
                args.append(self.parse_expression())
                while self.accept(TokenKind.COMMA):
                    args.append(self.parse_expression())
            self.expect(TokenKind.RPAREN)
            return CallExpr(name, tuple(args))
        return name

    def parse_show(self) -> ShowStmt:
        self.expect_word("show")
        path = [self.ident()]
        while self.accept(TokenKind.DOT):
            path.append(self.ident())
        self.terminator()
        return ShowStmt(tuple(path))


def parse_l0(source: str) -> ProgramAST:
    return Parser(lex(source)).parse()


# --------------------------- normalized semantic IR ------------------------

@dataclass(frozen=True)
class IREntity:
    name: str
    role: str = "value"
    type_kind: str | None = None
    dimension: int | str | None = None


@dataclass(frozen=True)
class IRRelation:
    name: str
    sources: tuple[str, ...]
    target: str
    properties: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    origin: str = "l0"
    resolved: bool = True

    def signature(self) -> tuple[str, tuple[str, ...], str]:
        return self.name, self.sources, self.target


@dataclass(frozen=True)
class IREvent:
    name: str
    inputs: tuple[str, ...]
    observed_target: str | None
    actions: tuple[Any, ...]


@dataclass(frozen=True)
class L0IR:
    name: str
    entities: tuple[IREntity, ...]
    relations: tuple[IRRelation, ...]
    events: tuple[IREvent, ...] = ()
    shows: tuple[tuple[str, ...], ...] = ()
    source_language: str = "l0"

    def relation_signatures(self) -> tuple[tuple[str, tuple[str, ...], str], ...]:
        return tuple(r.signature() for r in self.relations)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def l0_ast_to_ir(tree: ProgramAST) -> L0IR:
    entities: dict[str, IREntity] = {}
    for e in tree.entities:
        if e.name in entities:
            raise L0SemanticError(f"duplicate entity {e.name!r}")
        entities[e.name] = IREntity(
            e.name,
            e.role,
            e.type_spec.kind if e.type_spec else None,
            e.type_spec.dimension if e.type_spec else None,
        )

    decls: dict[str, RelationDecl] = {}
    for d in tree.relation_decls:
        if d.name in decls:
            raise L0SemanticError(f"duplicate relation declaration {d.name!r}")
        decls[d.name] = d

    rels: list[IRRelation] = []
    for link in tree.relation_links:
        for src in link.sources:
            if src not in entities:
                raise L0SemanticError(f"undeclared relation source {src!r}")
        if link.target not in entities:
            raise L0SemanticError(f"undeclared relation target {link.target!r}")
        props = decls[link.name].properties if link.name in decls else {}
        rels.append(IRRelation(link.name, link.sources, link.target, dict(props), "l0", True))

    linked_names = {x.name for x in tree.relation_links}
    unlinked = set(decls) - linked_names
    if unlinked:
        raise L0SemanticError(f"relation declaration(s) without link: {sorted(unlinked)!r}")

    # Event references are checked, but their backend meaning is intentionally not fixed here.
    for ev in tree.events:
        for x in ev.inputs:
            if x not in entities:
                raise L0SemanticError(f"event {ev.name!r} uses undeclared entity {x!r}")
        if ev.observed_target is not None and ev.observed_target not in entities:
            raise L0SemanticError(f"event {ev.name!r} observes undeclared target {ev.observed_target!r}")
        for a in ev.actions:
            if isinstance(a, LearnAction) and a.relation not in linked_names:
                raise L0SemanticError(f"learn references unknown relation {a.relation!r}")
            if isinstance(a, EmitAction) and a.name not in entities:
                raise L0SemanticError(f"emit references undeclared entity {a.name!r}")
            if isinstance(a, AssignAction) and a.target not in entities:
                raise L0SemanticError(f"assignment targets undeclared entity {a.target!r}")

    return L0IR(
        tree.name,
        tuple(entities.values()),
        tuple(rels),
        tuple(IREvent(e.name, e.inputs, e.observed_target, e.actions) for e in tree.events),
        tuple(s.path for s in tree.shows),
        "l0",
    )


def compile_l0_to_ir(source: str) -> L0IR:
    return l0_ast_to_ir(parse_l0(source))


# ---------------------------- Python frontend ------------------------------

_PY_BINOPS: dict[type[pyast.operator], str] = {
    pyast.Add: "add",
    pyast.Sub: "sub",
    pyast.Mult: "mul",
    pyast.Div: "div",
    pyast.FloorDiv: "floordiv",
    pyast.Mod: "mod",
    pyast.Pow: "pow",
    pyast.MatMult: "matmul",
}


def _py_name(node: pyast.AST) -> str | None:
    if isinstance(node, pyast.Name):
        return node.id
    return None


def _literal_name(value: Any) -> str:
    return f"const:{value!r}"


def _python_expr_relation(expr: pyast.AST, target: str) -> IRRelation:
    if isinstance(expr, pyast.BinOp):
        opname = _PY_BINOPS.get(type(expr.op))
        left = _py_name(expr.left) if not isinstance(expr.left, pyast.Constant) else _literal_name(expr.left.value)
        right = _py_name(expr.right) if not isinstance(expr.right, pyast.Constant) else _literal_name(expr.right.value)
        if opname is not None and left is not None and right is not None:
            return IRRelation(opname, (left, right), target, {}, "python", True)

    if isinstance(expr, pyast.Call):
        name = _py_name(expr.func)
        args: list[str] = []
        all_known = name is not None
        for a in expr.args:
            if isinstance(a, pyast.Name):
                args.append(a.id)
            elif isinstance(a, pyast.Constant):
                args.append(_literal_name(a.value))
            else:
                all_known = False
                args.append(f"unknown:{type(a).__name__}")
        if name is not None and all_known:
            return IRRelation(name, tuple(args), target, {}, "python", True)

    return IRRelation(
        f"unknown:{type(expr).__name__}",
        tuple(),
        target,
        {"python_ast": (type(expr).__name__,)},
        "python",
        False,
    )


def python_to_l0_ir(source: str, name: str = "python_module") -> L0IR:
    tree = pyast.parse(source)
    entities: dict[str, IREntity] = {}
    relations: list[IRRelation] = []

    def ensure_entity(x: str, role: str = "value") -> None:
        if x.startswith("const:") or x.startswith("unknown:"):
            return
        if x not in entities:
            entities[x] = IREntity(x, role)
        elif role == "target" and entities[x].role == "value":
            entities[x] = IREntity(x, "target")

    for stmt in tree.body:
        if isinstance(stmt, pyast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], pyast.Name):
            target = stmt.targets[0].id
            rel = _python_expr_relation(stmt.value, target)
            for src in rel.sources:
                ensure_entity(src, "input")
            ensure_entity(target, "target")
            relations.append(rel)
            continue

        if isinstance(stmt, pyast.AnnAssign) and isinstance(stmt.target, pyast.Name) and stmt.value is not None:
            target = stmt.target.id
            rel = _python_expr_relation(stmt.value, target)
            for src in rel.sources:
                ensure_entity(src, "input")
            ensure_entity(target, "target")
            relations.append(rel)
            continue

        # Preserve unsupported top-level semantics explicitly rather than guessing.
        synthetic_target = f"stmt_{getattr(stmt, 'lineno', len(relations)+1)}"
        ensure_entity(synthetic_target, "target")
        relations.append(IRRelation(
            f"unknown:{type(stmt).__name__}", (), synthetic_target,
            {"python_ast": (type(stmt).__name__,)}, "python", False
        ))

    return L0IR(name, tuple(entities.values()), tuple(relations), (), (), "python")


# -------------------------------- CLI ---------------------------------------

def _json_default(obj: Any):
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, Enum):
        return obj.name
    raise TypeError(type(obj).__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="L0 Language v0.2 frontend")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_native = sub.add_parser("l0", help="parse native L0 and emit normalized IR")
    p_native.add_argument("path", type=Path)

    p_py = sub.add_parser("python", help="parse Python and emit normalized L0 IR")
    p_py.add_argument("path", type=Path)

    args = ap.parse_args()
    text = args.path.read_text(encoding="utf-8")
    if args.cmd == "l0":
        ir = compile_l0_to_ir(text)
    else:
        ir = python_to_l0_ir(text, args.path.stem)
    print(json.dumps(ir, default=_json_default, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

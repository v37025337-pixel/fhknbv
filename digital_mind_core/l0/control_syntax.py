from __future__ import annotations

"""Parser for the native L0 v0.4 activation/settling surface syntax.

The syntax deliberately has no ``if`` or ``while`` keywords. Control is
expressed through activation regions and settling regions.
"""

from dataclasses import replace
import ast
from typing import Any

from .language import L0SyntaxError, Token, TokenKind, lex
from .semantic import SExpr
from .control import CAction, CRelation, ControlModule


_BINARY = {
    "add", "sub", "mul", "div", "floordiv", "mod", "pow",
    "lt", "le", "gt", "ge", "eq", "ne", "and", "or",
    "bit_and", "bit_or", "bit_xor",
    "value_and", "value_or", "c_div", "c_mod",
}
_UNARY = {"not", "neg", "pos", "bit_not", "c_int", "c_float"}


class ControlSyntaxError(L0SyntaxError):
    pass


class Parser:
    def __init__(self, source: str):
        self.tokens = lex(source)
        self.i = 0

    def cur(self) -> Token:
        return self.tokens[self.i]

    def at(self, kind: TokenKind, value: str | None = None) -> bool:
        t = self.cur()
        return t.kind == kind and (value is None or t.value == value)

    def take(self, kind: TokenKind, value: str | None = None) -> Token:
        t = self.cur()
        if not self.at(kind, value):
            want = kind.name + (f" {value!r}" if value is not None else "")
            raise ControlSyntaxError(f"expected {want} at {t.line}:{t.column}, got {t.kind.name} {t.value!r}")
        self.i += 1
        return t

    def skip_sep(self) -> None:
        while self.at(TokenKind.NEWLINE) or self.at(TokenKind.SEMICOLON):
            self.i += 1

    def ident(self, value: str | None = None) -> str:
        return self.take(TokenKind.IDENT, value).value

    def parse_expr(self) -> SExpr:
        t = self.cur()
        if t.kind == TokenKind.NUMBER:
            self.i += 1
            v: Any = float(t.value) if any(c in t.value for c in '.eE') else int(t.value)
            return SExpr.const(v)
        if t.kind == TokenKind.STRING:
            self.i += 1
            return SExpr.const(ast.literal_eval(t.value))
        if t.kind != TokenKind.IDENT:
            raise ControlSyntaxError(f"expected expression at {t.line}:{t.column}")
        name = t.value
        self.i += 1
        if name == "true":
            return SExpr.const(True)
        if name == "false":
            return SExpr.const(False)
        if name == "null":
            return SExpr.const(None)
        if not self.at(TokenKind.LPAREN):
            return SExpr.name(name)
        self.i += 1
        args: list[SExpr] = []
        if not self.at(TokenKind.RPAREN):
            args.append(self.parse_expr())
            while self.at(TokenKind.COMMA):
                self.i += 1
                args.append(self.parse_expr())
        self.take(TokenKind.RPAREN)
        if name in _BINARY and len(args) == 2:
            return SExpr.binary(name, args[0], args[1])
        if name in _UNARY and len(args) == 1:
            return SExpr.unary(name, args[0])
        if name == "load" and len(args) == 2:
            return SExpr.memory(args[0], args[1])
        return SExpr.call(name, args)

    def parse_block(self) -> tuple[CAction, ...]:
        self.take(TokenKind.LBRACE)
        self.skip_sep()
        out: list[CAction] = []
        while not self.at(TokenKind.RBRACE):
            out.append(self.parse_action())
            self.skip_sep()
        self.take(TokenKind.RBRACE)
        return tuple(out)

    def parse_relation_link(self) -> CAction:
        self.take(TokenKind.LPAREN)
        args: list[SExpr] = []
        if not self.at(TokenKind.RPAREN):
            args.append(self.parse_expr())
            while self.at(TokenKind.COMMA):
                self.i += 1
                args.append(self.parse_expr())
        self.take(TokenKind.RPAREN)
        self.take(TokenKind.REL_START)
        name = self.ident()
        self.take(TokenKind.REL_END)
        target = self.ident()
        return CAction("relation", target=target, operator=name, args=tuple(args))

    def parse_action(self) -> CAction:
        self.skip_sep()
        if self.at(TokenKind.LBRACKET):
            self.i += 1
            guard = self.parse_expr()
            self.take(TokenKind.RBRACKET)
            self.skip_sep()
            return CAction("active", guard=guard, body=self.parse_block())

        if self.at(TokenKind.LPAREN):
            return self.parse_relation_link()

        if not self.at(TokenKind.IDENT):
            t = self.cur()
            raise ControlSyntaxError(f"expected action at {t.line}:{t.column}")

        word = self.cur().value
        if word == "settle":
            self.i += 1
            self.take(TokenKind.LBRACKET)
            guard = self.parse_expr()
            self.take(TokenKind.RBRACKET)
            self.skip_sep()
            return CAction("settle", guard=guard, body=self.parse_block())

        if word == "yield":
            self.i += 1
            if self.at(TokenKind.NEWLINE) or self.at(TokenKind.SEMICOLON) or self.at(TokenKind.RBRACE):
                return CAction("effect", operator="yield")
            return CAction("effect", operator="yield", args=(self.parse_expr(),))

        if word == "state":
            self.i += 1
            return CAction("declare", target=self.ident())

        if word == "store":
            self.i += 1
            self.take(TokenKind.LPAREN)
            base = self.parse_expr()
            self.take(TokenKind.COMMA)
            idx = self.parse_expr()
            self.take(TokenKind.RPAREN)
            self.take(TokenKind.ASSIGN)
            value = self.parse_expr()
            return CAction("store", args=(base, idx, value))

        target = self.ident()
        self.take(TokenKind.ASSIGN)
        value = self.parse_expr()
        return CAction("flow", target=target, args=(value,))

    def parse(self) -> ControlModule:
        self.skip_sep()
        self.ident("module")
        name = self.ident()
        self.skip_sep()
        relations: list[CRelation] = []
        top: tuple[CAction, ...] = ()

        while not self.at(TokenKind.EOF):
            if self.at(TokenKind.IDENT, "relation"):
                self.i += 1
                rname = self.ident()
                self.take(TokenKind.LPAREN)
                params: list[str] = []
                defaults: list[SExpr] = []
                if not self.at(TokenKind.RPAREN):
                    while True:
                        params.append(self.ident())
                        if self.at(TokenKind.ASSIGN):
                            self.i+=1
                            value=self.parse_expr()
                            def is_literal(e):
                                return e.kind=='const' or (e.kind=='unary' and e.operator in {'neg','pos'} and len(e.args)==1 and is_literal(e.args[0]))
                            if not is_literal(value):raise ControlSyntaxError('parameter defaults must be literal values')
                            defaults.append(value)
                        elif defaults:raise ControlSyntaxError('required parameter follows a default')
                        if not self.at(TokenKind.COMMA):break
                        self.i += 1
                self.take(TokenKind.RPAREN)
                self.skip_sep()
                body = self.parse_block()
                relations.append(CRelation(rname, tuple(params), body, resolved=all(x.resolved for x in body), defaults=tuple(defaults)))
                self.skip_sep()
                continue
            if self.at(TokenKind.IDENT, "surface"):
                self.i += 1
                self.skip_sep()
                top = self.parse_block()
                self.skip_sep()
                continue
            t = self.cur()
            raise ControlSyntaxError(f"unexpected top-level token {t.value!r} at {t.line}:{t.column}")

        unresolved = sum(0 if r.resolved else 1 for r in relations)
        return ControlModule(
            name=name,
            source_language="l0-control",
            relations=tuple(relations),
            top_level=top,
            unresolved_count=unresolved,
            metadata={"control_semantics": "activation-settling-v0.4", "parser": "native-control-v0.4"},
        )


def parse_control(source: str) -> ControlModule:
    return Parser(source).parse()


"""
v0.28_mechanism_genesis.py

A small program-synthesis engine for pure scalar mechanisms.

This is deliberately NOT a library of named mutations. Given input/output
examples, it searches a generic expression grammar, prunes behaviorally
equivalent candidates, emits a minimal pure-Python function, and can return
NO_MECHANISM_FOUND when the specification is outside its supported grammar.

Current grammar:
- variables and numeric constants
- +, -, *, min, max
- abs, unary -
- <, <=, >, >=, ==
- piecewise if/else

No I/O, imports, reflection, network access, mutation, loops, or arbitrary exec
are part of the synthesized language.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import keyword


Number = int | float


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple
    cost: int


@dataclass(slots=True)
class SynthesisResult:
    status: str
    name: str
    args: List[str]
    expression: Optional[Expr]
    source: Optional[str]
    cost: Optional[int]
    examples_passed: int
    examples_total: int
    explored_numeric: int
    explored_boolean: int
    explanation: str


def _is_number(x: Any) -> bool:
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except (OverflowError, ValueError):
        return False


def _canon(x: Any):
    if isinstance(x, bool):
        return ("b", x)
    if _is_number(x):
        v = float(x)
        if abs(v) < 1e-12:
            v = 0.0
        return ("n", round(v, 10))
    return ("x", repr(x))


def _same(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if _is_number(a) and _is_number(b):
        return abs(float(a) - float(b)) <= tol
    return a == b


def evaluate(expr: Expr, env: Dict[str, Any]) -> Any:
    op = expr.op
    a = expr.args

    if op == "var":
        return env[a[0]]
    if op == "const":
        return a[0]

    if op == "neg":
        return -float(evaluate(a[0], env))
    if op == "abs":
        return abs(float(evaluate(a[0], env)))

    if op in {"add", "sub", "mul", "min", "max"}:
        x = float(evaluate(a[0], env))
        y = float(evaluate(a[1], env))
        if op == "add":
            return x + y
        if op == "sub":
            return x - y
        if op == "mul":
            return x * y
        if op == "min":
            return min(x, y)
        return max(x, y)

    if op in {"lt", "le", "gt", "ge", "eq"}:
        x = evaluate(a[0], env)
        y = evaluate(a[1], env)
        if op == "lt":
            return float(x) < float(y)
        if op == "le":
            return float(x) <= float(y)
        if op == "gt":
            return float(x) > float(y)
        if op == "ge":
            return float(x) >= float(y)
        return _same(x, y)

    if op == "if":
        return evaluate(a[1], env) if bool(evaluate(a[0], env)) else evaluate(a[2], env)

    raise ValueError(f"unknown op: {op}")


def to_source(expr: Expr) -> str:
    op = expr.op
    a = expr.args
    if op == "var":
        return a[0]
    if op == "const":
        return repr(a[0])
    if op == "neg":
        return f"(-float({to_source(a[0])}))"
    if op == "abs":
        return f"abs(float({to_source(a[0])}))"
    if op == "add":
        return f"(float({to_source(a[0])}) + float({to_source(a[1])}))"
    if op == "sub":
        return f"(float({to_source(a[0])}) - float({to_source(a[1])}))"
    if op == "mul":
        return f"(float({to_source(a[0])}) * float({to_source(a[1])}))"
    if op == "min":
        return f"min(float({to_source(a[0])}), float({to_source(a[1])}))"
    if op == "max":
        return f"max(float({to_source(a[0])}), float({to_source(a[1])}))"
    if op == "eq":
        return f"(abs(float({to_source(a[0])}) - float({to_source(a[1])})) <= 1e-9)"
    if op in {"lt", "le", "gt", "ge", "eq"}:
        symbol = {"lt":"<", "le":"<=", "gt":">", "ge":">=", "eq":"=="}[op]
        return f"(float({to_source(a[0])}) {symbol} float({to_source(a[1])}))"
    if op == "if":
        return f"({to_source(a[1])} if {to_source(a[0])} else {to_source(a[2])})"
    raise ValueError(op)


def explain(expr: Expr) -> str:
    op = expr.op
    if op == "var":
        return f"use input {expr.args[0]}"
    if op == "const":
        return f"use constant {expr.args[0]!r}"
    if op == "abs":
        return f"take absolute value of ({explain(expr.args[0])})"
    if op == "neg":
        return f"negate ({explain(expr.args[0])})"
    if op in {"add","sub","mul","min","max"}:
        label = {
            "add":"add", "sub":"subtract", "mul":"multiply",
            "min":"take minimum of", "max":"take maximum of",
        }[op]
        return f"{label} ({explain(expr.args[0])}) and ({explain(expr.args[1])})"
    if op in {"lt","le","gt","ge","eq"}:
        label = {"lt":"<","le":"<=","gt":">","ge":">=","eq":"=="}[op]
        return f"test whether ({explain(expr.args[0])}) {label} ({explain(expr.args[1])})"
    if op == "if":
        return (
            f"if {explain(expr.args[0])}, then {explain(expr.args[1])}; "
            f"otherwise {explain(expr.args[2])}"
        )
    return op


class MechanismGenesis:
    def __init__(
        self,
        *,
        max_numeric_cost: int = 7,
        max_boolean_cost: int = 8,
        max_numeric_candidates: int = 5000,
        max_boolean_candidates: int = 5000,
    ):
        self.max_numeric_cost = int(max_numeric_cost)
        self.max_boolean_cost = int(max_boolean_cost)
        self.max_numeric_candidates = int(max_numeric_candidates)
        self.max_boolean_candidates = int(max_boolean_candidates)

    @staticmethod
    def _examples(spec: Dict[str, Any]):
        examples = spec.get("examples", [])
        if not isinstance(examples, list) or not examples:
            raise ValueError("spec.examples must be a non-empty list")
        out = []
        for item in examples:
            if not isinstance(item, dict) or "inputs" not in item or "output" not in item:
                raise ValueError("each example needs inputs and output")
            if not isinstance(item["inputs"], dict):
                raise ValueError("example.inputs must be a dict")
            out.append((dict(item["inputs"]), item["output"]))
        return out

    @staticmethod
    def _validate_numeric_spec(args: List[str], examples):
        for env, target in examples:
            if set(env) != set(args):
                return False
            if not all(_is_number(env[a]) for a in args):
                return False
            if not _is_number(target):
                return False
        return True

    @staticmethod
    def _signature(expr: Expr, envs) -> Optional[tuple]:
        values = []
        try:
            for env in envs:
                v = evaluate(expr, env)
                if isinstance(v, bool):
                    values.append(("b", v))
                elif _is_number(v) and abs(float(v)) <= 1e12:
                    values.append(_canon(v))
                else:
                    return None
        except Exception:
            return None
        return tuple(values)

    def _enumerate(self, args: List[str], examples, constants: Sequence[Number]):
        envs = [env for env, _ in examples]

        numeric_by_cost: Dict[int, List[Expr]] = {}
        bool_by_cost: Dict[int, List[Expr]] = {}
        numeric_seen: Dict[tuple, Expr] = {}
        bool_seen: Dict[tuple, Expr] = {}

        def add_numeric(expr):
            if len(numeric_seen) >= self.max_numeric_candidates:
                return False
            sig = self._signature(expr, envs)
            if sig is None or any(k != "n" for k, _ in sig):
                return False
            if sig in numeric_seen:
                return False
            numeric_seen[sig] = expr
            numeric_by_cost.setdefault(expr.cost, []).append(expr)
            return True

        def add_bool(expr):
            if len(bool_seen) >= self.max_boolean_candidates:
                return False
            sig = self._signature(expr, envs)
            if sig is None or any(k != "b" for k, _ in sig):
                return False
            if sig in bool_seen:
                return False
            bool_seen[sig] = expr
            bool_by_cost.setdefault(expr.cost, []).append(expr)
            return True

        for arg in args:
            add_numeric(Expr("var", (arg,), 1))
        for c in constants:
            add_numeric(Expr("const", (c,), 1))

        for cost in range(2, self.max_numeric_cost + 1):
            # Unary operators.
            for child in list(numeric_by_cost.get(cost - 1, [])):
                add_numeric(Expr("abs", (child,), cost))
                add_numeric(Expr("neg", (child,), cost))

            # Binary numeric operators.
            for lc in range(1, cost - 1):
                rc = cost - 1 - lc
                if rc < 1:
                    continue
                lefts = numeric_by_cost.get(lc, [])
                rights = numeric_by_cost.get(rc, [])
                for left in lefts:
                    for right in rights:
                        add_numeric(Expr("add", (left, right), cost))
                        add_numeric(Expr("sub", (left, right), cost))
                        add_numeric(Expr("mul", (left, right), cost))
                        add_numeric(Expr("min", (left, right), cost))
                        add_numeric(Expr("max", (left, right), cost))

        # Boolean comparisons from the numeric pool.
        all_numeric = sorted(numeric_seen.values(), key=lambda e: (e.cost, to_source(e)))
        for left in all_numeric:
            for right in all_numeric:
                cost = 1 + left.cost + right.cost
                if cost > self.max_boolean_cost:
                    continue
                add_bool(Expr("lt", (left, right), cost))
                add_bool(Expr("le", (left, right), cost))
                add_bool(Expr("gt", (left, right), cost))
                add_bool(Expr("ge", (left, right), cost))
                add_bool(Expr("eq", (left, right), cost))

        return numeric_seen, bool_seen

    @staticmethod
    def _matches_targets(expr: Expr, examples, indices=None) -> bool:
        selected = range(len(examples)) if indices is None else indices
        for i in selected:
            env, target = examples[i]
            try:
                value = evaluate(expr, env)
            except Exception:
                return False
            if not _same(value, target):
                return False
        return True

    def synthesize(self, spec: Dict[str, Any]) -> SynthesisResult:
        name = str(spec.get("name", "generated_mechanism"))
        args = [str(x) for x in spec.get("args", [])]
        examples = self._examples(spec)

        if not args or len(set(args)) != len(args):
            raise ValueError("spec.args must contain unique argument names")

        if not self._validate_numeric_spec(args, examples):
            return SynthesisResult(
                "NO_MECHANISM_FOUND", name, args, None, None, None,
                0, len(examples), 0, 0,
                "current synthesis grammar supports finite numeric scalar examples only",
            )

        constants = spec.get("constants", [-2, -1, 0, 1, 2])
        constants = [c for c in constants if _is_number(c)]

        numeric, boolean = self._enumerate(args, examples, constants)
        numeric_exprs = sorted(
            numeric.values(), key=lambda e: (e.cost, len(to_source(e)), to_source(e))
        )

        # First search pure numeric expressions.
        for expr in numeric_exprs:
            if self._matches_targets(expr, examples):
                source = self._function_source(name, args, expr)
                return SynthesisResult(
                    "FOUND", name, args, expr, source, expr.cost,
                    len(examples), len(examples), len(numeric), len(boolean),
                    explain(expr),
                )

        # Then synthesize a piecewise function from a boolean partition.
        bool_exprs = sorted(
            boolean.values(), key=lambda e: (e.cost, len(to_source(e)), to_source(e))
        )
        for cond in bool_exprs:
            truth = []
            false = []
            for i, (env, _) in enumerate(examples):
                try:
                    flag = bool(evaluate(cond, env))
                except Exception:
                    flag = False
                (truth if flag else false).append(i)

            if not truth or not false:
                continue

            left_matches = [
                e for e in numeric_exprs
                if self._matches_targets(e, examples, truth)
            ]
            if not left_matches:
                continue

            right_matches = [
                e for e in numeric_exprs
                if self._matches_targets(e, examples, false)
            ]
            if not right_matches:
                continue

            then_expr = left_matches[0]
            else_expr = right_matches[0]
            expr = Expr(
                "if",
                (cond, then_expr, else_expr),
                1 + cond.cost + then_expr.cost + else_expr.cost,
            )
            if self._matches_targets(expr, examples):
                source = self._function_source(name, args, expr)
                return SynthesisResult(
                    "FOUND", name, args, expr, source, expr.cost,
                    len(examples), len(examples), len(numeric), len(boolean),
                    explain(expr),
                )

        return SynthesisResult(
            "NO_MECHANISM_FOUND", name, args, None, None, None,
            0, len(examples), len(numeric), len(boolean),
            "no expression in the current generic grammar satisfies all examples",
        )

    @staticmethod
    def _function_source(name: str, args: List[str], expr: Expr) -> str:
        reserved = {'float', 'abs', 'min', 'max'}
        if not name.isidentifier() or keyword.iskeyword(name) or name in reserved:
            raise ValueError("mechanism name must be a Python identifier")
        for arg in args:
            if not arg.isidentifier() or keyword.iskeyword(arg) or arg in reserved:
                raise ValueError(f"invalid argument name: {arg!r}")
        header = f"def {name}({', '.join(args)}):\n"
        if expr.op == 'if':
            cond, then, otherwise = expr.args
            return (header + f"    if {to_source(cond)}:\n"
                    + f"        return {to_source(then)}\n"
                    + f"    return {to_source(otherwise)}\n")
        return header + f"    return {to_source(expr)}\n"


def execute_result(result: SynthesisResult, inputs: Dict[str, Any]) -> Any:
    if result.status != "FOUND" or result.expression is None:
        raise ValueError("no synthesized mechanism is available")
    if set(inputs) != set(result.args) or not all(_is_number(v) for v in inputs.values()):
        raise ValueError('mechanism inputs must match its finite numeric arguments')
    value = evaluate(result.expression, dict(inputs))
    if not _is_number(value):
        raise ValueError('mechanism produced a non-finite scalar')
    return value


def result_dict(result: SynthesisResult) -> dict:
    return {
        "status": result.status,
        "name": result.name,
        "args": result.args,
        "source": result.source,
        "cost": result.cost,
        "examples_passed": result.examples_passed,
        "examples_total": result.examples_total,
        "explored_numeric": result.explored_numeric,
        "explored_boolean": result.explored_boolean,
        "explanation": result.explanation,
    }


def cegis_synthesize(
    engine: MechanismGenesis,
    spec: Dict[str, Any],
    validation_examples: List[Dict[str, Any]],
    *,
    max_rounds: int = 8,
) -> dict:
    """
    Counterexample-guided refinement.

    validation_examples are not part of the initial synthesis set. A failing
    validation case is promoted into the training set and synthesis restarts.
    This helps reject accidental formulas that merely interpolate the first
    few examples.
    """
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError('max_rounds must be a positive integer')
    work = {
        "name": spec.get("name", "generated_mechanism"),
        "args": list(spec.get("args", [])),
        "constants": list(spec.get("constants", [-2,-1,0,1,2])),
        "examples": [dict(x) for x in spec.get("examples", [])],
    }

    history = []

    for round_index in range(1, int(max_rounds) + 1):
        result = engine.synthesize(work)
        event = {
            "round": round_index,
            "status": result.status,
            "source": result.source,
            "cost": result.cost,
            "training_examples": len(work["examples"]),
        }

        if result.status != "FOUND":
            event["validation_passed"] = 0
            event["validation_total"] = len(validation_examples)
            history.append(event)
            return {
                "status": "NO_MECHANISM_FOUND",
                "result": result,
                "history": history,
                "counterexamples_added": len(work["examples"]) - len(spec.get("examples", [])),
            }

        failures = []
        passed = 0
        for item in validation_examples:
            env = dict(item["inputs"])
            target = item["output"]
            value = execute_result(result, env)
            if _same(value, target):
                passed += 1
            else:
                failures.append({
                    "inputs": env,
                    "output": target,
                    "observed": value,
                })

        event["validation_passed"] = passed
        event["validation_total"] = len(validation_examples)
        event["failure_count"] = len(failures)
        history.append(event)

        if not failures:
            return {
                "status": "FOUND",
                "result": result,
                "history": history,
                "counterexamples_added": len(work["examples"]) - len(spec.get("examples", [])),
            }

        # Add one counterexample per round. This makes the causal effect of each
        # refinement visible and prevents silently folding the whole validator
        # into the initial training set.
        first = failures[0]
        candidate = {
            "inputs": dict(first["inputs"]),
            "output": first["output"],
        }
        if candidate not in work["examples"]:
            work["examples"].append(candidate)
        else:
            # If the same counterexample is already present yet still fails,
            # the current grammar cannot satisfy the specification reliably.
            return {
                "status": "NO_MECHANISM_FOUND",
                "result": result,
                "history": history,
                "counterexamples_added": len(work["examples"]) - len(spec.get("examples", [])),
            }

    return {
        "status": "NO_MECHANISM_FOUND",
        "result": result,
        "history": history,
        "counterexamples_added": len(work["examples"]) - len(spec.get("examples", [])),
    }

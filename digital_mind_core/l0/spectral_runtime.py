from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .language import (
    AssignAction,
    CallExpr,
    EmitAction,
    IREvent,
    L0IR,
    LearnAction,
    compile_l0_to_ir,
)


@dataclass
class RelationSpec:
    name: str
    sources: list[str]
    target: str
    options: dict[str, str]


@dataclass
class ProgramSpec:
    name: str
    inputs: list[str]
    target: str
    relation: RelationSpec
    shows: list[str]


class L0BackendError(ValueError):
    pass


def _single_property(relation, key: str):
    values = relation.properties.get(key)
    if values is None:
        raise L0BackendError(f"missing relation property {key!r}")
    if len(values) != 1:
        raise L0BackendError(
            f"v0.24 backend requires one value for {key!r}, got {values!r}"
        )
    return values[0]


def _find_event(ir: L0IR, name: str) -> IREvent:
    matches = [e for e in ir.events if e.name == name]
    if len(matches) != 1:
        raise L0BackendError(
            f"v0.24 backend requires exactly one {name!r} event, got {len(matches)}"
        )
    return matches[0]


def compile_ir_to_backend(ir: L0IR) -> ProgramSpec:
    """Lower normalized L0 IR into the current numerical v0.24 backend contract.

    Grammar/semantic parsing belongs to l0_language_v02.  This function only
    checks what the present numerical backend can execute.
    """
    if ir.source_language != "l0":
        raise L0BackendError(
            "v0.24 numerical backend currently executes native L0 programs; "
            "foreign frontends must first produce an executable backend profile"
        )

    inputs = [e for e in ir.entities if e.role == "input"]
    targets = [e for e in ir.entities if e.role == "target"]
    if not inputs:
        raise L0BackendError("program requires at least one input")
    if len(targets) != 1:
        raise L0BackendError(
            f"v0.24 backend requires exactly one target, got {len(targets)}"
        )

    # The language can represent vectors/spaces already.  This backend still
    # consumes one scalar CSV column per source entity, so reject rather than
    # silently flattening higher-order values.
    for e in inputs + targets:
        if e.type_kind not in (None, "scalar"):
            raise L0BackendError(
                f"entity {e.name!r} has type {e.type_kind}[{e.dimension}]; "
                "v0.24 backend currently accepts scalar entities only"
            )

    if len(ir.relations) != 1:
        raise L0BackendError(
            f"v0.24 backend requires exactly one source relation, got {len(ir.relations)}"
        )
    rel = ir.relations[0]
    input_names = [e.name for e in inputs]
    target = targets[0].name

    if list(rel.sources) != input_names:
        raise L0BackendError(
            f"relation sources {list(rel.sources)!r} differ from declared inputs {input_names!r}"
        )
    if rel.target != target:
        raise L0BackendError(
            f"relation target {rel.target!r} differs from declared target {target!r}"
        )

    required = {
        "dimension": "adaptive",
        "geometry": "spectral",
        "law": "quadratic",
        "birth": "residual",
        "probation": "future",
        "death": "weak",
        "time": "frozen",
    }
    options: dict[str, str] = {}
    for key, expected in required.items():
        got = _single_property(rel, key)
        if got != expected:
            raise L0BackendError(
                f"v0.24 backend requires `{key} {expected}`, got {got!r}"
            )
        options[key] = str(got)

    observe = _find_event(ir, "observe")
    if list(observe.inputs) != input_names or observe.observed_target != target:
        raise L0BackendError(
            "observe signature must be (all inputs -> target) in declaration order"
        )
    if len(observe.actions) != 1:
        raise L0BackendError("v0.24 observe event currently supports exactly one action")
    learn = observe.actions[0]
    if not isinstance(learn, LearnAction) or learn.relation != rel.name or learn.source != "error":
        raise L0BackendError(f"observe must contain `learn {rel.name} from error`")

    predict = _find_event(ir, "predict")
    if list(predict.inputs) != input_names or predict.observed_target is not None:
        raise L0BackendError("predict signature must contain exactly the declared inputs")
    assigns = [a for a in predict.actions if isinstance(a, AssignAction)]
    emits = [a for a in predict.actions if isinstance(a, EmitAction)]
    if len(predict.actions) != 2 or len(assigns) != 1 or len(emits) != 1:
        raise L0BackendError("v0.24 predict event requires one assignment and one emit")
    assign = assigns[0]
    if assign.target != target or not isinstance(assign.expression, CallExpr):
        raise L0BackendError(f"predict must assign relation call to target {target!r}")
    call = assign.expression
    if call.name != rel.name or tuple(call.args) != tuple(input_names):
        raise L0BackendError(
            f"predict must call {rel.name}({', '.join(input_names)})"
        )
    if emits[0].name != target:
        raise L0BackendError(f"predict must emit target {target!r}")

    shows = [".".join(path) for path in ir.shows]
    return ProgramSpec(
        name=ir.name,
        inputs=input_names,
        target=target,
        relation=RelationSpec(rel.name, list(rel.sources), rel.target, options),
        shows=shows,
    )


def compile_source(text: str) -> tuple[L0IR, ProgramSpec]:
    ir = compile_l0_to_ir(text)
    return ir, compile_ir_to_backend(ir)


def load_core(path: Path):
    # Single-file canonical build: spectral core is embedded, not loaded from disk.
    from . import spectral_scalar
    return spectral_scalar


SMOKE_ROWS = [
    (0.344167,0.805833,0.160446,6,1,0,985),
    (0.363478,0.696087,0.248539,0,1,0,801),
    (0.196364,0.437273,0.248309,1,1,0,1349),
    (0.200000,0.590435,0.160296,2,1,0,1562),
    (0.226957,0.436957,0.186900,3,1,0,1600),
    (0.204348,0.518261,0.0895652,4,1,0,1606),
    (0.196522,0.498696,0.168726,5,1,0,1510),
    (0.165000,0.535833,0.266804,6,1,0,959),
    (0.138333,0.434167,0.361950,0,1,0,822),
    (0.150833,0.482917,0.223267,1,1,0,1321),
    (0.169091,0.686364,0.122132,2,1,0,1263),
    (0.172727,0.599545,0.304627,3,1,0,1162),
    (0.165000,0.470417,0.301000,4,1,0,1406),
    (0.160870,0.537826,0.126548,5,1,0,1421),
    (0.233333,0.498750,0.157963,6,1,0,1248),
    (0.231667,0.483750,0.188433,0,1,0,1204),
    (0.175833,0.537500,0.194017,1,1,1,1000),
    (0.216667,0.861667,0.146775,2,1,0,683),
    (0.292174,0.741739,0.208317,3,1,0,1650),
    (0.261667,0.538333,0.195904,4,1,0,1927),
    (0.177500,0.457083,0.353242,5,1,0,1543),
    (0.0591304,0.400000,0.171970,6,1,0,981),
    (0.0965217,0.436522,0.246600,0,1,0,986),
    (0.0973913,0.491739,0.158330,1,1,0,1416),
    (0.223478,0.616957,0.129796,2,1,0,1985),
    (0.217500,0.862500,0.293850,3,1,0,506),
    (0.195000,0.687500,0.113837,4,1,0,431),
    (0.203478,0.793043,0.123300,5,1,0,1167),
    (0.196522,0.651739,0.145365,6,1,0,1098),
    (0.216522,0.722174,0.0739826,0,1,0,1096),
    (0.180833,0.603750,0.187192,1,1,0,1501),
    (0.192174,0.829565,0.053213,2,1,0,1360),
    (0.260000,0.775417,0.264308,3,1,0,1526),
    (0.186957,0.437826,0.277752,4,1,0,1550),
    (0.211304,0.585217,0.127839,5,1,0,1708),
    (0.233333,0.929167,0.161079,6,1,0,1005),
    (0.285833,0.568333,0.141800,0,1,0,1623),
    (0.271667,0.738333,0.0454083,1,1,0,1712),
    (0.220833,0.537917,0.361950,2,1,0,1530),
    (0.134783,0.494783,0.188839,3,1,0,1605),
]


DEFAULT_ALIASES: dict[str, str] = {}



def parse_mapping(text: str | None):
    mapping = dict(DEFAULT_ALIASES)
    if not text:
        return mapping
    for item in text.split(","):
        left, right = item.split("=", 1)
        mapping[left.strip()] = right.strip()
    return mapping


def read_csv(program: ProgramSpec, path: Path, mapping: dict[str, str]):
    names = program.inputs + [program.target]
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                [float(row[mapping.get(name, name)]) for name in names]
            )
    return np.asarray(rows, dtype=float)


def compile_config(core, n_train: int):
    # Runtime schedule scales down for small streams; it does not alter
    # source-language semantics.
    if n_train < 80:
        return core.Config(
            block_size=4,
            history_window=8,
            plateau_blocks=3,
            probation_min=4,
            probation_max=8,
            probation_z=1.96,
            max_relations=2,
            mature_age=8,
        )
    return core.Config(
        block_size=20,
        history_window=80,
        plateau_blocks=4,
        probation_min=20,
        probation_max=56,
        probation_z=1.96,
        max_relations=2,
        mature_age=40,
    )


def execute(program: ProgramSpec, core, data: np.ndarray, train_fraction: float):
    if not math.isfinite(train_fraction) or not 0 < train_fraction < 1:
        raise ValueError('train_fraction must be finite and between 0 and 1')
    if not np.all(np.isfinite(data)):
        raise ValueError('spectral data must contain only finite values')
    D = len(program.inputs)
    if data.ndim != 2 or data.shape[1] != D + 1:
        raise ValueError(f"expected {D+1} columns, got shape {data.shape}")

    n = len(data)
    if n < 9:
        raise ValueError('spectral execution requires at least 9 rows (8 train, 1 test)')
    train_n = max(8, min(n - 1, int(round(train_fraction * n))))
    X0, y0 = data[:, :D], data[:, D]

    xm = X0[:train_n].mean(axis=0)
    xs = X0[:train_n].std(axis=0)
    xs = np.where(xs < 1e-12, 1.0, xs)
    ym = float(y0[:train_n].mean())
    ys = float(y0[:train_n].std())
    if ys < 1e-12:
        ys = 1.0

    X = (X0 - xm) / xs
    y = (y0 - ym) / ys

    kernel = core.Kernel(
        D,
        seed=7,
        config=compile_config(core, train_n),
        adapt_geometry=False,
    )

    # on observe(...) { learn demand from error }
    kernel.train(X[:train_n], y[:train_n])

    # on predict(...) { target <- demand(...); emit target }
    prediction = np.asarray(
        [kernel.field.predict(x) for x in X[train_n:]]
    ) * ys + ym
    truth = y0[train_n:]

    rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
    mae = float(np.mean(np.abs(prediction - truth)))

    rels = []
    for i, rel in enumerate(kernel.field.relations):
        eig = np.linalg.eigvalsh(0.5 * (rel.A + rel.A.T))[::-1]
        rels.append({
            "index": i,
            "effective_dimension": float(rel.effective_dim()),
            "active_rank": int(rel.active_rank()) if hasattr(rel, "active_rank") else int(np.sum(eig > 1e-8)),
            "spectral_eigenvalues": [float(v) for v in eig],
            "locality": float(rel.locality()),
            "rho": float(rel.rho()),
            "age": int(rel.age),
            "usage_ema": float(rel.usage_ema),
            "benefit_ema": float(rel.benefit_ema),
        })

    return {
        "program": asdict(program),
        "execution": {
            "rows": int(n),
            "train_rows": int(train_n),
            "test_rows": int(n - train_n),
            "external_dimension": D,
            "rmse": rmse,
            "mae": mae,
            "lifecycle": {
                "proposals": kernel.proposals,
                "admitted": kernel.admitted,
                "rejected": kernel.rejected,
                "pruned": kernel.pruned,
                "geometry_proposals": kernel.geometry_proposals,
                "geometry_admitted": kernel.geometry_admitted,
            },
            "relations": rels,
            "predictions": [
                {"predicted": float(p), "actual": float(t)}
                for p, t in zip(prediction, truth)
            ],
        },
    }


def print_report(report):
    p = report["program"]
    e = report["execution"]
    print("L0 SOURCE COMPILE: PASS")
    print("program:", p["name"])
    print("inputs:", p["inputs"])
    print("target:", p["target"])
    print()
    print(
        "lifecycle P/A/R/pruned = "
        f"{e['lifecycle']['proposals']}/"
        f"{e['lifecycle']['admitted']}/"
        f"{e['lifecycle']['rejected']}/"
        f"{e['lifecycle']['pruned']}"
    )
    print("relations alive =", len(e["relations"]))
    print(f"RMSE = {e['rmse']:.2f}")
    print(f"MAE  = {e['mae']:.2f}")
    print()

    for directive in p["shows"]:
        if directive.endswith(".dimension"):
            print(
                "show", directive, "=>",
                [round(r["effective_dimension"], 4) for r in e["relations"]]
            )
        elif directive.endswith(".geometry"):
            print("show", directive, "=>")
            for r in e["relations"]:
                print(
                    "  eig(A) =",
                    tuple(round(v, 4) for v in r["spectral_eigenvalues"])
                )
                print(
                    "  locality =", round(r["locality"], 4),
                    "rho =", round(r["rho"], 4)
                )
        elif directive.endswith(".relations"):
            print(
                "show", directive, "=>",
                len(e["relations"]), "alive relation(s)"
            )


def main():
    ap = argparse.ArgumentParser(description="L0 Language v0.2 runtime over numerical kernel v0.24")
    ap.add_argument("program", type=Path)
    ap.add_argument(
        "--core",
        type=Path,
        default=Path(__file__).with_name("spectral_scalar.py"),
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--smoke", action="store_true")
    source.add_argument("--csv", type=Path)
    ap.add_argument("--map", dest="mapping")
    ap.add_argument("--train-fraction", type=float, default=0.70)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    ir, program = compile_source(args.program.read_text(encoding="utf-8"))
    core = load_core(args.core)

    if args.smoke:
        data = np.asarray(SMOKE_ROWS, dtype=float)
    else:
        data = read_csv(program, args.csv, parse_mapping(args.mapping))

    report = execute(program, core, data, args.train_fraction)
    report["frontend"] = {"pipeline": "lexer->AST->IR->backend", "source_language": ir.source_language}
    print_report(report)

    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("report:", args.report)


if __name__ == "__main__":
    main()

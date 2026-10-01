from __future__ import annotations

"""L0 v0.7 — explicit unified frontend for ARC and spectral semantics.

There is one public entry point: ``compile_l0_v07``.  The second logical
statement of every source must select exactly one semantics profile:

    program name
    semantics arc

or

    program name
    semantics spectral

The selector is never guessed and there is no parser fallback.  The two
semantics profiles deliberately keep different body productions because they
have different meaning, but they share one source envelope, one compiler
entrypoint, one result type, and one CLI/runtime surface.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Iterable
import math

import numpy as np
import hashlib
import json

from .arc_syntax import (
    CompiledL0ARC,
    L0ARCParseError,
    L0ARCSemanticError,
    compile_l0_arc,
)
from .language import (
    L0IR,
    L0SyntaxError,
    L0SemanticError,
    compile_l0_to_ir,
)
from . import spectral_runtime as spectral_rt


class L0UnifiedError(ValueError):
    pass


class L0SemanticsError(L0UnifiedError):
    pass


MAX_SOURCE_BYTES = 2 * 1024 * 1024
_ALLOWED_SEMANTICS = frozenset({"arc", "spectral"})


@dataclass(frozen=True)
class Header:
    name: str
    semantics: str
    program_line: int
    semantics_line: int


@dataclass(frozen=True)
class CompiledL0V07:
    name: str
    semantics: str
    fingerprint: str
    header: Header
    arc: CompiledL0ARC | None = None
    spectral_ir: L0IR | None = None
    spectral_backend: spectral_rt.ProgramSpec | None = None

    def execute_arc(
        self,
        state: Mapping[str, Any] | None = None,
        *,
        requested: Iterable[str] | None = None,
    ):
        if self.semantics != "arc" or self.arc is None:
            raise L0SemanticsError("execute_arc requires `semantics arc`")
        supplied = dict(state or {})
        for key, value in supplied.items():
            _validate_finite_value(value, f"state[{key!r}]")
        return self.arc.execute(supplied, requested=requested)

    def execute_spectral(
        self,
        data,
        *,
        train_fraction: float = 0.70,
        core_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if self.semantics != "spectral" or self.spectral_backend is None:
            raise L0SemanticsError("execute_spectral requires `semantics spectral`")
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2:
            raise L0UnifiedError(f"spectral data must be a 2D matrix, got ndim={arr.ndim}")
        if arr.shape[0] < 9:
            raise L0UnifiedError("spectral execution requires at least 9 rows (8 train + 1 test)")
        if not np.all(np.isfinite(arr)):
            raise L0UnifiedError("spectral data contains NaN or infinite values")
        if not (0.0 < float(train_fraction) < 1.0):
            raise L0UnifiedError("train_fraction must be strictly between 0 and 1")
        core_p = Path(core_path) if core_path else Path(__file__).with_name(
            "spectral_scalar.py"
        )
        core = spectral_rt.load_core(core_p)
        report = spectral_rt.execute(self.spectral_backend, core, arr, train_fraction)
        report["frontend"] = {
            "pipeline": "L0-v0.7-envelope->semantic-profile->IR/backend",
            "semantics": "spectral",
            "fingerprint": self.fingerprint,
        }
        return report



def _validate_finite_value(value: Any, where: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise L0UnifiedError(f"{where} contains non-finite float {value!r}")
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for i, item in enumerate(value):
            _validate_finite_value(item, f"{where}[{i}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_value(item, f"{where}[{key!r}]")


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Return non-empty source lines after removing # comments outside strings.

    The header grammar itself has no strings, but doing this correctly prevents
    an unrelated quoted # later in a file from confusing duplicate-header
    checks or diagnostics.
    """
    out: list[tuple[int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        buf: list[str] = []
        quote: str | None = None
        escaped = False
        for ch in raw:
            if quote is not None:
                buf.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
                continue
            if ch in {"'", '"'}:
                quote = ch
                buf.append(ch)
                continue
            if ch == "#":
                break
            buf.append(ch)
        cleaned = "".join(buf).strip()
        if cleaned:
            out.append((lineno, cleaned))
    return out


def _valid_ident(s: str) -> bool:
    return bool(s) and (s[0].isalpha() or s[0] == "_") and all(
        c.isalnum() or c == "_" for c in s
    )


def parse_header(text: str) -> Header:
    size = len(text.encode("utf-8"))
    if size > MAX_SOURCE_BYTES:
        raise L0UnifiedError(
            f"source too large: {size} bytes > {MAX_SOURCE_BYTES} byte frontend limit"
        )
    lines = _logical_lines(text)
    if len(lines) < 2:
        raise L0UnifiedError("source requires `program <name>` then `semantics <arc|spectral>`")

    p_ln, p = lines[0]
    s_ln, s = lines[1]
    pparts = p.split()
    sparts = s.split()
    if len(pparts) != 2 or pparts[0] != "program" or not _valid_ident(pparts[1]):
        raise L0UnifiedError(
            f"first logical statement must be `program <identifier>` (line {p_ln})"
        )
    if len(sparts) != 2 or sparts[0] != "semantics":
        raise L0UnifiedError(
            f"second logical statement must be `semantics arc` or `semantics spectral` (line {s_ln})"
        )
    semantics = sparts[1]
    if semantics not in _ALLOWED_SEMANTICS:
        raise L0SemanticsError(
            f"unknown semantics {semantics!r}; expected one of {sorted(_ALLOWED_SEMANTICS)}"
        )
    return Header(pparts[1], semantics, p_ln, s_ln)


def _remove_semantics_line(text: str, line_number: int) -> str:
    lines = text.splitlines(keepends=True)
    if line_number < 1 or line_number > len(lines):
        raise AssertionError("invalid semantics line")
    # Preserve line count so diagnostics from the legacy-tested spectral parser
    # remain aligned with the v0.7 source.
    original = lines[line_number - 1]
    lines[line_number - 1] = "\n" if original.endswith("\n") else ""
    return "".join(lines)


def _spectral_fingerprint(ir: L0IR) -> str:
    canonical = {
        "language": "L0-v0.7",
        "semantics": "spectral",
        "backend_profile": "spectral-v0.24",
        "ir": asdict(ir),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compile_l0_v07(text: str) -> CompiledL0V07:
    header = parse_header(text)
    if header.semantics == "arc":
        # Explicit dispatch only.  Failure here is final; there is no spectral
        # fallback, even if the body happens to resemble spectral syntax.
        try:
            arc = compile_l0_arc(text)
        except (L0ARCParseError, L0ARCSemanticError) as exc:
            raise L0UnifiedError(f"ARC compile failed: {exc}") from exc
        if arc.ast.name != header.name:
            raise AssertionError("header/program mismatch")
        fp_payload = json.dumps(
            {
                "language": "L0-v0.7",
                "semantics": "arc",
                "arc_v06_fingerprint": arc.fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        fp = hashlib.sha256(fp_payload.encode("utf-8")).hexdigest()
        return CompiledL0V07(header.name, "arc", fp, header, arc=arc)

    # Spectral v0.24 parser predates the explicit semantics line.  v0.7 owns
    # the common envelope and passes only the profile body to that already
    # tested parser.  This is adapter reuse, not parser fallback.
    spectral_source = _remove_semantics_line(text, header.semantics_line)
    try:
        ir = compile_l0_to_ir(spectral_source)
        backend = spectral_rt.compile_ir_to_backend(ir)
    except (L0SyntaxError, L0SemanticError, spectral_rt.L0BackendError) as exc:
        raise L0UnifiedError(f"spectral compile failed: {exc}") from exc
    if ir.name != header.name:
        raise AssertionError("header/program mismatch")
    return CompiledL0V07(
        header.name,
        "spectral",
        _spectral_fingerprint(ir),
        header,
        spectral_ir=ir,
        spectral_backend=backend,
    )


def compile_file(path: str | Path) -> CompiledL0V07:
    return compile_l0_v07(Path(path).read_text(encoding="utf-8"))


def semantics_of(text: str) -> str:
    return parse_header(text).semantics

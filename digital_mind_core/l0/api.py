from __future__ import annotations
from pathlib import Path
from typing import Any

from .compiler import compile_l0_v10 as _compile, compile_file as _compile_file
from .semantic import compile_semantic, available_frontends
from .control import lower_semantic_to_control, compile_control, render_control

VERSION = "1.0.2-canonical"
LANGUAGE = "L0-Canonical"


def compile_source(source: str):
    """Compile current native L0. The semantics line explicitly selects arc/spectral/hybrid."""
    return _compile(source)


def compile_file(path: str | Path):
    return _compile_file(path)


def compile_foreign(source: str, language: str, name: str = "module"):
    """Translate Python/JS/C/C++/Rust/native L0 into the shared Semantic IR."""
    return compile_semantic(source, language, name)


def lower_foreign_control(source: str, language: str, name: str = "module"):
    """Translate a supported foreign language into L0 native control semantics."""
    return compile_control(source, language, name)


def migrate_legacy_source(source: str) -> str:
    """Upgrade an old native spectral source that predates the explicit semantics header.

    Migration is intentionally separate from compilation: current execution never guesses a
    semantics profile. If a semantics line already exists, source is returned unchanged.
    """
    logical=[]
    for raw in source.splitlines():
        code=raw.split('#',1)[0].strip()
        if code: logical.append(code)
    if not logical or not logical[0].startswith('program '):
        raise ValueError('legacy source must begin with `program <name>`')
    if len(logical)>1 and logical[1].startswith('semantics '):
        return source
    # v0.1/v0.2 spectral family has a relation arrow + observe/predict events.
    if '-[' in source and 'on observe' in source and 'on predict' in source:
        lines=source.splitlines(keepends=True)
        # insert after physical program declaration line, preserving comments before it if any
        idx=None
        for i,raw in enumerate(lines):
            if raw.split('#',1)[0].strip().startswith('program '): idx=i; break
        if idx is None: raise ValueError('program declaration not found')
        nl='\n' if lines[idx].endswith('\n') else '\n'
        lines.insert(idx+1, 'semantics spectral'+nl)
        return ''.join(lines)
    raise ValueError('legacy syntax is not uniquely migratable; add an explicit semantics line')

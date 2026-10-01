from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Iterable,Mapping
import hashlib,json
from .profiles import _logical_lines,_valid_ident,MAX_SOURCE_BYTES
from .profiles import compile_l0_v07,L0UnifiedError as L0V07Error,CompiledL0V07
from .hybrid import compile_hybrid_v10,CompiledHybridV10,L0HybridV10Error

class L0UnifiedV10Error(ValueError):pass
class L0SemanticsV10Error(L0UnifiedV10Error):pass
_ALLOWED=frozenset({'arc','spectral','hybrid'})
@dataclass(frozen=True)
class HeaderV10:
    name:str;semantics:str;program_line:int;semantics_line:int
@dataclass(frozen=True)
class CompiledL0V10:
    name:str;semantics:str;fingerprint:str;header:HeaderV10;legacy:CompiledL0V07|None=None;hybrid:CompiledHybridV10|None=None
    def execute_arc(self,state:Mapping[str,Any]|None=None,*,requested:Iterable[str]|None=None):
        if self.semantics!='arc' or self.legacy is None:raise L0SemanticsV10Error('execute_arc requires `semantics arc`')
        return self.legacy.execute_arc(state,requested=requested)
    def execute_spectral(self,data,*,train_fraction:float=.70,core_path:str|Path|None=None):
        if self.semantics!='spectral' or self.legacy is None:raise L0SemanticsV10Error('execute_spectral requires `semantics spectral`')
        return self.legacy.execute_spectral(data,train_fraction=train_fraction,core_path=core_path)
    def new_hybrid_runtime(self,*,training_horizon:int=256,core_path:str|Path|None=None,digital_geometry_kwargs:Mapping[str,Any]|None=None):
        if self.semantics!='hybrid' or self.hybrid is None:raise L0SemanticsV10Error('new_hybrid_runtime requires `semantics hybrid`')
        return self.hybrid.new_runtime(training_horizon=training_horizon,core_path=core_path,digital_geometry_kwargs=digital_geometry_kwargs)
def parse_header_v10(text:str)->HeaderV10:
    size=len(text.encode('utf-8'))
    if size>MAX_SOURCE_BYTES:raise L0UnifiedV10Error(f'source too large: {size} bytes > {MAX_SOURCE_BYTES}')
    lines=_logical_lines(text)
    if len(lines)<2:raise L0UnifiedV10Error('source requires `program <name>` then `semantics <arc|spectral|hybrid>`')
    p_ln,p=lines[0];s_ln,s=lines[1];pp=p.split();sp=s.split()
    if len(pp)!=2 or pp[0]!='program' or not _valid_ident(pp[1]):raise L0UnifiedV10Error(f'first logical statement must be `program <identifier>` (line {p_ln})')
    if len(sp)!=2 or sp[0]!='semantics':raise L0UnifiedV10Error(f'second logical statement must select semantics (line {s_ln})')
    if sp[1] not in _ALLOWED:raise L0SemanticsV10Error(f'unknown semantics {sp[1]!r}; expected one of {sorted(_ALLOWED)}')
    return HeaderV10(pp[1],sp[1],p_ln,s_ln)
def compile_l0_v10(text:str)->CompiledL0V10:
    h=parse_header_v10(text)
    if h.semantics in {'arc','spectral'}:
        try:c=compile_l0_v07(text)
        except L0V07Error as exc:raise L0UnifiedV10Error(str(exc)) from exc
        payload=json.dumps({'language':'L0-Canonical','runtime':'1.0.2-canonical','semantics':h.semantics,'v07':c.fingerprint},sort_keys=True,separators=(',',':'))
        return CompiledL0V10(h.name,h.semantics,hashlib.sha256(payload.encode()).hexdigest(),h,legacy=c)
    try:hy=compile_hybrid_v10(text)
    except L0HybridV10Error as exc:raise L0UnifiedV10Error(f'hybrid digital compile failed: {exc}') from exc
    return CompiledL0V10(h.name,'hybrid',hy.fingerprint,h,hybrid=hy)
def compile_file(path:str|Path)->CompiledL0V10:return compile_l0_v10(Path(path).read_text(encoding='utf-8'))

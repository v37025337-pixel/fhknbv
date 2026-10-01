from __future__ import annotations

"""L0 v0.9 — transactional vector learned nodes for ARC.

v0.9 extends v0.8 without weakening its transaction boundary.

A model may learn a scalar target (legacy-compatible) or a fixed ambient vector
``target y : vector[M]``.  A vector model owns one ARC cell whose value is an
atomic tuple of length M.  The tuple arity never changes during execution.
Instead, a spectral target-space activity operator A_out is learned only from
*past committed targets* and exposes a continuous effective output dimension

    d_out = tr(A_out),  0 <= d_out <= M.

This avoids changing the causal graph/type of an ARC cell while a transaction is
running.  Current targets cannot influence the current prediction because
output geometry and all scalar heads are frozen for the complete ARC fixed-point
transaction and are updated only after successful commit.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import copy
import hashlib
import json
import math

import numpy as np

from .arc_syntax import Lexer, compile_l0_arc, CompiledL0ARC, L0ARCParseError, L0ARCSemanticError
from .arc import ARCProgram, ARCRule, ARCResult, Proposal, RuleResult, execute_arc
from . import spectral_runtime as spectral_rt
from .vector_spectral import VectorKernel, VectorConfig


class L0HybridV09Error(ValueError):
    pass
class L0HybridV09ParseError(L0HybridV09Error):
    pass
class L0HybridV09SemanticError(L0HybridV09Error):
    pass


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str = "scalar"       # scalar | vector
    dimension: int = 1


@dataclass(frozen=True)
class ModelSpec:
    name: str
    inputs: tuple[str, ...]
    output: str
    target: str
    properties: tuple[tuple[str, str], ...]
    start_line: int
    end_line: int

    @property
    def property_map(self) -> dict[str, str]:
        return dict(self.properties)


@dataclass(frozen=True)
class HybridASTV09:
    name: str
    targets: tuple[TargetSpec, ...]
    models: tuple[ModelSpec, ...]

    @property
    def target_map(self) -> dict[str, TargetSpec]:
        return {t.name: t for t in self.targets}


_REQUIRED_MODEL_PROPERTIES = {
    "dimension": "adaptive",
    "geometry": "spectral",
    "law": "quadratic",
    "birth": "residual",
    "probation": "future",
    "death": "weak",
    "time": "frozen",
}
_ALLOWED_EXTRA_PROPERTIES = {"output_dimension"}


def _expect(tokens, i: int, value: str) -> int:
    if i >= len(tokens) or tokens[i].value != value:
        t = tokens[min(i, len(tokens)-1)]
        raise L0HybridV09ParseError(f"expected {value!r} at {t.line}:{t.col}, got {t.value!r}")
    return i + 1


def _expect_ident(tokens, i: int) -> tuple[str, int]:
    if i >= len(tokens):
        raise L0HybridV09ParseError("expected identifier at end of source")
    t = tokens[i]
    if t.kind != "IDENT":
        raise L0HybridV09ParseError(f"expected identifier at {t.line}:{t.col}, got {t.value!r}")
    return t.value, i + 1


def _skip_newlines(tokens, i: int) -> int:
    while i < len(tokens) and tokens[i].kind == "NEWLINE":
        i += 1
    return i


def _end_line(tokens, i: int) -> int:
    if i < len(tokens) and tokens[i].kind == "NEWLINE":
        return _skip_newlines(tokens, i)
    t = tokens[min(i, len(tokens)-1)]
    if t.kind == "EOF":
        return i
    raise L0HybridV09ParseError(f"expected end of line at {t.line}:{t.col}, got {t.value!r}")


def scan_hybrid_v09(text: str) -> HybridASTV09:
    try:
        tokens = Lexer(text).tokens()
    except L0ARCParseError as exc:
        raise L0HybridV09ParseError(str(exc)) from exc

    i = _skip_newlines(tokens, 0)
    i = _expect(tokens, i, "program")
    name, i = _expect_ident(tokens, i)
    i = _end_line(tokens, i)
    i = _expect(tokens, i, "semantics")
    if i >= len(tokens) or tokens[i].value != "hybrid":
        t = tokens[min(i, len(tokens)-1)]
        raise L0HybridV09ParseError(f"expected `semantics hybrid` at {t.line}:{t.col}")
    i += 1
    i = _end_line(tokens, i)

    targets: list[TargetSpec] = []
    models: list[ModelSpec] = []
    depth = 0
    while i < len(tokens) and tokens[i].kind != "EOF":
        if tokens[i].kind == "NEWLINE":
            i += 1; continue
        t = tokens[i]
        if t.value == "target" and depth == 0:
            i += 1
            tname, i = _expect_ident(tokens, i)
            kind, dim = "scalar", 1
            if tokens[i].value == ":":
                i += 1
                if tokens[i].value == "scalar":
                    i += 1
                elif tokens[i].value == "vector":
                    i += 1
                    i = _expect(tokens, i, "[")
                    if tokens[i].kind != "NUMBER":
                        tt=tokens[i]; raise L0HybridV09ParseError(f"vector dimension must be integer at {tt.line}:{tt.col}")
                    raw=tokens[i].value; i += 1
                    try:
                        fv=float(raw); dim=int(fv)
                    except Exception as exc:
                        raise L0HybridV09ParseError(f"invalid vector dimension {raw!r}") from exc
                    if fv != dim or dim <= 0:
                        raise L0HybridV09ParseError(f"vector dimension must be a positive integer, got {raw!r}")
                    i = _expect(tokens, i, "]")
                    kind="vector"
                else:
                    tt=tokens[i]; raise L0HybridV09ParseError(f"expected scalar or vector at {tt.line}:{tt.col}")
            targets.append(TargetSpec(tname,kind,dim))
            i = _end_line(tokens, i)
            continue

        if t.value == "model" and depth == 0:
            start_line=t.line; i += 1
            mname,i=_expect_ident(tokens,i); i=_expect(tokens,i,"(")
            ins=[]
            if tokens[i].value != ")":
                x,i=_expect_ident(tokens,i); ins.append(x)
                while tokens[i].value == ",":
                    i+=1; x,i=_expect_ident(tokens,i); ins.append(x)
            i=_expect(tokens,i,")"); i=_expect(tokens,i,"->"); output,i=_expect_ident(tokens,i)
            i=_expect(tokens,i,"{"); i=_end_line(tokens,i)
            props=[]; target_name=None
            while True:
                i=_skip_newlines(tokens,i)
                if tokens[i].value == "}":
                    end_line=tokens[i].line; i+=1; i=_end_line(tokens,i); break
                key=tokens[i].value; i+=1
                if key == "learn":
                    if target_name is not None:
                        raise L0HybridV09ParseError(f"model {mname}: duplicate learn directive")
                    target_name,i=_expect_ident(tokens,i); i=_end_line(tokens,i); continue
                if i >= len(tokens) or tokens[i].kind in {"NEWLINE","EOF"}:
                    raise L0HybridV09ParseError(f"model {mname}: property {key!r} requires a value")
                value=tokens[i].value; i+=1; i=_end_line(tokens,i)
                props.append((key,value))
            if target_name is None:
                raise L0HybridV09ParseError(f"model {mname}: missing `learn <target>`")
            models.append(ModelSpec(mname,tuple(ins),output,target_name,tuple(props),start_line,end_line))
            continue
        if t.value == "{": depth += 1
        elif t.value == "}": depth=max(0,depth-1)
        i += 1
    return HybridASTV09(name,tuple(targets),tuple(models))


def _transform_to_arc(text: str, ast: HybridASTV09) -> str:
    lines=text.splitlines(); semantics_done=False
    for idx,line in enumerate(lines):
        clean=line.split("#",1)[0].strip()
        if clean == "semantics hybrid":
            lines[idx]=line.replace("semantics hybrid","semantics arc",1); semantics_done=True; break
    if not semantics_done: raise L0HybridV09ParseError("missing semantics hybrid line")
    target_names={t.name for t in ast.targets}
    for idx,line in enumerate(lines):
        clean=line.split("#",1)[0].strip()
        if clean.startswith("target "):
            # Scanner already validated this declaration; strip all supervision channels.
            parts=clean.replace(":"," ").split()
            if len(parts)>=2 and parts[1] in target_names: lines[idx]=""
    for m in ast.models:
        if not m.inputs: raise L0HybridV09SemanticError(f"model {m.name}: at least one input is required")
        placeholder=[f"relation {m.name} ({', '.join(m.inputs)}) -> ({m.output}) {{",
                     f" propose {m.output} set {m.inputs[0]}","}"]
        lines[m.start_line-1]="\n".join(placeholder)
        for j in range(m.start_line,min(m.end_line,len(lines))): lines[j]=""
    return "\n".join(lines)+("\n" if text.endswith("\n") else "")


def _seed_for(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],"big") & 0x7fffffff


@dataclass
class OnlineNormalizer:
    dim:int
    n:int=0
    mean:np.ndarray=field(init=False)
    m2:np.ndarray=field(init=False)
    def __post_init__(self):
        self.mean=np.zeros(self.dim,float); self.m2=np.zeros(self.dim,float)
    def scale(self):
        if self.n<2: return np.ones(self.dim,float)
        v=self.m2/max(1,self.n-1); s=np.sqrt(np.maximum(v,0.0)); return np.where(s<1e-9,1.0,s)
    def transform(self,x):
        a=np.asarray(x,float).reshape(self.dim)
        return a.copy() if self.n==0 else (a-self.mean)/self.scale()
    def inverse(self,z):
        a=np.asarray(z,float).reshape(self.dim)
        return a.copy() if self.n==0 else self.mean+self.scale()*a
    def update(self,x):
        a=np.asarray(x,float).reshape(self.dim); self.n+=1
        d=a-self.mean; self.mean += d/self.n; d2=a-self.mean; self.m2 += d*d2


@dataclass
class OutputGeometry:
    """Orthogonally covariant target-space activity from committed history."""
    dim:int
    n:int=0
    mean:np.ndarray=field(init=False)
    m2:np.ndarray=field(init=False)
    def __post_init__(self):
        self.mean=np.zeros(self.dim,float); self.m2=np.zeros((self.dim,self.dim),float)
    def update(self,y):
        a=np.asarray(y,float).reshape(self.dim); self.n+=1
        d=a-self.mean; self.mean += d/self.n; d2=a-self.mean; self.m2 += np.outer(d,d2)
    def covariance(self):
        if self.n<2: return np.zeros((self.dim,self.dim),float)
        C=self.m2/max(1,self.n-1); return .5*(C+C.T)
    def _spectral(self):
        # Before n>D, sample covariance is necessarily rank-deficient; do not
        # infer dimensional collapse from insufficient evidence.
        if self.n <= self.dim:
            return np.ones(self.dim,float), np.eye(self.dim)
        e,V=np.linalg.eigh(self.covariance()); e=np.clip(e,0.0,None)
        mx=float(e.max()) if e.size else 0.0
        a=np.zeros_like(e) if mx < 1e-15 else e/mx
        return a,V
    def activity_eigenvalues(self):
        return self._spectral()[0]
    def effective_dim(self):
        return float(np.sum(self.activity_eigenvalues()))
    def operator(self):
        a,V=self._spectral(); return V@np.diag(a)@V.T
    def project(self,p):
        a=np.asarray(p,float).reshape(self.dim)
        if self.n <= self.dim: return a.copy()
        return self.mean + self.operator()@(a-self.mean)


@dataclass
class IsotropicVectorNormalizer:
    """Mean vector + one global scale; exactly covariant under output rotations."""
    dim:int
    n:int=0
    mean:np.ndarray=field(init=False)
    m2_total:float=0.0
    def __post_init__(self): self.mean=np.zeros(self.dim,float)
    def scale(self):
        if self.n<2:return 1.0
        v=self.m2_total/max(1,(self.n-1)*self.dim)
        return 1.0 if v<1e-18 else math.sqrt(v)
    def transform(self,x):
        a=np.asarray(x,float).reshape(self.dim)
        return a.copy() if self.n==0 else (a-self.mean)/self.scale()
    def inverse(self,z):
        a=np.asarray(z,float).reshape(self.dim)
        return a.copy() if self.n==0 else self.mean+self.scale()*a
    def update(self,x):
        a=np.asarray(x,float).reshape(self.dim); self.n+=1
        d=a-self.mean; self.mean += d/self.n; d2=a-self.mean; self.m2_total += float(np.dot(d,d2))


@dataclass
class ModelRuntimeStateV09:
    spec:ModelSpec
    target:TargetSpec
    kernel:Any
    xnorm:OnlineNormalizer
    ynorm:Any
    output_geometry:OutputGeometry
    version:int=0
    observations:int=0


@dataclass(frozen=True)
class HybridStepResultV09:
    arc:ARCResult
    predictions:Mapping[str,Any]
    model_versions_before:Mapping[str,int]
    model_versions_after:Mapping[str,int]
    learned_models:tuple[str,...]


@dataclass(frozen=True)
class CompiledHybridV09:
    ast:HybridASTV09
    arc:CompiledL0ARC
    fingerprint:str
    def new_runtime(self,*,training_horizon:int=256,core_path:str|Path|None=None):
        return HybridRuntimeV09(self,training_horizon=training_horizon,core_path=core_path)


class HybridRuntimeV09:
    def __init__(self,compiled:CompiledHybridV09,*,training_horizon:int=256,core_path:str|Path|None=None):
        if int(training_horizon)<=0: raise L0HybridV09SemanticError("training_horizon must be positive")
        self.compiled=compiled
        core_p=Path(core_path) if core_path else Path(__file__).with_name("spectral_scalar.py")
        self.core=spectral_rt.load_core(core_p); cfg=spectral_rt.compile_config(self.core,int(training_horizon))
        tmap=compiled.ast.target_map; self.models={}
        for m in compiled.ast.models:
            ts=tmap[m.target]
            if ts.kind=='scalar':
                kernel=self.core.Kernel(len(m.inputs),seed=_seed_for(m.name),config=cfg,adapt_geometry=False)
                ynorm=OnlineNormalizer(1)
            else:
                vcfg=VectorConfig(block_size=cfg.block_size,history_window=cfg.history_window,plateau_blocks=cfg.plateau_blocks,plateau_z=cfg.plateau_z,probation_min=cfg.probation_min,probation_max=cfg.probation_max,probation_z=cfg.probation_z,max_relations=cfg.max_relations,mature_age=cfg.mature_age)
                kernel=VectorKernel(len(m.inputs),ts.dimension,seed=_seed_for(m.name),config=vcfg)
                ynorm=IsotropicVectorNormalizer(ts.dimension)
            self.models[m.name]=ModelRuntimeStateV09(m,ts,kernel,OnlineNormalizer(len(m.inputs)),ynorm,OutputGeometry(ts.dimension))

    @staticmethod
    def _finite_scalar(value:Any,where:str)->float:
        try:v=float(value)
        except Exception as exc:raise L0HybridV09SemanticError(f"{where} must be numeric, got {value!r}") from exc
        if not math.isfinite(v):raise L0HybridV09SemanticError(f"{where} must be finite, got {value!r}")
        return v

    def _target_value(self,ts:TargetSpec,value:Any):
        if ts.kind=='scalar':return self._finite_scalar(value,f"target {ts.name}")
        if isinstance(value,(str,bytes)):raise L0HybridV09SemanticError(f"target {ts.name} must be vector[{ts.dimension}]")
        try:vals=list(value)
        except Exception as exc:raise L0HybridV09SemanticError(f"target {ts.name} must be vector[{ts.dimension}]") from exc
        if len(vals)!=ts.dimension:raise L0HybridV09SemanticError(f"target {ts.name} requires vector[{ts.dimension}], got length {len(vals)}")
        return tuple(self._finite_scalar(v,f"target {ts.name}[{i}]") for i,v in enumerate(vals))

    def _prepare_state(self,supplied):
        base=dict(self.compiled.arc.initial_state); supplied=dict(supplied or {})
        declared={i.name for i in self.compiled.arc.ast.inputs}|{c.name for c in self.compiled.arc.ast.cells}
        unknown=set(supplied)-declared
        if unknown:raise L0HybridV09SemanticError(f"unknown supplied state cells {sorted(unknown)}")
        owned={m.output for m in self.compiled.ast.models};illegal=set(supplied)&owned
        if illegal:raise L0HybridV09SemanticError(f"model-owned output cells cannot be supplied externally: {sorted(illegal)}")
        base.update(supplied);missing=self.compiled.arc.required_inputs-set(base)
        if missing:raise L0HybridV09SemanticError(f"missing required inputs {sorted(missing)}")
        return base

    def _predict_model(self,mr:ModelRuntimeStateV09,snapshot:Mapping[str,Any]):
        spec=mr.spec;raw=np.asarray([self._finite_scalar(snapshot[n],f"model {spec.name} input {n}") for n in spec.inputs],float);z=mr.xnorm.transform(raw)
        if mr.target.kind=='scalar':
            pz=float(mr.kernel.field.predict(z));pred=float(mr.ynorm.inverse([pz])[0]);value=pred
            finite=math.isfinite(pred)
        else:
            pz=np.asarray(mr.kernel.field.predict(z),float);raw_pred=mr.ynorm.inverse(pz);pred=mr.output_geometry.project(raw_pred);value=tuple(float(x) for x in pred);finite=bool(np.all(np.isfinite(pred)))
        if not finite:raise L0HybridV09SemanticError(f"model {spec.name} produced non-finite prediction")
        return raw,value

    def _program_for_transaction(self,captures):
        model_names=set(self.models);rules=[]
        for r in self.compiled.arc.program.rules:
            if r.name not in model_names:rules.append(r);continue
            mr=self.models[r.name];spec=mr.spec
            def evaluate(snapshot,*,_mr=mr,_spec=spec):
                raw,value=self._predict_model(_mr,snapshot);captures[_spec.name]=(raw.copy(),copy.deepcopy(value),_mr.version);return RuleResult((Proposal(_spec.name,0,_spec.output,value,'set'),))
            rules.append(ARCRule(spec.name,frozenset(spec.inputs),frozenset({spec.output}),evaluate,None,r.after,r.demand_tags))
        return ARCProgram(tuple(rules),merges=self.compiled.arc.program.merges,observables=self.compiled.arc.program.observables)

    def step(self,state=None,*,targets=None,requested:Iterable[str]|None=None,learn:bool=True,rule_order:Sequence[str]|None=None):
        base=self._prepare_state(state);raw_targets=dict(targets or {});tmap=self.compiled.ast.target_map;unknown=set(raw_targets)-set(tmap)
        if unknown:raise L0HybridV09SemanticError(f"unknown supervision targets {sorted(unknown)}")
        target_values={k:self._target_value(tmap[k],v) for k,v in raw_targets.items()}
        requested_set=set(requested) if requested is not None else set(self.compiled.arc.ast.observes)
        if self.compiled.arc.ast.effects:requested_set.add('__effects__')
        if learn:
            for m in self.compiled.ast.models:
                if m.target in target_values:requested_set.add(m.output)
        if requested_set&set(tmap):raise L0HybridV09SemanticError('supervision targets are not observable ARC state')
        before={n:m.version for n,m in self.models.items()};captures={};arc_result=execute_arc(self._program_for_transaction(captures),base,requested=(requested_set or None),config=self.compiled.arc.config,rule_order=rule_order);predictions={n:r[1] for n,r in captures.items()}
        learned=[]
        if learn and target_values:
            trial=copy.deepcopy(self.models)
            for name in sorted(trial):
                mr=trial[name];spec=mr.spec
                if spec.target not in target_values:continue
                if name not in captures:raise L0HybridV09SemanticError(f"model {name} has target {spec.target!r} but was not causally evaluated")
                raw,_pred,version_used=captures[name]
                if version_used!=self.models[name].version:raise AssertionError('model version changed during frozen ARC transaction')
                z=mr.xnorm.transform(raw);yv=target_values[spec.target]
                if mr.target.kind=='scalar':
                    ya=np.asarray([yv],float);yz=float(mr.ynorm.transform(ya)[0]);mr.kernel.step(z,yz)
                else:
                    ya=np.asarray(yv,float);yz=mr.ynorm.transform(ya);mr.kernel.step(z,yz)
                # Commit preprocessing/target geometry only after learner step succeeded.
                mr.xnorm.update(raw);mr.ynorm.update(ya)
                if mr.target.kind=='vector':mr.output_geometry.update(ya)
                mr.version+=1;mr.observations+=1;learned.append(name)
            self.models=trial
        after={n:m.version for n,m in self.models.items()};return HybridStepResultV09(arc_result,predictions,before,after,tuple(learned))

    def clone(self):
        other=object.__new__(HybridRuntimeV09);other.compiled=self.compiled;other.core=self.core;other.models=copy.deepcopy(self.models);return other

    def model_report(self):
        out={}
        for name,mr in sorted(self.models.items()):
            k=mr.kernel;row={'version':mr.version,'observations':mr.observations,'target_kind':mr.target.kind,'ambient_output_dimension':mr.target.dimension,'relations':len(k.field.relations),'effective_dimensions':[float(x) for x in k.effective_dims()],'proposals':int(k.proposals),'admitted':int(k.admitted),'rejected':int(k.rejected)}
            if mr.target.kind=='scalar':row['output_effective_dimension']=1.0;row['output_activity_eigenvalues']=[1.0]
            else:row['output_effective_dimension']=float(mr.output_geometry.effective_dim());row['output_activity_eigenvalues']=[float(x) for x in mr.output_geometry.activity_eigenvalues()]
            out[name]=row
        return out


def compile_hybrid_v09(text:str)->CompiledHybridV09:
    ast=scan_hybrid_v09(text)
    if not ast.targets: raise L0HybridV09SemanticError("hybrid program must declare at least one supervision `target`")
    names=[t.name for t in ast.targets]
    if len(names)!=len(set(names)): raise L0HybridV09SemanticError("target names must be unique")
    if not ast.models: raise L0HybridV09SemanticError("hybrid program must declare at least one `model`")
    if len({m.name for m in ast.models})!=len(ast.models): raise L0HybridV09SemanticError("model names must be unique")
    if len({m.output for m in ast.models})!=len(ast.models): raise L0HybridV09SemanticError("each model must own a unique output cell")
    tmap=ast.target_map
    for m in ast.models:
        if len(m.inputs)!=len(set(m.inputs)): raise L0HybridV09SemanticError(f"model {m.name}: duplicate input")
        if m.output in m.inputs: raise L0HybridV09SemanticError(f"model {m.name}: output cannot be its own input")
        if m.target not in tmap: raise L0HybridV09SemanticError(f"model {m.name}: learn target {m.target!r} is not declared")
        props=m.property_map
        if len(props)!=len(m.properties): raise L0HybridV09SemanticError(f"model {m.name}: duplicate model property")
        for k,v in _REQUIRED_MODEL_PROPERTIES.items():
            if props.get(k)!=v: raise L0HybridV09SemanticError(f"model {m.name}: requires `{k} {v}`, got {props.get(k)!r}")
        extra=set(props)-set(_REQUIRED_MODEL_PROPERTIES)-_ALLOWED_EXTRA_PROPERTIES
        if extra: raise L0HybridV09SemanticError(f"model {m.name}: unsupported properties {sorted(extra)}")
        ts=tmap[m.target]; od=props.get("output_dimension")
        if ts.kind=="vector" and od!="adaptive":
            raise L0HybridV09SemanticError(f"model {m.name}: vector target requires `output_dimension adaptive`")
        if ts.kind=="scalar" and od not in {None,"scalar"}:
            raise L0HybridV09SemanticError(f"model {m.name}: scalar target permits only `output_dimension scalar`")

    arc_source=_transform_to_arc(text,ast)
    try: arc=compile_l0_arc(arc_source)
    except (L0ARCParseError,L0ARCSemanticError) as exc: raise L0HybridV09SemanticError(f"hybrid ARC skeleton failed: {exc}") from exc
    state_inputs={i.name for i in arc.ast.inputs}; state_cells={c.name for c in arc.ast.cells}; state_names=state_inputs|state_cells
    tnames=set(tmap); overlap=tnames&state_names
    if overlap: raise L0HybridV09SemanticError(f"supervision targets cannot also be ARC state cells: {sorted(overlap)}")
    for m in ast.models:
        missing=set(m.inputs)-state_names
        if missing: raise L0HybridV09SemanticError(f"model {m.name}: undeclared input cells {sorted(missing)}")
        if m.output not in state_cells: raise L0HybridV09SemanticError(f"model {m.name}: output {m.output!r} must be declared as `cell`")
        cell_decl=next(c for c in arc.ast.cells if c.name==m.output)
        if cell_decl.initial is not None: raise L0HybridV09SemanticError(f"model {m.name}: owned output cell {m.output!r} cannot have an initializer")
        for r in arc.program.rules:
            if r.name!=m.name and m.output in r.writes: raise L0HybridV09SemanticError(f"model {m.name}: output {m.output!r} is also written by rule {r.name!r}")

    canonical={"language":"L0-v0.9","semantics":"hybrid","program":ast.name,"arc_fingerprint":arc.fingerprint,
      "targets":sorted(({"name":t.name,"kind":t.kind,"dimension":t.dimension} for t in ast.targets),key=lambda x:x["name"]),
      "models":sorted(({"name":m.name,"inputs":list(m.inputs),"output":m.output,"target":m.target,"properties":sorted(m.properties)} for m in ast.models),key=lambda x:x["name"]),
      "phase_rule":"frozen-vector-predict->arc-fixed-point->atomic-multihead-learn","vector_rule":"fixed-ambient-arity+adaptive-spectral-effective-dimension"}
    fp=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return CompiledHybridV09(ast,arc,fp)


__all__=["TargetSpec","ModelSpec","HybridASTV09","OutputGeometry","CompiledHybridV09","HybridRuntimeV09","HybridStepResultV09","L0HybridV09Error","L0HybridV09ParseError","L0HybridV09SemanticError","scan_hybrid_v09","compile_hybrid_v09"]

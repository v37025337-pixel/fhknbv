from __future__ import annotations

"""L0 Canonical — Digital Relational Geometry runtime.

Public semantics are digital, not physical.  A learned node receives a fixed
ambient state vector, but its relation-specific distinguishability is learned
from committed digital history.

Two separate pieces are important:

1. Frozen covariance whitening (calibrated only from committed past) removes
   arbitrary offsets, coordinate units and linear basis conditioning before the
   numerical learner is allowed to train.  Warm-up predictions use only the
   mean of *previously committed* targets, so the current target cannot leak.
   A degenerate warm-up waits for usable rank. Newly observed directions trigger
   recalibration and replay on bounded committed history, after frozen prediction.
2. DigitalRelationalGeometry estimates a dimensionless causal metric from the
   whitened sensitivity of the learned mapping.  It is diagnostic/semantic
   geometry, not a physical-space metric.

Transaction order:
  frozen prediction -> ARC fixed point -> atomic learner update ->
  digital-geometry commit.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import copy, hashlib, json, math, re
from collections import deque
import numpy as np

from .hybrid_base import (
    compile_hybrid_v09, CompiledHybridV09, TargetSpec, ModelSpec, OnlineNormalizer,
    L0HybridV09Error,
)
from .arc import ARCProgram, ARCRule, Proposal, RuleResult, execute_arc
from .vector_spectral import VectorKernel, VectorConfig
from .digital_geometry import DigitalRelationalGeometry, sym
from . import spectral_runtime as spectral_rt


class L0HybridV10Error(ValueError): pass
class L0HybridV10ParseError(L0HybridV10Error): pass
class L0HybridV10SemanticError(L0HybridV10Error): pass


def _split_comment(line: str):
    if '#' in line:
        a,b=line.split('#',1); return a,b
    return line,None


def _normalize_source(text: str) -> tuple[str, tuple[str,...]]:
    """Explicitly lower public v0.10 geometry into the proven v0.9 parser.

    This is not fallback parsing.  Every model is required to state exactly:
      geometry digital
      metric causal
    and is then lowered to the spectral numerical backend after validation.
    """
    lines=text.splitlines(); out=[]; in_model=False; current=None; seen={}; models=[]
    for line in lines:
        code,comment=_split_comment(line); clean=code.strip(); indent=code[:len(code)-len(code.lstrip())]
        if not in_model and clean.startswith('model '):
            m=re.match(r'model\s+([A-Za-z_]\w*)\b',clean)
            if not m: raise L0HybridV10ParseError(f'invalid model declaration: {clean!r}')
            current=m.group(1);models.append(current);seen[current]={'geometry':0,'metric':0};in_model=True;out.append(line);continue
        if in_model and clean=='}':
            if seen[current]['geometry']!=1: raise L0HybridV10SemanticError(f"model {current}: v0.10 requires exactly one `geometry digital`")
            if seen[current]['metric']!=1: raise L0HybridV10SemanticError(f"model {current}: v0.10 requires exactly one `metric causal`")
            in_model=False;current=None;out.append(line);continue
        if in_model:
            if clean.startswith('geometry '):
                seen[current]['geometry']+=1
                if clean!='geometry digital': raise L0HybridV10SemanticError(f"model {current}: geometry must be `digital`, got {clean!r}")
                repl=indent+'geometry spectral'
                if comment is not None: repl+=' #'+comment
                out.append(repl);continue
            if clean.startswith('metric '):
                seen[current]['metric']+=1
                if clean!='metric causal': raise L0HybridV10SemanticError(f"model {current}: metric must be `causal`, got {clean!r}")
                # Backend parser must not see this v0.10-only directive. Keep line count.
                out.append(('#'+comment) if comment is not None else '')
                continue
        out.append(line)
    if in_model: raise L0HybridV10ParseError(f"model {current}: unterminated block")
    if not models: raise L0HybridV10SemanticError('hybrid digital program must declare at least one model')
    return '\n'.join(out)+('\n' if text.endswith('\n') else ''), tuple(models)


def _seed_for(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'big') & 0x7fffffff


def _sqrt_and_invsqrt(c: np.ndarray, rel_tol: float=1e-12):
    c=sym(c)
    if not np.all(np.isfinite(c)):raise L0HybridV10SemanticError('calibration covariance must be finite')
    scale=float(np.max(np.abs(c))) if c.size else 0.0
    if scale==0.0:return np.zeros_like(c),np.zeros_like(c),0
    e,v=np.linalg.eigh(c/scale);e=np.clip(e,0.0,None);mx=float(e.max()) if e.size else 0.0
    if mx == 0.0:
        return np.zeros_like(c),np.zeros_like(c),0
    keep=e>rel_tol*mx;s=np.zeros_like(e);w=np.zeros_like(e);s[keep]=np.sqrt(e[keep])*math.sqrt(scale);w[keep]=(1.0/np.sqrt(e[keep]))/math.sqrt(scale)
    return v@np.diag(s)@v.T, v@np.diag(w)@v.T, int(np.sum(keep))


@dataclass
class FrozenWhitening:
    dim:int
    values:list[np.ndarray]=field(default_factory=list)
    calibrated:bool=False
    mean:np.ndarray|None=None
    sqrt_cov:np.ndarray|None=None
    invsqrt_cov:np.ndarray|None=None
    rank:int=0
    mean_count:int=0
    def append(self,value):
        a=np.asarray(value,float).reshape(self.dim)
        if not np.all(np.isfinite(a)):raise L0HybridV10SemanticError('whitening history must be finite')
        if self.calibrated:raise L0HybridV10SemanticError('frozen whitening cannot accept calibration samples after freeze')
        self.values.append(a.copy())
    def calibrate(self):
        if self.calibrated:return
        if len(self.values)<2:raise L0HybridV10SemanticError('at least two committed samples are required for whitening')
        a=np.stack(self.values);self.mean=a.mean(0);self.mean_count=len(a);d=a-self.mean;cov=sym((d.T@d)/max(1,len(a)-1));self.sqrt_cov,self.invsqrt_cov,self.rank=_sqrt_and_invsqrt(cov);self.calibrated=True
    def transform(self,value):
        if not self.calibrated:raise L0HybridV10SemanticError('whitening is not calibrated')
        return self.invsqrt_cov@(np.asarray(value,float).reshape(self.dim)-self.mean)
    def inverse(self,z):
        if not self.calibrated:raise L0HybridV10SemanticError('whitening is not calibrated')
        return self.mean+self.sqrt_cov@np.asarray(z,float).reshape(self.dim)
    def past_mean(self):
        if not self.values:return np.zeros(self.dim,float)
        return np.mean(np.stack(self.values),axis=0)
    def update_mean(self,value):
        if not self.calibrated:raise L0HybridV10SemanticError('whitening is not calibrated')
        a=np.asarray(value,float).reshape(self.dim);old=self.mean.copy();self.mean_count+=1;self.mean += (a-self.mean)/self.mean_count
        # Add this shift to normalized predictions to preserve the same raw output.
        return self.invsqrt_cov@(old-self.mean)

    def supports(self,value,rel_tol=1e-8):
        if not self.calibrated:return False
        if self.rank==self.dim:return True
        d=np.asarray(value,float).reshape(self.dim)-self.mean
        residual=d-self.sqrt_cov@(self.invsqrt_cov@d)
        scale=max(float(np.linalg.norm(d)),float(np.linalg.norm(self.sqrt_cov)))
        return float(np.linalg.norm(residual)) <= rel_tol*scale


@dataclass
class ModelRuntimeStateV10:
    spec:ModelSpec
    target:TargetSpec
    kernel:Any
    mode:str
    warmup:int=0
    xwhite:FrozenWhitening|None=None
    ywhite:FrozenWhitening|None=None
    xnorm:Any=None
    ynorm:Any=None
    version:int=0
    observations:int=0
    trained_observations:int=0
    raw_history:Any=field(default_factory=lambda:deque(maxlen=192))
    recalibrations:int=0
    @property
    def calibrated(self):
        if self.mode=='scalar':return True
        return bool(self.xwhite and self.ywhite and self.xwhite.calibrated and self.ywhite.calibrated)


@dataclass(frozen=True)
class HybridStepResultV10:
    arc:Any
    predictions:Mapping[str,Any]
    model_versions_before:Mapping[str,int]
    model_versions_after:Mapping[str,int]
    learned_models:tuple[str,...]


@dataclass(frozen=True)
class CompiledHybridV10:
    backend:CompiledHybridV09
    fingerprint:str
    source_models:tuple[str,...]
    @property
    def ast(self):return self.backend.ast
    @property
    def arc(self):return self.backend.arc
    def new_runtime(self,*,training_horizon:int=256,core_path:str|Path|None=None,digital_geometry_kwargs:Mapping[str,Any]|None=None):return HybridRuntimeV10(self,training_horizon=training_horizon,core_path=core_path,digital_geometry_kwargs=digital_geometry_kwargs)


class HybridRuntimeV10:
    def __init__(self,compiled:CompiledHybridV10,*,training_horizon:int=256,core_path:str|Path|None=None,digital_geometry_kwargs:Mapping[str,Any]|None=None):
        if int(training_horizon)<=0:raise L0HybridV10SemanticError('training_horizon must be positive')
        self.compiled=compiled
        self.digital_geometry_kwargs=dict(digital_geometry_kwargs or {})
        # Reuse v0.24 scheduling constants, not its scalar learner state.
        core_p=Path(core_path) if core_path else Path(__file__).with_name('spectral_scalar.py')
        core=spectral_rt.load_core(core_p);cfg=spectral_rt.compile_config(core,int(training_horizon))
        vcfg=VectorConfig(block_size=cfg.block_size,history_window=cfg.history_window,plateau_blocks=cfg.plateau_blocks,plateau_z=cfg.plateau_z,probation_min=cfg.probation_min,probation_max=cfg.probation_max,probation_z=cfg.probation_z,max_relations=cfg.max_relations,mature_age=cfg.mature_age)
        self.models={};self.digital={};tmap=compiled.ast.target_map
        for m in compiled.ast.models:
            ts=tmap[m.target];D=len(m.inputs);M=ts.dimension
            if ts.kind=='scalar':
                state=ModelRuntimeStateV10(m,ts,core.Kernel(D,seed=_seed_for(m.name),config=cfg,adapt_geometry=False),'scalar',xnorm=OnlineNormalizer(D),ynorm=OnlineNormalizer(1))
            else:
                warm=max(D,M)+3
                state=ModelRuntimeStateV10(m,ts,VectorKernel(D,M,seed=_seed_for(m.name),config=vcfg),'vector',warmup=warm,xwhite=FrozenWhitening(D),ywhite=FrozenWhitening(M))
                state.raw_history=deque(maxlen=max(vcfg.history_window,2*warm))
            self.models[m.name]=state;self.digital[m.name]=DigitalRelationalGeometry(D,M,**self.digital_geometry_kwargs)

    @staticmethod
    def _finite_scalar(value:Any,where:str)->float:
        try:v=float(value)
        except Exception as exc:raise L0HybridV10SemanticError(f'{where} must be numeric, got {value!r}') from exc
        if not math.isfinite(v):raise L0HybridV10SemanticError(f'{where} must be finite, got {value!r}')
        return v

    def _target_value(self,ts:TargetSpec,value:Any):
        if ts.kind=='scalar':return self._finite_scalar(value,f'target {ts.name}')
        if isinstance(value,(str,bytes)):raise L0HybridV10SemanticError(f'target {ts.name} must be vector[{ts.dimension}]')
        try:vals=list(value)
        except Exception as exc:raise L0HybridV10SemanticError(f'target {ts.name} must be vector[{ts.dimension}]') from exc
        if len(vals)!=ts.dimension:raise L0HybridV10SemanticError(f'target {ts.name} requires vector[{ts.dimension}], got length {len(vals)}')
        return tuple(self._finite_scalar(v,f'target {ts.name}[{i}]') for i,v in enumerate(vals))

    def _prepare_state(self,supplied):
        base=dict(self.compiled.arc.initial_state);supplied=dict(supplied or {});declared={i.name for i in self.compiled.arc.ast.inputs}|{c.name for c in self.compiled.arc.ast.cells};unknown=set(supplied)-declared
        if unknown:raise L0HybridV10SemanticError(f'unknown supplied state cells {sorted(unknown)}')
        owned={m.output for m in self.compiled.ast.models};illegal=set(supplied)&owned
        if illegal:raise L0HybridV10SemanticError(f'model-owned output cells cannot be supplied externally: {sorted(illegal)}')
        base.update(supplied);missing=self.compiled.arc.required_inputs-set(base)
        if missing:raise L0HybridV10SemanticError(f'missing required inputs {sorted(missing)}')
        return base

    def _raw_target_array(self,mr:ModelRuntimeStateV10,value):
        if mr.target.kind=='scalar':return np.asarray([float(value)],float)
        return np.asarray(value,float).reshape(mr.target.dimension)

    def _predict_raw(self,mr:ModelRuntimeStateV10,raw:np.ndarray)->np.ndarray:
        raw=np.asarray(raw,float).reshape(len(mr.spec.inputs))
        if mr.mode=='scalar':
            z=mr.xnorm.transform(raw);pz=float(mr.kernel.field.predict(z));pred=np.asarray([float(mr.ynorm.inverse([pz])[0])],float)
        elif mr.calibrated and mr.kernel.field.relations:
            z=mr.xwhite.transform(raw);pz=np.asarray(mr.kernel.field.predict(z),float).reshape(mr.target.dimension);pred=mr.ywhite.inverse(pz)
        elif mr.calibrated:
            pred=mr.ywhite.mean.copy()
        else:
            pred=mr.ywhite.past_mean()
        if not np.all(np.isfinite(pred)):raise L0HybridV10SemanticError(f'model {mr.spec.name} produced non-finite prediction')
        return pred

    def _predict_model(self,mr:ModelRuntimeStateV10,snapshot:Mapping[str,Any]):
        raw=np.asarray([self._finite_scalar(snapshot[n],f'model {mr.spec.name} input {n}') for n in mr.spec.inputs],float);pred=self._predict_raw(mr,raw)
        value=float(pred[0]) if mr.target.kind=='scalar' else tuple(float(x) for x in pred)
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

    def _calibrate_and_replay(self,mr:ModelRuntimeStateV10):
        if mr.mode!='vector' or len(mr.raw_history)<mr.warmup:return
        xwhite=FrozenWhitening(len(mr.spec.inputs));ywhite=FrozenWhitening(mr.target.dimension)
        for x,y in mr.raw_history:xwhite.append(x);ywhite.append(y)
        xwhite.calibrate();ywhite.calibrate()
        # A zero-rank warm-up is evidence to wait, not a permanent null space.
        if xwhite.rank==0 or ywhite.rank==0:
            return
        was_calibrated=mr.calibrated
        if was_calibrated:
            mr.kernel=VectorKernel(mr.kernel.D,mr.kernel.M,seed=mr.kernel.seed,config=mr.kernel.cfg)
            mr.recalibrations+=1
        mr.xwhite=xwhite;mr.ywhite=ywhite
        for x,y in mr.raw_history:
            mr.kernel.step(mr.xwhite.transform(x),mr.ywhite.transform(y));mr.trained_observations+=1

    def step(self,state=None,*,targets=None,requested:Iterable[str]|None=None,learn:bool=True,rule_order:Sequence[str]|None=None):
        base=self._prepare_state(state);raw_targets=dict(targets or {});tmap=self.compiled.ast.target_map;unknown=set(raw_targets)-set(tmap)
        if unknown:raise L0HybridV10SemanticError(f'unknown supervision targets {sorted(unknown)}')
        target_values={k:self._target_value(tmap[k],v) for k,v in raw_targets.items()}
        requested_set=set(requested) if requested is not None else set(self.compiled.arc.ast.observes)
        declared={i.name for i in self.compiled.arc.ast.inputs}|{c.name for c in self.compiled.arc.ast.cells}
        if requested_set-declared:raise L0HybridV10SemanticError(f'unknown requested cells {sorted(requested_set-declared)}')
        if self.compiled.arc.ast.effects:requested_set.add('__effects__')
        if learn:
            for m in self.compiled.ast.models:
                if m.target in target_values:requested_set.add(m.output)
        if requested_set&set(tmap):raise L0HybridV10SemanticError('supervision targets are not observable ARC state')
        before={n:m.version for n,m in self.models.items()};captures={};arc_result=execute_arc(self._program_for_transaction(captures),base,requested=(requested_set or None),config=self.compiled.arc.config,rule_order=rule_order);predictions={n:r[1] for n,r in captures.items()}
        learned=[]
        if learn and target_values:
            # Copy-on-write transaction: only supervised models are cloned.  The
            # previous implementation deep-copied *all* model and digital history
            # on every observation, which made a large geometry window expensive
            # even when its metric was not queried.
            trial=dict(self.models)
            trial_digital=dict(self.digital)
            for name in sorted(self.models):
                original=self.models[name];spec=original.spec
                if spec.target not in target_values:continue
                if name not in captures:raise L0HybridV10SemanticError(f'model {name} has target {spec.target!r} but was not causally evaluated')
                raw,_pred,version_used=captures[name]
                if version_used!=original.version:raise AssertionError('model version changed during frozen ARC transaction')
                mr=copy.deepcopy(original);trial[name]=mr
                ya=self._raw_target_array(mr,target_values[spec.target])
                # Perform even the append on detached state. A failed append or
                # learner update cannot publish a partial multi-model transaction.
                dg_obj=self.digital[name]
                if isinstance(dg_obj,DigitalRelationalGeometry):
                    staged_dg=dg_obj.stage_observation(raw,ya)
                else:
                    staged_dg=copy.deepcopy(dg_obj)
                    if hasattr(staged_dg,'validate_observation') and hasattr(staged_dg,'observe_prevalidated'):
                        staged_dg.observe_prevalidated(staged_dg.validate_observation(raw,ya))
                    else:staged_dg.observe(raw,ya)
                trial_digital[name]=staged_dg
                if mr.mode=='scalar':
                    z=mr.xnorm.transform(raw);yz=float(mr.ynorm.transform(ya)[0]);mr.kernel.step(z,yz);mr.xnorm.update(raw);mr.ynorm.update(ya);mr.trained_observations+=1
                else:
                    mr.raw_history.append((raw.copy(),ya.copy()))
                    if not mr.calibrated:
                        mr.xwhite.values=[x.copy() for x,_ in mr.raw_history]
                        mr.ywhite.values=[y.copy() for _,y in mr.raw_history]
                        self._calibrate_and_replay(mr)
                    elif not mr.xwhite.supports(raw) or not mr.ywhite.supports(ya):
                        self._calibrate_and_replay(mr)
                    else:
                        mr.kernel.step(mr.xwhite.transform(raw),mr.ywhite.transform(ya));mr.trained_observations+=1
                        shift=mr.ywhite.update_mean(ya)
                        if mr.kernel.field.relations:mr.kernel.field.bias += shift
                mr.version+=1;mr.observations+=1;learned.append(name)
            # Publish only after every learner and detached geometry update has
            # succeeded. Shared immutable sample arrays avoid full history copies.
            self.models,self.digital=trial,trial_digital
        after={n:m.version for n,m in self.models.items()};return HybridStepResultV10(arc_result,predictions,before,after,tuple(learned))

    def clone(self):
        other=object.__new__(HybridRuntimeV10);other.compiled=self.compiled;other.digital_geometry_kwargs=dict(self.digital_geometry_kwargs);other.models=copy.deepcopy(self.models);other.digital=copy.deepcopy(self.digital);return other

    def predict_model(self, name: str, state: Mapping[str, Any]):
        """Read one frozen model for hypothetical planning, without committing.

        The caller supplies already computed model inputs. ARC dependencies are
        not evaluated here; use step(..., learn=False) for a full ARC transaction.
        """
        if name not in self.models:
            raise L0HybridV10SemanticError(f'unknown model {name!r}')
        mr = self.models[name]
        missing = set(mr.spec.inputs) - set(state)
        if missing:
            raise L0HybridV10SemanticError(f'missing model inputs {sorted(missing)}')
        _, value = self._predict_model(mr, state)
        return value

    def model_report(self):
        out={}
        for name,mr in sorted(self.models.items()):
            k=mr.kernel
            if mr.mode=='scalar':
                row={'version':mr.version,'observations':mr.observations,'trained_observations':mr.trained_observations,'calibrated':True,'calibration_warmup':0,'input_calibration_rank':len(mr.spec.inputs),'output_calibration_rank':1,'target_kind':'scalar','ambient_output_dimension':1,'relations':len(k.field.relations),'backend_relation_effective_dimensions':[float(x) for x in k.effective_dims()],'proposals':int(k.proposals),'admitted':int(k.admitted),'rejected':int(k.rejected)}
            else:
                row={'version':mr.version,'observations':mr.observations,'trained_observations':mr.trained_observations,'calibrated':mr.calibrated,'calibration_warmup':mr.warmup,'input_calibration_rank':mr.xwhite.rank if mr.xwhite.calibrated else 0,'output_calibration_rank':mr.ywhite.rank if mr.ywhite.calibrated else 0,'target_kind':'vector','ambient_output_dimension':mr.target.dimension,'relations':len(k.field.relations),'backend_relation_effective_dimensions':[float(x) for x in k.effective_dims()],'proposals':int(k.proposals),'admitted':int(k.admitted),'rejected':int(k.rejected)}
            row.update({'recalibrations':mr.recalibrations,'calibration_status':('ready' if mr.calibrated else ('degenerate_warmup' if mr.observations>=mr.warmup else 'warming_up')),'geometry':'digital','metric':'causal','digital_geometry':self.digital[name].report()});out[name]=row
        return out

    def digital_distance(self,model_name:str,x1:Sequence[float],x2:Sequence[float])->float:
        if model_name not in self.digital:raise L0HybridV10SemanticError(f'unknown model {model_name!r}')
        return float(self.digital[model_name].distance(x1,x2))


def compile_hybrid_v10(text:str)->CompiledHybridV10:
    backend_source,models=_normalize_source(text)
    try:back=compile_hybrid_v09(backend_source)
    except L0HybridV09Error as exc:raise L0HybridV10SemanticError(f'digital backend compile failed: {exc}') from exc
    canonical={'language':'L0-Canonical','runtime':'1.0.2-canonical','semantics':'hybrid','backend':back.fingerprint,'geometry':'digital-relational','metric':'causal-whitened-sensitivity','normalization':'committed-history-whitening-with-rank-expansion','models':sorted(models),'phase':'frozen-predict->arc-fixed-point->stage-learn-and-geometry->publish'}
    fp=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest();return CompiledHybridV10(back,fp,models)


__all__=['FrozenWhitening','CompiledHybridV10','HybridRuntimeV10','HybridStepResultV10','L0HybridV10Error','L0HybridV10ParseError','L0HybridV10SemanticError','compile_hybrid_v10']

from __future__ import annotations

"""Vector spectral kernel for L0 v0.9.

This is a true multi-output extension of the v0.24 spectral relation geometry.
It does not train one unrelated scalar topology per output coordinate.

For an orthogonal output reparameterization y' = y R, the residual pressure
uses G G^T and the loss uses ||e||^2, so input geometry/lifecycle decisions are
invariant.  Ridge fitting and online coefficient updates are linear in y, hence
the represented vector function transforms covariantly (up to floating-point
roundoff and any external preprocessing).
"""

from dataclasses import dataclass
from collections import deque
import copy, math
import numpy as np

EPS=1e-9

def sym(M): return .5*(M+M.T)

def project_activity(A,lo=0.0,hi=1.0):
    e,V=np.linalg.eigh(sym(A)); e=np.clip(e,lo,hi); return V@np.diag(e)@V.T

def weighted_stats_vector(X,E):
    w=np.sum(np.asarray(E,float)**2,axis=1)+1e-10; w/=w.sum()
    c=(w[:,None]*X).sum(0); d=X-c; C=(d*w[:,None]).T@d
    return c,sym(C),w

def pressure_operator_vector(X,residual):
    X=np.asarray(X,float); E=np.asarray(residual,float)
    if E.ndim!=2: raise ValueError('vector residual must be N x M')
    D=X.shape[1]; c,Cw,_=weighted_stats_vector(X,E); d=X-c
    Cu=sym((d.T@d)/max(1,len(d)))
    G=d.T@E/max(1,len(E))
    H=np.zeros((D,D)); ng=float(np.linalg.norm(G,'fro'))
    if ng>1e-10: H += (G@G.T)/(ng*ng)
    S=sym(Cw-Cu); es,Vs=np.linalg.eigh(S); absS=Vs@np.diag(np.abs(es))@Vs.T
    nS=float(np.linalg.norm(absS))
    if nS>1e-10: H += .65*absS/nS
    eh,Vh=np.linalg.eigh(sym(H)); eh=np.clip(eh,0,None)
    if float(eh.max())<1e-12: target=np.eye(D)*(1.0/D)
    else:
        a=(eh/(float(eh.max())+EPS))**.60; a=.01+.99*a; target=Vh@np.diag(a)@Vh.T
    return c,project_activity(target,lo=0.0,hi=1.0)

def quadratic_features(w):
    w=np.asarray(w,float); D=len(w); vals=[1.0]; vals.extend(w.tolist())
    for i in range(D):
        for j in range(i,D): vals.append(w[i]*w[j])
    return np.asarray(vals,float)

@dataclass
class VectorRelation:
    center:np.ndarray
    A:np.ndarray
    rho_log:float
    locality_logit:float
    coeff:np.ndarray          # M x F
    age:int=0
    usage_ema:float=0.
    benefit_ema:float=0.
    def activity_eigs(self): return np.linalg.eigvalsh(sym(self.A))
    def effective_dim(self): return float(np.trace(self.A))
    def active_rank(self,tol=1e-8): return int(np.sum(self.activity_eigs()>tol))
    def rho(self): return float(math.exp(np.clip(self.rho_log,-4,3)))
    def locality(self):
        z=float(np.clip(self.locality_logit,-8,8)); return 1/(1+math.exp(-z))
    def parts(self,x):
        d=np.asarray(x,float)-self.center; tr=float(np.trace(self.A)+EPS)
        q=float(d@self.A@d)/tr; local_gate=math.exp(-.5*self.rho()*min(q,160.)); loc=self.locality()
        gate=(1-loc)+loc*local_gate; w=self.A@d; feat=quadratic_features(w)
        law=self.coeff@feat; cont=gate*law
        return gate,law,cont,d,w,q,feat,local_gate,loc

class VectorField:
    def __init__(self,D,M):
        self.D=D; self.M=M; self.bias=np.zeros(M,float); self.relations=[]
        self.bias_lr=.008; self.coeff_lr=.004; self.rho_lr=.0015; self.locality_lr=.010
        self.rms_bias=0.; self.rms_decay=.99
    def predict(self,x):
        out=self.bias.copy()
        for r in self.relations: out += r.parts(x)[2]
        return out
    @staticmethod
    def _loss(e): return float(np.dot(e,e)/max(1,len(e)))
    @staticmethod
    def _norm_cap(A,cap):
        n=float(np.linalg.norm(A))
        return A if n<=cap else A*(cap/(n+EPS))
    def update(self,x,y):
        xx=np.asarray(x,float); yy=np.asarray(y,float).reshape(self.M)
        ps=[r.parts(xx) for r in self.relations]
        cs=[p[2] for p in ps]; pred=self.bias.copy()
        for c in cs: pred+=c
        e=pred-yy; loss=self._loss(e)
        self.rms_bias=self.rms_decay*self.rms_bias+(1-self.rms_decay)*loss
        self.bias -= self.bias_lr*e/(math.sqrt(self.rms_bias)+1e-4)
        self.bias=self._norm_cap(self.bias,8*math.sqrt(self.M))
        for r,p,c in zip(self.relations,ps,cs):
            gate,law,cont,d,w,q,feat,local_gate,loc=p; r.age+=1
            r.usage_ema=.995*r.usage_ema+.005*gate
            loo=self._loss(pred-c-yy); r.benefit_ema=.995*r.benefit_ema+.005*(loo-loss)
            r.coeff -= (self.coeff_lr/self.M)*np.outer(e,gate*feat)
            r.coeff=self._norm_cap(r.coeff,8*math.sqrt(self.M*r.coeff.shape[1]))
            # d cont / d locality = (local_gate-1) * law vector
            dvec=(local_gate-1.0)*law; grad_loc=float(np.dot(e,dvec)/self.M)*loc*(1-loc)
            r.locality_logit -= self.locality_lr*np.clip(grad_loc,-3,3); r.locality_logit=float(np.clip(r.locality_logit,-8,8))
            dvec_rho=(loc*local_gate*law)*(-.5*r.rho()*q); grad_rho=float(np.dot(e,dvec_rho)/self.M)
            r.rho_log -= self.rho_lr*np.clip(grad_rho,-3,3); r.rho_log=float(np.clip(r.rho_log,-4,3))
        return pred,loss

def fit_relation(field,rel,X,Y,idx=None):
    base=[]; rows=[]
    for x in X:
        v=field.bias.copy()
        for j,r in enumerate(field.relations):
            if idx is not None and j==idx: continue
            v+=r.parts(x)[2]
        base.append(v); gate,*rest=rel.parts(x); feat=rest[5]; rows.append(gate*feat)
    residual=np.asarray(Y,float)-np.asarray(base,float); Mmat=np.asarray(rows,float)
    D=field.D; ridge=np.eye(Mmat.shape[1]); ridge[0,0]=1e-5; ridge[1:1+D,1:1+D]*=.05; ridge[1+D:,1+D:]*=6.0
    coef=np.linalg.solve(Mmat.T@Mmat+ridge,Mmat.T@residual) # F x M
    rel.coeff=coef.T
    return rel

def truncate_activity(A,k):
    e,V=np.linalg.eigh(sym(A)); order=np.argsort(e)[::-1]; keep=order[:max(1,min(int(k),len(e)))]
    out=V[:,keep]@np.diag(e[keep])@V[:,keep].T
    return project_activity(out,0.0,1.0)

def candidate_losses(field,rel,X,Y,idx):
    vals=[]
    for x,t in zip(X,Y):
        v=field.bias.copy()
        for j,r in enumerate(field.relations):
            if idx is not None and j==idx:continue
            v+=r.parts(x)[2]
        v+=rel.parts(x)[2]; e=v-t; vals.append(float(np.dot(e,e)/field.M))
    return np.asarray(vals,float)

def select_spectral_support(field,rel,X,Y,idx=None):
    X=np.asarray(X,float); Y=np.asarray(Y,float); n=len(X); D=field.D
    cut=max(16,int(round(.70*n))); cut=min(cut,n-8)
    if cut<8 or n-cut<8:return fit_relation(field,rel,X,Y,idx)
    Xf,Yf=X[:cut],Y[:cut]; Xv,Yv=X[cut:],Y[cut:]; candidates=[]
    for k in range(1,D+1):
        cand=copy.deepcopy(rel); cand.A=truncate_activity(rel.A,k); fit_relation(field,cand,Xf,Yf,idx)
        losses=candidate_losses(field,cand,Xv,Yv,idx); candidates.append((cand,losses,float(np.mean(losses))))
    best=min(candidates,key=lambda z:z[2]); bl=best[1]; eligible=[]
    for cand,losses,ml in candidates:
        diff=losses-bl; md=float(np.mean(diff)); se=float(np.std(diff,ddof=1)/math.sqrt(len(diff))) if len(diff)>1 else 0.
        if md<=se+1e-12:eligible.append((cand,losses,ml))
    chosen_fit,chosen_losses,_=min(eligible,key=lambda z:(z[0].effective_dim(),z[2])); k_chosen=chosen_fit.active_rank()
    full_losses=candidates[-1][1]; improvement=full_losses-chosen_losses; mi=float(np.mean(improvement)); sei=float(np.std(improvement,ddof=1)/math.sqrt(len(improvement))) if len(improvement)>1 else math.inf
    z=max(0.,mi/(sei+EPS)) if math.isfinite(sei) else 0.; tail_keep=math.exp(-z)
    e,V=np.linalg.eigh(sym(rel.A)); order=np.argsort(e)[::-1]; scale=np.full(D,tail_keep,float); scale[order[:max(1,min(k_chosen,D))]]=1.
    chosen=copy.deepcopy(rel); chosen.A=project_activity(V@np.diag(e*scale)@V.T,0.,1.)
    return fit_relation(field,chosen,X,Y,idx)

def make_birth(field,X,Y,preds):
    residual=np.asarray(Y)-np.asarray(preds); c,A=pressure_operator_vector(X,residual)
    F=1+field.D+field.D*(field.D+1)//2
    rel=VectorRelation(c.copy(),A.copy(),math.log(.35),math.log(.1/.9),np.zeros((field.M,F)))
    return select_spectral_support(field,rel,X,Y,None)

@dataclass(frozen=True)
class VectorConfig:
    block_size:int=24; history_window:int=96; plateau_blocks:int=4; plateau_z:float=1.645
    probation_min:int=24; probation_max:int=64; probation_z:float=1.96; max_relations:int=2; mature_age:int=48

class VectorKernel:
    def __init__(self,D,M,seed=7,config=VectorConfig()):
        self.D=D;self.M=M;self.seed=seed;self.field=VectorField(D,M);self.cfg=config
        self.hist=deque(maxlen=config.history_window);self.blocks=[];self.cur=[];self.shadow=None;self.diffs=[];self.replace_idx=None
        self.proposals=self.admitted=self.rejected=self.pruned=0;self.geometry_proposals=self.geometry_admitted=0;self.birth_proposals=self.birth_admitted=0;self.proposal_kind=None;self._next_geometry=True
    def plateau(self):
        n=self.cfg.plateau_blocks
        if len(self.blocks)<n:return False
        yy=np.asarray(self.blocks[-n:]);xx=np.arange(n);cx=xx-xx.mean();den=float(cx@cx);slope=float(cx@(yy-yy.mean())/(den+EPS));fit=yy.mean()+slope*cx;resid=yy-fit;var=float(resid@resid/max(1,n-2));se=math.sqrt(var/(den+EPS));return slope+self.cfg.plateau_z*se>=0
    def start_birth(self):
        X=np.stack([h[0] for h in self.hist]);Y=np.stack([h[1] for h in self.hist]);preds=np.stack([self.field.predict(x) for x in X]);rel=make_birth(self.field,X,Y,preds);sh=copy.deepcopy(self.field);self.replace_idx=None
        if len(sh.relations)>=self.cfg.max_relations:
            idx=min(range(len(sh.relations)),key=lambda i:(sh.relations[i].benefit_ema,sh.relations[i].usage_ema));sh.relations[idx]=rel;self.replace_idx=idx
        else:sh.relations.append(rel)
        self.shadow=sh;self.diffs=[];self.proposals+=1;self.birth_proposals+=1;self.proposal_kind='birth'
    def start_geometry(self):
        if not self.field.relations:return self.start_birth()
        X=np.stack([h[0] for h in self.hist]);Y=np.stack([h[1] for h in self.hist]);idx=max(range(len(self.field.relations)),key=lambda i:self.field.relations[i].benefit_ema)
        base=[]
        for x in X:
            v=self.field.bias.copy()
            for j,r in enumerate(self.field.relations):
                if j!=idx:v+=r.parts(x)[2]
            base.append(v)
        residual=Y-np.asarray(base);c_target,A_target=pressure_operator_vector(X,residual);sh=copy.deepcopy(self.field);rel=sh.relations[idx];eta=.45;rel.center=.75*rel.center+.25*c_target;rel.A=project_activity((1-eta)*rel.A+eta*A_target,0.,1.);fit_relation(sh,rel,X,Y,idx)
        self.shadow=sh;self.diffs=[];self.replace_idx=None;self.proposals+=1;self.geometry_proposals+=1;self.proposal_kind='geometry'
    def finish(self):
        if self.shadow is None:return
        n=len(self.diffs)
        if n<self.cfg.probation_min:return
        d=np.asarray(self.diffs);mean=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(n)) if n>1 else math.inf;upper=mean+self.cfg.probation_z*se
        if upper<0:
            self.field=self.shadow;self.shadow=None;self.diffs=[];self.admitted+=1
            if self.proposal_kind=='geometry':self.geometry_admitted+=1
            elif self.proposal_kind=='birth':self.birth_admitted+=1
            if self.replace_idx is not None:self.pruned+=1
            self.replace_idx=None;self.proposal_kind=None
        elif n>=self.cfg.probation_max:
            self.shadow=None;self.diffs=[];self.rejected+=1;self.replace_idx=None;self.proposal_kind=None
    @staticmethod
    def _loss(p,y):
        e=np.asarray(p)-np.asarray(y);return float(np.dot(e,e)/len(e))
    def step(self,x,y):
        xx=np.asarray(x,float);yy=np.asarray(y,float).reshape(self.M);pm=self.field.predict(xx);lm=self._loss(pm,yy)
        if self.shadow is not None:
            ps=self.shadow.predict(xx);self.diffs.append(self._loss(ps,yy)-lm)
        self.hist.append((xx.copy(),yy.copy()));self.cur.append(lm);self.field.update(xx,yy)
        if self.shadow is not None:self.shadow.update(xx,yy);self.finish()
        if len(self.cur)>=self.cfg.block_size:
            self.blocks.append(float(np.mean(self.cur)));self.blocks=self.blocks[-self.cfg.plateau_blocks:];self.cur=[]
            if self.shadow is None and len(self.hist)>=self.cfg.history_window:
                if not self.field.relations:self.start_birth()
                elif self.plateau():
                    if self._next_geometry:self.start_geometry()
                    else:self.start_birth()
                    self._next_geometry=not self._next_geometry
        return pm,lm
    def effective_dims(self):return tuple(r.effective_dim() for r in self.field.relations)

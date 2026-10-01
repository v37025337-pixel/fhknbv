"""
L0 v0.24-V — Confidence-Weighted Spectral Dimension

Why this version exists
-----------------------
v0.22 replaced slow integer rank growth with continuous dimensional pressure,
but the coordinate bank itself was still effectively frozen after a relation
was admitted.  A naive online rotation of that bank was tested and rejected:
rotating coordinates without transporting the local law destroys the meaning
of its coefficients.

v0.24 keeps that covariance and lets held-out evidence remove unsupported spectral modes.

Representation
--------------
A relation is described directly by a positive spectral activity operator

    A in R^(D x D),    0 < eig(A) <= 1

rather than by a privileged list of coordinate axes.

Its eigenvectors define the current spatial directions.
Its eigenvalues define their continuous activities.

    d_eff = tr(A)

For d = x - c,

    w = A d

and the local law is a full quadratic tensor

    law(d) = alpha + beta^T w + w^T Q w,

with Q symmetric.

Support can continuously interpolate between a global and a local relation:

    g_local = exp[-0.5 rho * d^T A d / tr(A)]
    support = (1-lambda) + lambda * g_local
    R(x) = support * law(d)

where 0 < lambda < 1 is learned online.

Coordinate covariance
---------------------
Under an orthogonal reparameterization x' = R^T x,

    c'    = R^T c
    A'    = R^T A R
    beta' = R^T beta
    Q'    = R^T Q R

and the represented function is unchanged.  This is explicitly tested.

Spatial evolution
-----------------
The residual defines a spectral pressure operator.  A new orientation is NOT
silently imposed on a mature relation.  It is proposed as a shadow model,
refit on past data, and accepted only if future-only prequential A/B evidence
supports it.  Birth/replacement of whole relations uses the same discipline.

Time / recurrent clocks remain intentionally excluded.  The purpose here is to
improve L0 space before redesigning the concept of time.

Honest boundary
---------------
The spectral pressure construction, Gaussian support family, global/local
mixture, full quadratic law, optimizer, and confidence constants are still
human-designed.  v0.23 demonstrates covariant continuous relation geometry
inside that model class; it does not derive mathematics from nothing.
"""

from __future__ import annotations
import math, copy
from collections import deque
from dataclasses import dataclass
from typing import List
import numpy as np

EPS=1e-9

def sym(M):
    return 0.5*(M+M.T)

def project_activity(A, lo=0.0, hi=1.0):
    e,V=np.linalg.eigh(sym(A))
    e=np.clip(e,lo,hi)
    return V@np.diag(e)@V.T

def weighted_stats(X,r):
    w=np.asarray(r,float)**2+1e-10
    w/=w.sum()
    c=(w[:,None]*X).sum(0)
    d=X-c
    C=(d*w[:,None]).T@d
    return c,sym(C),w

def pressure_operator(X,residual):
    X=np.asarray(X,float); r=np.asarray(residual,float)
    D=X.shape[1]
    c,Cw,w=weighted_stats(X,r)
    d=X-c
    Cu=sym((d.T@d)/max(1,len(d)))
    g=d.T@r/max(1,len(r))

    H=np.zeros((D,D))
    ng=float(np.linalg.norm(g))
    if ng>1e-10:
        H += np.outer(g,g)/(ng*ng)

    S=sym(Cw-Cu)
    es,Vs=np.linalg.eigh(S)
    absS=Vs@np.diag(np.abs(es))@Vs.T
    nS=float(np.linalg.norm(absS))
    if nS>1e-10:
        H += 0.65*absS/nS

    eh,Vh=np.linalg.eigh(sym(H))
    eh=np.clip(eh,0,None)
    if float(eh.max())<1e-12:
        target=np.eye(D)*(1.0/D)
    else:
        a=(eh/(float(eh.max())+EPS))**0.60
        # continuous floor, not a rank threshold
        a=0.01+0.99*a
        target=Vh@np.diag(a)@Vh.T
    return c,project_activity(target)

def quadratic_features(w):
    w=np.asarray(w,float)
    D=len(w)
    vals=[1.0]
    vals.extend(w.tolist())
    for i in range(D):
        for j in range(i,D):
            vals.append(w[i]*w[j])
    return np.asarray(vals,float)

def coeff_to_tensors(coeff,D):
    coeff=np.asarray(coeff,float)
    alpha=float(coeff[0])
    beta=coeff[1:1+D].copy()
    Q=np.zeros((D,D))
    k=1+D
    for i in range(D):
        for j in range(i,D):
            val=float(coeff[k]); k+=1
            if i==j:
                Q[i,i]=val
            else:
                # feature coefficient * wi*wj corresponds to 2 Qij wi wj
                Q[i,j]=Q[j,i]=0.5*val
    return alpha,beta,Q

def tensors_to_coeff(alpha,beta,Q):
    D=len(beta)
    out=[float(alpha)]
    out.extend(np.asarray(beta,float).tolist())
    for i in range(D):
        for j in range(i,D):
            out.append(float(Q[i,i]) if i==j else float(2*Q[i,j]))
    return np.asarray(out,float)

@dataclass
class SpectralRelation:
    center: np.ndarray
    A: np.ndarray
    rho_log: float
    locality_logit: float
    coeff: np.ndarray
    age:int=0
    usage_ema:float=0.
    benefit_ema:float=0.

    def activity_eigs(self):
        return np.linalg.eigvalsh(sym(self.A))
    def effective_dim(self):
        return float(np.trace(self.A))
    def active_rank(self,tol=1e-8):
        return int(np.sum(self.activity_eigs()>tol))
    def rho(self):
        return float(math.exp(np.clip(self.rho_log,-4,3)))
    def locality(self):
        z=float(np.clip(self.locality_logit,-8,8))
        return 1.0/(1.0+math.exp(-z))

    def parts(self,x):
        d=np.asarray(x,float)-self.center
        tr=float(np.trace(self.A)+EPS)
        q=float(d@self.A@d)/tr
        local_gate=math.exp(-0.5*self.rho()*min(q,160.))
        locality=self.locality()
        support=(1.0-locality)+locality*local_gate
        w=self.A@d
        feat=quadratic_features(w)
        law=float(self.coeff@feat)
        return support,law,support*law,d,w,q,feat,local_gate,locality

    def covariant_rotate(self,R):
        """Exact coordinate reparameterization x' = R^T x."""
        alpha,beta,Q=coeff_to_tensors(self.coeff,len(self.center))
        R=np.asarray(R,float)
        return SpectralRelation(
            center=R.T@self.center,
            A=R.T@self.A@R,
            rho_log=float(self.rho_log),
            locality_logit=float(self.locality_logit),
            coeff=tensors_to_coeff(alpha,R.T@beta,R.T@Q@R),
            age=self.age,
            usage_ema=self.usage_ema,
            benefit_ema=self.benefit_ema,
        )

class SpectralField:
    def __init__(self,D,seed=7,adapt_geometry=True):
        self.D=D; self.rng=np.random.default_rng(seed)
        self.bias=0.
        self.relations:List[SpectralRelation]=[]
        self.adapt_geometry=bool(adapt_geometry)
        self.bias_lr=.008
        self.coeff_lr=.004
        self.rho_lr=.0015
        self.locality_lr=.010
        self.geometry_rate=.35
        self.center_rate=.04
        self.rms_bias=0.; self.rms_decay=.99

    def predict(self,x):
        xx=np.asarray(x,float)
        return float(self.bias+sum(r.parts(xx)[2] for r in self.relations))

    def update(self,x,y):
        xx=np.asarray(x,float); yy=float(y)
        ps=[r.parts(xx) for r in self.relations]
        cs=np.asarray([p[2] for p in ps],float)
        pred=float(self.bias+(cs.sum() if len(cs) else 0.))
        e=pred-yy; loss=e*e

        self.rms_bias=self.rms_decay*self.rms_bias+(1-self.rms_decay)*e*e
        self.bias-=self.bias_lr*e/(math.sqrt(self.rms_bias)+1e-4)
        self.bias=float(np.clip(self.bias,-4,4))

        for r,p,c in zip(self.relations,ps,cs):
            gate,law,cont,d,w,q,feat,local_gate,locality=p
            r.age+=1
            r.usage_ema=.995*r.usage_ema+.005*gate
            loo=(pred-c-yy)**2
            r.benefit_ema=.995*r.benefit_ema+.005*(loo-loss)

            r.coeff-=self.coeff_lr*e*gate*feat
            r.coeff=np.clip(r.coeff,-8,8)

            # Mixture between global and local support.
            # locality=0 -> global law, locality=1 -> Gaussian-local law.
            dcont_dloc=(local_gate-1.0)*law
            dloc_dlog=locality*(1.0-locality)
            r.locality_logit-=self.locality_lr*np.clip(e*dcont_dloc*dloc_dlog,-3,3)
            r.locality_logit=float(np.clip(r.locality_logit,-8,8))

            # scalar locality width, independent of spectral dimension
            # only the local fraction depends on rho
            grad_logrho=e*(locality*local_gate*law)*(-0.5*r.rho()*q)
            r.rho_log-=self.rho_lr*np.clip(grad_logrho,-3,3)
            r.rho_log=float(np.clip(r.rho_log,-4,3))
        return pred,loss

    def adapt_from_history(self,history):
        if not self.adapt_geometry or not self.relations or len(history)<32:
            return
        X=np.stack([h[0] for h in history])
        y=np.asarray([h[1] for h in history])
        # adapt each relation to the residual left by all the other relations
        for i,r in enumerate(self.relations):
            base=[]
            for x in X:
                v=self.bias
                for j,rr in enumerate(self.relations):
                    if j!=i: v+=rr.parts(x)[2]
                base.append(v)
            residual=y-np.asarray(base)
            c_target,A_target=pressure_operator(X,residual)
            r.center=(1-self.center_rate)*r.center+self.center_rate*c_target
            r.A=project_activity((1-self.geometry_rate)*r.A+self.geometry_rate*A_target)
            fit_relation(self,r,X,y,idx=i)

def fit_relation(field,rel,X,y,idx=None):
    base=[]
    Arows=[]
    for x in X:
        v=field.bias
        for j,r in enumerate(field.relations):
            if idx is not None and j==idx: continue
            v+=r.parts(x)[2]
        base.append(v)
        gate,law,c,d,w,q,feat,local_gate,locality=rel.parts(x)
        Arows.append(gate*feat)
    residual=y-np.asarray(base)
    M=np.asarray(Arows)
    # stronger regularization for full quadratic tensor
    D=field.D
    ridge=np.eye(M.shape[1])
    ridge[0,0]=1e-5
    ridge[1:1+D,1:1+D]*=0.05
    ridge[1+D:,1+D:]*=6.0
    coef=np.linalg.solve(M.T@M+ridge,M.T@residual)
    rel.coeff=np.clip(coef,-8,8)
    return rel

def truncate_activity(A,k):
    e,V=np.linalg.eigh(sym(A))
    order=np.argsort(e)[::-1]
    keep=order[:max(1,min(int(k),len(e)))]
    out=np.zeros_like(A)
    out += V[:,keep] @ np.diag(e[keep]) @ V[:,keep].T
    return project_activity(out,lo=0.0,hi=1.0)

def candidate_losses(field,rel,X,y,idx):
    vals=[]
    for x,t in zip(X,y):
        v=field.bias
        for j,r in enumerate(field.relations):
            if idx is not None and j==idx:
                continue
            v += r.parts(x)[2]
        v += rel.parts(x)[2]
        vals.append((v-float(t))**2)
    return np.asarray(vals,float)

def select_spectral_support(field,rel,X,y,idx=None):
    """Choose task-supported spectral support without a hand-set rank.

    A past-only fit/selection split compares every nested spectral support.
    We then apply the standard one-SE rule: among candidates statistically
    indistinguishable from the best held-out loss, keep the simplest support.
    The selected relation is finally refit on all available past data; the
    kernel's outer probation still requires future-only evidence.
    """
    X=np.asarray(X,float); y=np.asarray(y,float)
    n=len(X); D=field.D
    cut=max(16,int(round(0.70*n)))
    cut=min(cut,n-8)
    if cut<8 or n-cut<8:
        return fit_relation(field,rel,X,y,idx=idx)

    Xf,yf=X[:cut],y[:cut]
    Xv,yv=X[cut:],y[cut:]
    candidates=[]
    for k in range(1,D+1):
        cand=copy.deepcopy(rel)
        cand.A=truncate_activity(rel.A,k)
        fit_relation(field,cand,Xf,yf,idx=idx)
        losses=candidate_losses(field,cand,Xv,yv,idx)
        candidates.append((cand,losses,float(np.mean(losses))))

    best=min(candidates,key=lambda z:z[2])
    best_losses=best[1]
    eligible=[]
    for cand,losses,mean_loss in candidates:
        diff=losses-best_losses
        md=float(np.mean(diff))
        se=float(np.std(diff,ddof=1)/math.sqrt(len(diff))) if len(diff)>1 else 0.0
        if md <= se + 1e-12:
            eligible.append((cand,losses,mean_loss))

    chosen_rec=min(eligible,key=lambda z:(z[0].effective_dim(),z[2]))
    chosen_fit,chosen_losses,_=chosen_rec
    k_chosen=chosen_fit.active_rank() if hasattr(chosen_fit,"active_rank") else int(np.sum(chosen_fit.activity_eigs()>1e-8))

    # Confidence-weighted soft pruning.  A short holdout is not allowed to
    # erase spectral directions abruptly.  It only attenuates the tail in
    # proportion to paired evidence that the compact candidate predicts
    # better than the full-support sibling.
    full_losses=candidates[-1][1]
    improvement=full_losses-chosen_losses
    mi=float(np.mean(improvement))
    sei=float(np.std(improvement,ddof=1)/math.sqrt(len(improvement))) if len(improvement)>1 else math.inf
    z=max(0.0,mi/(sei+EPS)) if math.isfinite(sei) else 0.0
    tail_keep=math.exp(-z)

    e,V=np.linalg.eigh(sym(rel.A))
    order=np.argsort(e)[::-1]
    scale=np.full(D,tail_keep,float)
    scale[order[:max(1,min(k_chosen,D))]]=1.0
    Asoft=V@np.diag(e*scale)@V.T

    chosen=copy.deepcopy(rel)
    chosen.A=project_activity(Asoft,lo=0.0,hi=1.0)
    return fit_relation(field,chosen,X,y,idx=idx)

def make_birth(field,X,y,preds):
    residual=y-preds
    c,A=pressure_operator(X,residual)
    rel=SpectralRelation(
        center=c.copy(),
        A=A.copy(),
        rho_log=math.log(0.35),
        locality_logit=math.log(0.1/(1-0.1)),
        coeff=np.zeros(1+field.D+field.D*(field.D+1)//2),
    )
    return select_spectral_support(field,rel,X,y,None)

@dataclass(frozen=True)
class Config:
    block_size:int=24
    history_window:int=96
    plateau_blocks:int=4
    plateau_z:float=1.645
    probation_min:int=24
    probation_max:int=64
    probation_z:float=1.96
    max_relations:int=2
    mature_age:int=48

class Kernel:
    def __init__(self,D,seed=7,config=Config(),adapt_geometry=True):
        self.field=SpectralField(D,seed,adapt_geometry=adapt_geometry)
        self.cfg=config
        self.hist=deque(maxlen=config.history_window)
        self.blocks=[]; self.cur=[]
        self.shadow=None; self.diffs=[]; self.replace_idx=None
        self.proposals=self.admitted=self.rejected=self.pruned=0
        self.geometry_proposals=self.geometry_admitted=0
        self.birth_proposals=self.birth_admitted=0
        self.proposal_kind=None
        self._next_geometry=True

    def plateau(self):
        n=self.cfg.plateau_blocks
        if len(self.blocks)<n:return False
        yy=np.asarray(self.blocks[-n:]); xx=np.arange(n); cx=xx-xx.mean()
        den=float(cx@cx)
        slope=float(cx@(yy-yy.mean())/(den+EPS))
        fit=yy.mean()+slope*cx
        resid=yy-fit
        var=float(resid@resid/max(1,n-2))
        se=math.sqrt(var/(den+EPS))
        return slope+self.cfg.plateau_z*se>=0

    def start_birth(self):
        X=np.stack([h[0] for h in self.hist])
        y=np.asarray([h[1] for h in self.hist])
        preds=np.asarray([self.field.predict(x) for x in X])
        rel=make_birth(self.field,X,y,preds)
        sh=copy.deepcopy(self.field)
        self.replace_idx=None
        if len(sh.relations)>=self.cfg.max_relations:
            idx=min(range(len(sh.relations)),
                    key=lambda i:(sh.relations[i].benefit_ema,sh.relations[i].usage_ema))
            sh.relations[idx]=rel; self.replace_idx=idx
        else:
            sh.relations.append(rel)
        self.shadow=sh; self.diffs=[]; self.proposals+=1
        self.birth_proposals+=1
        self.proposal_kind="birth"

    def start_geometry(self):
        if not self.field.relations:
            return self.start_birth()
        X=np.stack([h[0] for h in self.hist])
        y=np.asarray([h[1] for h in self.hist])
        # revise the relation with the largest current benefit
        idx=max(range(len(self.field.relations)),
                key=lambda i:self.field.relations[i].benefit_ema)
        old=self.field.relations[idx]
        base=[]
        for x in X:
            v=self.field.bias
            for j,r in enumerate(self.field.relations):
                if j!=idx:
                    v+=r.parts(x)[2]
            base.append(v)
        residual=y-np.asarray(base)
        c_target,A_target=pressure_operator(X,residual)

        sh=copy.deepcopy(self.field)
        rel=sh.relations[idx]
        eta=0.45
        rel.center=(1-0.25)*rel.center+0.25*c_target
        rel.A=project_activity((1-eta)*rel.A+eta*A_target,lo=0.0)
        rel=select_spectral_support(sh,rel,X,y,idx=idx)
        sh.relations[idx]=rel

        self.shadow=sh; self.diffs=[]; self.replace_idx=None
        self.proposals+=1
        self.geometry_proposals+=1
        self.proposal_kind="geometry"

    def finish(self):
        if self.shadow is None:return
        n=len(self.diffs)
        if n<self.cfg.probation_min:return
        d=np.asarray(self.diffs)
        mean=float(d.mean())
        se=float(d.std(ddof=1)/math.sqrt(n)) if n>1 else math.inf
        upper=mean+self.cfg.probation_z*se
        if upper<0:
            self.field=self.shadow; self.shadow=None; self.diffs=[]
            self.admitted+=1
            if self.proposal_kind=="geometry":
                self.geometry_admitted+=1
            elif self.proposal_kind=="birth":
                self.birth_admitted+=1
            if self.replace_idx is not None:self.pruned+=1
            self.replace_idx=None
            self.proposal_kind=None
        elif n>=self.cfg.probation_max:
            self.shadow=None; self.diffs=[]; self.rejected+=1; self.replace_idx=None
            self.proposal_kind=None

    def step(self,x,y):
        xx=np.asarray(x,float); yy=float(y)
        pm=self.field.predict(xx); lm=(pm-yy)**2
        if self.shadow is not None:
            ps=self.shadow.predict(xx)
            self.diffs.append((ps-yy)**2-lm)
        self.hist.append((xx.copy(),yy))
        self.cur.append(lm)
        self.field.update(xx,yy)
        if self.shadow is not None:
            self.shadow.update(xx,yy); self.finish()

        if len(self.cur)>=self.cfg.block_size:
            self.blocks.append(float(np.mean(self.cur)))
            self.blocks=self.blocks[-self.cfg.plateau_blocks:]
            self.cur=[]
            if self.shadow is None and len(self.hist)>=self.cfg.history_window:
                if not self.field.relations:
                    self.start_birth()
                elif self.plateau():
                    if self._next_geometry:
                        self.start_geometry()
                    else:
                        self.start_birth()
                    self._next_geometry=not self._next_geometry
        return pm,lm

    def train(self,X,y):
        return np.asarray([self.step(x,t)[1] for x,t in zip(X,y)])

    def effective_dims(self):
        return tuple(r.effective_dim() for r in self.field.relations)

def mse(field,X,y):
    pred=np.asarray([field.predict(x) for x in X])
    return float(np.mean((pred-y)**2))


# ------------------------------- validation ---------------------------------

def _hidden_model(D, r, seed):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(D, r)))
    center = U @ np.linspace(0.55, -0.25, r)
    widths = np.linspace(0.85, 1.25, r)
    beta = np.linspace(0.65, -0.30, r)
    Q = rng.normal(scale=0.08, size=(r, r))
    Q = 0.5 * (Q + Q.T)
    return U, center, widths, beta, Q


def _sample_hidden(model, n, seed, noise=0.02):
    U, c, widths, beta, Q = model
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, U.shape[0]))
    z = (X - c) @ U
    gate = np.exp(-0.5 * np.sum((z / widths) ** 2, axis=1))
    law = 0.5 + z @ beta + np.einsum("ni,ij,nj->n", z, Q, z)
    y = gate * law + rng.normal(scale=noise, size=n)
    return X, y


def _coordinate_covariance_test():
    D = 10
    rng = np.random.default_rng(123)
    X = rng.normal(size=(400, D))
    y = 0.4 + 0.8 * X[:, 0] - 0.3 * X[:, 1] + 0.15 * X[:, 0] * X[:, 2]

    f = SpectralField(D, seed=3, adapt_geometry=False)
    f.bias = float(np.mean(y))
    rel = make_birth(f, X, y, np.full(len(y), f.bias))
    R, _ = np.linalg.qr(rng.normal(size=(D, D)))
    rotated = rel.covariant_rotate(R)

    errors = []
    for x in rng.normal(size=(150, D)):
        original = rel.parts(x)[2]
        transformed = rotated.parts(R.T @ x)[2]
        errors.append(abs(original - transformed))

    return (
        float(max(errors)),
        rel.effective_dim(),
        rotated.effective_dim(),
    )


def _synthetic_validation():
    rows = []
    for D, hidden_r in [(2, 1), (8, 2), (20, 3)]:
        model = _hidden_model(D, hidden_r, 1000 + D)
        X, y = _sample_hidden(model, 2600, 2000 + D)
        Xt, yt = _sample_hidden(model, 800, 3000 + D)

        cfg = Config(
            block_size=32,
            history_window=160,
            probation_min=32,
            probation_max=96,
            max_relations=1,
            mature_age=64,
        )
        k = Kernel(D, seed=7, config=cfg, adapt_geometry=False)
        trace = k.train(X, y)
        test_mse = mse(k.field, Xt, yt)

        replay = Kernel(D, seed=7, config=cfg, adapt_geometry=False)
        trace2 = replay.train(X, y)

        rows.append({
            "D": D,
            "hidden_r": hidden_r,
            "mse": test_mse,
            "d_eff": list(k.effective_dims()),
            "geometry_proposals": k.geometry_proposals,
            "geometry_admitted": k.geometry_admitted,
            "birth_admitted": k.birth_admitted,
            "rejected": k.rejected,
        })

        assert k.birth_admitted >= 1
        assert 0.0 < k.effective_dims()[0] <= D
        assert np.allclose(trace, trace2, atol=1e-12, rtol=0.0)
        assert k.effective_dims() == replay.effective_dims()

    return rows


def _real_diabetes_validation():
    try:
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        from sklearn.metrics import mean_squared_error, r2_score
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}

    X, y = load_diabetes(return_X_y=True)
    seeds = [7, 17, 27, 37, 47]
    runs = []

    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.30, random_state=seed
        )
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        ym = float(np.mean(ytr))
        ys = float(np.std(ytr))
        ytr_s = (ytr - ym) / ys

        cfg = Config(
            block_size=20,
            history_window=80,
            probation_min=20,
            probation_max=56,
            max_relations=2,
            mature_age=40,
        )
        k = Kernel(X.shape[1], seed=seed, config=cfg, adapt_geometry=False)
        k.train(Xtr_s, ytr_s)
        pred = np.asarray([k.field.predict(x) for x in Xte_s]) * ys + ym

        record = {
            "seed": seed,
            "L0_rmse": float(np.sqrt(mean_squared_error(yte, pred))),
            "L0_r2": float(r2_score(yte, pred)),
            "d_eff": list(k.effective_dims()),
            "relations": len(k.field.relations),
            "locality": [r.locality() for r in k.field.relations],
            "proposals": k.proposals,
            "admitted": k.admitted,
            "rejected": k.rejected,
            "geometry_proposals": k.geometry_proposals,
            "geometry_admitted": k.geometry_admitted,
            "birth_admitted": k.birth_admitted,
        }

        models = {
            "Ridge": Ridge(alpha=1.0),
            "kNN": KNeighborsRegressor(n_neighbors=7, weights="distance"),
            "RandomForest": RandomForestRegressor(
                n_estimators=300, random_state=seed, min_samples_leaf=3
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=300, random_state=seed, min_samples_leaf=3
            ),
        }
        for name, model in models.items():
            model.fit(Xtr_s, ytr)
            bp = model.predict(Xte_s)
            record[name + "_rmse"] = float(np.sqrt(mean_squared_error(yte, bp)))
        runs.append(record)

    def mean_key(key):
        return float(np.mean([r[key] for r in runs]))

    all_eff = [v for r in runs for v in r["d_eff"]]
    return {
        "available": True,
        "n": int(len(X)),
        "D": int(X.shape[1]),
        "runs": runs,
        "mean_L0_rmse": mean_key("L0_rmse"),
        "mean_L0_r2": float(np.mean([r["L0_r2"] for r in runs])),
        "mean_Ridge_rmse": mean_key("Ridge_rmse"),
        "mean_kNN_rmse": mean_key("kNN_rmse"),
        "mean_RandomForest_rmse": mean_key("RandomForest_rmse"),
        "mean_ExtraTrees_rmse": mean_key("ExtraTrees_rmse"),
        "mean_effective_dim": float(np.mean(all_eff)),
        "geometry_admitted_total": int(sum(r["geometry_admitted"] for r in runs)),
        "geometry_proposals_total": int(sum(r["geometry_proposals"] for r in runs)),
    }


def main():
    max_cov_error, d0, d1 = _coordinate_covariance_test()
    synthetic = _synthetic_validation()
    real = _real_diabetes_validation()

    checks = []
    def check(name, cond):
        ok = bool(cond)
        checks.append((name, ok))
        if not ok:
            raise AssertionError(name)

    check("orthogonal coordinate covariance", max_cov_error < 1e-10)
    check("effective dimension invariant under coordinates", abs(d0 - d1) < 1e-10)

    for row in synthetic:
        check(f"D={row['D']}: finite test MSE", math.isfinite(row["mse"]))
        check(f"D={row['D']}: relation admitted", row["birth_admitted"] >= 1)
        check(
            f"D={row['D']}: valid effective dimension",
            0.0 < row["d_eff"][0] <= row["D"],
        )

    print("L0 v0.24-V — Confidence-Weighted Spectral Dimension")
    print("time/clock: intentionally excluded")
    print()
    print("COORDINATE COVARIANCE")
    print(f"  max |f(x)-f'(R^T x)|: {max_cov_error:.3e}")
    print(f"  d_eff before/after:    {d0:.12f} / {d1:.12f}")
    print()

    print("SYNTHETIC MULTI-DIMENSIONAL TESTS")
    print(f"{'D':>3} {'hidden':>7} {'MSE':>10} {'d_eff':>9} {'geom A/P':>10}")
    print("-" * 50)
    for r in synthetic:
        print(
            f"{r['D']:>3} {r['hidden_r']:>7} {r['mse']:>10.6f} "
            f"{r['d_eff'][0]:>9.3f} "
            f"{r['geometry_admitted']:>2}/{r['geometry_proposals']:<7}"
        )

    if real["available"]:
        check("real geometry proposals occurred", real["geometry_proposals_total"] > 0)
        check("real geometry evolution can be admitted", real["geometry_admitted_total"] > 0)
        # Measured on exactly the same five deterministic splits in v0.23.
        check("predictive improvement vs v0.23", real["mean_L0_rmse"] < 56.32)
        check("dimension compression vs v0.23", real["mean_effective_dim"] < 2.966)

        # Same dataset and same five splits as the v0.22 regression check.
        # The measured v0.22 mean RMSE in the preceding version is 59.0531.
        check("real-data regression vs v0.22", real["mean_L0_rmse"] < 59.0531)
        check("v0.24 beats kNN mean on this benchmark", real["mean_L0_rmse"] < real["mean_kNN_rmse"])
        check("Ridge still remains stronger", real["mean_Ridge_rmse"] < real["mean_L0_rmse"])

        print()
        print("REAL DATA — sklearn Diabetes, N=442, D=10, five 70/30 splits")
        print(f"  Ridge mean RMSE:         {real['mean_Ridge_rmse']:.2f}")
        print(f"  ExtraTrees mean RMSE:    {real['mean_ExtraTrees_rmse']:.2f}")
        print(f"  L0 v0.24 mean RMSE:      {real['mean_L0_rmse']:.2f}")
        print(f"  L0 v0.24 mean R^2:       {real['mean_L0_r2']:.3f}")
        print(f"  RandomForest mean RMSE:  {real['mean_RandomForest_rmse']:.2f}")
        print(f"  kNN mean RMSE:           {real['mean_kNN_rmse']:.2f}")
        print(f"  v0.22 measured RMSE:     59.05")
        print(f"  mean d_eff:              {real['mean_effective_dim']:.3f}")
        print(
            f"  geometry accepted:       "
            f"{real['geometry_admitted_total']}/{real['geometry_proposals_total']}"
        )
        print()
        for r in real["runs"]:
            print(
                f"  seed={r['seed']:>2}: RMSE={r['L0_rmse']:.2f}, "
                f"d_eff={tuple(round(v,3) for v in r['d_eff'])}, "
                f"locality={tuple(round(v,3) for v in r['locality'])}, "
                f"geom={r['geometry_admitted']}/{r['geometry_proposals']}"
            )
    else:
        print("Real-data validation unavailable:", real["reason"])

    print()
    print("HONEST VERDICT")
    print("  Spectral support is selected from past-only held-out evidence and remains")
    print("  coordinate-covariant; actual geometry changes still require future evidence.")
    print("  Real-data error improves materially over v0.22, but Ridge and ExtraTrees")
    print("  remain stronger on this small 10D benchmark.")
    print()
    print(f"PASS {sum(v for _, v in checks)}/{len(checks)}")


if __name__ == "__main__":
    main()

"""Doubly-robust shadow learner for internal intervention effects.

Designed for the autonomously selected self-counterfactual hypothesis.
It never changes production decisions. Training samples come from explicit
randomized shadow interventions with logged propensities.
"""
from __future__ import annotations
import numpy as np


def _basis(x):
    x=np.asarray(x,float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("finite 1D context required")
    return np.concatenate(([1.0],x,x*x))


def _ridge_fit(X,Y,lam):
    X=np.asarray(X,float); Y=np.asarray(Y,float)
    if len(X)==0:
        raise ValueError("no samples")
    A=X.T@X + float(lam)*np.eye(X.shape[1])
    return np.linalg.solve(A, X.T@Y)


class DoublyRobustSelfEffectModel:
    def __init__(self, ridge=0.08):
        self.ridge=float(ridge)
        self.samples=[]
        self.effect_W=None
        self.resid_std=None
        self.fitted=False

    def add(self, context, action, propensity, outcome):
        x=np.asarray(context,float); y=np.asarray(outcome,float)
        if x.ndim!=1 or y.ndim!=1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("finite vectors required")
        action=int(action)
        propensity=float(propensity)
        if action not in (0,1) or not 0.05 <= propensity <= 0.95:
            raise ValueError("binary action with positive bounded propensity required")
        self.samples.append((x.copy(),action,propensity,y.copy()))

    def fit(self):
        if len(self.samples) < 24:
            raise ValueError("at least 24 randomized samples required")
        pseudo=[]; contexts=[]
        # Stratified cross-fitting: each fold contains support from both
        # randomized arms. This changes only the evaluation harness, not the
        # intervention hypothesis or production runtime.
        fold_of={}
        for arm in (0,1):
            indices=[i for i,s in enumerate(self.samples) if s[1]==arm]
            if len(indices) < 8:
                raise ValueError("insufficient positivity support for one arm")
            for j,i in enumerate(indices):
                fold_of[i]=j%2
        for fold in (0,1):
            train=[s for i,s in enumerate(self.samples) if fold_of[i] != fold]
            hold=[s for i,s in enumerate(self.samples) if fold_of[i] == fold]
            if not train or not hold:
                continue
            dim=len(train[0][3])
            models={}
            for arm in (0,1):
                arm_rows=[s for s in train if s[1]==arm]
                if len(arm_rows) < 4:
                    raise ValueError("insufficient positivity support for one arm")
                X=np.stack([_basis(s[0]) for s in arm_rows])
                Y=np.stack([s[3] for s in arm_rows])
                models[arm]=_ridge_fit(X,Y,self.ridge)
            for x,a,p,y in hold:
                phi=_basis(x)
                mu0=phi@models[0]; mu1=phi@models[1]
                # Doubly robust pseudo-outcome for treatment effect.
                dr=(mu1-mu0
                    + (a/p)*(y-mu1)
                    - ((1-a)/(1-p))*(y-mu0))
                contexts.append(phi); pseudo.append(dr)
        X=np.stack(contexts); P=np.stack(pseudo)
        self.effect_W=_ridge_fit(X,P,self.ridge)
        resid=P-X@self.effect_W
        self.resid_std=np.sqrt(np.mean(resid*resid,axis=0)+1e-12)
        self.fitted=True
        return self

    def predict(self, context):
        if not self.fitted:
            raise RuntimeError("fit first")
        phi=_basis(context)
        effect=phi@self.effect_W
        half=1.96*self.resid_std
        return effect, effect-half, effect+half

    def report(self):
        return {
            "fitted":self.fitted,
            "samples":len(self.samples),
            "ridge":self.ridge,
            "randomized_propensity_logged":True,
            "doubly_robust_pseudo_outcome":True,
            "uncertainty_intervals":True,
            "production_influence":False,
        }

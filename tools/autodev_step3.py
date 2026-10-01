"""Autonomous development step 3: test the self-selected causal-self hypothesis."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import numpy as np
from digital_mind_core.kernel import KernelConfig,UnifiedMind
from digital_mind_core.causal_self import DoublyRobustSelfEffectModel

DESIGN=(131,132,133)
BLIND=(141,142,143,144)
FIELDS=("replan","reflect")

def decision_vector(mind,decision):
    return np.asarray([
        float(bool(decision.get("replan",False))),
        float(bool(decision.get("reflect",False))),
        float(bool(decision.get("consolidate",False))),
        float(bool(decision.get("allow_probe",False))),
        float(bool(decision.get("macro_reflect",False))),
        np.clip(float(decision.get("exploration",0.0)),0,1),
        np.clip(float(decision.get("budget",0.0))/1.2,0,1),
        0.0,
    ])

def preview(mind):
    m=copy.deepcopy(mind)
    return m._executive_decision(m.fast_steps)

def branch(mind,field,value):
    b=copy.deepcopy(mind); orig=b._executive_decision
    def forced(t):
        d=copy.deepcopy(orig(t)); d[field]=bool(value); return d
    b._executive_decision=forced
    before=b._self_state_vector().copy()
    b.step()
    return b._self_state_vector()-before

def context(mind,field):
    state=mind._self_state_vector()
    t=mind.fast_steps
    phases=np.asarray([
        ((t+1)%mind.episode_len)/max(1,mind.episode_len-1),
        (t%mind.macro_period)/max(1,mind.macro_period-1),
        (t%mind.config.control_period)/max(1,mind.config.control_period-1),
        ((t+1)%mind.slow_period)/max(1,mind.slow_period-1),
        float(field=="reflect"),
        float(field=="replan"),
    ])
    return np.concatenate([state,phases])

def baseline_effect(mind,field):
    state=mind._self_state_vector(); d=preview(mind)
    d0=copy.deepcopy(d); d1=copy.deepcopy(d)
    d0[field]=False; d1[field]=True
    p0=mind.predictive_self.predict_next(state,decision_vector(mind,d0))
    p1=mind.predictive_self.predict_next(state,decision_vector(mind,d1))
    return p1-p0

def run(seed,steps=640,train_end=400):
    mind=UnifiedMind(KernelConfig(seed=seed))
    rng=np.random.default_rng(seed+7000)
    models={f:DoublyRobustSelfEffectModel() for f in FIELDS}
    # Warm the ordinary kernel first.
    while mind.fast_steps < 128:
        mind.step()
    checkpoints=list(range(128,train_end,8))
    # Complete randomized design with marginal propensity 0.5: each field gets
    # equal numbers of 0/1 interventions, in a seed-randomized order.
    schedules={}
    for field in FIELDS:
        n=len(checkpoints)
        actions=np.asarray(([0]*(n//2))+([1]*(n//2)),dtype=int)
        if n%2:
            actions=np.concatenate([actions,[int(rng.integers(0,2))]])
        rng.shuffle(actions)
        schedules[field]=iter(actions.tolist())
    # Randomized shadow logging: one arm observed per field with known p=0.5.
    while mind.fast_steps < train_end:
        if mind.fast_steps % 8 == 0:
            for field in FIELDS:
                x=context(mind,field)
                a=int(next(schedules[field]))
                y=branch(mind,field,a)
                models[field].add(x,a,0.5,y)
        mind.step()
    for model in models.values():
        model.fit()
    rows=[]
    while mind.fast_steps < steps:
        if mind.fast_steps % 16 == 0:
            for field in FIELDS:
                x=context(mind,field)
                actual=branch(mind,field,True)-branch(mind,field,False)
                candidate,lo,hi=models[field].predict(x)
                baseline=baseline_effect(mind,field)
                energy=float(np.mean(actual*actual))
                if energy > 1e-12:
                    rows.append({
                        "field":field,
                        "candidate_mse":float(np.mean((actual-candidate)**2)),
                        "baseline_mse":float(np.mean((actual-baseline)**2)),
                        "zero_mse":energy,
                        "coverage":float(np.mean((actual>=lo)&(actual<=hi))),
                        "interval_width":float(np.mean(hi-lo)),
                    })
        mind.step()
    cand=sum(r["candidate_mse"] for r in rows)
    base=sum(r["baseline_mse"] for r in rows)
    zero=sum(r["zero_mse"] for r in rows)
    return {
        "seed":seed,
        "pairs":len(rows),
        "candidate_vs_current_ratio":cand/base if base>1e-15 else None,
        "candidate_vs_zero_ratio":cand/zero if zero>1e-15 else None,
        "coverage":float(np.mean([r["coverage"] for r in rows])) if rows else None,
        "mean_interval_width":float(np.mean([r["interval_width"] for r in rows])) if rows else None,
        "models":{k:v.report() for k,v in models.items()},
    }

def aggregate(rows):
    valid=[r for r in rows if r["candidate_vs_current_ratio"] is not None]
    return {
        "runs":len(rows),
        "mean_ratio_vs_current":float(np.mean([r["candidate_vs_current_ratio"] for r in valid])),
        "better_than_current_runs":sum(r["candidate_vs_current_ratio"]<1 for r in valid),
        "mean_ratio_vs_zero":float(np.mean([r["candidate_vs_zero_ratio"] for r in valid])),
        "mean_coverage":float(np.mean([r["coverage"] for r in valid])),
        "pairs":sum(r["pairs"] for r in valid),
    }

def verdict(rows,needed):
    a=aggregate(rows)
    passed=(a["mean_ratio_vs_current"]<0.98 and
            a["better_than_current_runs"]>=needed and
            a["mean_coverage"]>=0.80 and a["pairs"]>=20)
    return {"pass":passed,"reason":"DR_SELF_EFFECT_PASS" if passed else "DR_SELF_EFFECT_REJECT",**a}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--hypothesis",type=Path,required=True); p.add_argument("--report",type=Path,required=True)
    a=p.parse_args(); h=json.loads(a.hypothesis.read_text())
    text=" ".join(x["mechanism"] for x in h["selected_mechanisms"]).lower()
    required={"doubly_robust":"doubly robust" in text,"randomized":"randomized" in text,"uncertainty":"uncertainty" in text}
    if not all(required.values()):
        raise SystemExit("hypothesis does not support candidate")
    design=[run(s) for s in DESIGN]
    dver=verdict(design,2)
    # Freeze method/hyperparameters before untouched blind seeds.
    blind=[run(s) for s in BLIND]
    bver=verdict(blind,3)
    report={
        "schema":"digital-mind.autodev-step3.v1",
        "action":"EXPERIMENT",
        "hypothesis":h,
        "required_mechanisms":required,
        "design_runs":design,"design_verdict":dver,
        "blind_runs":blind,"blind_verdict":bver,
        "promotion":bool(dver["pass"] and bver["pass"]),
        "production_runtime_modified":False,
        "final_decision":"PROMOTE_CANDIDATE" if dver["pass"] and bver["pass"] else "REJECT_CANDIDATE",
    }
    a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"design_pass":dver["pass"],"blind_pass":bver["pass"],"decision":report["final_decision"],"blind_ratio":bver["mean_ratio_vs_current"]},sort_keys=True))
if __name__=="__main__": main()

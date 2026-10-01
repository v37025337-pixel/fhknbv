from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

from .api import compile_file, compile_foreign, lower_foreign_control, migrate_legacy_source, VERSION
from .compiler import L0SemanticsV10Error
from . import spectral_runtime as spectral_rt
from .control import render_control


def _parse_state(raw):
    if not raw: return {}
    v=json.loads(raw)
    if not isinstance(v,dict): raise ValueError('--state must be a JSON object')
    return v


def _row_target(row,t):
    if t.kind=='scalar': return float(row[t.name])
    cols=[f'{t.name}[{i}]' for i in range(t.dimension)]
    miss=[c for c in cols if c not in row]
    if miss: raise ValueError(f'CSV missing vector target columns {miss}')
    return tuple(float(row[c]) for c in cols)


def _hybrid_csv(compiled,path,train_fraction):
    if not(0<train_fraction<1): raise ValueError('train_fraction must be between 0 and 1')
    with path.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    if len(rows)<2: raise ValueError('hybrid CSV requires at least 2 rows')
    inp=[x.name for x in compiled.hybrid.arc.ast.inputs]
    targets=list(compiled.hybrid.ast.targets)
    missing=set(inp)-set(rows[0])
    if missing: raise ValueError(f'CSV missing columns {sorted(missing)}')
    n=max(1,min(len(rows)-1,int(round(train_fraction*len(rows)))))
    rt=compiled.new_hybrid_runtime(training_horizon=n)
    for row in rows[:n]:
        rt.step({k:float(row[k]) for k in inp},targets={t.name:_row_target(row,t) for t in targets})
    pred={m.name:[] for m in compiled.hybrid.ast.models}; truth={m.name:[] for m in compiled.hybrid.ast.models}
    for row in rows[n:]:
        z=rt.step({k:float(row[k]) for k in inp},learn=False)
        for m in compiled.hybrid.ast.models:
            pred[m.name].append(z.arc.state[m.output])
            truth[m.name].append(_row_target(row,compiled.hybrid.ast.target_map[m.target]))
    metrics={}
    for m in compiled.hybrid.ast.models:
        p=np.asarray(pred[m.name],float); y=np.asarray(truth[m.name],float); e=p-y
        metrics[m.name]={'target':m.target,'output':m.output,'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(np.abs(e)))}
    return {'program':compiled.name,'semantics':'hybrid','language':'L0-Canonical','fingerprint':compiled.fingerprint,'rows':len(rows),'train_rows':n,'test_rows':len(rows)-n,'metrics':metrics,'models':rt.model_report()}


def run_cmd(a):
    c=compile_file(a.program)
    if c.semantics=='arc':
        if a.smoke or a.csv or a.mapping: raise L0SemanticsV10Error('ARC does not accept data options')
        r=c.execute_arc(_parse_state(a.state),requested=(a.request or None))
        return {'program':c.name,'semantics':'arc','language':'L0-Canonical','fingerprint':c.fingerprint,'status':r.status,'state':r.state,'effects':[e.__dict__ for e in r.effects],'stats':r.stats.__dict__}
    if c.semantics=='spectral':
        if a.state or a.request: raise L0SemanticsV10Error('spectral does not accept state/request')
        if a.smoke==bool(a.csv): raise ValueError('spectral requires exactly one of --smoke or --csv')
        data=np.asarray(spectral_rt.SMOKE_ROWS,float) if a.smoke else spectral_rt.read_csv(c.legacy.spectral_backend,a.csv,spectral_rt.parse_mapping(a.mapping))
        out=c.execute_spectral(data,train_fraction=a.train_fraction)
        out.update({'program_name':c.name,'semantics':'spectral','language':'L0-Canonical','fingerprint':c.fingerprint})
        return out
    if a.state or a.request or a.smoke or a.mapping: raise L0SemanticsV10Error('hybrid CSV runner accepts only --csv/--train-fraction')
    if not a.csv: raise ValueError('hybrid program requires --csv')
    return _hybrid_csv(c,a.csv,a.train_fraction)


def main(argv=None):
    ap=argparse.ArgumentParser(prog='l0',description='L0 Canonical — one frontend for ARC, spectral, hybrid/digital, and foreign semantic IR')
    ap.add_argument('--version',action='version',version=VERSION)
    sub=ap.add_subparsers(dest='cmd',required=True)

    run=sub.add_parser('run',help='compile and execute native L0')
    run.add_argument('program',type=Path); run.add_argument('--state'); run.add_argument('--request',action='append',default=[])
    run.add_argument('--smoke',action='store_true'); run.add_argument('--csv',type=Path); run.add_argument('--map',dest='mapping')
    run.add_argument('--train-fraction',type=float,default=.70); run.add_argument('--report',type=Path)

    ins=sub.add_parser('inspect',help='compile native L0 and print canonical metadata')
    ins.add_argument('program',type=Path)

    mig=sub.add_parser('migrate',help='upgrade old pre-semantics native L0 source')
    mig.add_argument('program',type=Path); mig.add_argument('-o','--output',type=Path)

    fr=sub.add_parser('foreign',help='compile Python/JS/C/C++/Rust to shared Semantic IR')
    fr.add_argument('language'); fr.add_argument('path',type=Path); fr.add_argument('--control',action='store_true')

    a=ap.parse_args(argv)
    if a.cmd=='run':
        out=run_cmd(a); text=json.dumps(out,indent=2,default=str); print(text)
        if a.report: a.report.write_text(text,encoding='utf-8')
    elif a.cmd=='inspect':
        c=compile_file(a.program); print(json.dumps({'name':c.name,'semantics':c.semantics,'fingerprint':c.fingerprint,'language':'L0-Canonical'},indent=2))
    elif a.cmd=='migrate':
        text=migrate_legacy_source(a.program.read_text(encoding='utf-8'))
        if a.output: a.output.write_text(text,encoding='utf-8')
        else: print(text,end='' if text.endswith('\n') else '\n')
    else:
        source=a.path.read_text(encoding='utf-8')
        if a.control:
            print(render_control(lower_foreign_control(source,a.language,a.path.stem)))
        else:
            m=compile_foreign(source,a.language,a.path.stem)
            print(json.dumps(m.to_dict() if hasattr(m,'to_dict') else m.__dict__,indent=2,default=str))

if __name__=='__main__': main()

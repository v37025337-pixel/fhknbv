from __future__ import annotations
import argparse, json
from pathlib import Path
from digital_mind_core.autodev import step1

p=argparse.ArgumentParser()
p.add_argument("--state",type=Path,required=True)
p.add_argument("--report",type=Path,required=True)
a=p.parse_args()
state=json.loads(a.state.read_text(encoding="utf-8"))
out=step1(state)
a.report.parent.mkdir(parents=True,exist_ok=True)
a.report.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps({"target":out["selected_target"]["deficit_id"],"query":out["capability_request"]["query"]},ensure_ascii=False))

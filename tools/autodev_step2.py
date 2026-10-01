from __future__ import annotations
import argparse,json
from pathlib import Path
from digital_mind_core.autodev import step2
p=argparse.ArgumentParser()
p.add_argument("--step1",type=Path,required=True)
p.add_argument("--evidence",type=Path,required=True)
p.add_argument("--report",type=Path,required=True)
a=p.parse_args()
out=step2(json.loads(a.step1.read_text()),json.loads(a.evidence.read_text()))
a.report.write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
print(json.dumps({"target":out["selected_target"]["deficit_id"],"mechanisms":[x["mechanism"] for x in out["selected_mechanisms"]]},ensure_ascii=False))

from __future__ import annotations
import argparse,json
from pathlib import Path
from digital_mind_core.self_diagnostic import diagnose

p=argparse.ArgumentParser()
p.add_argument("--root",type=Path,default=Path("."))
p.add_argument("--report",type=Path,required=True)
a=p.parse_args()
report=diagnose(a.root.resolve())
a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps({
  "top_risk":report["self_verdict"]["top_risk"],
  "promotion_safe_now":report["self_verdict"]["promotion_safe_now"],
  "finding_count":len(report["findings"]),
  "strengths":report["strengths"]
},ensure_ascii=False,sort_keys=True))

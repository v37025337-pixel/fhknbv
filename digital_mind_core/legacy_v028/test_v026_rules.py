
from pathlib import Path
import importlib.util, sys
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("rules_core",root/"v0.18_cognitive_core.py")
m=importlib.util.module_from_spec(spec);sys.modules["rules_core"]=m;spec.loader.exec_module(m)
c=m.CognitiveCore()
rule=([("edge","$x","$y"),("edge","$y","$z")],("reach","$x","$z"))
c.process({"rules":[rule],"persist_rules":True})
r=c.process({
    "facts":[("edge","a","b"),("edge","b","c")],
    "queries":[("reach","a","c")],
    "persist_rules":True,
})
assert any(x["value"] for x in r["inferences"])
print("V026_RULES_PASS")

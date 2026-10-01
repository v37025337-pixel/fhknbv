
from pathlib import Path
import importlib.util, sys
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("dedupe_core",root/"v0.18_cognitive_core.py")
m=importlib.util.module_from_spec(spec);sys.modules["dedupe_core"]=m;spec.loader.exec_module(m)
c=m.CognitiveCore()
rule=([("p","$x")],("q","$x"))
for _ in range(50):
    c.process({"rules":[rule],"persist_rules":True})
assert len(c.state["persistent_rules"]) == 1
r=c.process({"facts":[("p","a")],"queries":[("q","a")],"persist_rules":True})
assert any(x["value"] for x in r["inferences"])
print("V027_DEDUPE_PASS")


from pathlib import Path
import importlib.util, sys
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("mem_core",root/"v0.18_cognitive_core.py")
m=importlib.util.module_from_spec(spec);sys.modules["mem_core"]=m;spec.loader.exec_module(m)
c=m.CognitiveCore()
c.process({"facts":[("knows","a","map")],"persist_facts":True})
r=c.process({"queries":[("knows","a","map")],"persist_facts":True})
assert any(x["value"] for x in r["inferences"])
c.process({"context":"mission","preserve_context":True,"candidate_actions":["x"]})
c.process({"preserve_context":True,"candidate_actions":["x"]})
assert c.state["context"]=="mission"
print("V021_MEMORY_PASS")

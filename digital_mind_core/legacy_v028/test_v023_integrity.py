
from pathlib import Path
import importlib.util, sys
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("int_core",root/"v0.18_cognitive_core.py")
m=importlib.util.module_from_spec(spec);sys.modules["int_core"]=m;spec.loader.exec_module(m)
c=m.CognitiveCore()
r=c.process({"observations":[{"nested":[1,2,("x","y")]}]})
before=len(c.state["trace"])
r["state"]["trace"].append({"corrupt":True})
assert len(c.state["trace"])==before
assert isinstance(r["state"]["observations"][0]["nested"][2],tuple)
print("V023_INTEGRITY_PASS")

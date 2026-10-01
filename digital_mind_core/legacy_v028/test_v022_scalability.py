
from pathlib import Path
import importlib.util, sys
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("scale_core",root/"v0.18_cognitive_core.py")
m=importlib.util.module_from_spec(spec);sys.modules["scale_core"]=m;spec.loader.exec_module(m)
c=m.CognitiveCore()
for i in range(1200):
    c.process({"observations":[i]})
assert len(c.state["trace"]) <= 1000
assert len(c.memory) <= 1000
assert c.state["cycle"] == 1200
print("V022_SCALE_PASS")

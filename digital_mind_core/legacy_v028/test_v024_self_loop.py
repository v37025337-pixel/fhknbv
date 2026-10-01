
from pathlib import Path
import importlib.util, sys
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("self_core",root/"v0.18_cognitive_core.py")
m=importlib.util.module_from_spec(spec);sys.modules["self_core"]=m;spec.loader.exec_module(m)
c=m.CognitiveCore()
before=c.self_model.estimate("plan")
for _ in range(5):
    c.process({"outcome":{"capability":"plan","success":True}})
after=c.self_model.estimate("plan")
assert after > before + .15
for _ in range(5):
    c.process({"outcome":{"capability":"fragile","success":False}})
assert c.self_model.estimate("fragile") < .35
print("V024_SELF_LOOP_PASS")


from pathlib import Path
import importlib.util, sys
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("batch_core",root/"v0.18_cognitive_core.py")
m=importlib.util.module_from_spec(spec);sys.modules["batch_core"]=m;spec.loader.exec_module(m)
c=m.CognitiveCore()
c.process({"outcomes":[
    {"capability":"reader","success":True},
    {"capability":"writer","success":False},
]})
assert c.self_model.estimate("reader") > .5
assert c.self_model.estimate("writer") < .5
print("V025_BATCH_PASS")

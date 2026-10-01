
from pathlib import Path
import importlib.util, sys
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v020_core",root/"v0.18_cognitive_core.py")
m=importlib.util.module_from_spec(spec)
sys.modules["v020_core"]=m
spec.loader.exec_module(m)
core=m.CognitiveCore()

ctx="strong"
for _ in range(8):
    core.process({"context":ctx,"candidate_actions":["safe","fast"],"feedback":{"safe":1.0,"fast":0.0}})
r=core.process({
    "context":ctx,
    "candidate_actions":["safe","fast"],
    "proposals":[{"source":"sensor","topic":"action","value":"fast","action":"fast","confidence":.99,"urgency":.99}],
    "feedback":{"safe":1.0,"fast":0.0},
})
assert r["action"]=="safe"
assert r["challenge"] is not None
assert any(x["stage"]=="challenge" for x in core.reason_trace())
print("V020_META_PASS")

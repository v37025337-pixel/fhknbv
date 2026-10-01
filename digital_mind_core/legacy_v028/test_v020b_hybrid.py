
from pathlib import Path
import importlib.util, sys
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v020b_core",root/"v0.18_cognitive_core.py")
m=importlib.util.module_from_spec(spec)
sys.modules["v020b_core"]=m
spec.loader.exec_module(m)
core=m.CognitiveCore()

# Old learned preference, then immediate regime flip + matching current signal.
ctx="rapid"
for _ in range(12):
    core.process({"context":ctx,"candidate_actions":["A","B"],"feedback":{"A":1.0,"B":0.0}})
r=core.process({
    "context":ctx,
    "candidate_actions":["A","B"],
    "feedback":{"A":0.0,"B":1.0},
    "proposals":[{"source":"sensor","topic":"action","value":"B","action":"B","confidence":1.0,"urgency":1.0}],
})
assert r["action"]=="B"

# Stable reward should still resist noisy contradictory signal.
ctx2="stable"
for _ in range(10):
    core.process({"context":ctx2,"candidate_actions":["A","B"],"feedback":{"A":1.0,"B":0.0}})
r2=core.process({
    "context":ctx2,
    "candidate_actions":["A","B"],
    "feedback":{"A":1.0,"B":0.0},
    "proposals":[{"source":"noise","topic":"action","value":"B","action":"B","confidence":.95,"urgency":.95}],
})
assert r2["action"]=="A"
print("V020B_HYBRID_PASS")

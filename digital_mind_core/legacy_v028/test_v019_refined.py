
from pathlib import Path
import importlib.util, sys

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("refined_core",root/"v0.18_cognitive_core.py")
m=importlib.util.module_from_spec(spec)
sys.modules["refined_core"]=m
spec.loader.exec_module(m)
core=m.CognitiveCore()

# Adaptive policy must affect a later choice.
for _ in range(4):
    core.process({
        "context":"ctx",
        "candidate_actions":["L","R"],
        "feedback":{"L":0.0,"R":1.0},
    })
r=core.process({"context":"ctx","candidate_actions":["L","R"]})
assert r["action"]=="R"

# Integrated challenge must be present.
before=len(core.reason_trace())
core.process({"proposals":[
    {"source":"a","topic":"route","value":"left","action":"left","confidence":.9,"urgency":.8},
    {"source":"b","topic":"route","value":"right","action":"right","confidence":.8,"urgency":.7},
]})
assert any(x["stage"]=="challenge" for x in core.reason_trace()[before:])

# Full trace preserves observation content.
before=len(core.reason_trace())
core.process({"observations":["marker-917"]})
obs=[x for x in core.reason_trace()[before:] if x["stage"]=="observation"][-1]
assert "marker-917" in obs["payload"]["items"]

print("V019_REFINED_PASS")

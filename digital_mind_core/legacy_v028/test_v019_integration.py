
from pathlib import Path
import importlib.util, sys
root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("integration_test_core", root/"v0.18_cognitive_core.py")
m = importlib.util.module_from_spec(spec)
sys.modules["integration_test_core"] = m
spec.loader.exec_module(m)
core = m.CognitiveCore()
assert callable(getattr(core, "process", None))
assert callable(getattr(core, "reason_trace", None))
assert hasattr(core, "state") and hasattr(core, "blackboard")
r = core.process({
    "facts": [("edge","A","B"),("edge","B","C")],
    "rules": [([("edge","$x","$y"),("edge","$y","$z")],("reachable","$x","$z"))],
    "queries": [("reachable","A","C")],
    "graph": {"A":[("B",1)],"B":[("C",1)],"C":[]},
    "start": "A",
    "goal": "C",
})
assert r["plan"] == ["A","B","C"]
assert any(x["value"] for x in r["inferences"])
stages = [x["stage"] for x in core.reason_trace()]
for stage in ("observation","inference","planning","decision"):
    assert stage in stages
assert core.self_model.last_decision()["action"] == r["action"]
print("INTEGRATION_PASS")


from pathlib import Path
import importlib.util, sys

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v028_persisted_core",root/"v0.18_cognitive_core.py")
m=importlib.util.module_from_spec(spec)
sys.modules["v028_persisted_core"]=m
spec.loader.exec_module(m)

core=m.CognitiveCore()

# Mechanism must be available on a fresh process/core instance.
info=core.mechanism_info("stability_switch")
assert info["status"]=="PERSISTED"

cases=[
    (10.0,10.2,10.0),
    (10.0,12.0,12.0),
    (-5.0,-4.2,-5.0),
    (-5.0,-6.2,-6.2),
    (0.0,-1.0,0.0),
    (0.0,-1.01,-1.01),
    (2.5,1.5,2.5),
    (2.5,1.49,1.49),
]
for history,current,expected in cases:
    got=core.run_mechanism(
        "stability_switch",
        history=history,
        current=current,
    )
    assert abs(float(got)-float(expected)) < 1e-9

print("V028_PERSISTED_MECHANISM_PASS")

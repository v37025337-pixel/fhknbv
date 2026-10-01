
from pathlib import Path
import importlib.util, sys

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("v028_core",root/"v0.18_cognitive_core.py")
m=importlib.util.module_from_spec(spec)
sys.modules["v028_core"]=m
spec.loader.exec_module(m)

core=m.CognitiveCore()

spec1={
    "name":"novel_rule",
    "args":["x","y"],
    "constants":[-2,-1,0,1,2],
    "examples":[
        {"inputs":{"x":0,"y":1},"output":1},
        {"inputs":{"x":1,"y":2},"output":3},
        {"inputs":{"x":2,"y":3},"output":7},
        {"inputs":{"x":-2,"y":5},"output":9},
        {"inputs":{"x":3,"y":-1},"output":8},
    ],
}
r=core.synthesize_mechanism(spec1)
assert r["status"]=="FOUND"
assert core.run_mechanism("novel_rule",x=4,y=10)==26
assert core.run_mechanism("novel_rule",x=-3,y=0)==9

# An unsupported domain must fail honestly instead of producing arbitrary code.
bad={
    "name":"reverse_text",
    "args":["text"],
    "examples":[
        {"inputs":{"text":"abc"},"output":"cba"},
        {"inputs":{"text":"hello"},"output":"olleh"},
    ],
}
r2=core.synthesize_mechanism(bad)
assert r2["status"]=="NO_MECHANISM_FOUND"

info=core.mechanism_info("novel_rule")
assert "source" in info and "return" in info["source"]
print("V028_GENESIS_PASS")

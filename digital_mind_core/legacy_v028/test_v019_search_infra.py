
from pathlib import Path
import importlib.util, sys
root = Path(__file__).resolve().parent
p = root / "v0.19_integration_evolver.py"
spec = importlib.util.spec_from_file_location("v019_infra_test", p)
m = importlib.util.module_from_spec(spec)
sys.modules["v019_infra_test"] = m
spec.loader.exec_module(m)
space = m.generate_architecture_space()
assert len(space) == 16
assert len({s.architecture_id for s in space}) == 16
print("V019_INFRA_PASS")

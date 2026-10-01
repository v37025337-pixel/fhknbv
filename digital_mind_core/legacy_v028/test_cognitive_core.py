
from pathlib import Path
import importlib.util, json, sys

root = Path(__file__).resolve().parent
p = root / "v0.18_cognitive_lab.py"
spec = importlib.util.spec_from_file_location("test_cognitive_lab_runtime", p)
lab = importlib.util.module_from_spec(spec)
sys.modules["test_cognitive_lab_runtime"] = lab
spec.loader.exec_module(lab)

result = lab.run_benchmark(root)
enabled = set(result.get("enabled", []))

for dim in enabled:
    assert result["scores"][dim] >= 0.99, (dim, result["scores"][dim])

# Selection must be a real remaining deficit when the core is incomplete.
if len(enabled) < len(lab.DIMENSIONS):
    choice = lab.choose_deficit(result, root)
    assert choice is not None
    assert choice["dimension"] not in enabled
    assert 0.0 <= choice["score"] <= 1.0

print(json.dumps({
    "scores": result["scores"],
    "mean_score": result["mean_score"],
    "enabled": result["enabled"],
}, sort_keys=True))
print("COGNITIVE_LAB_PASS")

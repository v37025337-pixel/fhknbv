
from pathlib import Path
import importlib.util, sys, tempfile

def load(name, filename):
    p = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

v14 = load("baseline_v14", "v0.14_autoconnect_runtime.py")

root = Path(tempfile.mkdtemp())
(root/"lib").mkdir()
(root/"pubspec.yaml").write_text("name: demo\n")
(root/"lib/main.dart").write_text("class App {}\\nvoid main() {}\\n")

rt = v14.AutoConnectRuntime(
    runtime_kwargs={
        "memory_max_episodes": 100,
        "memory_target_episodes": 50,
        "memory_max_concepts": 200,
    }
)
res = rt.connect(root)
assert res.domain == "code"
assert res.attached
print("BASELINE_PASS")

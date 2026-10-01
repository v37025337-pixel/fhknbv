
from pathlib import Path
import importlib.util, sys, tempfile

def load(name, filename):
    p = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

v14 = load("autodiscovery_v14", "v0.14_autoconnect_runtime.py")

root = Path(__file__).resolve().parent
providers = root / "providers"
assert providers.exists()

# No perception callback and no URL fetcher are passed here.
rt = v14.AutoConnectRuntime(
    provider_dirs=[str(providers)],
    runtime_kwargs={
        "memory_max_episodes": 100,
        "memory_target_episodes": 50,
        "memory_max_concepts": 200,
    },
)

tmp = Path(tempfile.mkdtemp())
img = tmp / "dns.jpg"
img.write_bytes(b"fake image")

r1 = rt.connect(img)
assert r1.domain == "network_config"
assert r1.attached is True

r2 = rt.connect("https://example.org/doc")
assert r2.domain == "document"
assert r2.attached is True
assert r2.details["read_only"] is True

snap = rt.provider_registry.snapshot()
assert "vision.perceive" in snap
assert "web.fetch.readonly" in snap
print("AUTODISCOVERY_PASS")

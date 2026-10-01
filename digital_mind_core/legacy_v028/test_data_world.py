
from pathlib import Path
import importlib.util, sys, tempfile, json

root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("data_test_v14",root/"v0.14_autoconnect_runtime.py")
m=importlib.util.module_from_spec(spec);sys.modules["data_test_v14"]=m;spec.loader.exec_module(m)

tmp=Path(tempfile.mkdtemp())
csvf=tmp/"states.csv"
csvf.write_text(
    "Rank,State,Population\n"
    "1,Alabama,4849377\n"
    "2,Alaska,736732\n"
    "3,Arizona,6731484\n",
    encoding="utf-8",
)
jsonf=tmp/"repo.json"
jsonf.write_text(json.dumps({
    "name":"flask","language":"Python","archived":False
}),encoding="utf-8")

rt=m.AutoConnectRuntime(
    runtime_kwargs={
        "memory_max_episodes":100,
        "memory_target_episodes":50,
        "memory_max_concepts":200,
    }
)

a=rt.connect(csvf)
assert a.domain=="data" and a.attached
fields={f["name"]:f["type"] for f in a.details["schema"]["fields"]}
assert fields["Rank"]=="int"
assert fields["Population"]=="int"
assert a.details["schema"]["row_count"]==3

b=rt.connect(jsonf)
assert b.domain=="structured_data" and b.attached
assert b.details["schema"]["row_count"]==1

c=rt.connect({"alpha":1,"beta":"x"})
assert c.domain=="structured_data" and c.attached

snap=rt.snapshot()
assert snap["data_world"]["datasets"]==3
print("DATA_WORLD_PASS")

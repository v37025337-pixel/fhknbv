"""Repository self-diagnostic for DIGITAL_MIND / UnifiedKernel.

The diagnostic uses only repository/runtime evidence. It does not mutate the
runtime and it does not promote any fix. Findings are ranked from measured
signals so the kernel can compare its own diagnosis with an external audit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable


@dataclass(frozen=True)
class Finding:
    finding_id: str
    severity: float
    category: str
    evidence: tuple[str, ...]
    diagnosis: str
    proposed_check: str

    def document(self):
        return asdict(self)


def _py_files(root: Path):
    return sorted(
        p for p in (root / "digital_mind_core").rglob("*.py")
        if "legacy_v028" not in p.parts and "__pycache__" not in p.parts
    )


def _module_name(root: Path, path: Path):
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _parse(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _internal_imports(tree: ast.AST, current: str):
    out = set()
    package = "digital_mind_core"
    current_parts = current.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == package or alias.name.startswith(package + "."):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = current_parts[:-1]
                up = max(0, node.level - 1)
                if up:
                    base = base[:-up]
                module = node.module.split(".") if node.module else []
                target = ".".join(base + module)
            else:
                target = node.module or ""
            if target == package or target.startswith(package + "."):
                out.add(target)
    return out


def _tarjan(graph: dict[str, set[str]]):
    index = 0
    stack = []
    on_stack = set()
    indices = {}
    low = {}
    sccs = []

    def visit(v):
        nonlocal index
        indices[v] = low[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph.get(v, ()):
            if w not in graph:
                continue
            if w not in indices:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], indices[w])
        if low[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1 or (len(comp) == 1 and v in graph.get(v, ())):
                sccs.append(sorted(comp))

    for v in graph:
        if v not in indices:
            visit(v)
    return sorted(sccs)


def _test_imports(root: Path):
    direct = set()
    test_files = sorted((root / "tests").glob("test_*.py"))
    for path in test_files:
        try:
            tree = _parse(path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("digital_mind_core"):
                        direct.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("digital_mind_core"):
                    direct.add(node.module)
    return test_files, direct


def _covered(module: str, imports: set[str]):
    return any(
        imp == module or imp.startswith(module + ".") or module.startswith(imp + ".")
        for imp in imports
    )


def _function_lengths(tree: ast.AST):
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            rows.append((node.name, int(end - node.lineno + 1)))
    return sorted(rows, key=lambda x: (-x[1], x[0]))


def _workflow_info(root: Path):
    result = []
    for p in sorted((root / ".github" / "workflows").glob("*.yml")):
        text = p.read_text(encoding="utf-8")
        result.append({
            "path": str(p.relative_to(root)),
            "push": bool(re.search(r"(?m)^\s*push\s*:", text)),
            "workflow_run": "workflow_run:" in text,
            "workflow_call": "workflow_call:" in text,
            "needs": bool(re.search(r"(?m)^\s*needs\s*:", text)),
            "artifact_download": "download-artifact" in text,
        })
    return result


def _dependency_status(root: Path):
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    deps = re.findall(r'"([A-Za-z0-9_.-]+\s*[^"]*)"', text)
    relevant = [d for d in deps if d.lower().startswith(("numpy", "scipy"))]
    exact = [d for d in relevant if "==" in d]
    return {"declared": relevant, "exact_pins": exact}


def _run_tests(root: Path):
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
    )
    tail = proc.stdout[-4000:]
    match = re.search(r"Ran\s+(\d+)\s+tests?", proc.stdout)
    return {
        "pass": proc.returncode == 0,
        "test_count": int(match.group(1)) if match else None,
        "returncode": proc.returncode,
        "tail": tail,
    }


def _import_smoke(modules: Iterable[str]):
    failures = []
    imported = 0
    for name in modules:
        if name.endswith(".__main__"):
            continue
        try:
            importlib.import_module(name)
            imported += 1
        except Exception as exc:
            failures.append({"module": name, "error": f"{type(exc).__name__}: {exc}"})
    return {"imported": imported, "failures": failures}


def diagnose(root: Path):
    files = _py_files(root)
    modules = {}
    graph = {}
    oversized = []
    long_functions = []
    syntax_failures = []

    for path in files:
        module = _module_name(root, path)
        if not module:
            continue
        try:
            tree = _parse(path)
        except SyntaxError as exc:
            syntax_failures.append({"path": str(path.relative_to(root)), "error": str(exc)})
            continue
        lines = path.read_text(encoding="utf-8").count("\n") + 1
        functions = _function_lengths(tree)
        modules[module] = {
            "path": str(path.relative_to(root)),
            "lines": lines,
            "functions": len(functions),
            "classes": sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree)),
        }
        graph[module] = _internal_imports(tree, module)
        if lines >= 600:
            oversized.append({"module": module, "lines": lines})
        for name, length in functions:
            if length >= 100:
                long_functions.append({"module": module, "function": name, "lines": length})

    test_files, direct_imports = _test_imports(root)
    untested = [
        {"module": name, "lines": meta["lines"]}
        for name, meta in modules.items()
        if name not in {"digital_mind_core", "digital_mind_core.__main__"}
        and not _covered(name, direct_imports)
    ]
    untested.sort(key=lambda x: (-x["lines"], x["module"]))

    cycles = _tarjan(graph)
    workflows = _workflow_info(root)
    push_workflows = [w for w in workflows if w["push"]]
    autodev_workflows = [w for w in workflows if "autodev_step" in w["path"]]
    chained_autodev = any(
        w["workflow_run"] or w["needs"] or w["artifact_download"]
        for w in autodev_workflows
    )

    development = root / "development"
    autodev_step3_committed = (development / "autodev_step3.json").exists()
    state_path = development / "autodev_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    history = state.get("attempt_history", []) if isinstance(state, dict) else []

    dependency = _dependency_status(root)
    tests = _run_tests(root)
    smoke = _import_smoke(sorted(modules))

    init_text = (root / "digital_mind_core" / "__init__.py").read_text(encoding="utf-8")
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)', init_text)
    package_version = version_match.group(1) if version_match else None
    binding = json.loads((root / "UNIFIED_KERNEL_BINDING.json").read_text(encoding="utf-8"))
    binding_runtime = binding.get("runtime_line")

    findings = []
    def add(fid, sev, cat, evidence, diagnosis, check):
        findings.append(Finding(fid, sev, cat, tuple(evidence), diagnosis, check))

    if not tests["pass"]:
        add(
            "test_suite_failure", 1.0, "correctness",
            [f"unittest return code={tests['returncode']}", tests["tail"][-1200:]],
            "The active regression suite is failing.",
            "Fix the first reproducible failure before any capability promotion.",
        )
    if syntax_failures or smoke["failures"]:
        add(
            "module_integrity_failure", 1.0, "correctness",
            [f"syntax_failures={len(syntax_failures)}", f"import_failures={len(smoke['failures'])}"],
            "One or more active modules cannot be parsed/imported cleanly.",
            "Require zero syntax/import failures in a fresh environment.",
        )
    if len(push_workflows) >= 4:
        add(
            "ci_fanout", 0.88, "operations",
            [f"push-triggered workflows={len(push_workflows)}",
             ", ".join(w["path"] for w in push_workflows)],
            "A normal code push starts several independent workflows, including development stages; this wastes compute and can duplicate experiments.",
            "Make expensive development workflows dispatch/chained and keep one lightweight push gate.",
        )
    if autodev_workflows and not chained_autodev:
        add(
            "autodev_not_end_to_end_chained", 0.96, "autonomy",
            ["autodev workflows are separate", "no workflow_run/needs/download-artifact dependency detected"],
            "The autonomous development stages are not a closed repository-native pipeline; an external host must materialize outputs and start the next stage.",
            "One orchestrator must pass immutable artifacts step1->research->step2->step3 without host choosing or rewriting stage outputs.",
        )
    if not autodev_step3_committed:
        add(
            "autodev_state_incomplete", 0.91, "persistence",
            ["development/autodev_step3.json is absent",
             f"committed attempt_history entries={len(history)}"],
            "The committed development state does not contain the latest experiment verdict, so future target selection can run from stale history.",
            "Atomically persist every completed development verdict before selecting the next deficit.",
        )
    if untested:
        top = untested[:8]
        add(
            "direct_test_gaps", min(0.92, 0.55 + 0.02 * len(untested)), "testing",
            [f"active modules without direct test import={len(untested)}",
             ", ".join(f"{x['module']}({x['lines']}L)" for x in top)],
            "Several active modules have no direct test ownership; regressions may be caught only indirectly or not at all.",
            "Add module-level tests for highest-risk unowned modules, then mutation/fault tests for critical boundaries.",
        )
    if oversized:
        oversized.sort(key=lambda x: -x["lines"])
        add(
            "oversized_modules", min(0.85, 0.55 + 0.05 * len(oversized)), "maintainability",
            [", ".join(f"{x['module']}={x['lines']}L" for x in oversized[:8])],
            "Several modules concentrate too many responsibilities, increasing regression and review cost.",
            "Split only where dependency/behavior tests preserve public contracts; compare complexity before/after.",
        )
    if long_functions:
        add(
            "long_functions", min(0.78, 0.50 + 0.03 * len(long_functions)), "maintainability",
            [", ".join(f"{x['module']}:{x['function']}={x['lines']}L" for x in long_functions[:8])],
            "Long functions create hidden coupling and make self-rewrite validation coarse.",
            "Refactor one high-risk function at a time under characterization tests.",
        )
    if cycles:
        add(
            "import_cycles", min(0.90, 0.60 + 0.08 * len(cycles)), "architecture",
            [json.dumps(cycles[:6])],
            "Active package import cycles reduce modular isolation.",
            "Break cycles at interfaces and verify import graph becomes acyclic.",
        )
    if dependency["declared"] and not dependency["exact_pins"]:
        add(
            "dependency_reproducibility", 0.68, "reproducibility",
            [f"dependencies={dependency['declared']}", "no exact numpy/scipy pins detected"],
            "Fresh CI can silently change numerical behavior when dependency releases move.",
            "Introduce a tested lock/constraints file while retaining declared compatibility ranges.",
        )
    if binding_runtime and package_version and binding_runtime not in package_version:
        add(
            "version_identity_drift", 0.72, "identity",
            [f"package_version={package_version}", f"binding_runtime_line={binding_runtime}"],
            "Repository/package versioning and the external UnifiedKernel runtime identity use different version lines.",
            "Define one canonical build identity containing package version, runtime lineage and git SHA.",
        )

    google_text = (root / "digital_mind_core" / "google_search.py").read_text(encoding="utf-8")
    if "GOOGLE_API_KEY" in google_text and "host_search" in google_text:
        add(
            "host_mediated_search_boundary", 0.58, "integration",
            ["GoogleSearchClient supports native credentials or host callback",
             "GitHub runner has no verified host browser callback by itself"],
            "Public search capability is available through the ChatGPT/TinyFish host, but the standalone Python runtime does not inherit that browser connector.",
            "Expose a signed provider interface/service endpoint or configure official Google credentials in the actual long-lived runtime.",
        )

    findings.sort(key=lambda f: (-f.severity, f.finding_id))
    strengths = {
        "unit_suite_pass": tests["pass"],
        "unit_tests": tests["test_count"],
        "syntax_failures": len(syntax_failures),
        "import_failures": len(smoke["failures"]),
        "active_modules": len(modules),
        "import_cycles": len(cycles),
        "directly_unowned_modules": len(untested),
        "push_workflows": len(push_workflows),
    }

    return {
        "schema": "digital-mind.self-diagnostic.v1",
        "scope": "active repository code; legacy_v028 excluded from architecture scoring",
        "strengths": strengths,
        "findings": [f.document() for f in findings],
        "module_metrics": modules,
        "import_graph": {k: sorted(v) for k, v in graph.items()},
        "import_cycles": cycles,
        "direct_test_imports": sorted(direct_imports),
        "untested_modules": untested,
        "oversized_modules": oversized,
        "long_functions": long_functions,
        "workflows": workflows,
        "dependency_status": dependency,
        "tests": tests,
        "import_smoke": smoke,
        "state": {
            "package_version": package_version,
            "binding_runtime_line": binding_runtime,
            "autodev_step3_committed": autodev_step3_committed,
            "committed_attempt_history": len(history),
        },
        "self_verdict": {
            "top_risk": findings[0].finding_id if findings else None,
            "promotion_safe_now": bool(
                tests["pass"] and not syntax_failures and not smoke["failures"]
                and not any(f.severity >= 0.95 for f in findings)
            ),
            "next_action": findings[0].proposed_check if findings else "No critical repair selected.",
        },
    }

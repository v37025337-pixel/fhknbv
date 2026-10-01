"""Run the uploaded branch's test scripts independently in fresh processes."""
import json
from pathlib import Path
import subprocess
import sys
import time


root = Path(__file__).resolve().parents[1]
legacy = root / 'digital_mind_core/legacy_v028'
rows = []
for path in sorted(legacy.glob('test_*.py')):
    started = time.monotonic()
    try:
        result = subprocess.run([sys.executable, str(path)], cwd=legacy,
                                capture_output=True, text=True, timeout=60)
        row = {'test': path.name, 'exit_code': result.returncode,
               'passed': result.returncode == 0, 'seconds': round(time.monotonic() - started, 3),
               'output': (result.stdout + result.stderr).strip()}
    except subprocess.TimeoutExpired:
        row = {'test': path.name, 'passed': False, 'output': 'timeout after 60 seconds'}
    rows.append(row)
    print(path.name, 'PASS' if row['passed'] else 'FAIL', flush=True)
report = {'passed': sum(r['passed'] for r in rows), 'total': len(rows), 'tests': rows}
(root / 'examples/legacy_test_results.json').write_text(json.dumps(report, indent=2) + '\n')
if not all(r['passed'] for r in rows):
    raise SystemExit(1)

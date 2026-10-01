"""Generate one bounded technical hypothesis from the kernel's own audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from digital_mind_core.hypothesis_genesis import synthesize_hypothesis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assessment = json.loads(args.assessment.read_text(encoding="utf-8"))
    result = synthesize_hypothesis(assessment)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "deficit": result["deficit"]["deficit_id"],
        "mechanisms": [x["mechanism"] for x in result["selected_mechanisms"]],
        "candidate_mode": result["candidate_mode"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

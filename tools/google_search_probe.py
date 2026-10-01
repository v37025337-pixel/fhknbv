"""Probe whether this runtime has an actual Google search transport."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from digital_mind_core.google_search import GoogleSearchClient, GoogleSearchUnavailable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--query", default="UnifiedKernel self research")
    args = parser.parse_args()

    client = GoogleSearchClient()
    status = client.status()
    result = None
    error = None
    if status["status"] == "READY_NATIVE_GOOGLE_API":
        try:
            result = client.search(args.query, 3)
        except GoogleSearchUnavailable as exc:
            error = str(exc)

    report = {
        "schema": "digital-mind.google-search-probe.v1",
        "status": client.status(),
        "query": args.query,
        "search_attempted": result is not None or error is not None,
        "result": result,
        "error": error,
        "html_scraping_fallback": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"]["status"],
        "search_attempted": report["search_attempted"],
        "result_count": (
            0 if result is None else len(result.get("results", []))
        ),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

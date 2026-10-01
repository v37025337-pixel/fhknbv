"""Opt-in smoke test of real HTTP data entering the cognitive kernel.

This is separate from offline CI tests because public websites can be unavailable.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from digital_mind_core import __version__
from digital_mind_core.internet import WebReader
from digital_mind_core.kernel import KernelConfig
from digital_mind_core.persistence import source_fingerprint
from digital_mind_core.tool_module import MindModule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('examples/internet_live_report.json'))
    parser.add_argument('--memory', type=Path, default=Path('examples/web_memory.json'))
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.memory.parent.mkdir(parents=True, exist_ok=True)
    module = MindModule(KernelConfig(l0_learning=False), web_reader=WebReader(), web_memory_path=args.memory)
    results = []
    for url in ('https://example.com/', 'https://www.python.org/',
                'https://api.github.com/repos/v37025337-pixel/fhknbv'):
        response = module.execute({'op': 'fetch_url', 'url': url})
        source = response['source']
        assert response['kernel']['inferences'][0]['value']
        assert source['http_status'] == 200 and source['text']
        results.append(response)
        print(json.dumps({'url': source['url'], 'http_status': source['http_status'],
                          'title': source['title'], 'bytes_received': source['bytes_received'],
                          'kernel_cycle': response['kernel']['cycle']}), flush=True)
    github_id = results[-1]['source']['id']
    logic = module.execute({'op': 'reason', 'web_source_id': github_id, 'task': {
        'queries': [['web_field', github_id, 'full_name', 'v37025337-pixel/fhknbv']]}})
    assert logic['inferences'][0]['value']
    restarted = MindModule(KernelConfig(l0_learning=False), web_reader=WebReader(), web_memory_path=args.memory)
    restored = restarted.execute({'op': 'web_memory', 'source_id': github_id})
    assert restored['source']['content_sha256'] == results[-1]['source']['content_sha256']
    assert restarted.execute({'op': 'internet_status'})['successful_fetches'] == 0
    document = {'version': __version__, 'source_sha256': source_fingerprint(),
                'real_http_requests': len(results), 'all_observed_by_kernel': True,
                'saved_source_reused_for_reasoning': True, 'web_memory_restored_without_network': True,
                'internet_status': module.internet_status(), 'fetches': results,
                'reasoning': {k: logic.get(k) for k in ('cycle', 'action', 'inferences')},
                'scope': 'real HTTP text and structured JSON observations; no LLM or free-form understanding'}
    args.report.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

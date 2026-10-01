"""A callable Python module and a stateful JSON-lines interface to the kernel.

Each instance owns one UnifiedMind. Its output is evidence from the experimental
kernel; a caller decides how to use it. This interface is not an LLM or MCP server.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from . import __version__
from .kernel import KernelConfig, UnifiedMind
from .google_search import GoogleSearchClient


MAX_REQUEST_CHARS = 1_000_000
MAX_STEPS_PER_CALL = 1024


def _json_result(value):
    # Legacy planning traces use infinity for an absent/unreachable path.
    # JSON represents an unavailable numeric result with null instead.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_result(item) for item in value]
    return value


class MindModule:
    def __init__(self, config=None, *, google_client=None):
        self.mind = UnifiedMind(config if config is not None else KernelConfig())
        self.google = google_client if google_client is not None else GoogleSearchClient()

    def execute(self, request):
        if not isinstance(request, dict):
            raise ValueError('request must be a JSON object')
        operation = request.get('op')
        fields = {
            'status': {'op', 'id'},
            'simulate': {'op', 'id', 'steps'},
            'reason': {'op', 'id', 'task', 'include_state'},
            'run_mechanism': {'op', 'id', 'name', 'inputs'},
            'google_status': {'op', 'id'},
            'google_search': {'op', 'id', 'query', 'num'},
        }
        if not isinstance(operation, str) or operation not in fields:
            raise ValueError('op must be status, simulate, reason, run_mechanism, google_status or google_search')
        if set(request) - fields[operation]:
            raise ValueError('unknown request fields')
        identifier = request.get('id')
        if identifier is not None and type(identifier) not in (str, int):
            raise ValueError('id must be a string, integer or null')
        if operation == 'status':
            return self.mind.report()
        if operation == 'simulate':
            steps = request.get('steps', 96)
            if type(steps) is not int or not 1 <= steps <= MAX_STEPS_PER_CALL:
                raise ValueError(f'steps must be an integer from 1 to {MAX_STEPS_PER_CALL}')
            self.mind.run(steps)
            return self.mind.report()
        if operation == 'reason':
            task = request.get('task')
            if not isinstance(task, dict):
                raise ValueError('task must be a JSON object')
            include_state = request.get('include_state', False)
            if type(include_state) is not bool:
                raise ValueError('include_state must be boolean')
            result = self.mind.process(task)
            if not include_state:
                result.pop('state', None)
            return _json_result(result)
        if operation == 'google_status':
            return self.google.status()
        if operation == 'google_search':
            query = request.get('query')
            num = request.get('num', 5)
            return self.google.search(query, num)
        name, inputs = request.get('name'), request.get('inputs')
        if not isinstance(name, str) or not isinstance(inputs, dict):
            raise ValueError('name must be a string and inputs must be a JSON object')
        if not all(isinstance(key, str) for key in inputs):
            raise ValueError('input names must be strings')
        return {'name': name, 'value': self.mind.run_mechanism(name, **inputs),
                'execution': self.mind.cognition.mechanism_info(name).get('execution')}


def _reject_nonfinite(value):
    raise ValueError(f'non-finite JSON number: {value}')


def serve(module, input_stream, output_stream):
    """One JSON response per nonempty request line, preserving instance state."""
    while True:
        raw = input_stream.readline(MAX_REQUEST_CHARS + 1)
        if not raw:
            return
        identifier = None
        try:
            if len(raw) > MAX_REQUEST_CHARS:
                while raw and not raw.endswith('\n'):
                    raw = input_stream.readline(MAX_REQUEST_CHARS + 1)
                raise ValueError('request exceeds the character limit')
            if not raw.strip():
                continue
            request = json.loads(raw, parse_constant=_reject_nonfinite)
            if isinstance(request, dict) and type(request.get('id')) in (str, int):
                identifier = request['id']
            result = module.execute(request)
            response = {'id': identifier, 'ok': True, 'result': result}
            encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
        except Exception as exc:
            response = {'id': identifier, 'ok': False,
                        'error': {'type': type(exc).__name__, 'message': str(exc)}}
            encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
        output_stream.write(encoded + '\n')
        output_stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Call DIGITAL_MIND as a JSON-lines module')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--version', action='version', version=__version__)
    args = parser.parse_args(argv)
    try:
        config = KernelConfig(seed=args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    serve(MindModule(config), sys.stdin, sys.stdout)


if __name__ == '__main__':
    main()

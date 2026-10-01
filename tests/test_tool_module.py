import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

from digital_mind_core.kernel import KernelConfig
from digital_mind_core.tool_module import MAX_REQUEST_CHARS, MindModule, serve


class ToolModuleTests(unittest.TestCase):
    def setUp(self):
        self.module = MindModule(KernelConfig(l0_learning=False))

    def test_calls_share_one_world_instance(self):
        first = self.module.execute({'op': 'simulate', 'steps': 2})
        second = self.module.execute({'op': 'simulate', 'steps': 3})
        self.assertEqual(first['steps'], 2)
        self.assertEqual(second['steps'], 5)
        self.assertEqual(second['cognition']['cycles'], 5)
        self.assertEqual(second['l0']['executive_ticks'], 5)
        self.assertEqual(self.module.execute({'op': 'status'})['steps'], 5)

    def test_logic_uses_persistent_state_across_requests(self):
        self.module.execute({'op': 'reason', 'task': {
            'facts': [['p', 'a']], 'rules': [[[['p', '$x']], ['q', '$x']]],
            'persist_facts': True, 'persist_rules': True}})
        result = self.module.execute({'op': 'reason', 'task': {
            'queries': [['q', 'a']], 'persist_facts': True, 'persist_rules': True}})
        self.assertTrue(result['inferences'][0]['value'])
        self.assertNotIn('state', result)

    def test_mechanism_uses_l0(self):
        result = self.module.execute({'op': 'run_mechanism', 'name': 'stability_switch',
                                     'inputs': {'history': 10., 'current': 12.}})
        self.assertEqual(result['value'], 12.)
        self.assertEqual(result['execution'], 'L0-Control-IR')

    def test_invalid_requests_do_not_advance_world(self):
        for request in ({'op': 'simulate', 'steps': True}, {'op': 'simulate', 'steps': 0},
                        {'op': 'simulate', 'steps': 1025}, {'op': 'status', 'extra': 1},
                        {'op': 'reason', 'task': []},
                        {'op': 'reason', 'task': {}, 'include_state': 'yes'}, {'op': 'unknown'}, []):
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.module.execute(request)
        self.assertEqual(self.module.mind.fast_steps, 0)

    def test_stream_recovers_after_error_and_preserves_request_ids(self):
        inputs = io.StringIO('\nnot json\n{"id":"bad","op":"simulate","steps":false}\n'
                            '{"id":3,"op":"simulate","steps":2}\n{"id":4,"op":"status"}\n')
        output = io.StringIO()
        serve(self.module, inputs, output)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r['ok'] for r in rows], [False, False, True, True])
        self.assertEqual([r['id'] for r in rows], [None, 'bad', 3, 4])
        self.assertEqual(rows[-1]['result']['steps'], 2)

    def test_oversized_request_is_drained_before_next_request(self):
        inputs = io.StringIO('x' * (MAX_REQUEST_CHARS + 10) + '\n{"op":"status"}\n')
        output = io.StringIO()
        serve(self.module, inputs, output)
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertFalse(rows[0]['ok'])
        self.assertTrue(rows[1]['ok'])

    def test_nonfinite_json_is_rejected(self):
        inputs = io.StringIO('{"op":"run_mechanism","name":"stability_switch",'
                            '"inputs":{"history":NaN,"current":2}}\n')
        output = io.StringIO()
        serve(self.module, inputs, output)
        self.assertFalse(json.loads(output.getvalue())['ok'])
        self.assertEqual(self.module.mind.cognition.mechanism_calls, 0)

    def test_real_cli_accepts_multiple_requests(self):
        root = Path(__file__).resolve().parents[1]
        requests = [{'op': 'simulate', 'steps': 2}, {'op': 'status'},
                    {'op': 'reason', 'task': {'facts': [['p', 'a']], 'queries': [['p', 'a']]}}]
        result = subprocess.run([sys.executable, '-m', 'digital_mind_core.tool_module'],
                                input=''.join(json.dumps(row) + '\n' for row in requests),
                                cwd=root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r['ok'] for r in rows))
        self.assertEqual(rows[1]['result']['steps'], 2)
        self.assertTrue(rows[2]['result']['inferences'][0]['value'])

    def test_logic_result_is_strict_json_without_changing_legacy_trace(self):
        result = self.module.execute({'op': 'reason', 'include_state': True, 'task': {
            'facts': [['p', 'a']], 'queries': [['p', 'a']]}})
        json.dumps(result, allow_nan=False)
        self.assertTrue(result['inferences'][0]['value'])
        unavailable = [entry for entry in result['state']['trace']
                       if isinstance(entry.get('payload'), dict) and 'cost' in entry['payload']]
        self.assertTrue(unavailable)
        self.assertTrue(all(entry['payload']['cost'] is None for entry in unavailable))


if __name__ == '__main__':
    unittest.main()

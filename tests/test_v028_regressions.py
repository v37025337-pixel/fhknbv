import importlib.util
from pathlib import Path
import sys
import unittest


def load(name, filename):
    root = Path(__file__).resolve().parents[1] / 'digital_mind_core/legacy_v028'
    spec = importlib.util.spec_from_file_location(name, root / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load('unified_regression_core', 'v0.18_cognitive_core.py')
genesis = load('unified_regression_genesis', 'v0.28_mechanism_genesis.py')


class V028RegressionTests(unittest.TestCase):
    def test_failed_cegis_is_not_reported_found(self):
        c = core.CognitiveCore()
        result = c.synthesize_mechanism(
            {'name': 'unvalidated', 'args': ['x'],
             'examples': [{'inputs': {'x': 0}, 'output': 0}]},
            validation_examples=[{'inputs': {'x': 1}, 'output': 7}], max_rounds=1)
        self.assertEqual(result['status'], 'NO_MECHANISM_FOUND')
        self.assertNotIn('unvalidated', c.mechanism_info())

    def test_persistent_rule_snapshot_cannot_change_live_rule(self):
        c = core.CognitiveCore()
        result = c.process({'rules': [([('p', '$x')], ('q', '$x'))], 'persist_rules': True})
        result['state']['persistent_rules'][0][0].append(('tampered', '$x'))
        later = c.process({'facts': [('p', 'a')], 'queries': [('q', 'a')], 'persist_rules': True})
        self.assertTrue(later['inferences'][0]['value'])

    def test_user_rules_are_detached_on_ingress(self):
        c = core.CognitiveCore()
        premises = [('p', '$x')]
        c.process({'rules': [(premises, ('q', '$x'))], 'persist_rules': True})
        premises.append(('tampered', '$x'))
        later = c.process({'facts': [('p', 'a')], 'queries': [('q', 'a')], 'persist_rules': True})
        self.assertTrue(later['inferences'][0]['value'])

    def test_other_public_result_fields_are_detached(self):
        c = core.CognitiveCore()
        result = c.process({'facts': [('p', 'a')], 'queries': [('p', 'a')]})
        result['inferences'][0]['value'] = False
        self.assertTrue(c.state['inferences'][0]['value'])
        trace = c.reason_trace()
        trace[0]['stage'] = 'tampered'
        self.assertEqual(c.reason_trace()[0]['stage'], 'observation')

    def test_repeated_variable_keeps_none_binding(self):
        logic = core.RelationalLogic()
        logic.add_fact('pair', None, 'different')
        logic.add_rule([('pair', '$x', '$x')], ('equal_pair', '$x'))
        logic.infer()
        self.assertFalse(logic.query('equal_pair', 'different'))

    def test_exported_equality_agrees_with_interpreter(self):
        E = genesis.Expr
        expr = E('if', (E('eq', (E('var', ('x',), 1), E('var', ('y',), 1)), 3),
                        E('const', (1,), 1), E('const', (0,), 1)), 6)
        source = genesis.MechanismGenesis._function_source('comparison', ['x', 'y'], expr)
        namespace = {}
        exec(compile(source, '<generated>', 'exec'), namespace)
        for x, y in ((0., 5e-10), (0., 2e-9), (4., 4.)):
            self.assertEqual(namespace['comparison'](x, y), genesis.evaluate(expr, {'x': x, 'y': y}))

    def test_exported_integer_arithmetic_agrees_with_float_interpreter(self):
        E = genesis.Expr
        expr = E('sub', (E('var', ('x',), 1), E('const', (1,), 1)), 3)
        source = genesis.MechanismGenesis._function_source('subtract', ['x'], expr)
        namespace = {}
        exec(compile(source, '<generated>', 'exec'), namespace)
        x = 2 ** 53 + 1
        self.assertEqual(namespace['subtract'](x), genesis.evaluate(expr, {'x': x}))


if __name__ == '__main__':
    unittest.main()

import copy
import json
from pathlib import Path
import tempfile
import unittest

from digital_mind_core.cognition import CognitiveCore
from digital_mind_core.evolution import MechanismEvolution
from digital_mind_core.kernel import KernelConfig, UnifiedMind


OPTIONS = {'max_numeric_cost': 3, 'max_boolean_cost': 3,
           'max_numeric_candidates': 256, 'max_boolean_candidates': 64}


def minimum_spec(name='minimum'):
    return {'name': name, 'args': ['history_cost', 'current_cost'], 'constants': [0, 1],
            'examples': [{'inputs': {'history_cost': a, 'current_cost': b}, 'output': min(a, b)}
                         for a, b in ((1., 3.), (4., 2.), (-2., 7.), (8., -1.))]}


class CognitionTests(unittest.TestCase):
    def test_legacy_mechanism_is_executed_by_l0(self):
        core = CognitiveCore()
        for h, c, expected in ((10., 10.2, 10.), (10., 12., 12.), (0., -1., 0.),
                               (0., -1.01, -1.01)):
            self.assertEqual(core.run_mechanism('stability_switch', history=h, current=c), expected)
        self.assertEqual(core.mechanism_info('stability_switch')['execution'], 'L0-Control-IR')
        self.assertEqual(core.mechanism_calls, 4)

    def test_rejected_holdout_is_not_registered(self):
        core = CognitiveCore()
        result = core.synthesize_mechanism(minimum_spec(), engine_options=OPTIONS,
            holdout_examples=[{'inputs': {'history_cost': 5., 'current_cost': 3.}, 'output': 7.},
                              {'inputs': {'history_cost': 0., 'current_cost': 4.}, 'output': 9.}])
        self.assertEqual(result['status'], 'REJECTED')
        self.assertNotIn('minimum', core.mechanism_info())

    def test_generated_mechanism_survives_new_core_instance(self):
        first = CognitiveCore()
        result = first.synthesize_mechanism(minimum_spec(), engine_options=OPTIONS,
            holdout_examples=[{'inputs': {'history_cost': 5., 'current_cost': 3.}, 'output': 3.},
                              {'inputs': {'history_cost': 0., 'current_cost': 4.}, 'output': 0.}])
        self.assertEqual(result['status'], 'FOUND')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mechanisms.json'
            first.save_mechanisms(path)
            restored = CognitiveCore()
            restored.load_mechanisms(path)
            self.assertEqual(restored.run_mechanism('minimum', history_cost=100., current_cost=-9.), -9.)
            self.assertEqual(restored.mechanism_info('minimum')['admission'], 'holdout_passed')

    def test_corrupt_store_cannot_partially_publish(self):
        source = CognitiveCore()
        source.synthesize_mechanism(minimum_spec('aaa'), engine_options=OPTIONS)
        source.synthesize_mechanism(minimum_spec('bbb'), engine_options=OPTIONS)
        data = source.mechanism_document()
        data['mechanisms'][1]['expression']['op'] = 'exec'
        restored = CognitiveCore()
        before = restored.mechanism_document()
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            restored.restore_mechanism_document(data)
        self.assertEqual(restored.mechanism_document(), before)

    def test_future_proposal_uses_synthesized_rule_and_changes_choice(self):
        core = CognitiveCore()
        evolution = MechanismEvolution(core)
        for i in range(16):
            h, c = (i + 1., i + 3.) if i % 2 else (i + 4., i + 2.)
            evolution.choose(h, c)
        self.assertFalse(evolution.active)
        self.assertTrue(evolution.choose(2., 9.))
        self.assertTrue(evolution.active)
        self.assertEqual(evolution.hold_selections, 1)
        self.assertEqual(evolution.model_cost_saved, 7.)
        self.assertEqual(evolution.calls, 1)

    def test_runtime_counterexample_rolls_back_underfit_rule(self):
        core = CognitiveCore()
        spec = {'name': 'choose_plan_cost', 'args': ['history_cost', 'current_cost'],
                'examples': [{'inputs': {'history_cost': 3., 'current_cost': 1.}, 'output': 1.},
                             {'inputs': {'history_cost': 4., 'current_cost': 2.}, 'output': 2.}]}
        result = core.synthesize_mechanism(spec, engine_options=OPTIONS, holdout_examples=[
            {'inputs': {'history_cost': 5., 'current_cost': 2.}, 'output': 2.},
            {'inputs': {'history_cost': 9., 'current_cost': 3.}, 'output': 3.}])
        self.assertEqual(result['status'], 'FOUND')
        evolution = MechanismEvolution(core)
        evolution.adopt_loaded()
        self.assertFalse(evolution.choose(1., 5.))
        self.assertFalse(evolution.active)
        self.assertEqual(evolution.events[-1]['event'], 'rolled_back')

    def test_world_cycle_reaches_all_three_branches(self):
        mind = UnifiedMind(KernelConfig(seed=1, l0_learning=False))
        mind.run(96)
        self.assertEqual(mind.cognition.state['cycle'], 96)
        self.assertEqual(mind.executive_ticks, 96)
        self.assertGreater(mind.evolution.calls, 0)
        self.assertEqual(mind.cognition.mechanism_info('choose_plan_cost')['execution'], 'L0-Control-IR')
        result = mind.process({'facts': [('p', 'a')],
                               'rules': [([('p', '$x')], ('q', '$x'))],
                               'queries': [('q', 'a')]})
        self.assertTrue(result['inferences'][0]['value'])

    def test_logic_request_reaches_l0_replanning(self):
        mind = UnifiedMind(KernelConfig(l0_learning=False, mechanism_evolution=False))
        mind.model.ensemble_losses.extend([.001] * 31 + [.1])
        decision = mind._executive_decision(41)
        self.assertTrue(decision['replan'])
        self.assertTrue(mind.cognition.state['inferences'][0]['value'])

    def test_legacy_data_adapter_feeds_shared_logic(self):
        mind = UnifiedMind(KernelConfig(l0_learning=False))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'states.csv'
            path.write_text('Rank,State,Population\n1,Alabama,4849377\n2,Alaska,736732\n')
            connection = mind.connect(path)
            self.assertEqual(connection.domain, 'data')
            self.assertTrue(connection.attached)
            result = mind.process({'persist_facts': True,
                'queries': [('dataset_field', 'connection_1', 'Population', 'int'),
                            ('dataset_rows', 'connection_1', 2)]})
            self.assertTrue(all(q['value'] for q in result['inferences']))


    def test_evolution_has_no_arbitrary_three_attempt_ceiling(self):
        class AlwaysRejectCore:
            def synthesize_mechanism(self, *args, **kwargs):
                return {'status': 'NO_MECHANISM_FOUND',
                        'source': None, 'explanation': 'deliberate rejection'}

        evolution = MechanismEvolution(AlwaysRejectCore())
        # _advance() runs before the current sample is appended.  Enough
        # observations should therefore drive more than the former 3 attempts.
        for i in range(81):
            evolution.choose(float(i + 3), float(i + 1))
        self.assertGreater(evolution.attempts, 3)
        self.assertIsNone(evolution.report()['attempt_limit'])



if __name__ == '__main__':
    unittest.main()

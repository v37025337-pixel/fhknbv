"""Bounded autonomous mechanism development from the agent's own planning trace.

The specification is explicit: select the lower of two model-predicted rollout
costs. Training examples come from actual planning decisions; swapped examples
encode the required symmetry. A later fixed holdout is never fed to the search.
This learns a local software arbiter, not a physical law or arbitrary algorithm.
"""
from collections import deque
import copy
import math


class MechanismEvolution:
    name = 'choose_plan_cost'

    def __init__(self, core, enabled=True):
        self.core = core
        self.enabled = enabled
        self.samples = deque(maxlen=96)
        self.events = []
        self.active = False
        self.attempts = 0
        self.calls = 0
        self.hold_selections = 0
        self.model_cost_saved = 0.
        self._last_attempt_samples = 0
        self.observations = 0

    def adopt_loaded(self):
        info = self.core.mechanism_info().get(self.name)
        self.active = bool(self.enabled and info and info.get('admission') == 'holdout_passed')
        if self.active:
            self.events.append({'event': 'restored', 'name': self.name})

    def _advance(self):
        if not self.enabled or self.active or len(self.samples) < 16:
            return
        if self.attempts and self.observations - self._last_attempt_samples < 16:
            return
        samples = list(self.samples)
        train = copy.deepcopy(samples[:6])
        # Symmetry belongs to the specified operation; these are labelled as
        # transformed training examples, not unseen evaluation data.
        train += [{'inputs': {'history_cost': x['inputs']['current_cost'],
                              'current_cost': x['inputs']['history_cost']},
                   'output': x['output']} for x in samples[:6]]
        validation = copy.deepcopy(samples[6:10])
        holdout = copy.deepcopy(samples[10:16])
        self.attempts += 1
        self._last_attempt_samples = self.observations
        result = self.core.synthesize_mechanism(
            {'name': self.name, 'args': ['history_cost', 'current_cost'],
             'constants': [0, 1], 'examples': train},
            validation_examples=validation, holdout_examples=holdout, max_rounds=4,
            engine_options={'max_numeric_cost': 3, 'max_boolean_cost': 3,
                            'max_numeric_candidates': 256, 'max_boolean_candidates': 64})
        self.active = result['status'] == 'FOUND' and result.get('admission') == 'holdout_passed'
        self.events.append({'event': 'admitted' if self.active else 'rejected',
                            'name': self.name, 'status': result['status'],
                            'source': result.get('source'), 'training_count': len(train),
                            'validation_count': len(validation), 'holdout_count': len(holdout),
                            'explanation': result.get('explanation')})

    def choose(self, history_cost, current_cost):
        if not math.isfinite(history_cost) or not math.isfinite(current_cost):
            raise ValueError('rollout costs must be finite')
        if not self.enabled:
            return False  # Original behavior: execute the newly proposed plan.
        # Synthesis sees only PREVIOUS proposals, not the current decision.
        self._advance()
        expected = min(history_cost, current_cost)
        self.samples.append({'inputs': {'history_cost': float(history_cost),
                                         'current_cost': float(current_cost)},
                             'output': float(expected)})
        self.observations += 1
        if not self.active:
            return False
        try:
            selected = self.core.run_mechanism(self.name, history_cost=float(history_cost),
                                               current_cost=float(current_cost))
            self.calls += 1
            # A known invariant guards a generated software policy. Its benefit
            # in the real world is evaluated separately, not inferred from this.
            if selected != expected:
                raise ValueError('generated arbiter violated minimum-cost invariant')
        except (ValueError, RuntimeError, TypeError, OverflowError) as exc:
            self.active = False
            self.core.mechanism_registry[self.name]['admission'] = 'rolled_back'
            self.events.append({'event': 'rolled_back', 'reason': str(exc)})
            return False
        hold = history_cost < current_cost
        if hold:
            self.hold_selections += 1
            self.model_cost_saved += current_cost - history_cost
        return hold

    def report(self):
        return {'enabled': self.enabled, 'active': self.active, 'attempts': self.attempts,
                'planning_examples': self.observations, 'l0_calls': self.calls,
                'attempt_limit': None,
                'hold_selections': self.hold_selections,
                'model_predicted_cost_saved': self.model_cost_saved,
                'label_source': 'frozen world-model rollout costs; not real-world counterfactual outcomes',
                'events': copy.deepcopy(self.events)}

    def state_document(self):
        return {**self.report(), 'samples': list(self.samples),
                'last_attempt_samples': self._last_attempt_samples}

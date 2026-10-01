"""Choose predictive hypotheses using errors of forecasts made before outcomes.

The hypothesis family is supplied by the developer. Authority is earned from
future observed transitions; search/planning predictions do not earn evidence.
The rolling block score is a conservative policy heuristic, not a significance
test or a guarantee of improvement on an arbitrary future distribution.
"""
from collections import Counter, deque
import copy

import numpy as np


class ProspectiveSelection:
    window = 64
    recent_window = 8

    def __init__(self, candidates):
        candidates = tuple(candidates)
        if (not candidates or len(set(candidates)) != len(candidates)
                or any(not isinstance(name, str) or name == 'baseline' for name in candidates)):
            raise ValueError('candidate names must be unique strings other than baseline')
        self.gains = {name: deque(maxlen=self.window) for name in candidates}
        self.active = 'baseline'
        self.observations = 0
        self.used_steps = Counter({'baseline': 0, **{name: 0 for name in candidates}})
        self.loss_sums = Counter({'baseline': 0., 'selected': 0.})
        self.events = deque(maxlen=128)

    def scores(self):
        scores = {}
        for name, history in self.gains.items():
            values = np.asarray(history, float)
            ready = len(values) == self.window
            if ready:
                blocks = values.reshape(4, 16).mean(axis=1)
                conservative = float(blocks.mean() - 2. * blocks.std(ddof=1) / np.sqrt(4))
            else:
                conservative = None
            recent = float(values[-self.recent_window:].mean()) if len(values) else None
            scores[name] = {'future_forecasts': len(values),
                            'mean_paired_gain': float(values.mean()) if len(values) else None,
                            'conservative_gain': conservative, 'recent_gain': recent,
                            'eligible': bool(ready and conservative > 1e-8 and recent > 0.)}
        return scores

    def observe(self, target, predictions, selected):
        if set(predictions) != {'baseline', *self.gains} or selected != self.active:
            raise ValueError('forecasts must cover the frozen baseline and every candidate')
        target = np.asarray(target, float)
        if target.ndim != 1 or not target.size or not np.all(np.isfinite(target)):
            raise ValueError('target must be a finite nonempty vector')
        losses = {}
        for name, prediction in predictions.items():
            prediction = np.asarray(prediction, float)
            if prediction.shape != target.shape or not np.all(np.isfinite(prediction)):
                raise ValueError('forecast must be finite and match the target shape')
            with np.errstate(over='ignore', invalid='ignore'):
                loss = float(np.mean((target - prediction) ** 2))
            if not np.isfinite(loss):
                raise ValueError('forecast loss must be finite')
            losses[name] = loss
        # Every loss was checked before any evidence or authority is changed.
        self.observations += 1
        self.used_steps[selected] += 1
        self.loss_sums['baseline'] += losses['baseline']
        self.loss_sums['selected'] += losses[selected]
        for name, history in self.gains.items():
            history.append(losses['baseline'] - losses[name])
        scores = self.scores()
        eligible = [name for name, score in scores.items() if score['eligible']]
        active = max(eligible, key=lambda name: scores[name]['conservative_gain']) if eligible else 'baseline'
        if active != self.active:
            event = ('returned_to_baseline' if active == 'baseline' else
                     'candidate_admitted' if self.active == 'baseline' else 'candidate_changed')
            self.events.append({'event': event, 'after_observation': self.observations,
                                'previous': self.active, 'next': active,
                                'evidence': copy.deepcopy(scores.get(active))})
            self.active = active

    def report(self):
        count = self.observations
        return {'enabled': True, 'active_model': self.active, 'observations': count,
                'minimum_future_forecasts': self.window,
                'used_steps': dict(self.used_steps), 'scores': self.scores(),
                'baseline_mse_on_same_transitions': self.loss_sums['baseline'] / count if count else None,
                'selected_mse_on_same_transitions': self.loss_sums['selected'] / count if count else None,
                'events': copy.deepcopy(list(self.events)),
                'evidence_kind': 'prequential paired errors; rolling score is a policy heuristic'}

    def state_document(self):
        return {**self.report(), 'gain_history': {name: list(values) for name, values in self.gains.items()},
                'loss_sums': dict(self.loss_sums)}

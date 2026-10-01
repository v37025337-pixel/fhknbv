"""One closed cognitive loop, with real ARC decisions and L0 online learning.

The RLS/fast model remains the baseline. L0 learns its vector innovation using
frozen predictions and committed observations. Its influence is estimated only
from earlier errors. Causal experiments and predictive explanation are recorded
separately: neither is evidence of phenomenal consciousness.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from . import __version__, l0
from .components import DigitalMindV07, MultiTimescalePlanner, PredictiveModel
from .cognition import CognitiveCore
from .evolution import MechanismEvolution
from .predictive_self import PredictiveSelfModel


@dataclass(frozen=True)
class KernelConfig:
    seed: int = 0
    horizon: int = 7
    beam: int = 6
    control_period: int = 4
    l0_learning: bool = True
    questions: bool = True
    experiment_samples: int = 48
    action_budget: float = 1.2
    experiment_fraction: float = 0.25
    cognition: bool = True
    mechanism_evolution: bool = True

    def __post_init__(self):
        for name in ('seed', 'horizon', 'beam', 'control_period', 'experiment_samples'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.seed < 0:
            raise ValueError('seed must be nonnegative')
        for name in ('l0_learning', 'questions', 'cognition', 'mechanism_evolution'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if self.horizon < 1 or self.beam < 1 or self.control_period < 1:
            raise ValueError('horizon, beam and control_period must be positive')
        if self.experiment_samples < 24:
            raise ValueError('experiment_samples must be at least 24')
        if not np.isfinite(self.action_budget) or not 0 < self.action_budget <= 1.2:
            raise ValueError('action_budget must be in (0, 1.2]')
        if not np.isfinite(self.experiment_fraction) or not 0 <= self.experiment_fraction <= 1:
            raise ValueError('experiment_fraction must be in [0, 1]')


def innovation_source(d_action, d_obs, projected=4):
    names = ([f'p{i}' for i in range(projected)]
             + [f'a{i}' for i in range(d_action)]
             + ['delay0', 'delay1'])
    return '\n'.join([
        'program learned_innovation', 'semantics hybrid',
        *(f'input {name}' for name in names),
        f'target residual : vector[{d_obs}]',
        'cell innovation', 'observe innovation',
        f'model world_innovation ({", ".join(names)}) -> innovation {{',
        '    learn residual', '    dimension adaptive', '    output_dimension adaptive',
        '    geometry digital', '    metric causal', '    law quadratic',
        '    birth residual', '    probation future', '    death weak',
        '    time frozen', '}', '',
    ])


class L0PredictiveModel(PredictiveModel):
    """Hybrid innovation model with a shrinkage weight from past observations."""
    def __init__(self, d_obs, d_action, seed, enabled=True):
        super().__init__(d_obs, d_action, seed)
        self.enabled = enabled
        projection = np.random.default_rng(seed + 100).normal(size=(d_obs, 4))
        self.projection = np.linalg.qr(projection)[0].T
        self.source = innovation_source(d_action, d_obs)
        self.compiled = l0.compile_source(self.source)
        self.runtime = self.compiled.new_hybrid_runtime(
            training_horizon=80, digital_geometry_kwargs={'max_history': 96, 'kd_workers': 1})
        self.cross_ema = 0.
        self.norm_ema = 0.
        self.baseline_losses = []
        self.ensemble_losses = []
        self.weights = []
        self.shadow_benefits = deque(maxlen=80)
        # Read-only observability for external shadow experiments. These fields
        # expose the already-computed transition pieces and do not affect policy.
        self.last_baseline_error = None
        self.last_correction = None
        self.last_runtime_error = None
        self.last_runtime_weight = 0.0
        self.last_shadow_weight = 0.0

    @property
    def l0_weight(self):
        if not self.enabled or len(self.baseline_losses) < 64:
            return 0.
        if not self.runtime.models['world_innovation'].kernel.field.relations:
            return 0.
        if len(self.shadow_benefits) < 64:
            return 0.
        blocks = np.asarray(list(self.shadow_benefits)[-64:]).reshape(4, 16).mean(axis=1)
        conservative_gain = blocks.mean() - 2. * blocks.std(ddof=1) / np.sqrt(4)
        if conservative_gain <= 0:
            return 0.
        return self.shadow_weight

    @property
    def shadow_weight(self):
        return float(np.clip(self.cross_ema / (self.norm_ema + 1e-12), 0., 0.5))

    def state_for(self, obs, action, prev3=None):
        projected = self.projection @ np.asarray(obs, float)
        delayed = self.prev_actions[-3] if prev3 is None else prev3
        return dict(zip(
            [f'p{i}' for i in range(4)] + [f'a{i}' for i in range(self.d_action)]
            + ['delay0', 'delay1'],
            [*projected, *np.asarray(action, float), *np.asarray(delayed, float)[:2]],
        ))

    def predict(self, obs, action, prev1=None, prev2=None, prev3=None):
        baseline = super().predict(obs, action, prev1, prev2, prev3)
        if self.l0_weight == 0:
            return baseline
        state = self.state_for(obs, action, prev3)
        # Planning reads the frozen learner directly, without committing history.
        correction = np.asarray(self.runtime.predict_model('world_innovation', state), float)
        return baseline + self.l0_weight * correction

    def update(self, obs, action, next_obs):
        phi = self.features(obs, action)
        baseline = super().predict(obs, action)
        weight = self.l0_weight  # Freeze before seeing this transition's target.
        shadow_weight = self.shadow_weight
        correction = np.zeros(self.d_obs)
        if self.enabled:
            result = self.runtime.step(self.state_for(obs, action),
                                       targets={'residual': next_obs - baseline})
            correction = np.asarray(result.predictions['world_innovation'], float)
        pred = baseline + weight * correction
        err = np.asarray(next_obs) - pred
        baseline_err = next_obs - baseline
        shadow_err = baseline_err - shadow_weight * correction
        self.last_baseline_error = np.asarray(baseline_err, float).copy()
        self.last_correction = np.asarray(correction, float).copy()
        self.last_runtime_error = np.asarray(err, float).copy()
        self.last_runtime_weight = float(weight)
        self.last_shadow_weight = float(shadow_weight)
        self.shadow_benefits.append(float(np.mean(baseline_err ** 2) - np.mean(shadow_err ** 2)))
        super().update(obs, action, next_obs)
        self.cross_ema = 0.98 * self.cross_ema + 0.02 * float(np.mean(baseline_err * correction))
        self.norm_ema = 0.98 * self.norm_ema + 0.02 * float(np.mean(correction ** 2))
        self.baseline_losses.append(float(np.mean(baseline_err ** 2)))
        self.ensemble_losses.append(float(np.mean(err ** 2)))
        self.weights.append(weight)
        return pred, err, phi


@dataclass
class Intervention:
    qid: int
    action_dim: int
    target_direction: np.ndarray
    started_t: int
    samples: list[tuple[int, float]] = field(default_factory=list)
    status: str = 'collecting'
    effect: float | None = None
    confidence_interval: tuple[float, float] | None = None

    def observe(self, sign, outcome, required):
        if self.status != 'collecting':
            return
        self.samples.append((int(sign), float(outcome)))
        # One fixed look: do not stop repeatedly on a favorable p-value.
        if len(self.samples) < required:
            return
        groups = [np.array([v for s, v in self.samples if s == sign])
                  for sign in (-1, 1)]
        if min(map(len, groups)) < 8:
            self.status = 'inconclusive'
            return
        neg, pos = groups
        self.effect = float(pos.mean() - neg.mean())
        variances = [float(g.var(ddof=1) / len(g)) for g in groups]
        se2 = sum(variances)
        if se2 <= 1e-18:
            margin = 0.
        else:
            df = se2 ** 2 / sum(v ** 2 / (len(g) - 1) for v, g in zip(variances, groups))
            margin = float(student_t.ppf(.975, df) * np.sqrt(se2))
        self.confidence_interval = self.effect - margin, self.effect + margin
        self.status = ('supported' if self.confidence_interval[0] > 0 or
                       self.confidence_interval[1] < 0 else 'inconclusive')


class UnifiedMind(DigitalMindV07):
    def __init__(self, config=None):
        self.config = config or KernelConfig()
        super().__init__(self.config.seed)
        self.model = L0PredictiveModel(self.d_obs, self.d_action,
                                       self.config.seed + 1, self.config.l0_learning)
        self.questions_enabled = self.config.questions
        self.questions.feature_names = tuple(self.model.feature_names)
        self.executive = l0.compile_file(Path(__file__).with_name('executive.l0'))
        self.executive_ticks = 0
        self.executive_statuses = Counter()
        self.experiments = []
        self.experiment = None
        self.experiment_rng = np.random.default_rng(self.config.seed + 900)
        self.probe_steps = 0
        self.pending_treatment = None
        self.consolidated_self = {}
        self.trace = deque(maxlen=256)
        self._decision = None
        self._step_errors = []
        self._step_rewards = []
        self.plan_ticks = 0
        self.planned_goal_id = None
        self.cognition = CognitiveCore()
        self.evolution = MechanismEvolution(self.cognition, self.config.mechanism_evolution)
        self.predictive_self = PredictiveSelfModel()
        self._initial_mechanisms = None
        self._external_tasks = 0
        self._connections = 0

    def process(self, task):
        """Use the same symbolic state for explicit logical/planning tasks."""
        result = self.cognition.process(task)
        self._external_tasks += 1
        return result

    def _self_state_vector(self):
        recent_loss = self._step_errors[-1] if self._step_errors else 0.0
        unresolved = sum(not q.resolved for q in self.questions.questions)
        active_goal = self.goals.active_leaf()
        confidence = (
            self.cognition.self_model.estimate('world_prediction')
            if self.config.cognition else 0.5
        )
        return np.asarray([
            np.clip(float(np.mean(self.self_trace)), 0.0, 1.0),
            np.clip(float(np.mean(self.competence)), 0.0, 1.0),
            np.clip(float(np.mean(self.narrative.social_trust)), 0.0, 1.0),
            np.clip(float(confidence), 0.0, 1.0),
            np.clip(float(np.tanh(max(0.0, recent_loss) / 0.05)), 0.0, 1.0),
            np.clip(float(unresolved) / 8.0, 0.0, 1.0),
            np.clip(float(active_goal.depth) / 3.0, 0.0, 1.0),
            np.clip(float(self.model.shadow_weight) / 0.5, 0.0, 1.0),
            np.clip(float(self.model.fast_gate), 0.0, 1.0),
        ], dtype=float)

    def _self_decision_vector(self):
        decision = self._decision or {}
        return np.asarray([
            float(bool(decision.get('replan', False))),
            float(bool(decision.get('reflect', False))),
            float(bool(decision.get('consolidate', False))),
            float(bool(decision.get('allow_probe', False))),
            float(bool(decision.get('macro_reflect', False))),
            np.clip(float(decision.get('exploration', 0.0)), 0.0, 1.0),
            np.clip(float(decision.get('budget', 0.0)) / 1.2, 0.0, 1.0),
            float(self.pending_treatment is not None),
        ], dtype=float)

    def predict_self_counterfactual(self, **decision_overrides):
        """Model-based self counterfactual; this is not a causal claim."""
        if self._decision is None:
            raise RuntimeError('run at least one internal decision first')
        return self.predictive_self.counterfactual(
            self._self_state_vector(),
            self._self_decision_vector(),
            **decision_overrides,
        )

    def connect(self, target, **kwargs):
        result = self.cognition.connect(target, **kwargs)
        self._connections += 1
        namespace = f'connection_{self._connections}'
        schema = result.details.get('schema', {})
        facts = [('source_domain', namespace, result.domain)]
        if result.attached and schema:
            facts.append(('dataset_rows', namespace, schema['row_count']))
            facts.extend(('dataset_field', namespace, field['name'], field['type'])
                         for field in schema.get('fields', []))
        self.process({'observations': [{'source': namespace, 'domain': result.domain,
                                        'attached': result.attached, 'schema': schema}],
                      'facts': facts, 'persist_facts': True,
                      'context': 'external:' + result.domain})
        return result

    def save_mechanisms(self, path):
        self.cognition.save_mechanisms(path)

    def load_mechanisms(self, path):
        if self.fast_steps:
            raise ValueError('load mechanisms before running; use replay to resume an existing agent')
        self.cognition.load_mechanisms(path)
        self._initial_mechanisms = self.cognition.mechanism_document()
        self.evolution.adopt_loaded()

    def _symbolic_decision(self, t, surprise):
        if not self.config.cognition:
            return False
        selected = self.goals.active_leaf()
        facts = [('observed',), ('goal', selected.gid)]
        if t > 40 and surprise > 2.4:
            facts.append(('prediction_surprise',))
        if self.current_goal.status != 'active':
            facts.append(('goal_terminal',))
        task = {'context': 'world', 'observations': self.obs.tolist(), 'facts': facts,
                'rules': [([('prediction_surprise',)], ('request_replan',)),
                          ([('goal_terminal',)], ('request_replan',))],
                'queries': [('request_replan',)],
                'start': self.goals.root.gid, 'goal': selected.gid,
                'graph': {gid: [(child, 1.) for child in children]
                          for gid, children in self.goals.children.items()}}
        result = self.cognition.process(task)
        return bool(result['inferences'][0]['value'])

    def _executive_decision(self, t):
        losses = self.model.ensemble_losses
        surprise = (losses[-1] / (np.mean(losses[-32:]) + 1e-9)) if losses else 0.
        symbolic_replan = self._symbolic_decision(t, surprise)
        trust = float(self.narrative.social_trust.mean())
        if self.config.cognition and self.cognition.self_model._stats.get('world_prediction'):
            trust = .8 * trust + .2 * self.cognition.self_model.estimate('world_prediction')
        result = self.executive.execute_arc({
            'surprise': float(min(surprise, 5.)),
            'trust': trust,
            'symbolic_replan': symbolic_replan,
            'unresolved': sum(not q.resolved for q in self.questions.questions) +
                          int(bool(self.experiment and self.experiment.status == 'collecting')),
            # Period flags are supplied as integer phase counters.
            'episode_phase': (t + 1) % self.episode_len,
            'macro_phase': t % self.macro_period,
            'control_phase': t % self.config.control_period,
            'slow_phase': (t + 1) % self.slow_period,
            'action_budget': self.config.action_budget,
        })
        if result.status not in {'STABLE_EXACT', 'STABLE_TOLERANCE'}:
            raise RuntimeError(f'L0 executive failed: {result.status}')
        self.executive_ticks += 1
        self.executive_statuses[result.status] += 1
        return result.state

    def _start_experiment(self, t):
        if not self.config.questions or (self.experiment and self.experiment.status == 'collecting'):
            return
        used = {e.qid for e in self.experiments}
        candidates = [q for q in self.questions.questions
                      if not q.resolved and q.qid not in used and t - q.created_t < 128]
        if not candidates:
            return
        q = candidates[-1]
        action_start = 1 + self.d_obs
        indices = [i - action_start for i in q.predictor_indices
                   if action_start <= i < action_start + self.d_action]
        if indices:
            dim = indices[0]
        else:
            effects = np.abs(q.target_direction @ self.model.action_sensitivity())
            dim = int(np.argmax(effects))
        self.experiment = Intervention(q.qid, dim, q.target_direction.copy(), t)
        self.experiments.append(self.experiment)

    def _choose_action(self, t):
        self.goals.refresh(t, self.obs, {q.qid for q in self.questions.questions if q.resolved})
        decision = self._executive_decision(t)
        self._decision = decision
        selected = self.goals.active_leaf()
        if decision['macro_reflect']:
            self.macro_ticks += 1
        if decision['replan'] or selected.gid != self.planned_goal_id:
            self.current_goal = selected
            previous = self.macro_action.copy()
            self.macro_action = self.planner.plan(
                self.obs, self.model, selected, self.config.horizon, self.config.beam,
                action_limit=decision['budget'])
            history_cost = self._rollout_cost([previous] * self.config.horizon, selected)
            current_cost = -self.planner.last_score
            if self.evolution.choose(history_cost, current_cost):
                self.macro_action = previous
            self.plan_ticks += 1
            self.planned_goal_id = selected.gid
        action = self.macro_action.copy()
        action += self.rng.normal(scale=decision['exploration'], size=self.d_action)
        self._start_experiment(t)
        self.pending_treatment = None
        budget_available = self.probe_steps < (t + 1) * self.config.experiment_fraction
        if (budget_available and decision['allow_probe'] and self.experiment
                and self.experiment.status == 'collecting'):
            sign = int(self.experiment_rng.choice((-1, 1)))
            dim = self.experiment.action_dim
            action[dim] = sign * min(0.9, decision['budget'])
            neutral = action.copy()
            neutral[dim] = 0.
            neutral_outcome = float(self.model.predict(self.obs, neutral) @ self.experiment.target_direction)
            self.pending_treatment = sign, neutral_outcome
            self.probe_steps += 1
        return np.clip(action, -decision['budget'], decision['budget'])

    def _rollout_cost(self, actions, goal):
        obs = self.obs.copy()
        history = [a.copy() for a in reversed(self.model.prev_actions)]
        cost = 0.
        for depth, action in enumerate(actions):
            obs = self.model.predict(obs, action, *history)
            cost += .98 ** depth * (float(np.mean((obs - goal.target) ** 2))
                                   + .025 * float(action @ action))
            history = [action.copy(), history[0], history[1]]
        return cost

    def _update_self(self):
        # Zero sensitivity must provide zero ownership evidence.
        sensitivity = self.model.action_sensitivity()
        strength = np.linalg.norm(sensitivity, axis=1)
        evidence = strength / (strength + .05)
        self.self_trace += .04 * (evidence - self.self_trace)
        per_action = np.linalg.norm(sensitivity, axis=0)
        self.max_sensitivity = np.maximum(self.max_sensitivity, per_action)
        loss = np.mean(self.model.ensemble_losses[-32:]) if self.model.ensemble_losses else 0.
        reliability = 1. / (1. + loss / .02)
        self.competence = np.clip(per_action / (self.max_sensitivity + 1e-9) * reliability, 0., 1.)

    def _consolidate(self):
        total = max(1, sum(self.narrative.theme_counts.values()))
        self.consolidated_self = {
            't': self.fast_steps,
            'theme_weights': {str(k): v / total for k, v in self.narrative.theme_counts.items()},
            'social_trust': self.narrative.social_trust.tolist(),
            'competence': self.competence.tolist(),
            'goal_outcomes': dict(Counter(g.status for g in self.goals.goals.values() if g.depth >= 2)),
            'operational_prediction_confidence': self.cognition.self_model.estimate('world_prediction'),
            'generated_mechanisms': sorted(self.cognition._mechanism_objects),
        }
        # Learned predictability feeds the structural social goal's priority.
        self.goals.social.priority = float(.3 + .4 * self.narrative.social_trust.mean())

    def step(self, t=None):
        if t is None:
            t = self.fast_steps
        if t != self.fast_steps:
            raise ValueError('step index must match the persistent clock')
        action = self._choose_action(t)
        self_state_before = self._self_state_vector()
        self_decision = self._self_decision_vector()
        ws = self.world.step(action)
        _, err, phi = self.model.update(self.obs, action, ws.obs)
        self.questions.observe(err, phi, t)
        self._update_self()
        self.ep_actions.append(action.copy())
        self.ep_errors.append(float(np.mean(err ** 2)))
        self.ep_rewards.append(ws.reward_components.copy())
        self.ep_social.append(ws.social.copy())
        self.obs = ws.obs.copy()
        self.last_world = ws
        self.fast_steps += 1
        if t < self.world.regime_change:
            if t > 60:
                self.loss_pre.append(float(np.mean(err ** 2)))
        else:
            self.loss_post.append(float(np.mean(err ** 2)))
        if self._decision['reflect']:
            self._close_episode(t)
        loss = self.model.ensemble_losses[-1]
        self._step_errors.append(loss)
        if self.config.cognition:
            past = self._step_errors[max(0, len(self._step_errors) - 33):-1]
            if past:
                self.cognition.self_model.observe('world_prediction', loss <= 1.3 * float(np.mean(past)))
        self._step_rewards.append(ws.reward_components.tolist())
        if self.pending_treatment:
            sign, neutral_outcome = self.pending_treatment
            outcome = float(ws.obs @ self.experiment.target_direction) - neutral_outcome
            self.experiment.observe(sign, outcome, self.config.experiment_samples)
        if self._decision['consolidate']:
            self.slow_ticks += 1
            self._consolidate()
        self.predictive_self.observe(
            self_state_before,
            self_decision,
            self._self_state_vector(),
        )
        self.trace.append({'t': t, 'loss': loss, 'goal': self.current_goal.gid,
                           'l0_weight': self.model.weights[-1],
                           'experiment': self.experiment.status if self.experiment else None})
        return ws

    def run(self, T=640):
        if not isinstance(T, int) or T < 0:
            raise ValueError('T must be a nonnegative integer')
        return [self.step() for _ in range(T)]

    def report(self):
        losses = np.asarray(self._step_errors)
        split = self.world.regime_change
        def average(a):
            return float(np.mean(a)) if len(a) else None
        experiments = [
            {'question': e.qid, 'action_dim': e.action_dim, 'status': e.status,
             'samples': len(e.samples), 'effect': e.effect, 'ci95': e.confidence_interval}
            for e in self.experiments]
        lr = self.model.runtime.model_report()['world_innovation']
        # Keep the CLI report readable; full geometry is accessible through runtime.
        lr['digital_geometry'].pop('operator', None)
        return {
            'version': __version__, 'seed': self.config.seed, 'steps': self.fast_steps,
            'cognition': {'cycles': self.cognition.state['cycle'],
                          'logic_queries': len(self.cognition.state.get('inferences', [])),
                          'prediction_confidence': self.cognition.self_model.estimate('world_prediction'),
                          'trace_events': len(self.cognition.state['trace']),
                          'mechanisms': self.cognition.mechanism_info(),
                          'generated_l0_calls': self.cognition.mechanism_calls},
            'mechanism_evolution': self.evolution.report(),
            'predictive_self': self.predictive_self.report(),
            'l0': {'version': l0.VERSION, 'executive_ticks': self.executive_ticks,
                   'executive_statuses': dict(self.executive_statuses),
                   'learner': lr, 'mean_innovation_weight': average(self.model.weights)},
            'prediction': {
                'mean_loss': average(losses),
                'baseline_loss_on_same_transitions': average(self.model.baseline_losses),
                'pre_change_last80': average(losses[max(0, split - 80):split]),
                'post_change_first40': average(losses[split:split + 40]),
                'post_change_final80': average(losses[max(split, len(losses) - 80):]),
                'change_events': self.model.change_events,
            },
            'reward_components_mean': np.mean(self._step_rewards, axis=0).tolist() if self._step_rewards else None,
            'questions': {'created': len(self.questions.questions),
                          'predictively_explained': sum(q.resolved for q in self.questions.questions),
                          'examples': [q.text for q in self.questions.questions[:3]]},
            'interventions': experiments,
            'intervention_steps': self.probe_steps,
            'goals': dict(Counter(g.status for g in self.goals.goals.values() if g.depth >= 2)),
            'memory': {'episodes': len(self.narrative.nodes), 'edges': len(self.narrative.edges),
                       'themes': len(self.narrative.theme_centroids)},
            'clocks': {'fast': self.fast_steps, 'meso': self.meso_ticks,
                       'macro': self.macro_ticks, 'plans': self.plan_ticks, 'slow': self.slow_ticks},
            'consolidated_self': self.consolidated_self,
        }

    def write_report(self, path):
        Path(path).write_text(json.dumps(self.report(), ensure_ascii=False,
                                        indent=2, allow_nan=False) + '\n', encoding='utf-8')


def planning_benchmark(steps=16):
    """Compare actual controllers over the SAME executed horizon and rewards."""
    def transition(state, a):
        resource, buffer = state
        return .84 * resource + .72 * buffer, .72 * buffer + a
    def utility(state, a, nxt):
        return state[0] - .30 * a
    terminal = lambda state: 1.6 * state[0]
    results = {}
    for horizon in (1, 8):
        state = (0., 0.)
        total = 0.
        actions = []
        for _ in range(steps):
            _, seq = MultiTimescalePlanner.search(
                state, (0., 1.), transition, utility,
                horizon=horizon, beam=6, discount=1., terminal=terminal)
            action = seq[0]
            nxt = transition(state, action)
            total += utility(state, action, nxt)
            state = nxt
            actions.append(action)
        results[str(horizon)] = {'steps': steps, 'score': total + terminal(state),
                                 'actions': actions}
    return results

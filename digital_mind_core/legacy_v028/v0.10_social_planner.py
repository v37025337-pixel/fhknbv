
"""
v0.10_social_planner.py

Social planning bridge:
OtherModel -> predicted response -> own action selection.

The planner evaluates several horizons and stops when deeper look-ahead
no longer changes the preferred action and normalized action values
have stabilized.

Expected environment API:
    n_self_actions
    n_other_actions
    state_key(state)
    context_key(state, self_action)
    other_action_features(state, self_action)
    transition(state, self_action, other_action)
    reward(state, self_action, other_action, next_state)
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np


@dataclass(slots=True)
class PlanResult:
    action: int
    horizon: int
    q_values: np.ndarray
    action_changed_with_depth: bool
    social_uncertainty: float
    unknown_probability: float


class MultiTimescalePlanner:
    def __init__(
        self,
        other_model,
        *,
        gamma: float = 0.92,
        horizons=(1, 2, 3),
        value_tolerance: float = 0.055,
        unknown_risk_penalty: float = 0.30,
    ):
        hs = tuple(sorted(set(int(h) for h in horizons)))
        if not hs or hs[0] < 1:
            raise ValueError("horizons must contain positive integers")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0,1]")
        self.other_model = other_model
        self.gamma = float(gamma)
        self.horizons = hs
        self.value_tolerance = float(value_tolerance)
        self.unknown_risk_penalty = float(unknown_risk_penalty)

    def _discount_norm(self, h: int) -> float:
        if self.gamma == 1.0:
            return float(h)
        return (1.0 - self.gamma**h) / (1.0 - self.gamma)

    @staticmethod
    def _std3(x: np.ndarray) -> float:
        # Faster than dispatching np.std for the tiny branch arrays used here.
        m = float((x[0] + x[1] + x[2]) / 3.0)
        return float(
            ((x[0]-m)**2 + (x[1]-m)**2 + (x[2]-m)**2) / 3.0
        ) ** 0.5

    def plan(self, state, env) -> PlanResult:
        pred_cache: Dict[Tuple[Any, int], Any] = {}
        q_cache: Dict[Tuple[Any, int], np.ndarray] = {}
        v_cache: Dict[Tuple[Any, int], float] = {}

        def prediction(s, a_self):
            key = (env.state_key(s), int(a_self))
            if key not in pred_cache:
                pred_cache[key] = self.other_model.predict(
                    env.other_action_features(s, a_self),
                    context_key=env.context_key(s, a_self),
                )
            return pred_cache[key]

        def q_values(s, depth):
            key = (env.state_key(s), int(depth))
            if key in q_cache:
                return q_cache[key]

            out = np.empty(env.n_self_actions, dtype=np.float64)
            for a_self in range(env.n_self_actions):
                pred = prediction(s, a_self)
                p = pred.action_prob
                ret = np.empty(env.n_other_actions, dtype=np.float64)

                for a_other in range(env.n_other_actions):
                    ns = env.transition(s, a_self, a_other)
                    r = env.reward(s, a_self, a_other, ns)
                    if depth > 1:
                        vk = (env.state_key(ns), depth - 1)
                        if vk not in v_cache:
                            v_cache[vk] = float(np.max(q_values(ns, depth - 1)))
                        r += self.gamma * v_cache[vk]
                    ret[a_other] = r

                q = float(p[0]*ret[0] + p[1]*ret[1] + p[2]*ret[2])
                if self.unknown_risk_penalty:
                    q -= (
                        self.unknown_risk_penalty
                        * float(getattr(pred, "unknown_prob", 0.0))
                        * self._std3(ret)
                    )
                out[a_self] = q

            q_cache[key] = out
            return out

        probe = prediction(state, 0)
        prev_action = None
        prev_norm = None
        changed = False
        chosen_h = self.horizons[-1]
        chosen_q = None

        for h in self.horizons:
            q = q_values(state, h)
            action = int(np.argmax(q))

            # Compare *relative action preference*, not the common future-value
            # offset that grows naturally with horizon.  This makes the stop
            # test answer the real question: does deeper look-ahead still alter
            # the decision boundary?
            # Common future-value offsets do not matter for choosing an action.
            # Compare centered Q-vectors directly. If deeper look-ahead only adds
            # the same continuation value to every action, preference is unchanged.
            preference = q - float(np.mean(q))

            if prev_action is not None:
                changed |= action != prev_action
                delta = float(np.max(np.abs(preference - prev_norm)))
                if action == prev_action and delta <= self.value_tolerance:
                    chosen_h, chosen_q = h, q
                    break

            prev_action, prev_norm = action, preference
            chosen_h, chosen_q = h, q

        return PlanResult(
            action=int(np.argmax(chosen_q)),
            horizon=chosen_h,
            q_values=chosen_q.copy(),
            action_changed_with_depth=changed,
            social_uncertainty=float(getattr(probe, "uncertainty", 0.0)),
            unknown_probability=float(getattr(probe, "unknown_prob", 0.0)),
        )


class UniformOtherModel:
    """Baseline: the other actor is not modeled, responses are uniform."""
    def __init__(self, n_other_actions: int):
        self.n = int(n_other_actions)

    class _Prediction:
        pass

    def predict(self, action_features, context_key=None):
        x = self._Prediction()
        x.action_prob = np.full(self.n, 1.0 / self.n)
        x.goal_prob = np.array([], dtype=float)
        x.unknown_prob = 1.0
        x.uncertainty = 1.0
        return x

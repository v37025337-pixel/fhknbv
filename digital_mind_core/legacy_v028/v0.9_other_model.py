
"""
v0.9_other_model.py

Open-set online model of another agent.

Tracks:
- posterior over explicit latent goal hypotheses
- an explicit UNKNOWN-goal state (prevents false certainty)
- reliability/trust as a Beta posterior
- per-action capability as Beta posteriors
- normalized social-state uncertainty
- context-specific empirical residuals

The "goal" states are predictive hypotheses supplied by the host system.
They are not claims about human consciousness or privileged access to intent.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Hashable
import math
import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-15)


@dataclass(slots=True)
class Prediction:
    action_prob: np.ndarray
    goal_prob: np.ndarray
    unknown_prob: float
    uncertainty: float
    trust_mean: float
    capability_mean: np.ndarray


class OtherModel:
    def __init__(
        self,
        goal_vectors: np.ndarray,
        n_actions: int,
        *,
        rationality: float = 4.0,
        hazard: float = 0.02,
        unknown_prior: float = 0.10,
        empirical_strength: float = 0.20,
        empirical_prior: float = 0.5,
    ):
        goals = np.asarray(goal_vectors, dtype=np.float64)
        if goals.ndim != 2:
            raise ValueError("goal_vectors must have shape [n_goals, n_features]")
        if n_actions < 2:
            raise ValueError("n_actions must be >= 2")
        if not (0.0 <= hazard < 1.0):
            raise ValueError("hazard must be in [0,1)")
        if not (0.0 < unknown_prior < 1.0):
            raise ValueError("unknown_prior must be in (0,1)")

        self.goal_vectors = goals
        self.n_goals, self.n_features = goals.shape
        self.n_actions = int(n_actions)
        self.rationality = float(rationality)
        self.hazard = float(hazard)
        self.unknown_prior = float(unknown_prior)
        self.empirical_strength = float(empirical_strength)
        self.empirical_prior = float(empirical_prior)

        self.base_state_prior = np.empty(self.n_goals + 1, dtype=np.float64)
        self.base_state_prior[:-1] = (1.0 - self.unknown_prior) / self.n_goals
        self.base_state_prior[-1] = self.unknown_prior
        self.state_prob = self.base_state_prior.copy()

        self.trust_alpha = 1.0
        self.trust_beta = 1.0
        self.cap_alpha = np.ones(self.n_actions, dtype=np.float64)
        self.cap_beta = np.ones(self.n_actions, dtype=np.float64)
        self.context_counts: Dict[Hashable, np.ndarray] = {}
        self.observations = 0

    @property
    def goal_prob(self) -> np.ndarray:
        # Absolute mass on each known goal. Sum may be < 1 because UNKNOWN has mass.
        return self.state_prob[:-1]

    @property
    def unknown_prob(self) -> float:
        return float(self.state_prob[-1])

    def _goal_policy(self, action_features: np.ndarray) -> np.ndarray:
        af = np.asarray(action_features, dtype=np.float64)
        expected = (self.n_actions, self.n_features)
        if af.shape != expected:
            raise ValueError(f"action_features must have shape {expected}, got {af.shape}")
        utilities = self.goal_vectors @ af.T
        return np.vstack([_softmax(self.rationality * u) for u in utilities])

    def predict(
        self,
        action_features: np.ndarray,
        *,
        context_key: Optional[Hashable] = None,
    ) -> Prediction:
        policies = self._goal_policy(action_features)
        unknown_policy = np.full(self.n_actions, 1.0 / self.n_actions)

        model_prob = self.state_prob[:-1] @ policies
        model_prob += self.state_prob[-1] * unknown_policy

        if context_key is not None and context_key in self.context_counts:
            counts = self.context_counts[context_key]
            empirical = counts / counts.sum()
            evidence = max(0.0, counts.sum() - self.n_actions * self.empirical_prior)
            weight = self.empirical_strength * (1.0 - math.exp(-evidence / 12.0))
            action_prob = (1.0 - weight) * model_prob + weight * empirical
        else:
            action_prob = model_prob

        action_prob /= action_prob.sum() + 1e-15

        h = -float(np.sum(self.state_prob * np.log(self.state_prob + 1e-15)))
        uncertainty = h / math.log(self.n_goals + 1)

        return Prediction(
            action_prob=action_prob.copy(),
            goal_prob=self.state_prob[:-1].copy(),
            unknown_prob=float(self.state_prob[-1]),
            uncertainty=float(uncertainty),
            trust_mean=self.trust_mean,
            capability_mean=self.capability_mean.copy(),
        )

    def observe(
        self,
        action_features: np.ndarray,
        chosen_action: int,
        *,
        context_key: Optional[Hashable] = None,
        promised_action: Optional[int] = None,
        success: Optional[bool] = None,
    ) -> None:
        a = int(chosen_action)
        if not (0 <= a < self.n_actions):
            raise ValueError("chosen_action out of range")

        policies = self._goal_policy(action_features)

        # Goal can change. A small transition toward the original open-set prior
        # prevents irreversible certainty and enables recovery after switches.
        prior = (1.0 - self.hazard) * self.state_prob + self.hazard * self.base_state_prior

        likelihood = np.empty(self.n_goals + 1, dtype=np.float64)
        likelihood[:-1] = np.maximum(policies[:, a], 1e-12)
        likelihood[-1] = 1.0 / self.n_actions  # conservative UNKNOWN model

        post = prior * likelihood
        self.state_prob = post / (post.sum() + 1e-15)

        if context_key is not None:
            counts = self.context_counts.setdefault(
                context_key,
                np.full(self.n_actions, self.empirical_prior, dtype=np.float64),
            )
            counts[a] += 1.0

        if promised_action is not None:
            p = int(promised_action)
            if not (0 <= p < self.n_actions):
                raise ValueError("promised_action out of range")
            if a == p:
                self.trust_alpha += 1.0
            else:
                self.trust_beta += 1.0

        if success is not None:
            if bool(success):
                self.cap_alpha[a] += 1.0
            else:
                self.cap_beta[a] += 1.0

        self.observations += 1

    @property
    def trust_mean(self) -> float:
        return float(self.trust_alpha / (self.trust_alpha + self.trust_beta))

    @property
    def capability_mean(self) -> np.ndarray:
        return self.cap_alpha / (self.cap_alpha + self.cap_beta)

    def snapshot(self) -> dict:
        zero_features = np.zeros((self.n_actions, self.n_features), dtype=np.float64)
        p = self.predict(zero_features)
        return {
            "observations": self.observations,
            "goal_prob": p.goal_prob.tolist(),
            "unknown_prob": p.unknown_prob,
            "uncertainty": p.uncertainty,
            "trust_mean": self.trust_mean,
            "capability_mean": self.capability_mean.tolist(),
            "contexts_seen": len(self.context_counts),
        }

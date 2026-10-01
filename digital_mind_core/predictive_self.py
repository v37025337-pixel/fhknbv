"""Operational predictive self-model.

This module models the kernel's own measurable state transition:

    self_state_t + internal_decision_t -> predicted self_state_{t+1}

It is deliberately operational.  It does not claim subjective awareness.
Counterfactuals are model predictions only and are not treated as causal facts.
"""
from __future__ import annotations

from collections import deque
import copy
import math

import numpy as np


class PredictiveSelfModel:
    STATE_NAMES = (
        "causal_boundary",
        "competence",
        "social_trust",
        "prediction_confidence",
        "recent_loss",
        "unresolved_questions",
        "active_goal_depth",
        "l0_shadow_authority",
        "fast_adapter_gate",
    )
    DECISION_NAMES = (
        "replan",
        "reflect",
        "consolidate",
        "allow_probe",
        "macro_reflect",
        "exploration",
        "budget",
        "probe_used",
    )

    def __init__(self, *, ridge=6.0, forgetting=0.997, history=512):
        self.state_dim = len(self.STATE_NAMES)
        self.decision_dim = len(self.DECISION_NAMES)
        self.feature_dim = 1 + self.state_dim + self.decision_dim
        self.W = np.zeros((self.state_dim, self.feature_dim), dtype=float)
        self.P = np.eye(self.feature_dim, dtype=float) * float(ridge)
        self.forgetting = float(forgetting)
        if not 0.9 < self.forgetting <= 1.0:
            raise ValueError("forgetting must be in (0.9, 1]")
        self.observations = 0
        self.prediction_mse = deque(maxlen=int(history))
        self.persistence_mse = deque(maxlen=int(history))
        self.transitions = deque(maxlen=int(history))
        self.counterfactual_calls = 0

    @staticmethod
    def _vector(values, size, label):
        vector = np.asarray(values, dtype=float)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} must be a finite vector of length {size}")
        return vector

    def features(self, state, decision):
        state = self._vector(state, self.state_dim, "state")
        decision = self._vector(decision, self.decision_dim, "decision")
        return np.concatenate(([1.0], state, decision))

    def predict_next(self, state, decision):
        state = self._vector(state, self.state_dim, "state")
        phi = self.features(state, decision)
        delta = self.W @ phi
        return np.clip(state + delta, 0.0, 1.0)

    def observe(self, state, decision, next_state):
        state = self._vector(state, self.state_dim, "state")
        next_state = self._vector(next_state, self.state_dim, "next_state")
        phi = self.features(state, decision)

        predicted = np.clip(state + self.W @ phi, 0.0, 1.0)
        persistence = state.copy()
        target_delta = next_state - state
        pred_error = next_state - predicted

        pphi = self.P @ phi
        denom = self.forgetting + float(phi @ pphi)
        gain = pphi / max(denom, 1e-12)
        delta_error = target_delta - self.W @ phi
        self.W += np.outer(delta_error, gain)
        self.P = (self.P - np.outer(gain, phi @ self.P)) / self.forgetting
        self.P = 0.5 * (self.P + self.P.T)

        pred_mse = float(np.mean(pred_error ** 2))
        persist_mse = float(np.mean((next_state - persistence) ** 2))
        self.prediction_mse.append(pred_mse)
        self.persistence_mse.append(persist_mse)
        self.observations += 1

        self.transitions.append({
            "prediction_mse": pred_mse,
            "persistence_mse": persist_mse,
            "improvement": persist_mse - pred_mse,
        })
        return {
            "predicted": predicted,
            "actual": next_state.copy(),
            "prediction_mse": pred_mse,
            "persistence_mse": persist_mse,
        }

    def counterfactual(self, state, decision, **overrides):
        state = self._vector(state, self.state_dim, "state")
        decision = self._vector(decision, self.decision_dim, "decision").copy()
        index = {name: i for i, name in enumerate(self.DECISION_NAMES)}
        for name, value in overrides.items():
            if name not in index:
                raise ValueError(f"unknown decision field: {name}")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("counterfactual values must be finite")
            decision[index[name]] = value
        self.counterfactual_calls += 1
        predicted = self.predict_next(state, decision)
        return {
            name: float(predicted[i])
            for i, name in enumerate(self.STATE_NAMES)
        }

    def report(self, *, warmup=128, window=256):
        pred = list(self.prediction_mse)
        base = list(self.persistence_mse)
        start = min(int(warmup), len(pred))
        pred_eval = pred[start:][-int(window):]
        base_eval = base[start:][-int(window):]
        pred_mean = float(np.mean(pred_eval)) if pred_eval else None
        base_mean = float(np.mean(base_eval)) if base_eval else None
        improvement = None
        if pred_mean is not None and base_mean is not None:
            improvement = float(base_mean - pred_mean)
        ratio = None
        if pred_mean is not None and base_mean is not None and base_mean > 1e-15:
            ratio = float(pred_mean / base_mean)
        return {
            "mode": "SHADOW",
            "observations": self.observations,
            "state_names": list(self.STATE_NAMES),
            "decision_names": list(self.DECISION_NAMES),
            "evaluation_count": len(pred_eval),
            "prediction_mse": pred_mean,
            "persistence_baseline_mse": base_mean,
            "absolute_improvement": improvement,
            "mse_ratio_vs_persistence": ratio,
            "counterfactual_calls": self.counterfactual_calls,
            "causal_claim": False,
        }

    def state_document(self):
        return {
            "report": self.report(),
            "W": self.W.tolist(),
            "P": self.P.tolist(),
            "transitions": copy.deepcopy(list(self.transitions)),
        }

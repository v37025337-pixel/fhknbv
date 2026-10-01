"""Shadow-only drift-aware L0 correction router.

This candidate is not wired into production policy.  It implements the first
self-selected research hypothesis using only pre-update context and past losses.
"""
from __future__ import annotations

import numpy as np


class DriftAwareCorrectionRouter:
    def __init__(
        self,
        *,
        drift_threshold=0.5,
        decay=0.98,
        warmup=16,
        margin_ratio=0.0,
        scales=(0.0, 0.25, 0.5, 1.0),
    ):
        self.drift_threshold = float(drift_threshold)
        self.decay = float(decay)
        self.warmup = int(warmup)
        self.margin_ratio = float(margin_ratio)
        self.scales = tuple(float(x) for x in scales)
        if not 0.0 <= self.drift_threshold <= 1.0:
            raise ValueError("drift_threshold must be in [0,1]")
        if not 0.0 < self.decay < 1.0:
            raise ValueError("decay must be in (0,1)")
        if self.warmup < 1:
            raise ValueError("warmup must be positive")
        if self.margin_ratio < 0.0:
            raise ValueError("margin_ratio must be nonnegative")
        if not self.scales or self.scales[0] != 0.0:
            raise ValueError("first scale must be zero for fail-closed routing")
        if any(x < 0.0 or x > 1.0 for x in self.scales):
            raise ValueError("scales must be in [0,1]")

        n = len(self.scales)
        self.loss = {
            0: np.zeros(n, dtype=float),
            1: np.zeros(n, dtype=float),
        }
        self.count = {0: 0, 1: 0}
        self.selected = {0: np.zeros(n, dtype=int), 1: np.zeros(n, dtype=int)}
        self.last = None

    def context(self, drift_score):
        score = float(drift_score)
        if not np.isfinite(score):
            raise ValueError("drift score must be finite")
        return int(score >= self.drift_threshold)

    def select(self, *, drift_score, shadow_weight):
        context = self.context(drift_score)
        shadow_weight = float(shadow_weight)
        if not np.isfinite(shadow_weight) or shadow_weight < 0.0:
            raise ValueError("shadow_weight must be finite and nonnegative")

        if self.count[context] < self.warmup:
            index = 0
        else:
            losses = self.loss[context]
            best = int(np.argmin(losses))
            # Fail closed on ties or marginal wins against scale=0.
            if best != 0:
                baseline = float(losses[0])
                candidate = float(losses[best])
                required = abs(baseline) * self.margin_ratio
                if not candidate < baseline - required:
                    best = 0
            index = best

        self.selected[context][index] += 1
        authority = self.scales[index] * shadow_weight
        self.last = {
            "context": context,
            "index": index,
            "scale": self.scales[index],
            "authority": authority,
            "shadow_weight": shadow_weight,
            "drift_score": float(drift_score),
        }
        return authority

    def observe(self, *, baseline_error, correction, shadow_weight, drift_score):
        baseline_error = np.asarray(baseline_error, dtype=float)
        correction = np.asarray(correction, dtype=float)
        if baseline_error.shape != correction.shape or baseline_error.ndim != 1:
            raise ValueError("baseline_error and correction must be same 1D shape")
        if not np.all(np.isfinite(baseline_error)) or not np.all(np.isfinite(correction)):
            raise ValueError("transition arrays must be finite")

        context = self.context(drift_score)
        shadow_weight = float(shadow_weight)
        losses = np.asarray([
            np.mean((baseline_error - scale * shadow_weight * correction) ** 2)
            for scale in self.scales
        ], dtype=float)
        if self.count[context] == 0:
            self.loss[context] = losses
        else:
            self.loss[context] = (
                self.decay * self.loss[context]
                + (1.0 - self.decay) * losses
            )
        self.count[context] += 1
        return losses.copy()

    def report(self):
        return {
            "drift_threshold": self.drift_threshold,
            "decay": self.decay,
            "warmup": self.warmup,
            "margin_ratio": self.margin_ratio,
            "scales": list(self.scales),
            "context_counts": dict(self.count),
            "loss": {str(k): self.loss[k].tolist() for k in self.loss},
            "selected": {str(k): self.selected[k].tolist() for k in self.selected},
            "mode": "SHADOW_ONLY",
        }

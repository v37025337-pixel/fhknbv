
"""
DIGITAL_MIND_v0_7.py

v0.7 — Rich-world cognitive architecture

This version addresses four concrete limitations of earlier toy prototypes:

1) Richer world:
   - 24 latent state variables
   - 18 observations
   - 6 continuous actions
   - nonlinear dynamics
   - delayed action effects
   - 3 social actors
   - hidden regime change

2) Open-structure question formation:
   - no learn_effect / causal_anomaly / self_change enum
   - questions are generated from residual subspaces and discovered predictors
   - each question is a mathematical object: target direction + candidate causes
   - natural-language text is only a rendering of that generated structure

3) Richer Self:
   - causal boundary
   - current competence
   - episodic autobiographical graph
   - learned narrative themes
   - social trust/expectation model
   - hierarchical goals

4) Multiple time scales and planning:
   - fast control: every step
   - meso reflection: every episode
   - macro planning: every 24 steps
   - slow identity consolidation: every 64 steps
   - separate delayed-reward planning benchmark

This remains a functional architecture experiment, not evidence of phenomenal consciousness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque, defaultdict
import math
import numpy as np


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def unit(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x)
    return x / (n + 1e-12)


def stable_matrix(rng, n, radius=0.78, density=0.18):
    M = rng.normal(scale=0.25, size=(n, n))
    mask = rng.random((n, n)) < density
    M *= mask
    M += np.eye(n) * 0.45
    vals = np.linalg.eigvals(M)
    rho = max(abs(vals)) + 1e-12
    return M * (radius / rho)


def top_signed_terms(vec, prefix, k=3):
    vec = np.asarray(vec, dtype=float)
    idx = np.argsort(np.abs(vec))[-k:][::-1]
    return [f"{prefix}{int(i)}:{vec[i]:+.2f}" for i in idx]


# ---------------------------------------------------------------------
# Rich nonlinear world
# ---------------------------------------------------------------------

@dataclass
class WorldStep:
    obs: np.ndarray
    social: np.ndarray
    reward_components: np.ndarray
    regime: int


class RichWorld:
    def __init__(
        self,
        seed=0,
        d_state=24,
        d_obs=18,
        d_action=6,
        n_social=3,
        regime_change=320,
    ):
        self.rng = np.random.default_rng(seed)
        self.d_state = d_state
        self.d_obs = d_obs
        self.d_action = d_action
        self.n_social = n_social
        self.regime_change = regime_change

        self.A0 = stable_matrix(self.rng, d_state, radius=0.76, density=0.16)
        self.A1 = self.A0.copy()

        # Hidden regime change modifies a structured subspace, not all weights.
        delta_A = self.rng.normal(scale=0.05, size=(d_state, d_state))
        delta_A *= (self.rng.random((d_state, d_state)) < 0.06)
        self.A1 = self.A1 + delta_A
        vals = np.linalg.eigvals(self.A1)
        rho = max(abs(vals)) + 1e-12
        self.A1 *= min(1.0, 0.80 / rho)

        self.B0 = self.rng.normal(scale=0.22, size=(d_state, d_action))
        self.B0 *= (self.rng.random((d_state, d_action)) < 0.45)

        # Rank-3 hidden change in action consequences.
        U = self.rng.normal(size=(d_state, 3))
        V = self.rng.normal(size=(d_action, 3))
        self.delta_B = 0.15 * U @ V.T / math.sqrt(3.0)
        self.B1 = self.B0 + self.delta_B

        self.delay_B = self.rng.normal(scale=0.16, size=(d_state, 2))
        self.delay_B *= (self.rng.random((d_state, 2)) < 0.35)

        self.N = self.rng.normal(scale=0.12, size=(d_state, d_state))
        self.N *= (self.rng.random((d_state, d_state)) < 0.12)

        # Observation projection: mostly local identity + mixed hidden features.
        self.C = self.rng.normal(scale=0.08, size=(d_obs, d_state))
        for i in range(min(d_obs, d_state)):
            self.C[i, i] += 0.85

        # Social actors: response to the last two action channels + context.
        base_bias = np.linspace(-0.45, 0.45, n_social)
        self.social_bias = base_bias + self.rng.normal(scale=0.08, size=n_social)

        self.social_response = self.rng.normal(scale=0.28, size=(n_social, 2))
        if n_social >= 3:
            # Distinct response styles: avoid accidental clone-like actors.
            self.social_response[0] += np.array([+0.55, -0.10])
            self.social_response[1] += np.array([-0.20, +0.50])
            self.social_response[2] += np.array([-0.45, -0.35])
        self.social_context = self.rng.normal(scale=0.20, size=(n_social, 4))
        self.social_to_state = self.rng.normal(
            scale=0.12, size=(d_state, n_social)
        )

        self.x = self.rng.normal(scale=0.15, size=d_state)
        self.action_delay = deque(
            [np.zeros(2), np.zeros(2), np.zeros(2)],
            maxlen=3,
        )
        self.t = 0

    def _regime(self):
        return int(self.t >= self.regime_change)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -1.5, 1.5)

        regime = self._regime()
        A = self.A1 if regime else self.A0
        B = self.B1 if regime else self.B0

        delayed = self.action_delay[0].copy()
        self.action_delay.append(action[:2].copy())

        context = np.array([
            self.x[0],
            self.x[1],
            self.x[4],
            math.sin(self.t / 17.0),
        ])

        social = np.tanh(
            self.social_bias
            + self.social_response @ action[-2:]
            + self.social_context @ context
            + self.rng.normal(scale=0.08, size=self.n_social)
        )

        nonlinear = 0.07 * self.x * np.tanh(self.N @ self.x)

        x_next = (
            A @ self.x
            + B @ action
            + self.delay_B @ delayed
            + nonlinear
            + self.social_to_state @ social
            + self.rng.normal(scale=0.035, size=self.d_state)
        )

        self.x = np.tanh(x_next)

        obs = (
            self.C @ self.x
            + self.rng.normal(scale=0.025, size=self.d_obs)
        )

        # Make social signals explicitly observable in the final three channels.
        obs[-self.n_social:] = (
            social + self.rng.normal(scale=0.02, size=self.n_social)
        )

        # Multi-objective environment signal (not collapsed to one scalar).
        reward_components = np.array([
            -float(np.mean(self.x[:4] ** 2)),          # stability
            -float(np.mean(np.abs(self.x[4:8]))),     # resource stress
            float(np.mean(social)),                   # social response
            float(self.x[8] - 0.4 * abs(self.x[9])),  # delayed progress
        ])

        self.t += 1
        return WorldStep(
            obs=obs,
            social=social,
            reward_components=reward_components,
            regime=regime,
        )


# ---------------------------------------------------------------------
# Predictive intelligence
# ---------------------------------------------------------------------

class PredictiveModel:
    """
    Online nonlinear local model:
        obs_{t+1} = W phi(obs_t, action_t, a_{t-1}, a_{t-2})

    phi contains linear, squared, and selected interaction features.
    """

    def __init__(self, d_obs, d_action, seed=0):
        self.rng = np.random.default_rng(seed)
        self.d_obs = d_obs
        self.d_action = d_action

        self.prev_actions = deque(
            [np.zeros(d_action) for _ in range(3)],
            maxlen=3,
        )

        self.cross_pairs = [
            (i, (3 * i + 5) % d_obs)
            for i in range(min(10, d_obs))
        ]

        self.feature_names = ["bias"]
        self.feature_names += [f"o{i}" for i in range(d_obs)]
        self.feature_names += [f"a{i}" for i in range(d_action)]
        self.feature_names += [f"a1_{i}" for i in range(d_action)]
        self.feature_names += [f"a2_{i}" for i in range(d_action)]
        self.feature_names += [f"a3_{i}" for i in range(d_action)]
        self.feature_names += [f"o{i}^2" for i in range(d_obs)]
        self.feature_names += [
            f"o{i}*o{j}" for i, j in self.cross_pairs
        ]

        self.n_features = len(self.feature_names)
        self.W = np.zeros((d_obs, self.n_features))

        # Slow/stable causal model.
        self.P = np.eye(self.n_features) * 8.0
        self.forgetting_normal = 0.994

        # Fast residual adapter.  It learns only the recent discrepancy
        # between the slow model and the changed world, avoiding the numerical
        # instability of making the whole RLS forget too aggressively.
        self.W_fast = np.zeros((d_obs, self.n_features))
        self.fast_gate = 0.0
        self.fast_lr = 0.085

        self.residual_var = np.ones(d_obs) * 0.15
        self.loss_history = []

        # Meta-adaptation: structural surprise switches the estimator into a
        # temporary high-forgetting regime.
        self.loss_ema = None
        self.fast_adapt_left = 0
        self.change_events = 0

    def features(self, obs, action, prev1=None, prev2=None, prev3=None):
        if prev1 is None:
            prev1 = self.prev_actions[-1]
        if prev2 is None:
            prev2 = self.prev_actions[-2]
        if prev3 is None:
            prev3 = self.prev_actions[-3]

        obs = np.asarray(obs, dtype=float)
        action = np.asarray(action, dtype=float)

        parts = [
            np.array([1.0]),
            obs,
            action,
            np.asarray(prev1),
            np.asarray(prev2),
            np.asarray(prev3),
            obs ** 2,
            np.array([obs[i] * obs[j] for i, j in self.cross_pairs]),
        ]
        phi = np.concatenate(parts)
        return np.clip(phi, -3.0, 3.0)

    def predict(self, obs, action, prev1=None, prev2=None, prev3=None):
        phi = self.features(obs, action, prev1, prev2, prev3)
        W_eff = self.W + self.fast_gate * self.W_fast
        return W_eff @ phi

    def update(self, obs, action, next_obs):
        phi = self.features(obs, action)
        slow_pred = self.W @ phi
        fast_pred = self.W_fast @ phi
        pred = slow_pred + self.fast_gate * fast_pred
        err = next_obs - pred
        loss = float(np.mean(err ** 2))

        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            if (
                len(self.loss_history) > 40
                and loss > max(0.010, 2.4 * self.loss_ema)
            ):
                if self.fast_adapt_left <= 0:
                    self.change_events += 1
                    # A new local adapter starts from zero discrepancy.
                    self.W_fast[:] = 0.0
                self.fast_adapt_left = max(self.fast_adapt_left, 52)
                self.fast_gate = 1.0

        # Slow RLS learns the long-run model with a conservative forgetting rate.
        slow_err = next_obs - slow_pred
        Pphi = self.P @ phi
        denom = self.forgetting_normal + float(phi @ Pphi)
        gain = Pphi / max(denom, 1e-9)

        self.W += np.outer(slow_err, gain)
        self.P = (
            self.P - np.outer(gain, phi @ self.P)
        ) / self.forgetting_normal

        self.P = 0.5 * (self.P + self.P.T)
        diag = np.clip(np.diag(self.P), 1e-5, 1e4)
        np.fill_diagonal(self.P, diag)

        # Fast adapter learns the *residual* left by the slow model.
        if self.fast_adapt_left > 0:
            residual_target = next_obs - (self.W @ phi)
            fast_err = residual_target - self.W_fast @ phi
            norm = 1.0 + 0.10 * float(phi @ phi)
            self.W_fast += self.fast_lr * np.outer(fast_err, phi) / norm
            self.W_fast *= 0.999
            self.fast_adapt_left -= 1
            self.fast_gate = 1.0
        else:
            # Once the slow model catches up, transfer authority back smoothly.
            self.fast_gate *= 0.965
            self.W_fast *= 0.997
            if self.fast_gate < 1e-3:
                self.fast_gate = 0.0

        # Recompute the actual post-update ensemble residual for monitoring.
        post_pred = (self.W + self.fast_gate * self.W_fast) @ phi
        post_err = next_obs - post_pred

        self.residual_var = (
            0.985 * self.residual_var + 0.015 * (post_err ** 2)
        )

        capped = loss if self.loss_ema is None else min(loss, 2.5 * self.loss_ema)
        self.loss_ema = 0.985 * self.loss_ema + 0.015 * capped

        self.loss_history.append(loss)

        self.prev_actions.append(np.asarray(action, dtype=float).copy())
        return pred, err, phi

    def action_sensitivity(self):
        start = 1 + self.d_obs
        W_eff = self.W + self.fast_gate * self.W_fast
        return W_eff[:, start:start + self.d_action].copy()


# ---------------------------------------------------------------------
# Open-structure question generation
# ---------------------------------------------------------------------

@dataclass
class OpenQuestion:
    qid: int
    created_t: int
    target_direction: np.ndarray
    predictor_indices: tuple
    predictor_scores: np.ndarray
    feature_names: tuple
    uncertainty: float
    text: str
    evidence: list = field(default_factory=list)
    resolved: bool = False
    resolution_score: float = 0.0

    def signature(self):
        top_target = tuple(
            np.argsort(np.abs(self.target_direction))[-3:][::-1].tolist()
        )
        return top_target, self.predictor_indices


class FreeQuestionGenerator:
    """
    No discrete question classes.

    A question is generated by:
      1) finding the dominant unexplained residual direction;
      2) projecting residuals onto that direction;
      3) finding candidate predictors in the agent's actual feature history;
      4) asking which discovered combination explains the gap.

    The *content* of the question is therefore generated from data.
    """

    def __init__(self, d_obs, feature_names, max_history=96):
        self.d_obs = d_obs
        self.feature_names = tuple(feature_names)
        self.residuals = deque(maxlen=max_history)
        self.features = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.samples_seen = 0
        self.questions = []
        self._qid = 0
        self.last_question_t = -10_000
        self.cooldown = 16

    def observe(self, residual, phi, t=None):
        self.residuals.append(np.asarray(residual, dtype=float).copy())
        self.features.append(np.asarray(phi, dtype=float).copy())
        self.timestamps.append(self.samples_seen if t is None else int(t))
        self.samples_seen += 1

    def update_resolution(self, t, min_samples=24):
        """Predictive explanation only; this does not establish a causal law."""
        for q in self.questions:
            if q.resolved:
                continue
            recent = [r for rt, r in zip(self.timestamps, self.residuals)
                      if rt > q.created_t]
            if len(recent) < min_samples:
                continue
            projected = np.stack(recent[-min_samples:]) @ q.target_direction
            energy = float(np.var(projected, ddof=1))
            q.evidence.append(energy)
            if len(q.evidence) >= 3 and energy < 0.72 * q.uncertainty:
                q.resolved = True
                q.resolution_score = 1.0 - energy / (q.uncertainty + 1e-9)

    def generate(self, t, min_samples=24):
        if len(self.residuals) < min_samples:
            return None

        if t - self.last_question_t < self.cooldown:
            return None

        # Keep only a small active frontier of unresolved questions.
        if sum(not q.resolved for q in self.questions[-8:]) >= 4:
            return None

        R = np.stack(self.residuals, axis=0)
        X = np.stack(self.features, axis=0)

        Rc = R - R.mean(axis=0, keepdims=True)
        cov = Rc.T @ Rc / max(1, len(Rc) - 1)

        vals, vecs = np.linalg.eigh(cov)
        u = vecs[:, -1]
        # Eigenvectors have arbitrary sign; normalize it before creating goals.
        if u[np.argmax(np.abs(u))] < 0:
            u = -u
        uncertainty = float(max(vals[-1], 0.0))

        # Ask only when there is a genuine dominant unexplained direction,
        # rather than manufacturing a question every reflection cycle.
        baseline = float(np.median(np.maximum(vals, 0.0))) + 1e-9
        if uncertainty < max(0.0035, 1.8 * baseline):
            return None

        y = Rc @ u
        ystd = np.std(y) + 1e-9

        Xc = X - X.mean(axis=0, keepdims=True)
        xstd = np.std(Xc, axis=0) + 1e-9
        corr = (Xc.T @ y) / (len(y) * xstd * ystd)

        # Bias cannot be a causal candidate.
        corr[0] = 0.0

        top = np.argsort(np.abs(corr))[-4:][::-1]
        scores = corr[top]

        # Avoid making the same structural query repeatedly.
        target_idx = tuple(
            np.argsort(np.abs(u))[-3:][::-1].tolist()
        )
        signature = (target_idx, tuple(int(i) for i in top))

        recent_signatures = {
            q.signature() for q in self.questions[-8:]
        }
        if signature in recent_signatures:
            return None

        target_terms = ", ".join(top_signed_terms(u, "o", 3))
        predictors = ", ".join(
            f"{self.feature_names[int(i)]}:{scores[j]:+.2f}"
            for j, i in enumerate(top)
        )

        text = (
            "Какая зависимость между обнаруженными факторами "
            f"[{predictors}] объясняет необъяснённое направление "
            f"[{target_terms}]?"
        )

        self._qid += 1
        q = OpenQuestion(
            qid=self._qid,
            created_t=int(t),
            target_direction=u.copy(),
            predictor_indices=tuple(int(i) for i in top),
            predictor_scores=scores.copy(),
            feature_names=tuple(self.feature_names[int(i)] for i in top),
            uncertainty=uncertainty,
            text=text,
        )
        self.questions.append(q)
        self.last_question_t = int(t)
        return q


# ---------------------------------------------------------------------
# Narrative / social self
# ---------------------------------------------------------------------

@dataclass
class EpisodeNode:
    eid: int
    t0: int
    t1: int
    goal_id: int
    embedding: np.ndarray
    action_mean: np.ndarray
    obs_delta: np.ndarray
    error_mean: float
    reward_mean: np.ndarray
    social_mean: np.ndarray
    self_relevance: float
    theme_id: int | None = None


class NarrativeSelf:
    """
    Identity is not one moving average.

    It contains:
      - an episodic graph;
      - temporal, similarity, and goal edges;
      - learned unlabeled themes (online clustering);
      - social expectations/trust;
      - causal boundary and competence outside this class.
    """

    def __init__(self, n_social=3, theme_threshold=0.76):
        self.nodes = []
        self.edges = []  # (src, dst, relation)
        self.theme_threshold = theme_threshold
        self.theme_centroids = {}
        self.theme_counts = defaultdict(int)
        self._next_theme = 0

        self.social_expectation = np.zeros(n_social)
        self.social_trust = np.ones(n_social) * 0.5
        self.social_error_ema = np.ones(n_social) * 0.25

    def _assign_theme(self, emb):
        if not self.theme_centroids:
            tid = self._next_theme
            self._next_theme += 1
            self.theme_centroids[tid] = emb.copy()
            self.theme_counts[tid] = 1
            return tid

        sims = {
            tid: cosine(emb, c)
            for tid, c in self.theme_centroids.items()
        }
        tid, sim = max(sims.items(), key=lambda kv: kv[1])

        if sim < self.theme_threshold:
            tid = self._next_theme
            self._next_theme += 1
            self.theme_centroids[tid] = emb.copy()
            self.theme_counts[tid] = 1
            return tid

        n = self.theme_counts[tid]
        self.theme_centroids[tid] = (
            (n * self.theme_centroids[tid] + emb) / (n + 1)
        )
        self.theme_counts[tid] = n + 1
        return tid

    def add_episode(self, node):
        node.theme_id = self._assign_theme(node.embedding)

        if self.nodes:
            self.edges.append((self.nodes[-1].eid, node.eid, "temporal"))

            # Similarity edge to the most similar older episode.
            candidates = self.nodes[:-1] if len(self.nodes) > 1 else self.nodes
            if candidates:
                best = max(
                    candidates,
                    key=lambda n: cosine(n.embedding, node.embedding)
                )
                if cosine(best.embedding, node.embedding) > 0.55:
                    self.edges.append((best.eid, node.eid, "similarity"))

            # Goal-thread edge.
            same_goal = [
                n for n in reversed(self.nodes)
                if n.goal_id == node.goal_id
            ]
            if same_goal:
                self.edges.append((same_goal[0].eid, node.eid, "goal-thread"))

        self.nodes.append(node)

        # Social self: expectation and trust learned from interaction history.
        err = node.social_mean - self.social_expectation
        self.social_expectation += 0.08 * err
        self.social_error_ema = (
            0.92 * self.social_error_ema + 0.08 * np.abs(err)
        )
        self.social_trust = np.exp(-2.2 * self.social_error_ema)

    def theme_persistence(self, split_t):
        pre = {
            n.theme_id for n in self.nodes if n.t1 < split_t
        }
        post = [
            n.theme_id for n in self.nodes if n.t0 >= split_t
        ]
        if not post:
            return 0.0
        return float(np.mean([tid in pre for tid in post]))

    def graph_density_ratio(self):
        if not self.nodes:
            return 0.0
        return len(self.edges) / len(self.nodes)

    def continuity_score(self, split_t):
        """
        Narrative identity need not mean "same cluster forever".
        Continuity is supported either by persistent themes or by explicit
        cross-regime similarity / goal-thread links in the autobiographical graph.
        """
        post_nodes = [n for n in self.nodes if n.t0 >= split_t]
        if not post_nodes:
            return 0.0

        persistence = self.theme_persistence(split_t)

        by_id = {n.eid: n for n in self.nodes}
        cross_links = 0
        for src, dst, rel in self.edges:
            if rel not in ("similarity", "goal-thread"):
                continue
            a = by_id.get(src)
            b = by_id.get(dst)
            if a is None or b is None:
                continue
            if a.t1 < split_t and b.t0 >= split_t:
                cross_links += 1

        bridge = min(1.0, cross_links / max(1.0, 0.20 * len(post_nodes)))
        return float(0.60 * persistence + 0.40 * bridge)


# ---------------------------------------------------------------------
# Hierarchical goals
# ---------------------------------------------------------------------

@dataclass
class GoalNode:
    gid: int
    parent: int | None
    depth: int
    created_t: int
    horizon: int
    priority: float
    target: np.ndarray
    origin_vector: np.ndarray
    description: str
    status: str = "active"
    question_id: int | None = None


class GoalHierarchy:
    def __init__(self, d_obs):
        self.d_obs = d_obs
        self.goals = {}
        self.children = defaultdict(list)
        self._gid = 0
        self.last_state_goal_t = {}

        # Meta-root is structural, not a concrete task.
        self.root = self._new(
            parent=None,
            depth=0,
            t=0,
            horizon=256,
            priority=1.0,
            target=np.zeros(d_obs),
            origin=np.array([1.0, 0.0, 0.0]),
            description="preserve coherent long-run agency",
        )

        self.stability = self._new(
            parent=self.root.gid,
            depth=1,
            t=0,
            horizon=96,
            priority=0.8,
            target=np.zeros(d_obs),
            origin=np.array([0.7, 0.2, 0.1]),
            description="reduce persistent state deviation",
        )

        self.epistemic = self._new(
            parent=self.root.gid,
            depth=1,
            t=0,
            horizon=96,
            priority=0.7,
            target=np.zeros(d_obs),
            origin=np.array([0.1, 0.8, 0.1]),
            description="reduce unresolved predictive structure",
        )

        self.social = self._new(
            parent=self.root.gid,
            depth=1,
            t=0,
            horizon=96,
            priority=0.6,
            target=np.zeros(d_obs),
            origin=np.array([0.1, 0.1, 0.8]),
            description="maintain workable social relations",
        )

    def _new(
        self, parent, depth, t, horizon, priority,
        target, origin, description, question_id=None
    ):
        self._gid += 1
        g = GoalNode(
            gid=self._gid,
            parent=parent,
            depth=depth,
            created_t=int(t),
            horizon=int(horizon),
            priority=float(priority),
            target=np.asarray(target, dtype=float).copy(),
            origin_vector=np.asarray(origin, dtype=float).copy(),
            description=description,
            question_id=question_id,
        )
        self.goals[g.gid] = g
        if parent is not None:
            self.children[parent].append(g.gid)
        return g

    def spawn_from_question(self, q, t):
        # No question "type": derive the subgoal from generated structure.
        target = np.zeros(self.d_obs)
        idx = np.argsort(np.abs(q.target_direction))[-3:]
        target[idx] = -0.15 * np.sign(q.target_direction[idx])

        mid = self._new(
            parent=self.epistemic.gid,
            depth=2,
            t=t,
            horizon=128,
            priority=min(1.0, 0.55 + 2.0 * q.uncertainty),
            target=target,
            origin=np.array([0.0, 1.0, 0.0]),
            description=f"resolve generated question {q.qid}",
            question_id=q.qid,
        )

        leaf = self._new(
            parent=mid.gid,
            depth=3,
            t=t,
            horizon=128,
            priority=mid.priority,
            target=target,
            origin=np.array([0.0, 1.0, 0.0]),
            description=f"collect discriminating evidence for q{q.qid}",
            question_id=q.qid,
        )
        return leaf

    def spawn_state_goal(self, obs, t):
        # Endogenous concrete goal from the strongest current deviation.
        idx = int(np.argmax(np.abs(obs[: min(8, len(obs))])))
        if abs(obs[idx]) < 0.45:
            return None

        if t - self.last_state_goal_t.get(idx, -10_000) < 56:
            return None
        self.last_state_goal_t[idx] = int(t)

        target = obs.copy()
        target[idx] = 0.0

        parent = self._new(
            parent=self.stability.gid,
            depth=2,
            t=t,
            horizon=40,
            priority=min(1.0, 0.5 + abs(obs[idx])),
            target=target,
            origin=np.array([1.0, 0.0, 0.0]),
            description=f"stabilize observation dimension {idx}",
        )

        return self._new(
            parent=parent.gid,
            depth=3,
            t=t,
            horizon=10,
            priority=parent.priority,
            target=target,
            origin=np.array([1.0, 0.0, 0.0]),
            description=f"near-term correction of observation {idx}",
        )

    def active_leaf(self):
        leaves = [
            g for g in self.goals.values()
            if g.status == "active" and not self.children[g.gid]
        ]
        if not leaves:
            return self.stability
        return max(leaves, key=lambda g: g.priority)

    def refresh(self, t, obs, resolved_questions):
        """Maintain generated goals; the four structural roots stay persistent."""
        for g in self.goals.values():
            if g.depth < 2 or g.status != 'active' or self.children[g.gid]:
                continue
            if g.question_id is not None:
                completed = g.question_id in resolved_questions
            else:
                completed = float(np.mean((obs - g.target) ** 2)) < 0.005
            if completed:
                g.status = 'completed'
            elif t >= g.created_t + g.horizon:
                g.status = 'expired'
        for g in sorted(self.goals.values(), key=lambda x: -x.depth):
            if g.depth < 2 or g.status != 'active' or not self.children[g.gid]:
                continue
            children = [self.goals[i] for i in self.children[g.gid]]
            if all(c.status != 'active' for c in children):
                g.status = ('completed' if all(c.status == 'completed' for c in children)
                            else 'expired')

    def max_depth(self):
        return max(g.depth for g in self.goals.values())


# ---------------------------------------------------------------------
# Multi-timescale planning
# ---------------------------------------------------------------------

class MultiTimescalePlanner:
    def __init__(self, d_action, seed=0):
        self.d_action = d_action
        self.rng = np.random.default_rng(seed)
        self.last_macro_action = np.zeros(d_action)
        self.macro_plans = 0

    def _candidate_actions(self):
        actions = [np.zeros(self.d_action)]

        # Six basis directions in both signs would be 13 candidates;
        # select a compact but broad basis.
        for i in range(self.d_action):
            e = np.zeros(self.d_action)
            e[i] = 0.8
            actions.append(e)
            actions.append(-e)

        for _ in range(4):
            actions.append(
                np.clip(self.rng.normal(scale=0.55, size=self.d_action), -1, 1)
            )
        return actions

    @staticmethod
    def search(initial, actions, transition, utility, *, horizon, beam=6,
               discount=0.98, terminal=None):
        """Shared beam search used by both the agent and planning benchmark."""
        if horizon < 1 or (beam is not None and beam < 1):
            raise ValueError('horizon and beam must be positive')
        if not 0 < discount <= 1:
            raise ValueError('discount must be in (0, 1]')
        actions = tuple(actions)
        if not actions:
            raise ValueError('at least one candidate action is required')
        states = [(0.0, initial, [])]
        for depth in range(horizon):
            expanded = []
            for score, state, seq in states:
                for action in actions:
                    nxt = transition(state, action)
                    value = score + discount ** depth * utility(state, action, nxt)
                    if not np.isfinite(value):
                        raise ValueError('planner received a non-finite score')
                    expanded.append((value, nxt, seq + [action]))
            expanded.sort(key=lambda row: row[0] + (
                discount ** (depth + 1) * terminal(row[1]) if terminal else 0.0
            ), reverse=True)
            states = expanded if beam is None else expanded[:beam]
        best = states[0]
        score = best[0] + (discount ** horizon * terminal(best[1]) if terminal else 0.)
        return score, best[2]

    def plan(self, obs, model, goal, horizon=5, beam=5, action_limit=1.2):
        initial = (obs.copy(), *(a.copy() for a in reversed(model.prev_actions)))

        def transition(state, action):
            o, p1, p2, p3 = state
            nxt = model.predict(o, action, p1, p2, p3)
            return nxt, action.copy(), p1.copy(), p2.copy()

        def utility(state, action, nxt):
            return -float(np.mean((nxt[0] - goal.target) ** 2)) - 0.025 * float(action @ action)

        candidates = [np.clip(a, -action_limit, action_limit) for a in self._candidate_actions()]
        score, seq = self.search(initial, candidates, transition, utility,
                             horizon=horizon, beam=beam)
        self.last_sequence = [a.copy() for a in seq]
        self.last_score = score
        self.last_macro_action = seq[0].copy()
        self.macro_plans += 1
        return self.last_macro_action.copy()


# ---------------------------------------------------------------------
# Full v0.7 agent
# ---------------------------------------------------------------------

class DigitalMindV07:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.world = RichWorld(seed=seed)

        self.d_obs = self.world.d_obs
        self.d_action = self.world.d_action

        self.model = PredictiveModel(
            self.d_obs, self.d_action, seed=seed + 1
        )
        self.questions = FreeQuestionGenerator(
            self.d_obs, self.model.feature_names
        )
        self.narrative = NarrativeSelf(
            n_social=self.world.n_social
        )
        self.goals = GoalHierarchy(self.d_obs)
        self.planner = MultiTimescalePlanner(
            self.d_action, seed=seed + 2
        )

        self.obs = np.zeros(self.d_obs)
        self.last_world = WorldStep(
            obs=self.obs.copy(),
            social=np.zeros(self.world.n_social),
            reward_components=np.zeros(4),
            regime=0,
        )

        # Causal self boundary / competence.
        self.self_trace = np.zeros(self.d_obs)
        self.competence = np.ones(self.d_action) * 0.5
        self.max_sensitivity = np.ones(self.d_action) * 1e-4

        # Time hierarchy.
        self.fast_steps = 0
        self.meso_ticks = 0
        self.macro_ticks = 0
        self.slow_ticks = 0

        self.episode_len = 8
        self.macro_period = 24
        self.slow_period = 64

        self.current_goal = self.goals.active_leaf()
        self.macro_action = np.zeros(self.d_action)

        self.ep_obs0 = self.obs.copy()
        self.ep_actions = []
        self.ep_errors = []
        self.ep_rewards = []
        self.ep_social = []

        self.loss_pre = []
        self.loss_post = []

    def _update_self(self):
        S = self.model.action_sensitivity()
        causal_strength = np.linalg.norm(S, axis=1)

        # Historical ownership reacts faster upward than downward.
        evidence = sigmoid(7.0 * (causal_strength - 0.04))
        for i in range(self.d_obs):
            rate = 0.045 if evidence[i] > self.self_trace[i] else 0.003
            self.self_trace[i] += rate * (evidence[i] - self.self_trace[i])

        per_action = np.linalg.norm(S, axis=0)
        self.max_sensitivity = np.maximum(self.max_sensitivity, per_action)
        self.competence = np.clip(
            per_action / (self.max_sensitivity + 1e-9),
            0.0, 1.0
        )

    def _question_probe_action(self, q):
        # Map generated feature indices back to action dimensions if present.
        action_start = 1 + self.d_obs
        action_end = action_start + self.d_action

        scores = np.zeros(self.d_action)
        for idx, sc in zip(q.predictor_indices, q.predictor_scores):
            if action_start <= idx < action_end:
                scores[idx - action_start] += abs(sc)

        if np.max(scores) <= 0:
            return None

        d = int(np.argmax(scores))
        a = np.zeros(self.d_action)
        a[d] = 0.9 if (q.qid % 2) else -0.9
        return a

    def _choose_action(self, t):
        # Macro plan provides slow guidance.
        if t % self.macro_period == 0:
            self.current_goal = self.goals.active_leaf()
            self.macro_action = self.planner.plan(
                self.obs,
                self.model,
                self.current_goal,
                horizon=7,
                beam=6,
            )
            self.macro_ticks += 1

        # A generated epistemic question can temporarily override one action
        # dimension with an intervention suggested by its own predictors.
        q = self.questions.questions[-1] if self.questions.questions else None
        probe = None
        if q is not None and not q.resolved:
            probe = self._question_probe_action(q)

        action = self.macro_action.copy()

        if probe is not None:
            action = 0.55 * action + 0.75 * probe

        # Small exploration prevents collapse to a single deterministic path.
        action += self.rng.normal(scale=0.10, size=self.d_action)
        return np.clip(action, -1.2, 1.2)

    def _close_episode(self, t):
        self.meso_ticks += 1

        actions = np.stack(self.ep_actions, axis=0)
        rewards = np.stack(self.ep_rewards, axis=0)
        social = np.stack(self.ep_social, axis=0)

        action_mean = actions.mean(axis=0)
        reward_mean = rewards.mean(axis=0)
        social_mean = social.mean(axis=0)
        error_mean = float(np.mean(self.ep_errors))
        obs_delta = self.obs - self.ep_obs0

        self_relevance = float(np.mean(self.self_trace[: min(8, self.d_obs)]))

        # Episode embedding = structured record, not one self-vector.
        emb = np.concatenate([
            np.tanh(action_mean),
            np.tanh(obs_delta[:8]),
            np.tanh(reward_mean),
            np.tanh(social_mean),
            np.array([
                min(error_mean, 2.0) / 2.0,
                self_relevance,
                float(self.current_goal.depth) / 3.0,
            ]),
        ])

        node = EpisodeNode(
            eid=len(self.narrative.nodes) + 1,
            t0=t - self.episode_len + 1,
            t1=t,
            goal_id=self.current_goal.gid,
            embedding=emb,
            action_mean=action_mean,
            obs_delta=obs_delta,
            error_mean=error_mean,
            reward_mean=reward_mean,
            social_mean=social_mean,
            self_relevance=self_relevance,
        )
        self.narrative.add_episode(node)

        # Generate a data-derived question.
        q = self.questions.generate(t) if getattr(self, 'questions_enabled', True) else None
        if q is not None:
            self.current_goal = self.goals.spawn_from_question(q, t)
        else:
            state_goal = self.goals.spawn_state_goal(self.obs, t)
            if state_goal is not None:
                self.current_goal = state_goal

        self.questions.update_resolution(t)
        self.goals.refresh(t, self.obs, {q.qid for q in self.questions.questions if q.resolved})

        self.ep_obs0 = self.obs.copy()
        self.ep_actions.clear()
        self.ep_errors.clear()
        self.ep_rewards.clear()
        self.ep_social.clear()

    def step(self, t):
        action = self._choose_action(t)
        ws = self.world.step(action)

        pred, err, phi = self.model.update(
            self.obs, action, ws.obs
        )

        self.questions.observe(err, phi)
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

        if (t + 1) % self.episode_len == 0:
            self._close_episode(t)

        # Slow consolidation is structural: no identity vector averaging.
        if (t + 1) % self.slow_period == 0:
            self.slow_ticks += 1
            # Lower confidence in relationships that became unpredictable.
            self.narrative.social_trust = np.clip(
                self.narrative.social_trust, 0.05, 0.98
            )

        return ws

    def run(self, T=640):
        return [self.step(self.fast_steps) for _ in range(T)]


# ---------------------------------------------------------------------
# Long-horizon planning benchmark
# ---------------------------------------------------------------------

class DelayedPlanningBenchmark:
    """
    Exact small benchmark for the planning mechanism.

    invest_t has an immediate cost, but produces delayed resource through a
    buffer. A myopic controller dislikes investment; a longer-horizon planner
    can discover that investment is worthwhile.
    """

    @staticmethod
    def rollout(seq):
        resource = 0.0
        buffer = 0.0
        total = 0.0

        for a in seq:
            a = float(a)
            # Immediate cost.
            reward = resource - 0.30 * a
            total += reward

            new_resource = 0.84 * resource + 0.72 * buffer
            new_buffer = 0.72 * buffer + a

            resource = new_resource
            buffer = new_buffer

        total += 1.6 * resource
        return total, resource

    @classmethod
    def best_sequence(cls, horizon):
        if not 1 <= horizon <= 16:
            raise ValueError('exact benchmark supports horizons 1..16')
        from itertools import product
        sequences = product((0., 1.), repeat=horizon)
        seq = max(sequences, key=lambda s: cls.rollout(s)[0])
        return cls.rollout(seq)[0], list(seq)

from __future__ import annotations

"""L0 Canonical — Digital Relational Geometry, indexed backend.

This is *digital* geometry: a dimensionless geometry of state differences for a
declared directed relation x -> y.  It has no physical-space interpretation.

Dense histories
---------------
Committed samples are full-covariance whitened.  Nearest-neighbour discovery is
performed by :class:`scipy.spatial.cKDTree`; the old O(N^2 D) all-pairs scan and
per-row ``np.argsort`` are gone.  Local Jacobians and the causal metric keep the
same mathematics as the previous dense implementation.

Sparse high-dimensional histories
----------------------------------
CSR/CSC inputs are accepted.  For sparse/high-D inputs a dense D x D covariance
would defeat sparsity, so the sparse backend uses variance whitening and a
*diagonal* causal metric.  Neighbour discovery uses cKDTree on either the exact
scaled dense representation (small D) or a deterministic CountSketch projection
(large D), followed by exact sparse reranking inside the candidate set.

That sparse backend deliberately trades full affine/rotational covariance for
memory/runtime scalability.  The selected backend and representation are
reported explicitly; it is never silently presented as the full-covariance
metric.
"""

from dataclasses import dataclass, field
from collections import deque
import copy
from typing import Any, Sequence
import math
import numpy as np

try:
    from scipy import sparse as sp
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover - dependency failure is explicit
    raise ImportError("L0 Digital Relational Geometry requires scipy") from exc

EPS = 1e-12


def sym(a):
    a = np.asarray(a, float)
    return .5 * (a + a.T)


def _cov(rows, dim):
    a = np.asarray(rows, float).reshape(len(rows), dim)
    if len(a) < 2:
        return np.eye(dim)
    d = a - a.mean(0)
    return sym((d.T @ d) / max(1, len(a) - 1))


def psd_invsqrt(c, rel_floor=1e-14):
    matrix = sym(c)
    if not np.all(np.isfinite(matrix)):
        raise ValueError('covariance must be finite')
    scale = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    if scale == 0.0:
        return np.zeros_like(matrix)
    e, v = np.linalg.eigh(matrix / scale)
    e = np.clip(e, 0., None)
    mx = float(e.max()) if e.size else 0.
    if mx == 0.0:
        return np.zeros_like(c)
    keep = e > rel_floor * mx
    w = np.zeros_like(e)
    w[keep] = (1 / np.sqrt(e[keep])) / math.sqrt(scale)
    return v @ np.diag(w) @ v.T


def _is_sparse(value: Any) -> bool:
    return sp.issparse(value)


def _dense_vector(value: Any, dim: int, *, scalar_ok: bool = False) -> np.ndarray:
    if _is_sparse(value):
        a = np.asarray(value.toarray(), float).reshape(-1)
    elif scalar_ok and np.asarray(value).ndim == 0:
        a = np.asarray([value], float)
    else:
        a = np.asarray(value, float).reshape(-1)
    if a.size != dim:
        raise ValueError(f"expected vector[{dim}], got {a.size} values")
    return a


def _sparse_row(value: Any, dim: int) -> sp.csr_matrix:
    if _is_sparse(value):
        row = value.tocsr().astype(float)
        if row.shape == (dim, 1):
            row = row.T.tocsr()
        elif row.shape != (1, dim):
            flat = np.asarray(row.toarray(), float).reshape(-1)
            if flat.size != dim:
                raise ValueError(f"expected vector[{dim}], got sparse shape {row.shape}")
            row = sp.csr_matrix(flat.reshape(1, dim))
        return row
    a = np.asarray(value, float).reshape(-1)
    if a.size != dim:
        raise ValueError(f"expected vector[{dim}], got {a.size} values")
    return sp.csr_matrix(a.reshape(1, dim))


def _finite_sparse(row: sp.csr_matrix) -> bool:
    return bool(np.all(np.isfinite(row.data)))


def _sparse_mean_var(a: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Sample mean/variance without densifying N x D."""
    n = int(a.shape[0])
    mean = np.asarray(a.mean(axis=0)).ravel()
    mean_sq = np.asarray(a.multiply(a).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean * mean, 0.0)
    if n > 1:
        var *= n / (n - 1)
    return mean, var


def _invstd(var: np.ndarray, rel_floor: float = 1e-14) -> np.ndarray:
    var = np.asarray(var, float)
    mx = float(var.max()) if var.size else 0.0
    out = np.zeros_like(var)
    if mx == 0.0:
        return out
    keep = var > rel_floor * mx
    out[keep] = 1.0 / np.sqrt(var[keep])
    return out


def _countsketch_dense(a: sp.csr_matrix, out_dim: int, seed: int = 0x9E3779B1) -> np.ndarray:
    """Deterministic one-hash CountSketch, returned dense for cKDTree.

    The matrix itself is never materialized; only non-zeros are touched.
    """
    n, d = a.shape
    p = max(2, min(int(out_dim), d))
    coo = a.tocoo(copy=False)
    cols = coo.col.astype(np.uint64, copy=False)
    # Two independent-ish integer hashes. Overflow is intentional in uint64.
    h1 = cols * np.uint64(11400714819323198485) + np.uint64(seed)
    h2 = cols * np.uint64(14029467366897019727) + np.uint64(seed ^ 0x85EBCA6B)
    buckets = np.asarray(h1 % np.uint64(p), dtype=np.intp)
    signs = np.where((h2 & np.uint64(1)) == 0, 1.0, -1.0)
    proj = sp.coo_matrix((coo.data * signs, (coo.row, buckets)), shape=(n, p)).tocsr()
    # 1-hash CountSketch is unbiased for squared norm up to collisions; no extra
    # scaling is required for neighbour ranking.
    return proj.toarray()


def _drop_self(row_idx: np.ndarray, i: int, k: int) -> np.ndarray:
    row = np.asarray(row_idx, dtype=np.intp).reshape(-1)
    row = row[row != i]
    if row.size > k:
        row = row[:k]
    return row


def _sparse_take_differences(a: sp.csr_matrix, rows: np.ndarray, i: int) -> sp.csr_matrix:
    rows = np.asarray(rows, dtype=np.intp).reshape(-1)
    if rows.size == 0:
        return sp.csr_matrix((0, a.shape[1]), dtype=float)
    base = a.getrow(int(i))
    repeated = sp.vstack([base] * int(rows.size), format='csr')
    return (a[rows] - repeated).tocsr()


@dataclass
class DigitalRelationalGeometry:
    input_dim: int
    output_dim: int
    max_history: int = 192
    neighbors: int = 12
    kd_leafsize: int = 32
    kd_workers: int = -1
    sparse_density_threshold: float = 0.12
    sparse_min_dim: int = 256
    sparse_exact_tree_max_dim: int = 64
    sparse_projection_dim: int = 32
    sparse_candidate_multiplier: int = 4
    dense_output_max_dim: int = 256
    report_operator_max_dim: int = 256
    _x: deque = field(init=False, repr=False)
    _y: deque = field(init=False, repr=False)
    _cache: dict | None = field(default=None, init=False, repr=False)
    _sparse_input: bool = field(default=False, init=False, repr=False)
    _sparse_output: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError('digital geometry dimensions must be positive')
        if self.max_history < 8:
            raise ValueError('max_history must be >= 8')
        if self.neighbors < 2:
            raise ValueError('neighbors must be >= 2')
        if self.kd_leafsize < 1:
            raise ValueError('kd_leafsize must be positive')
        if not (0.0 <= self.sparse_density_threshold <= 1.0):
            raise ValueError('sparse_density_threshold must be in [0,1]')
        self._x = deque(maxlen=self.max_history)
        self._y = deque(maxlen=self.max_history)

    @property
    def observations(self):
        return len(self._x)

    def _auto_sparse_dense(self, value: Any, dim: int) -> bool:
        if _is_sparse(value):
            return True
        if dim < self.sparse_min_dim:
            return False
        a = np.asarray(value)
        if a.size != dim:
            return False
        nnz = int(np.count_nonzero(a))
        return (nnz / max(1, dim)) <= self.sparse_density_threshold

    def ready(self):
        # Full-covariance local Jacobians need enough samples to span D.  Sparse
        # diagonal geometry intentionally does not: regularized dual solves work
        # with a fixed local neighbourhood in very high ambient dimension.
        if self._sparse_input or self._sparse_output or self.output_dim > self.dense_output_max_dim:
            out_requirement = self.output_dim if (not self._sparse_output and self.output_dim <= self.dense_output_max_dim) else 2
            return self.observations > max(out_requirement, self.neighbors, 4) + 2
        return self.observations > max(self.input_dim, self.output_dim) + 2

    def validate_observation(self, x: Sequence[float], y: Sequence[float] | float):
        sparse_x = self._sparse_input or self._auto_sparse_dense(x, self.input_dim)
        sparse_y = self._sparse_output or self._auto_sparse_dense(y, self.output_dim)

        if sparse_x:
            xa = _sparse_row(x, self.input_dim)
            if not _finite_sparse(xa):
                raise ValueError('digital geometry accepts finite committed values only')
        else:
            xa = _dense_vector(x, self.input_dim)
            if not np.all(np.isfinite(xa)):
                raise ValueError('digital geometry accepts finite committed values only')

        if self.output_dim == 1 and np.asarray(y).ndim == 0 and not _is_sparse(y):
            ya = np.asarray([y], float)
            if not np.all(np.isfinite(ya)):
                raise ValueError('digital geometry accepts finite committed values only')
            sparse_y = False
        elif sparse_y:
            ya = _sparse_row(y, self.output_dim)
            if not _finite_sparse(ya):
                raise ValueError('digital geometry accepts finite committed values only')
        else:
            ya = _dense_vector(y, self.output_dim)
            if not np.all(np.isfinite(ya)):
                raise ValueError('digital geometry accepts finite committed values only')
        return xa, ya, sparse_x, sparse_y

    def _commit_prepared(self, prepared):
        xa, ya, sparse_x, sparse_y = prepared
        # Once a sparse representation is selected, keep the history homogeneous.
        if sparse_x and not self._sparse_input and self._x:
            self._x = deque((_sparse_row(v, self.input_dim) for v in self._x), maxlen=self.max_history)
        if sparse_y and not self._sparse_output and self._y:
            self._y = deque((_sparse_row(v, self.output_dim) for v in self._y), maxlen=self.max_history)
        self._sparse_input = self._sparse_input or sparse_x
        self._sparse_output = self._sparse_output or sparse_y
        if self._sparse_input and not _is_sparse(xa):
            xa = _sparse_row(xa, self.input_dim)
        if self._sparse_output and not _is_sparse(ya):
            ya = _sparse_row(ya, self.output_dim)
        self._x.append(xa.copy())
        self._y.append(ya.copy())
        self._cache = None

    def observe_prevalidated(self, prepared):
        """Commit a tuple returned by :meth:`validate_observation`.

        Hybrid transactions use this two-phase form so validation happens before
        any committed geometry is mutated.
        """
        self._commit_prepared(prepared)

    def stage_observation(self, x, y):
        """Prepare a detached commit without copying immutable history arrays.

        Deques and cache ownership are separate; stored sample arrays are never
        mutated by observe. Subclasses use deep copies for their own state.
        """
        if type(self) is DigitalRelationalGeometry:
            staged = copy.copy(self)
            staged._x = deque(self._x, maxlen=self.max_history)
            staged._y = deque(self._y, maxlen=self.max_history)
            staged._cache = None
        else:
            staged = copy.deepcopy(self)
        staged.observe_prevalidated(staged.validate_observation(x, y))
        return staged

    def observe(self, x: Sequence[float], y: Sequence[float] | float):
        self._commit_prepared(self.validate_observation(x, y))

    def _empty_cache(self):
        D = self.input_dim
        if self._sparse_input:
            return {
                'representation': 'diagonal-sparse',
                'operator_diag': np.zeros(D), 'raw_diag': np.zeros(D),
                'wx_diag': np.zeros(D), 'updates': 0,
                'neighbor_backend': 'unready', 'neighbor_k': 0,
            }
        return {
            'representation': 'dense-full',
            'operator': np.zeros((D, D)), 'raw': np.zeros((D, D)),
            'wx': np.zeros((D, D)), 'updates': 0,
            'neighbor_backend': 'unready', 'neighbor_k': 0,
        }

    def _dense_compute(self):
        D = self.input_dim
        n = self.observations
        X = np.stack(self._x)
        if self._sparse_output or self.output_dim > self.dense_output_max_dim:
            # Sparse outputs are handled by the sparse backend to avoid dense MxM
            # covariance even when input happened to be dense.
            self._sparse_input = True
            self._x = deque((_sparse_row(v, D) for v in self._x), maxlen=self.max_history)
            return self._sparse_compute()
        Y = np.stack(self._y)
        wx = psd_invsqrt(_cov(X, D))
        wy = psd_invsqrt(_cov(Y, self.output_dim))
        Z = (X - X.mean(0)) @ wx.T
        U = (Y - Y.mean(0)) @ wy.T
        k = min(max(self.neighbors, D + 2), n - 1)

        tree = cKDTree(Z, leafsize=self.kd_leafsize, compact_nodes=True, balanced_tree=True)
        _, idx_all = tree.query(Z, k=min(k + 1, n), workers=self.kd_workers)
        if np.ndim(idx_all) == 1:
            idx_all = np.asarray(idx_all)[:, None]

        G = np.zeros((D, D), float)
        updates = 0
        for i in range(n):
            idx = _drop_self(idx_all[i], i, k)
            if not len(idx):
                continue
            DX = Z[idx] - Z[i]
            DY = U[idx] - U[i]
            J = (np.linalg.pinv(DX, rcond=1e-9) @ DY).T
            local = sym(J.T @ J)
            if np.all(np.isfinite(local)):
                G += local
                updates += 1

        raw = sym(G / max(1, updates))
        e, v = np.linalg.eigh(raw)
        e = np.clip(e, 0., None)
        mx = float(e.max()) if e.size else 0.
        op = np.zeros_like(raw) if mx <= EPS else sym(v @ np.diag(np.clip(e / mx, 0., 1.)) @ v.T)
        return {
            'representation': 'dense-full', 'operator': op, 'raw': raw, 'wx': wx,
            'updates': updates, 'neighbor_backend': 'ckdtree-exact', 'neighbor_k': k,
        }

    def _sparse_neighbors(self, Z: sp.csr_matrix, k: int):
        n, D = Z.shape
        if D <= self.sparse_exact_tree_max_dim:
            tree_data = Z.toarray()
            backend = 'ckdtree-exact-sparse-dense'
            q = min(k + 1, n)
            tree = cKDTree(tree_data, leafsize=self.kd_leafsize, compact_nodes=True, balanced_tree=True)
            _, idx_all = tree.query(tree_data, k=q, workers=self.kd_workers)
            if np.ndim(idx_all) == 1:
                idx_all = np.asarray(idx_all)[:, None]
            return [_drop_self(idx_all[i], i, k) for i in range(n)], backend

        projection = _countsketch_dense(Z, self.sparse_projection_dim)
        tree = cKDTree(projection, leafsize=self.kd_leafsize, compact_nodes=True, balanced_tree=True)
        candidate_k = min(n - 1, max(k, self.sparse_candidate_multiplier * k))
        _, candidates = tree.query(projection, k=min(candidate_k + 1, n), workers=self.kd_workers)
        if np.ndim(candidates) == 1:
            candidates = np.asarray(candidates)[:, None]
        out = []
        for i in range(n):
            cand = _drop_self(candidates[i], i, candidate_k)
            if cand.size <= k:
                out.append(cand)
                continue
            diff = _sparse_take_differences(Z, cand, i)
            ds = np.asarray(diff.multiply(diff).sum(axis=1)).ravel()
            # O(candidate_k) selection, not a full sort.
            keep = np.argpartition(ds, k - 1)[:k]
            # Deterministic order among selected neighbours helps reproducibility.
            sel = cand[keep]
            order = np.lexsort((sel, ds[keep]))
            out.append(sel[order])
        return out, 'ckdtree-countsketch-rerank'

    def _sparse_compute(self):
        D = self.input_dim
        M = self.output_dim
        n = self.observations
        X = sp.vstack([_sparse_row(v, D) for v in self._x], format='csr')
        _, var_x = _sparse_mean_var(X)
        wx_diag = _invstd(var_x)
        Z = X.multiply(wx_diag).tocsr()  # centering cancels in pair differences

        use_sparse_y = self._sparse_output or M > self.dense_output_max_dim
        if use_sparse_y:
            Ysp = sp.vstack([_sparse_row(v, M) for v in self._y], format='csr')
            _, var_y = _sparse_mean_var(Ysp)
            wy_diag = _invstd(var_y)
            Usp = Ysp.multiply(wy_diag).tocsr()
            U = None
            wy = None
        else:
            Y = np.stack([_dense_vector(v, M, scalar_ok=(M == 1)) for v in self._y])
            wy = psd_invsqrt(_cov(Y, M))
            U = (Y - Y.mean(0)) @ wy.T
            Usp = None
            wy_diag = None

        # Fixed local support is essential in very high D; requiring D+2 neighbours
        # would make sparse geometry unusable and is unnecessary for the dual solve.
        k = min(max(self.neighbors, 4), n - 1)
        neighbour_rows, backend = self._sparse_neighbors(Z, k)
        raw_diag = np.zeros(D, float)
        updates = 0

        for i, idx in enumerate(neighbour_rows):
            if not len(idx):
                continue
            DX = _sparse_take_differences(Z, idx, i)
            active_x = np.unique(DX.indices)
            if active_x.size == 0:
                continue
            DXa = DX[:, active_x].toarray()
            gram = DXa @ DXa.T
            gram_pinv = np.linalg.pinv(gram, rcond=1e-9)

            if use_sparse_y:
                DY = _sparse_take_differences(Usp, idx, i)
                active_y = np.unique(DY.indices)
                if active_y.size == 0:
                    continue
                DYa = DY[:, active_y].toarray()
            else:
                DYa = U[idx] - U[i]

            alpha = gram_pinv @ DYa
            # J^T = DX^T (DX DX^T)^+ DY.  Only active input columns are formed.
            jt = DXa.T @ alpha
            contrib = np.einsum('ij,ij->i', jt, jt)
            if np.all(np.isfinite(contrib)):
                raw_diag[active_x] += contrib
                updates += 1

        raw_diag /= max(1, updates)
        raw_diag = np.clip(raw_diag, 0.0, None)
        mx = float(raw_diag.max()) if raw_diag.size else 0.0
        op_diag = np.zeros_like(raw_diag) if mx <= EPS else np.clip(raw_diag / mx, 0.0, 1.0)
        return {
            'representation': 'diagonal-sparse',
            'operator_diag': op_diag, 'raw_diag': raw_diag, 'wx_diag': wx_diag,
            'wy_diag': wy_diag, 'wy': wy, 'updates': updates,
            'neighbor_backend': backend, 'neighbor_k': k,
        }

    def _compute(self):
        if self._cache is not None:
            return self._cache
        if not self.ready():
            self._cache = self._empty_cache()
            return self._cache
        self._cache = self._sparse_compute() if self._sparse_input else self._dense_compute()
        return self._cache

    @property
    def storage_mode(self):
        return 'sparse' if self._sparse_input or self._sparse_output else 'dense'

    @property
    def metric_representation(self):
        return self._compute()['representation']

    @property
    def neighbor_backend(self):
        return self._compute().get('neighbor_backend', 'unknown')

    def operator(self):
        c = self._compute()
        if c['representation'] == 'diagonal-sparse':
            return sp.diags(c['operator_diag'], format='csr')
        return c['operator'].copy()

    def raw_operator(self):
        c = self._compute()
        if c['representation'] == 'diagonal-sparse':
            return sp.diags(c['raw_diag'], format='csr')
        return c['raw'].copy()

    @property
    def metric_updates(self):
        return int(self._compute()['updates'])

    def activity_eigenvalues(self):
        c = self._compute()
        if c['representation'] == 'diagonal-sparse':
            return np.sort(np.clip(c['operator_diag'], 0., 1.))[::-1]
        return np.clip(np.linalg.eigvalsh(c['operator']), 0., 1.)[::-1]

    def effective_dimension(self):
        c = self._compute()
        if c['representation'] == 'diagonal-sparse':
            return float(np.sum(c['operator_diag']))
        return float(np.trace(c['operator']))

    def _distance_sparse(self, x1, x2, c):
        a = _sparse_row(x1, self.input_dim)
        b = _sparse_row(x2, self.input_dim)
        d = (b - a).tocsr()
        if d.nnz == 0:
            return 0.0
        cols = d.indices
        vals = d.data * c['wx_diag'][cols]
        return max(0.0, float(np.dot(vals * vals, c['operator_diag'][cols])))

    def distance2(self, x1, x2):
        c = self._compute()
        if not self.ready() or c['updates'] == 0:
            return 0.
        if c['representation'] == 'diagonal-sparse':
            return self._distance_sparse(x1, x2, c)
        d = _dense_vector(x2, self.input_dim) - _dense_vector(x1, self.input_dim)
        z = c['wx'] @ d
        return max(0., float(z @ c['operator'] @ z))

    def distance(self, x1, x2):
        return math.sqrt(self.distance2(x1, x2))

    def report(self):
        c = self._compute()
        base = {
            'meaning': 'dimensionless digital distinguishability for this directed relation',
            'observations': int(self.observations),
            'metric_updates': int(self.metric_updates),
            'effective_dimension': float(self.effective_dimension()),
            'storage_mode': self.storage_mode,
            'metric_representation': c['representation'],
            'neighbor_backend': c.get('neighbor_backend', 'unknown'),
            'neighbor_k': int(c.get('neighbor_k', 0)),
        }
        eig = self.activity_eigenvalues()
        if c['representation'] == 'dense-full' or self.input_dim <= self.report_operator_max_dim:
            base['activity_eigenvalues'] = [float(x) for x in eig]
        else:
            topn=min(32,len(eig));base['activity_eigenvalues_top']=[float(x) for x in eig[:topn]]
            base['activity_eigenvalues_omitted']=int(max(0,len(eig)-topn))
        if c['representation'] == 'dense-full':
            base['operator'] = [[float(x) for x in row] for row in c['operator']]
            base['covariance_mode'] = 'full'
            base['invariance'] = 'full linear-basis whitening (up to numerical rank)'
        else:
            diag = c['operator_diag']
            base['covariance_mode'] = 'diagonal'
            base['invariance'] = 'per-coordinate scale/offset; not arbitrary rotations'
            if self.input_dim <= self.report_operator_max_dim:
                base['operator_diagonal'] = [float(x) for x in diag]
            else:
                nz = np.flatnonzero(diag > 0)
                if nz.size:
                    top = nz[np.argsort(diag[nz])[-min(32, nz.size):][::-1]]
                    base['operator_top_diagonal'] = [[int(i), float(diag[i])] for i in top]
                else:
                    base['operator_top_diagonal'] = []
                base['operator_diagonal_omitted'] = int(self.input_dim - len(base['operator_top_diagonal']))
        return base


__all__ = ['DigitalRelationalGeometry', 'psd_invsqrt']

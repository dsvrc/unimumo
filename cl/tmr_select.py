"""TMR selection core (paper Algorithm 1, OMP variant), copied unchanged from DiscoForcing
scripts/evidence/tmr_select.py (the functions its scripts/build_replay_buffer.py uses): eigen-decomposition of the
temporal operator, the PPCA BIC rank, whitened modal atoms, simplex-constrained least squares and OMP.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


# ----------------------------------------------------------------------------------------------
# Operators and rank
# ----------------------------------------------------------------------------------------------
def temporal_operator(phi: np.ndarray, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """C_k = 1/(N D) sum_i phi~_i phi~_i^T (T x T, Eq. operator) and the mean trajectory phi_bar."""
    N, T, D = phi.shape
    mean = phi.mean(0, dtype=np.float64)
    C = np.zeros((T, T))
    for s in range(0, N, chunk):
        x = (phi[s:s + chunk] - mean).transpose(1, 0, 2).reshape(T, -1)
        C += x @ x.T
    return C / (N * D), mean


def channel_operator(phi: np.ndarray, mean: np.ndarray, chunk: int = 64) -> np.ndarray:
    """C^chan = 1/(N T) sum_i phi~_i^T phi~_i (D x D), the feature-axis operator of the Tab. 4 ablation."""
    N, T, D = phi.shape
    C = np.zeros((D, D))
    for s in range(0, N, chunk):
        x = (phi[s:s + chunk] - mean).reshape(-1, D)
        C += x.T @ x
    return C / (N * T)


def eig_desc(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lam, U = np.linalg.eigh(C)
    return np.clip(lam[::-1], 0, None), U[:, ::-1]


def bic_rank(lam: np.ndarray, n_eff: float) -> tuple[int, dict]:
    """Schwarz criterion on the PPCA model (Eq. bic). lam: descending eigenvalues of a d x d operator."""
    d = len(lam)
    rank = int((lam > lam[0] * 1e-12).sum()) if lam[0] > 0 else 0
    if rank <= 1:
        return 1, {"rank": rank, "M_max": 1, "criterion": []}
    M_max = min(d, rank) - 1
    crit = []
    for M in range(1, M_max + 1):
        s2 = lam[M:].sum() / (d - M)
        fit = np.log(lam[:M]).sum() + (d - M) * np.log(s2)
        crit.append(n_eff * fit + (d * M - M * (M - 1) / 2 + 1) * np.log(n_eff))
    return int(np.argmin(crit)) + 1, {"rank": rank, "M_max": M_max, "criterion": crit}


# ----------------------------------------------------------------------------------------------
# Representations (dictionaries) used for selection
# ----------------------------------------------------------------------------------------------
def modal_dictionary(phi: np.ndarray, kind: str = "temporal", whiten: bool = True, basis: str = "eig",
                     M: int | None = None, rng=None, ops: dict | None = None) -> tuple[np.ndarray, dict]:
    """Atoms z_i = vec(gamma_i) (Eq. whiten) with gamma_i = Lambda^{-1/2} V^T phi_i on the *uncentred* phi_i.
    kind='temporal': V = top-M eigenvectors of C_k (T x T); kind='channel': of C^chan, acting on phi_i^T.
    basis='random': a random orthonormal basis of the same rank, whitened by its own coordinate variances."""
    N, T, D = phi.shape
    ops = ops if ops is not None else operators(phi, channel=(kind == "channel"))
    if kind == "temporal":
        lam, U, n_eff = ops["lam"], ops["U"], N * D
    else:
        lam, U, n_eff = ops["lam_c"], ops["U_c"], N * T
    M = M or bic_rank(lam, n_eff)[0]
    if basis == "eig":
        V, var = U[:, :M], lam[:M]
    else:
        V, _ = np.linalg.qr((rng or np.random.default_rng(0)).standard_normal((U.shape[0], M)))
        Cop = ops["C"] if kind == "temporal" else ops["C_c"]
        var = np.einsum("tm,ts,sm->m", V, Cop, V)
    scale = 1.0 / np.sqrt(np.maximum(var, 1e-30)) if whiten else np.ones(M)
    Vs = (V * scale).astype(np.float32)  # columns already whitened
    if kind == "temporal":
        Z = np.einsum("tm,ntd->nmd", Vs, phi, optimize=True).reshape(N, -1)
    else:
        Z = np.einsum("dm,ntd->nmt", Vs, phi, optimize=True).reshape(N, -1)
    return np.ascontiguousarray(Z, dtype=np.float32), {"M": int(M), "kind": kind, "whiten": whiten, "basis": basis}


def operators(phi: np.ndarray, channel: bool = True) -> dict:
    C, mean = temporal_operator(phi)
    lam, U = eig_desc(C)
    out = {"C": C, "mean": mean, "lam": lam, "U": U}
    if channel:
        Cc = channel_operator(phi, mean)
        lam_c, U_c = eig_desc(Cc)
        out.update(C_c=Cc, lam_c=lam_c, U_c=U_c)
    return out


# ----------------------------------------------------------------------------------------------
# Simplex-constrained refit and OMP (Algorithm 1)
# ----------------------------------------------------------------------------------------------
def project_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    r = ind[u - css / ind > 0][-1]
    return np.maximum(v - css[r - 1] / r, 0.0)


def simplex_lsq(G: np.ndarray, b: np.ndarray, rho: float = 0.0, w0=None, iters: int = 5000,
                tol: float = 1e-13) -> np.ndarray:
    """argmin_{w >= 0, 1^T w = 1} w^T (G + rho I) w - 2 b^T w, i.e. ||y - Z_S w||^2 + rho ||w||^2 with G = Z_S^T Z_S,
    b = Z_S^T y (Algorithm 1, line refit). FISTA with exact simplex projection and adaptive restart, then an exact
    KKT solve on the identified support."""
    n = len(b)
    if n == 1:
        return np.ones(1)
    H = G + rho * np.eye(n)
    f = lambda w: w @ H @ w - 2 * b @ w
    step = 1.0 / (2 * max(np.linalg.eigvalsh(H)[-1], 1e-300))
    w = project_simplex(np.asarray(w0, float)) if w0 is not None else np.full(n, 1.0 / n)
    z, t = w.copy(), 1.0
    for _ in range(iters):
        w_new = project_simplex(z - step * 2 * (H @ z - b))
        if np.dot(z - w_new, w_new - w) > 0:  # restart
            t, z = 1.0, w.copy()
            continue
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        z = w_new + (t - 1) / t_new * (w_new - w)
        done = np.linalg.norm(w_new - w) < tol
        w, t = w_new, t_new
        if done:
            break
    A = np.where(w > 1e-12)[0]
    K = np.zeros((len(A) + 1, len(A) + 1))
    K[:-1, :-1], K[:-1, -1], K[-1, :-1] = 2 * H[np.ix_(A, A)], 1.0, 1.0
    sol = np.linalg.lstsq(K, np.concatenate([2 * b[A], [1.0]]), rcond=None)[0][:-1]
    if (sol >= 0).all():
        w_pol = np.zeros(n)
        w_pol[A] = sol
        if f(w_pol) <= f(w) + 1e-12 * abs(f(w)):
            w = w_pol
    return w


def omp_simplex(Z: np.ndarray, budget: int, groups: np.ndarray | None = None, quotas: dict | None = None,
                y: np.ndarray | None = None, score: str = "raw", rho: float = 0.0, trace: bool = False,
                stop: str = "fw") -> tuple[np.ndarray, np.ndarray, list[float]]:
    """OMP variant of Algorithm 1 on atoms Z (N, d) with target y (default: mean atom): add the atom of largest
    correlation with the residual u (within the per-session quotas), refit simplex weights on the whole support, stop
    at the budget or when no atom can lower the error. trace: modal error ||y - Z_S w|| per iteration (Prop. fw).

    stop='fw': an atom lowers ||y - Z w||^2 on the simplex iff <z_i - Z_S w_S, u> > 0 (the Frank-Wolfe gap), i.e.
    <z_i, u> > <Z_S w_S, u>; this is the fully-corrective Frank-Wolfe of Proposition fw. stop='positive' is the
    literal augment filter <z_i, u> > 0 of Algorithm 1, which is the test for unconstrained / non-negative pursuit:
    with uncentred atoms sharing a large common component every correlation turns negative after the first pick, so
    it can return a one-atom buffer. Both choose the same atom whenever they continue. stop='none' (fixed budget)
    also chooses that atom, and keeps going to the budget when no atom lowers the error any more."""
    N = len(Z)
    y = Z.mean(0, dtype=np.float64) if y is None else np.asarray(y, np.float64)
    norms = np.linalg.norm(Z, axis=1) + 1e-30 if score == "normalized" else None
    S, w, b = [], np.zeros(0), np.zeros(0)
    G = np.zeros((0, 0))
    used = np.zeros(N, bool)
    count = defaultdict(int)
    u = y.copy()
    hist = []
    while len(S) < budget:
        corr = (Z @ u.astype(np.float32)).astype(np.float64)
        if stop == "fw":
            thr = float((y - u) @ u) if S else -np.inf
        elif stop == "none":
            thr = -np.inf
        else:
            thr = 0.0
        ok = corr > thr + 1e-9 * (abs(thr) if np.isfinite(thr) else 0.0) + 1e-12 * float(u @ u)
        ok &= ~used
        if quotas is not None:
            for g, q in quotas.items():
                if count[g] >= q:
                    ok &= groups != g
        if not ok.any():
            break
        sc = corr / norms if norms is not None else corr
        i = int(np.argmax(np.where(ok, sc, -np.inf)))
        zi = Z[i].astype(np.float64)
        gi = Z[S].astype(np.float64) @ zi if S else np.zeros(0)
        G = np.block([[G, gi[:, None]], [gi[None, :], np.array([[zi @ zi]])]])
        b = np.append(b, zi @ y)
        S.append(i)
        used[i] = True
        if groups is not None:
            count[groups[i]] += 1
        w = simplex_lsq(G, b, rho, np.append(w, 0.0))
        u = y - Z[S].astype(np.float64).T @ w
        if trace:
            hist.append(float(np.linalg.norm(u)))
    return np.array(S, dtype=int), w, hist

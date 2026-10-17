# -*- coding: utf-8 -*-
"""
Paper 4 prototype v2 library: EXACT counterfactual lambda-path for the capacity-coupled fleet MILP + calibration on it.

Why exact: deadlines d_k(lambda) = rhat_k - lambda * s_k enter the MILP only through
  (a) which in-house periods p are allowed for unit k:  (p-1) L <= d_k     and
  (b) whether deferral is allowed:                      d_k >= P L.
The costs do not depend on lambda. Hence the optimal plan (with a deterministic solver) is a function of the
'state vector' q(lambda) in {0..P+1}^K, which only changes at the breakpoints lambda = (rhat_k - j L) / s_k,
j = 0..P, lambda > 0: at most K (P+1) breakpoints per fleet instance. Solving the MILP once per interval between
breakpoints gives the loss of every lambda >= 0 exactly (no grid error). This answers the 'counterfactual decisions'
bottleneck of the batch mode of Conformal Decision Theory (Lekeufack et al., ICRA 2024).

Convention: a path is (b, V): b = sorted breakpoints (> 0); V[i] = outcome on the open interval (b_i, b_{i+1})
(b_0 = 0, b_{m+1} = inf). We use the RIGHT-LIMIT loss l+(lam) = V[searchsorted(b, lam, 'right')], which is
right-continuous; the deployed margin is lam_hat + EPS (EPS tiny), whose realised loss is <= the right-limit envelope
at lam_hat, so CRC / LTT guarantees hold exactly (see Section 4 of the paper).

V columns: 0 n_fail, 1 any_fail, 2 cost, 3 model P(any) under N(rhat, sigma^2) independent units (Poisson-binomial),
           4 model E[#fail] (= Markov bound on P(any)), 5 #units outsourced, 6 #units maintained in-house.
"""
import os, sys
import numpy as np
from scipy.stats import binom, norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "prototype_v1"))
import p4v1_lib as V1  # noqa: E402

EPS = 1e-7
NCOL = 7
FLOORS = {"add": None, "sc": 2.0, "s05": 0.5}     # scale s_k: add -> 1; sc -> max(sigma, 2); s05 -> max(sigma, 0.5)


def scale(sig, family):
    if family == "add": return np.ones_like(sig)
    return np.maximum(sig, FLOORS[family])


def deadlines(rhat, sig, lam, family):
    return rhat - lam * scale(sig, family)


def states(d, P, L):
    """State of every unit for deadline vector d (same tolerance as the MILP)."""
    q = np.where(d + 1e-9 >= 0, np.minimum(P, np.floor((d + 1e-9) / L) + 1), 0).astype(int)
    return np.where(d >= P * L, P + 1, q)


def rep_deadline(q, P, L):
    """A deadline vector that reproduces state q exactly in FleetMILP.solve."""
    return np.where(q == 0, -1.0, np.where(q == P + 1, float(P * L), (q - 1) * float(L)))


class PathSolver:
    def __init__(self, milp):
        self.m = milp

    def outcome(self, a, rhat, sig_model, r):
        P, L = self.m.P, self.m.L
        nf, af, cost = self.m.evaluate(a, r)
        start = np.where(a > 0, (a - 1) * L, np.where(a == 0, 0, P * L)).astype(float)
        p = norm.cdf((start - rhat) / sig_model)          # model P(r < start), r ~ N(rhat, sigma^2)
        p = np.where(a == 0, 0.0, p)                      # outsourced now: cannot fail (same as evaluate: start 0, r >= 0)
        pany = 1.0 - float(np.prod(1.0 - p))
        return [nf, af, cost, pany, float(p.sum()), int(np.sum(a == 0)), int(np.sum(a > 0))]

    def path(self, rhat, sig, r, family, sig_model_floor=0.5):
        """Exact path for one fleet instance. Returns (b, V, n_solves)."""
        P, L = self.m.P, self.m.L
        s = scale(sig, family)
        j = np.arange(P + 1)
        bp = ((rhat[:, None] - j[None, :] * L) / s[:, None]).ravel()
        b = np.unique(bp[bp > 0])
        reps = np.concatenate([[b[0] / 2 if len(b) else 0.0], (b[:-1] + b[1:]) / 2, [b[-1] + 1.0] if len(b) else []])
        sm = np.maximum(sig, sig_model_floor)
        cache, Vout = {}, np.zeros((len(reps), NCOL))
        for i, lam in enumerate(reps):
            q = states(rhat - lam * s, P, L)
            key = q.tobytes()
            if key not in cache:
                a = self.m.solve(rhat, rep_deadline(q, P, L))
                cache[key] = self.outcome(a, rhat, sm, r)
            Vout[i] = cache[key]
        return b, Vout, len(cache)


# ------------------------------------------------------------------ evaluation on paths
def at(path, lam):
    b, V = path[0], path[1]
    return V[np.searchsorted(b, lam, side="right")]


def suffix_max(x):
    return np.maximum.accumulate(x[::-1])[::-1]


def candidates(paths):
    return np.unique(np.concatenate([[0.0]] + [p[0] for p in paths]))


def loss_matrix(paths, col, cand, K=None, envelope=True):
    """n x |cand| matrix of right-limit (optionally monotonized) losses at the candidate margins."""
    M = np.zeros((len(paths), len(cand)))
    for i, (b, V, *_) in enumerate(paths):
        v = V[:, col] / (K if K else 1.0)
        if envelope: v = suffix_max(v)
        M[i] = v[np.searchsorted(b, cand, side="right")]
    return M


def crc_path(paths, alpha, K, B=1.0):
    """CRC on the mean failure fraction (col 0 / K) over continuous lambda. Returns deployed margin (inf = outsource all)."""
    cand = candidates(paths); n = len(paths)
    R = loss_matrix(paths, 0, cand, K).mean(0)
    ok = np.where((n * R + B) / (n + 1) <= alpha)[0]
    return float(cand[ok[0]] + EPS) if len(ok) else np.inf


def ltt_path(paths, alpha, delta):
    """LTT (binomial p-values, fixed sequence from large lambda) on the any-failure indicator."""
    cand = candidates(paths); n = len(paths)
    R = loss_matrix(paths, 1, cand).mean(0)
    pv = binom.cdf(np.floor(n * R + 1e-9), n, alpha)
    j = len(cand) - 1
    if pv[j] > delta: return np.inf
    while j > 0 and pv[j - 1] <= delta: j -= 1
    return float(cand[j] + EPS)


def empirical_path(paths, alpha, col, K=None):
    """Validation-tuned: smallest lambda whose raw calibration mean loss is <= alpha (no correction, no envelope)."""
    cand = candidates(paths)
    R = loss_matrix(paths, col, cand, K, envelope=False).mean(0)
    ok = np.where(R <= alpha)[0]
    return float(cand[ok[0]] + EPS) if len(ok) else np.inf


def cheapest_valid_path(paths, lam_hat):
    """Any lambda >= lam_hat keeps the guarantee (Paper 3, Prop. 1): pick the one with the lowest mean calibration cost."""
    if not np.isfinite(lam_hat): return lam_hat
    cand = candidates(paths); cand = np.concatenate([[lam_hat], cand[cand > lam_hat]])
    C = loss_matrix(paths, 2, cand, envelope=False).mean(0)
    return float(cand[int(np.argmin(C))] + (EPS if np.argmin(C) > 0 else 0.0))


def unit_cp(paths_data, alpha, family):
    """Per-unit split conformal on the score (rhat - r)/s over all calibration units."""
    sc = np.concatenate([(rh - r) / scale(sg, family) for rh, sg, r in paths_data])
    s = np.sort(sc); m = len(s); k = int(np.ceil((m + 1) * (1 - alpha)))
    return float(max(s[k - 1], 0.0)) if k <= m else np.inf


def joint_cp(paths_data, alpha, family):
    """Joint (fleet-wide) split conformal on the max score over the K units of each scenario:
       P(all units covered: r_k >= rhat_k - lambda s_k for all k) >= 1 - alpha.  (Lindemann et al. 2025 style)"""
    sc = np.array([np.max((rh - r) / scale(sg, family)) for rh, sg, r in paths_data])
    s = np.sort(sc); n = len(s); k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(max(s[k - 1], 0.0)) if k <= n else np.inf


def scenario_max(paths):
    """Scenario approach with d = 1 (Campi & Garatti 2008) on the monotonized any-failure loss:
       lambda* = max_i lambda_i, lambda_i = smallest lambda from which scenario i has no failure."""
    lam = 0.0
    for b, V, *_ in paths:
        e = suffix_max(V[:, 1]); idx = np.where(e == 0)[0]
        li = 0.0 if idx[0] == 0 else b[idx[0] - 1]
        lam = max(lam, li)
    return float(lam + EPS)


def model_rule(path, alpha, col, K=None):
    """Per-instance model-based chance constraint (no calibration data): smallest lambda whose plan meets the model
       risk under independent N(rhat, sigma^2) RULs. col 3 = exact Poisson-binomial P(any) (Okumusoglu et al. style);
       col 4 = E[#fail] (Markov bound on P(any), Fallahi et al. style; / K = model mean failure fraction)."""
    b, V = path[0], path[1]
    v = V[:, col] / (K if K else 1.0)
    ok = np.where(v <= alpha)[0]
    i = ok[0] if len(ok) else len(v) - 1
    return 0.0 if i == 0 else float(b[i - 1] + EPS)


def nonmono(path):
    """(#increases of n_fail, #increases of any_fail) along the path."""
    V = path[1]
    return int(np.sum(np.diff(V[:, 0]) > 0)), int(np.sum(np.diff(V[:, 1]) > 0))

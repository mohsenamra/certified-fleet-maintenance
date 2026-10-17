# -*- coding: utf-8 -*-
"""
Paper 4 prototype library: C-MAPSS loading, RUL predictor, fleet maintenance MILP, decision losses, calibration rules.

Fleet maintenance model (one planning instance = one fleet scenario):
  K engines, each at its current cycle with predicted RUL rhat_i and (hidden) true RUL r_i.
  Horizon of P periods of L cycles; in-house capacity C slots per period.
  Decisions: x[i,p] maintain engine i in-house at the start of period p (cycle (p-1)L),
             o[i]   outsource now (cycle 0, extra cost),
             u[i]   defer beyond the horizon (allowed only if predicted safe for the whole horizon).
  Safety margin lambda (cycles): maintenance of i must start no later than rhat_i - lambda.
  Objective: min wasted predicted life (c_w per cycle) + outsourcing cost.
Decision loss at lambda (evaluated with the true RUL):
  engine i fails before its planned maintenance  <=>  r_i < start_i   (deferred: r_i < P*L; outsourced: never).
  L_mean = fraction of the fleet that fails, L_any = 1{at least one failure}.
Both losses are bounded in [0, 1]. They are NOT guaranteed monotone in lambda because of the shared capacity
(a tighter deadline for one engine can push another engine to a later slot); the monotone envelope
L~(lambda) = max_{lambda' >= lambda} L(lambda') restores monotonicity and keeps CRC / LTT valid (as in Paper 3).
"""
import os, time
import numpy as np
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "CMAPSS")
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]   # informative sensors (constant ones dropped)
RUL_CAP = 125
WIN = 30


# ------------------------------------------------------------------ data
def load_fd(fd):
    """Return dict unit_id -> array (cycles x [cycle, op1..3, s1..s21])."""
    a = np.loadtxt(os.path.join(DATA, f"train_FD00{fd}.txt"))
    return {(fd, int(u)): a[a[:, 0] == u, 1:] for u in np.unique(a[:, 0])}


def engine_features(arr, mu, sd):
    """Features for every cycle t of one engine, using only data up to t (causal).
    Per sensor: mean of last 5 cycles, mean over window, slope over window; plus current cycle."""
    X = (arr[:, 4 + np.array(SENSORS) - 1] - mu) / sd          # T x S
    T = len(X); t = np.arange(T, dtype=float)
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
    ctx = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X * t[:, None], 0)])
    ct = np.concatenate([[0], np.cumsum(t)]); ct2 = np.concatenate([[0], np.cumsum(t * t)])
    hi = np.arange(1, T + 1); lo_w = np.maximum(0, hi - WIN); lo_5 = np.maximum(0, hi - 5)
    n = (hi - lo_w)[:, None].astype(float)
    sx = cs[hi] - cs[lo_w]; stx = ctx[hi] - ctx[lo_w]
    st = (ct[hi] - ct[lo_w])[:, None]; st2 = (ct2[hi] - ct2[lo_w])[:, None]
    var_t = st2 - st * st / n
    slope = np.where(var_t > 1e-9, (stx - st * sx / n) / np.maximum(var_t, 1e-9), 0.0)
    last5 = (cs[hi] - cs[lo_5]) / (hi - lo_5)[:, None]
    return np.hstack([last5, sx / n, slope, arr[:, [0]]])


class Predictor:
    """Gradient-boosted RUL regressor on causal window features (target capped at RUL_CAP)."""
    def fit(self, engines):
        from sklearn.ensemble import HistGradientBoostingRegressor
        allx = np.vstack([e[:, 4 + np.array(SENSORS) - 1] for e in engines.values()])
        self.mu, self.sd = allx.mean(0), allx.std(0) + 1e-9
        Xs, ys = [], []
        for e in engines.values():
            F = engine_features(e, self.mu, self.sd); T = len(e)
            Xs.append(F); ys.append(np.minimum(T - 1 - np.arange(T), RUL_CAP))
        self.m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
        self.m.fit(np.vstack(Xs), np.concatenate(ys)); return self

    def predict_all(self, e):
        """Predicted RUL at every cycle of engine e (clipped to [0, RUL_CAP])."""
        return np.clip(self.m.predict(engine_features(e, self.mu, self.sd)), 0, RUL_CAP)


# ------------------------------------------------------------------ fleet scenarios
def make_scenarios(pool, preds, n, K, rng, min_hist=31):
    """pool: list of engine keys; preds: key -> predicted RUL per cycle; returns list of (keys, rhat, r)."""
    out = []
    for _ in range(n):
        keys = [pool[i] for i in rng.choice(len(pool), K, replace=False)]
        rh, r = [], []
        for k in keys:
            T = len(preds[k]); t = int(rng.integers(min(min_hist, T - 2), T - 1))   # current cycle index, true RUL >= 1
            rh.append(float(preds[k][t])); r.append(float(T - 1 - t))
        out.append((keys, np.array(rh), np.array(r)))
    return out


# ------------------------------------------------------------------ MILP
class FleetMILP:
    def __init__(self, K, P=6, L=10, C=2, c_waste=1.0, c_out=300.0, c_fail=1000.0):
        self.K, self.P, self.L, self.C = K, P, L, C
        self.c_waste, self.c_out, self.c_fail = c_waste, c_out, c_fail

    def solve(self, rhat, lam):
        """Return assignment per engine: period p in 1..P, 0 = outsourced now, -1 = deferred."""
        import highspy
        K, P, L = self.K, self.P, self.L
        dl = rhat - lam                                    # latest allowed start (cycles from now)
        nv = K * (P + 2)                                   # x[i,p] (P), o[i], u[i]
        idx_x = lambda i, p: i * (P + 2) + (p - 1); idx_o = lambda i: i * (P + 2) + P; idx_u = lambda i: i * (P + 2) + P + 1
        c = np.zeros(nv); ub = np.ones(nv)
        for i in range(K):
            for p in range(1, P + 1):
                start = (p - 1) * L
                c[idx_x(i, p)] = self.c_waste * max(rhat[i] - start, 0.0)
                if start > dl[i] + 1e-9: ub[idx_x(i, p)] = 0      # deadline
            c[idx_o(i)] = self.c_out + self.c_waste * rhat[i]
            if dl[i] < P * L: ub[idx_u(i)] = 0                     # may defer only if predicted safe over horizon
        rows, lo, up = [], [], []
        for i in range(K):                                         # each engine: exactly one option
            rows.append([idx_x(i, p) for p in range(1, P + 1)] + [idx_o(i), idx_u(i)]); lo.append(1); up.append(1)
        for p in range(1, P + 1):                                  # capacity
            rows.append([idx_x(i, p) for i in range(K)]); lo.append(-np.inf); up.append(self.C)
        h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
        lp = highspy.HighsLp(); lp.num_col_, lp.num_row_ = nv, len(rows)
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, np.zeros(nv), ub
        lp.row_lower_, lp.row_upper_ = np.array(lo, float), np.array(up, float)
        start, index = [0], []
        for rw in rows: index += rw; start.append(len(index))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = start, index, [1.0] * len(index)
        lp.integrality_ = [highspy.HighsVarType.kInteger] * nv
        h.passModel(lp); h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal: raise RuntimeError("MILP not optimal")
        x = np.round(np.array(h.getSolution().col_value)).astype(int)
        a = np.full(K, -1)
        for i in range(K):
            if x[idx_o(i)]: a[i] = 0
            for p in range(1, P + 1):
                if x[idx_x(i, p)]: a[i] = p
        return a

    def evaluate(self, a, r):
        """Realised outcome with true RUL r: (n_fail, any_fail, cost, n_out, waste)."""
        P, L = self.P, self.L
        start = np.where(a > 0, (a - 1) * L, np.where(a == 0, 0, P * L)).astype(float)
        fail = (r < start)                                  # fails before planned maintenance / end of horizon
        maint = (a >= 0) & ~fail
        waste = float(np.sum(np.where(maint, r - start, 0)))
        cost = self.c_waste * waste + self.c_out * np.sum(a == 0) + self.c_fail * np.sum(fail)
        return int(fail.sum()), int(fail.any()), float(cost), int(np.sum(a == 0)), waste


# ------------------------------------------------------------------ calibration rules (input: per-scenario loss matrix)
def envelope(Lm):
    """Monotone envelope along lambda (axis 1, increasing lambda): max over lambda' >= lambda."""
    return np.maximum.accumulate(Lm[:, ::-1], axis=1)[:, ::-1]


def crc(Lcal, alpha, B=1.0):
    """Conformal risk control on envelope losses; returns index of smallest valid lambda (or last index)."""
    n = Lcal.shape[0]; R = envelope(Lcal).mean(0)
    ok = np.where((n * R + B) / (n + 1) <= alpha)[0]
    return int(ok[0]) if len(ok) else Lcal.shape[1] - 1


def ltt(Lcal_bin, alpha, delta):
    """Learn-then-Test for a binary loss with fixed-sequence testing from the largest lambda down."""
    n = Lcal_bin.shape[0]; R = envelope(Lcal_bin).mean(0); best = Lcal_bin.shape[1] - 1
    for j in range(Lcal_bin.shape[1] - 1, -1, -1):
        if binom.cdf(np.floor(n * R[j] + 1e-9), n, alpha) <= delta: best = j
        else: break
    return best


def cheapest_valid(j_hat, cost_cal):
    """Any lambda >= lambda_hat keeps the guarantee (envelope is monotone): pick the cheapest on calibration data."""
    m = cost_cal.mean(0); return int(j_hat + np.argmin(m[j_hat:]))


def empirical(Lcal, alpha):
    """Validation-tuned baseline: smallest lambda with empirical risk <= alpha (no finite-sample correction, no envelope)."""
    ok = np.where(Lcal.mean(0) <= alpha)[0]; return int(ok[0]) if len(ok) else Lcal.shape[1] - 1


def engine_cp(res_cal, alpha, grid):
    """Per-engine split conformal lower bound: margin q = (1-alpha) quantile of (rhat - r) with finite-sample correction."""
    s = np.sort(res_cal); m = len(s); k = int(np.ceil((m + 1) * (1 - alpha)))
    q = s[min(k, m) - 1] if k <= m else np.inf
    j = np.searchsorted(grid, q - 1e-9); return int(min(j, len(grid) - 1))

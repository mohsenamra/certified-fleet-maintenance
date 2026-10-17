# -*- coding: utf-8 -*-
"""
Paper 4 prototype v1 library.

New in v1 (vs prototype v0):
  * all four C-MAPSS subsets; sensors normalised per operating condition (6 regimes in FD002/FD004);
  * ensemble predictor (bootstrap GBMs) -> point RUL rhat and a spread sigma (for sigma-scaled margins and for the
    Zhu et al. (2027) / Mitici et al. (2023) style baselines);
  * two margin families: additive  deadline_i = rhat_i - lambda          (lambda in cycles)
                         scaled    deadline_i = rhat_i - lambda * s_i      (s_i = max(sigma_i, SIG_FLOOR))
  * the MILP takes an arbitrary deadline vector, so every baseline uses the same planning model;
  * rolling-horizon fleet simulator for the online (adaptive-margin) experiments.

Fleet planning model (one planning instance):
  K engines, horizon P periods of L cycles, C in-house slots per period.
  x[i,p] maintain i in-house at the start of period p (cycle (p-1)L); o[i] outsource now; u[i] defer beyond horizon.
  Deadline: in-house maintenance of i must start no later than deadline_i; deferral only if deadline_i >= P*L.
  Objective: min wasted predicted life (c_waste per cycle) + outsourcing cost c_out.
  Outsourcing is always feasible (the fallback), so every instance has a solution.
Realised outcome with the true RUL r_i: engine i fails before its planned maintenance iff r_i < start_i.
"""
import os
import numpy as np
from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "CMAPSS")
RUL_CAP = 125
WIN = 30
SIG_FLOOR = 2.0
N_SENS = 21


# ------------------------------------------------------------------ data
def load_fd(fd):
    """dict key=(fd, unit) -> array T x 25 [cycle, set1, set2, set3, s1..s21] (train file: full run-to-failure)."""
    a = np.loadtxt(os.path.join(DATA, f"train_FD00{fd}.txt"))
    return {(fd, int(u)): a[a[:, 0] == u, 1:26] for u in np.unique(a[:, 0])}


class Regimes:
    """Operating-condition clustering on the three settings (6 regimes in C-MAPSS)."""
    def fit(self, engines, k=6):
        S = np.vstack([e[:, 1:4] for e in engines.values()])
        self.scale = S.std(0) + 1e-9
        Z = S / self.scale
        # deterministic k-means (few iterations, init from distinct rounded points)
        uniq = np.unique(np.round(Z, 1), axis=0)
        rng = np.random.default_rng(0)
        C = uniq[rng.choice(len(uniq), min(k, len(uniq)), replace=False)]
        for _ in range(50):
            lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), 1)
            C = np.array([Z[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(len(C))])
        self.C = C
        return self

    def label(self, settings):
        Z = settings / self.scale
        return np.argmin(((Z[:, None, :] - self.C[None]) ** 2).sum(-1), 1)


class Normaliser:
    """Per-regime z-scores of the 21 sensors, fitted on the predictor's training engines."""
    def fit(self, engines, regimes):
        self.reg = regimes; R = len(regimes.C)
        X = np.vstack([e[:, 4:4 + N_SENS] for e in engines.values()])
        L = np.concatenate([regimes.label(e[:, 1:4]) for e in engines.values()])
        g_mu, g_sd = X.mean(0), X.std(0)
        self.mu = np.tile(g_mu, (R, 1)); self.sd = np.tile(g_sd, (R, 1))
        for j in range(R):
            if np.sum(L == j) > 50:
                self.mu[j] = X[L == j].mean(0); self.sd[j] = X[L == j].std(0)
        self.keep = np.where(np.median(self.sd, 0) > 1e-6)[0]    # drop sensors that are constant within regimes
        self.sd = np.where(self.sd > 1e-6, self.sd, 1.0)
        return self

    def __call__(self, e):
        lab = self.reg.label(e[:, 1:4])
        Z = (e[:, 4:4 + N_SENS] - self.mu[lab]) / self.sd[lab]
        return Z[:, self.keep]


def engine_features(Z, cycle):
    """Causal window features per cycle: mean of last 5, window mean, window slope (per sensor) + cycle."""
    X = Z; T = len(X); t = np.arange(T, dtype=float)
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
    return np.hstack([last5, sx / n, slope, cycle[:, None]])


class EnsemblePredictor:
    """M gradient-boosted regressors on engine-level bootstrap samples; returns mean (rhat) and std (sigma)."""
    def __init__(self, M=5, seed=0):
        self.M, self.seed = M, seed

    def fit(self, engines, regimes):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.norm = Normaliser().fit(engines, regimes)
        keys = sorted(engines); rng = np.random.default_rng(self.seed)
        F = {k: engine_features(self.norm(engines[k]), engines[k][:, 0]) for k in keys}
        Y = {k: np.minimum(len(engines[k]) - 1 - np.arange(len(engines[k])), RUL_CAP) for k in keys}
        self.models = []
        for m in range(self.M):
            b = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=m)
            mdl.fit(np.vstack([F[k] for k in b]), np.concatenate([Y[k] for k in b]))
            self.models.append(mdl)
        return self

    def predict_all(self, e):
        F = engine_features(self.norm(e), e[:, 0])
        P = np.clip(np.vstack([m.predict(F) for m in self.models]), 0, RUL_CAP)
        return P.mean(0), P.std(0)


# ------------------------------------------------------------------ fleet scenarios (offline experiments)
def make_scenarios(pool, preds, n, K, rng, min_hist=31):
    """Each scenario: K distinct engines from pool, each observed at a random cycle. Returns list of (rhat, sigma, r)."""
    out = []
    for _ in range(n):
        keys = [pool[i] for i in rng.choice(len(pool), K, replace=False)]
        rh, sg, r = [], [], []
        for k in keys:
            mu, sd = preds[k]; T = len(mu)
            t = int(rng.integers(min(min_hist, T - 2), T - 1))
            rh.append(float(mu[t])); sg.append(float(sd[t])); r.append(float(T - 1 - t))
        out.append((np.array(rh), np.array(sg), np.array(r)))
    return out


# ------------------------------------------------------------------ MILP
class FleetMILP:
    def __init__(self, K, P=6, L=10, C=2, c_waste=1.0, c_out=300.0, c_fail=1000.0):
        self.K, self.P, self.L, self.C = K, P, L, C
        self.c_waste, self.c_out, self.c_fail = c_waste, c_out, c_fail

    def solve(self, rhat, dl):
        """rhat: predicted RUL (for the waste cost); dl: deadline vector. Returns assignment: p in 1..P, 0 = outsource, -1 = defer."""
        import highspy
        K, P, L = len(rhat), self.P, self.L
        nv = K * (P + 2)
        ix = lambda i, p: i * (P + 2) + (p - 1); io = lambda i: i * (P + 2) + P; iu = lambda i: i * (P + 2) + P + 1
        c = np.zeros(nv); ub = np.ones(nv)
        for i in range(K):
            for p in range(1, P + 1):
                s = (p - 1) * L
                c[ix(i, p)] = self.c_waste * max(rhat[i] - s, 0.0)
                if s > dl[i] + 1e-9: ub[ix(i, p)] = 0
            c[io(i)] = self.c_out + self.c_waste * rhat[i]
            if dl[i] < P * L: ub[iu(i)] = 0
        rows, lo, up = [], [], []
        for i in range(K):
            rows.append([ix(i, p) for p in range(1, P + 1)] + [io(i), iu(i)]); lo.append(1); up.append(1)
        for p in range(1, P + 1):
            rows.append([ix(i, p) for i in range(K)]); lo.append(-np.inf); up.append(self.C)
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
            if x[io(i)]: a[i] = 0
            for p in range(1, P + 1):
                if x[ix(i, p)]: a[i] = p
        return a

    def evaluate(self, a, r):
        """(n_fail, any_fail, cost) with true RUL r."""
        P, L = self.P, self.L
        start = np.where(a > 0, (a - 1) * L, np.where(a == 0, 0, P * L)).astype(float)
        fail = r < start
        waste = float(np.sum(np.where((a >= 0) & ~fail, r - start, 0)))
        cost = self.c_waste * waste + self.c_out * np.sum(a == 0) + self.c_fail * np.sum(fail)
        return int(fail.sum()), int(fail.any()), float(cost)


def deadlines(rhat, sig, lam, family):
    """add: rhat - lam; sc: rhat - lam * max(sigma, floor); raw: rhat - lam * sigma (Zhu / Mitici style, no floor)."""
    if family == "add": return rhat - lam
    if family == "raw": return rhat - lam * sig
    return rhat - lam * np.maximum(sig, SIG_FLOOR)


# ------------------------------------------------------------------ calibration rules (offline)
def envelope(Lm):
    """Monotonized loss (Angelopoulos et al., CRC, Cor. 1): max over lambda' >= lambda (axis 1 = increasing lambda)."""
    return np.maximum.accumulate(Lm[:, ::-1], axis=1)[:, ::-1]


def crc(Lcal, alpha, B=1.0):
    n = Lcal.shape[0]; R = envelope(Lcal).mean(0)
    ok = np.where((n * R + B) / (n + 1) <= alpha)[0]
    return int(ok[0]) if len(ok) else Lcal.shape[1] - 1


def ltt(Lbin, alpha, delta):
    n = Lbin.shape[0]; R = envelope(Lbin).mean(0); best = Lbin.shape[1] - 1
    for j in range(Lbin.shape[1] - 1, -1, -1):
        if binom.cdf(np.floor(n * R[j] + 1e-9), n, alpha) <= delta: best = j
        else: break
    return best


def cheapest_valid(j_hat, cost_cal):
    m = cost_cal.mean(0); return int(j_hat + np.argmin(m[j_hat:]))


def empirical(Lcal, alpha):
    ok = np.where(Lcal.mean(0) <= alpha)[0]; return int(ok[0]) if len(ok) else Lcal.shape[1] - 1


def engine_cp(scores, alpha, grid):
    """Per-engine split conformal: (1-alpha) quantile of the score with finite-sample correction -> grid index."""
    s = np.sort(scores); m = len(s); k = int(np.ceil((m + 1) * (1 - alpha)))
    q = s[min(k, m) - 1] if k <= m else np.inf
    return int(min(np.searchsorted(grid, q - 1e-9), len(grid) - 1))


# ------------------------------------------------------------------ rolling-horizon simulator (online experiments)
class FleetSim:
    """Rolling-horizon fleet: every period (L cycles) the planner sees (rhat, sigma) of the K engines in service,
    solves the MILP and executes only the first period: engines planned in period 1 or outsourced are replaced,
    engines whose true remaining life is shorter than L fail during the period and are replaced as well.
    Replacement engines are drawn from `pool_fn(t)` (allows a change of fleet composition = distribution shift)."""
    def __init__(self, milp, preds, lengths, rng, min_start=31, max_start_frac=0.35):
        self.m, self.preds, self.len, self.rng = milp, preds, lengths, rng
        self.min_start, self.max_frac = min_start, max_start_frac

    def new_engine(self, pool):
        k = pool[self.rng.integers(len(pool))]; T = self.len[k]
        hi = max(self.min_start + 1, int(self.max_frac * T))
        return [k, int(self.rng.integers(min(self.min_start, T - 2), min(hi, T - 1)))]

    def run(self, pool_fn, T_periods, rule):
        """rule(t, rhat, sig) -> deadline vector; rule.update(err_mean, err_any) called after each period.
        Returns per-period arrays: failures, maintenances, cost."""
        K, L = self.m.K, self.m.L
        fleet = [self.new_engine(pool_fn(0)) for _ in range(K)]
        F = np.zeros(T_periods, int); M = np.zeros(T_periods, int); Cst = np.zeros(T_periods); LAM = np.zeros(T_periods)
        for t in range(T_periods):
            rh = np.array([self.preds[k][0][c] for k, c in fleet]); sg = np.array([self.preds[k][1][c] for k, c in fleet])
            r = np.array([self.len[k] - 1 - c for k, c in fleet], float)
            dl = rule.deadline(rh, sg); LAM[t] = rule.lam
            a = self.m.solve(rh, dl)
            fails = 0; maint = 0; cost = 0.0
            for i in range(K):
                if a[i] == 1 or a[i] == 0:                  # maintained now (in-house slot 1 or outsourced)
                    maint += 1; cost += self.m.c_waste * r[i] + (self.m.c_out if a[i] == 0 else 0.0)
                    fleet[i] = self.new_engine(pool_fn(t))
                elif r[i] < L:                               # fails during this period
                    fails += 1; cost += self.m.c_fail
                    fleet[i] = self.new_engine(pool_fn(t))
                else:
                    fleet[i][1] += L
            F[t], M[t], Cst[t] = fails, maint, cost
            rule.update(fails / K, float(fails > 0))
        return F, M, Cst, LAM


class FixedRule:
    def __init__(self, lam, family): self.lam, self.family = lam, family
    def deadline(self, rh, sg): return deadlines(rh, sg, self.lam, self.family)
    def update(self, e_mean, e_any): pass


class AdaptiveRule:
    """Online margin update (adaptive conformal / online risk control):
       lambda_{t+1} = max(0, lambda_t + eta * (err_t - alpha)).
    Key fact: if lambda >= lam_max, every deadline is negative, all engines are outsourced now and err_t = 0.
    Hence lambda_t <= lam_max + eta for all t and, telescoping, (1/T) sum_t err_t <= alpha + (lam_max + eta) / (eta * T)
    for ANY sequence of fleet states (no exchangeability, no monotonicity in lambda needed). lam_max is only used in
    the analysis (no upper clipping)."""
    def __init__(self, lam0, family, alpha, eta, lam_max, target="mean"):
        self.lam, self.family, self.alpha, self.eta, self.lam_max, self.target = lam0, family, alpha, eta, lam_max, target
    def deadline(self, rh, sg): return deadlines(rh, sg, self.lam, self.family)
    def update(self, e_mean, e_any):
        e = e_mean if self.target == "mean" else e_any
        self.lam = float(max(0.0, self.lam + self.eta * (e - self.alpha)))

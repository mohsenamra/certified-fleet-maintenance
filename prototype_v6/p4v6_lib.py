# -*- coding: utf-8 -*-
"""
Paper 4 prototype v6: N-CMAPSS (Arias Chao et al. 2021, ten sub-data sets, real flight conditions) case.
Data: data/ncmapss/cycles_<subset>.npz from data/prep_ncmapss.py (one record per unit and flight).
Predictor (fixed in PROTOCOLS.md, P3, before any result):
  * flight-condition correction: for each of the 14 measured sensors, a ridge regression of the flight mean on the
    flight's scenario descriptors (alt, Mach, TRA, T2 means, their squares and pairwise products, flight class one-hot),
    fitted on HEALTHY flights (hs >= 0.99) of the training units; the residuals are degradation signals;
  * causal window features over flights: mean of the last 3 flights, mean and least-squares slope over the last
    10 flights of the residuals, plus the flight index and flight class;
  * ensemble of M = 5 gradient-boosted regressors on unit-level bootstrap samples, target RUL capped at 80 flights;
    rhat = ensemble mean, sigma = ensemble std (same construction as the C-MAPSS predictor of v1).
Renewal-reward baseline (Mitici et al. 2023 style): per engine, replacement time minimising the long-run cost rate
  [c_p + (c_c - c_p) F(t)] / int_0^t (1 - F(u)) du under F = N(rhat, max(sigma, 0.5)^2) truncated at 0.
"""
import os, sys, glob
import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "prototype_v5"))
import p4v5_lib as L5  # noqa: E402

V1, L2 = L5.V1, L5.L2
DATA = os.path.join(HERE, "..", "data", "ncmapss")
SUBSETS = ["DS01-005", "DS02-006", "DS03-012", "DS04", "DS05", "DS06", "DS07", "DS08a-009", "DS08c-008", "DS08d-010"]
RUL_CAP, WIN = 80, 10


def load_units():
    """dict unit -> dict(subset, xs (T x 14), w (T x 4), fc (T), hs (T), rul (T)) with flights sorted by cycle."""
    U = {}
    for sub in SUBSETS:
        f = os.path.join(DATA, f"cycles_{sub}.npz")
        if not os.path.exists(f): continue
        r = np.load(f)
        for u in np.unique(r["unit"]):
            m = np.where(r["unit"] == u)[0]; o = m[np.argsort(r["cycle"][m])]
            U[int(u)] = dict(subset=sub, xs=r["xs_mean"][o].astype(float), w=r["w_mean"][o].astype(float),
                             fc=r["fc"][o].astype(int), hs=r["hs"][o].astype(float), rul=r["rul"][o].astype(float))
    return U


def design(w, fc):
    q = [w, w ** 2] + [(w[:, i] * w[:, j])[:, None] for i in range(4) for j in range(i + 1, 4)]
    oh = np.stack([(fc == k).astype(float) for k in (1, 2, 3)], 1)
    return np.hstack(q + [oh])


class FlightCorrection:
    def fit(self, units, ridge=1e-3):
        Z = np.vstack([design(u["w"], u["fc"])[u["hs"] >= 0.99] for u in units])
        X = np.vstack([u["xs"][u["hs"] >= 0.99] for u in units])
        self.zm, self.zs = Z.mean(0), Z.std(0) + 1e-9
        Zs = np.hstack([np.ones((len(Z), 1)), (Z - self.zm) / self.zs])
        self.B = np.linalg.solve(Zs.T @ Zs + ridge * len(Z) * np.eye(Zs.shape[1]), Zs.T @ X)
        R = X - Zs @ self.B
        self.rs = R.std(0) + 1e-9
        return self

    def __call__(self, u):
        Z = design(u["w"], u["fc"]); Zs = np.hstack([np.ones((len(Z), 1)), (Z - self.zm) / self.zs])
        return (u["xs"] - Zs @ self.B) / self.rs


def window_features(R, fc):
    T = len(R); t = np.arange(T, dtype=float)
    cs = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(R, 0)])
    ctx = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(R * t[:, None], 0)])
    ct = np.concatenate([[0], np.cumsum(t)]); ct2 = np.concatenate([[0], np.cumsum(t * t)])
    hi = np.arange(1, T + 1); lo = np.maximum(0, hi - WIN); lo3 = np.maximum(0, hi - 3)
    n = (hi - lo)[:, None].astype(float)
    sx = cs[hi] - cs[lo]; stx = ctx[hi] - ctx[lo]; st = (ct[hi] - ct[lo])[:, None]; st2 = (ct2[hi] - ct2[lo])[:, None]
    vt = st2 - st * st / n
    slope = np.where(vt > 1e-9, (stx - st * sx / n) / np.maximum(vt, 1e-9), 0.0)
    last3 = (cs[hi] - cs[lo3]) / (hi - lo3)[:, None]
    return np.hstack([last3, sx / n, slope, t[:, None], fc[:, None]])


class NCPredictor:
    def __init__(self, M=5, seed=0):
        self.M, self.seed = M, seed

    def fit(self, units):
        from sklearn.ensemble import HistGradientBoostingRegressor
        keys = sorted(units)
        self.corr = FlightCorrection().fit([units[k] for k in keys])
        F = {k: window_features(self.corr(units[k]), units[k]["fc"]) for k in keys}
        Y = {k: np.minimum(units[k]["rul"], RUL_CAP) for k in keys}
        rng = np.random.default_rng(self.seed); self.models = []
        for m in range(self.M):
            b = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=m)
            mdl.fit(np.vstack([F[k] for k in b]), np.concatenate([Y[k] for k in b]))
            self.models.append(mdl)
        return self

    def predict(self, u):
        F = window_features(self.corr(u), u["fc"])
        P = np.clip(np.vstack([m.predict(F) for m in self.models]), 0, RUL_CAP)
        return P.mean(0), P.std(0)


def renewal_deadline(rhat, sig, c_p=300.0, c_c=1000.0, tmax=None, floor=0.5):
    """Mitici et al. (2023)-style renewal-reward replacement time per engine (in flights from now)."""
    s = np.maximum(sig, floor); out = np.zeros(len(rhat))
    for i, (m, sd) in enumerate(zip(rhat, s)):
        hi = tmax if tmax is not None else m + 4 * sd + 1
        t = np.arange(0.0, max(hi, 1.0) + 1e-9, 0.25)
        z0 = norm.cdf(-m / sd)                                  # truncation at 0
        F = np.clip((norm.cdf((t - m) / sd) - z0) / (1 - z0), 0, 1)
        S = 1 - F
        EL = np.concatenate([[0.0], np.cumsum((S[1:] + S[:-1]) / 2 * np.diff(t))])
        rate = np.where(EL > 0, (c_p + (c_c - c_p) * F) / np.maximum(EL, 1e-12), np.inf)
        out[i] = t[int(np.argmin(rate))]
    return out

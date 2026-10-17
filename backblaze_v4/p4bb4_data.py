# -*- coding: utf-8 -*-
"""
Paper 4, Backblaze v4 step 1: stronger failure-risk predictor (protocol P1 in PROTOCOLS.md).

Changes vs v3b (backblaze_v3/p4bb3_data.py), evaluation period UNCHANGED:
  * longer training history: decision days 2024-02-01 .. TRAIN_END = 2025-10-15 (v3b: 2025-07-01 .. 2025-10-15);
  * two candidate feature sets, chosen per drive model on TRAINING data only:
      A = the v3b features (SMART 5,187,188,197,198,199,193 raw + 7/30-day changes, 9,12,194 raw, age)
      B = every SMART attribute the model reports (raw + normalized) + changes over 3/7/14/30 days (raw) and
          7/30 days (normalized), age;
    selection: fit on decision days <= VAL_START - 29, validate on VAL_START .. TRAIN_END (2025-08-01 .. 2025-10-15),
    choose the higher mean AUC over h7..h28 (B must win by > 0.005, otherwise A); then refit on all training days;
  * per-horizon HistGB bagged over 5 seeds (v3b: 3), each seed with its own disjoint 1% negative subsample.
Leakage-free: features are causal (rows of the same drive up to the decision day); labels of fitting rows end before
the validation days; nothing after TRAIN_END + 28 days is used for training or selection.
Input: data/backblaze/panel4_<Q>_<model>.pkl.gz (data/get_backblaze_v4.py).
Output: data/backblaze/p4bb4_<model>.npz (same format as p4bb3b_*) and p4bb4_<model>_info.json. Resumable per model.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
BB = os.path.join(HERE, "..", "data", "backblaze")
QUARTERS = ["Q1_2024", "Q2_2024", "Q3_2024", "Q4_2024", "Q1_2025", "Q2_2025", "Q3_2025", "Q4_2025", "Q1_2026", "Q2_2026"]
MODELS = ["ST16000NM001G", "TOSHIBA MG07ACA14TA", "TOSHIBA MG08ACA16TA", "WDC WUH722222ALE6L4"]
DAY0 = pd.Timestamp("2024-01-01")
D = lambda s: int((pd.Timestamp(s) - DAY0).days)
TRAIN_START, TRAIN_END, EVAL_START = D("2024-02-01"), D("2025-10-15"), D("2025-11-15")
VAL_START = D("2025-08-01"); FIT_END = VAL_START - 29
RATE_START = D("2025-07-01")            # failures/day for the online target: same window as v3b
H = [7, 14, 21, 28]
RHO, SEEDS_SEL, SEEDS_FINAL = 0.01, 3, 5
RHO_STORE = RHO * SEEDS_FINAL           # stored negatives; seed s uses u in [s*RHO, (s+1)*RHO)
COUNTERS_A = ["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw", "smart_199_raw", "smart_193_raw"]
OTHER_A = ["smart_9_raw", "smart_12_raw", "smart_194_raw"]
NO_DIFF = {"smart_9_raw", "smart_12_raw", "smart_194_raw", "smart_190_raw"}   # age, power cycles, temperature
CONTEXT = 31                            # days of history carried into each window (30-day changes)
WIN = 15                                # decision days per feature window (memory)


def log(*a): print(*a, flush=True)


def tag(m): return m.replace(" ", "_")


def panel(q, m): return pd.read_pickle(os.path.join(BB, f"panel4_{q}_{tag(m)}.pkl.gz"))


# ------------------------------------------------------------------ pass 1: outcomes and columns
def pass1(m):
    fail, last, frac, rate_n, n_tq = {}, {}, {}, [0, set()], 0
    for q in QUARTERS:
        p = panel(q, m)
        f = p.loc[p["failure"] == 1].groupby("sid")["day"].min()
        for s, d in f.items(): fail[s] = min(fail.get(s, 1 << 30), int(d))
        l = p.groupby("sid")["day"].max()
        for s, d in l.items(): last[s] = max(last.get(s, -1), int(d))
        lo, hi = int(p["day"].min()), int(p["day"].max())
        if lo <= TRAIN_END:
            n_tq += 1
            for c in p.columns:
                if c.startswith("smart_"): frac.setdefault(c, []).append(float(p[c].notna().mean()))
        w = p[(p["day"] >= RATE_START) & (p["day"] <= TRAIN_END)]
        rate_n[0] += int(w["failure"].sum()); rate_n[1].update(w["day"].unique().tolist())
        log(f"    pass 1 {q}: {len(p):,} rows, days {lo}..{hi}")
        del p
    n = max(max(fail.keys(), default=0), max(last.keys(), default=0)) + 1
    FA = np.full(n, 1 << 30, np.int64); LA = np.full(n, -1, np.int64)
    for s, d in fail.items(): FA[s] = d
    for s, d in last.items(): LA[s] = d
    smart = sorted((c for c, v in frac.items() if np.mean(v) > 0.5 and len(v) >= 0.5 * n_tq),
                   key=lambda c: (int(c.split("_")[1]), c))
    return FA, LA, smart, rate_n[0] / max(len(rate_n[1]), 1)


def feature_lists(smart):
    A = [c for c in COUNTERS_A + OTHER_A if c in smart]
    fa = []
    for c in A:
        fa.append(c)
        if c in COUNTERS_A: fa += [c + "_d7", c + "_d30"]
    fa.append("age_days")
    fb = []
    for c in smart:
        fb.append(c)
        if c.endswith("_raw") and c not in NO_DIFF: fb += [c + "_d3", c + "_d7", c + "_d14", c + "_d30"]
        if c.endswith("_normalized"): fb += [c + "_d7", c + "_d30"]
    fb.append("age_days")
    return fa, sorted(set(fb), key=fb.index)


def features(fr, fb, keep):
    """fr sorted by (sid, day); returns a float32 array (rows x len(fb)) with all B features (A is a subset).
    Changes are row lags within the same drive (as v3b's groupby-shift), computed with numpy to save memory."""
    n = len(fr); sid = fr["sid"].values
    F = np.full((int(keep.sum()), len(fb)), np.nan, np.float32)      # only the rows that are returned
    same = {}
    for j, c in enumerate(fb):
        if c == "age_days":
            if "smart_9_raw" in fr: F[:, j] = fr["smart_9_raw"].values[keep].astype(np.float32) / 24.0
            continue
        last = c.split("_")[-1]
        if last.startswith("d") and last[1:].isdigit(): base, lag = c[:-(len(last) + 1)], int(last[1:])
        else: base, lag = c, 0
        if base not in fr: continue
        x = fr[base].values.astype(np.float32)
        if lag == 0: F[:, j] = x[keep]; continue
        if lag >= n: continue
        if lag not in same:
            s_ = np.zeros(n, bool); s_[lag:] = sid[lag:] == sid[:-lag]; same[lag] = s_
        d = np.full(n, np.nan, np.float32); d[lag:] = x[lag:] - x[:-lag]
        d[~same[lag]] = np.nan
        F[:, j] = d[keep]
    return F


def windows(m, quarters, fb, day_lo, day_hi):
    """Yields (frame rows, feature frame) for days in [day_lo, day_hi], month by month, with CONTEXT days of history."""
    prev = None
    for q in quarters:
        p = panel(q, m)
        lo_q, hi_q = int(p["day"].min()), int(p["day"].max())
        if hi_q < day_lo - CONTEXT or lo_q > day_hi: prev = p[p["day"] > hi_q - CONTEXT]; continue
        fr = p if prev is None else pd.concat([prev, p], ignore_index=True)
        prev = p[p["day"] > hi_q - CONTEXT]; del p
        fr = fr.sort_values(["sid", "day"], kind="stable", ignore_index=True)
        for a in range(max(lo_q, day_lo), min(hi_q, day_hi) + 1, WIN):
            b = min(a + WIN - 1, hi_q, day_hi)
            sub = fr[(fr["day"] >= a - CONTEXT) & (fr["day"] <= b)].reset_index(drop=True)
            keep = (sub["day"] >= a).values
            F = features(sub, fb, keep)
            yield sub[keep].reset_index(drop=True), F
            del F, sub
        del fr


def outcome(rows, FA, LA):
    s = rows["sid"].values; d = rows["day"].values.astype(np.int64)
    T = FA[s] - d; T = np.where(FA[s] >= (1 << 30), 9999, T)
    return T.astype(np.int64), (LA[s] - d).astype(np.int64)


def labels(T, obs, h):
    y = np.full(len(T), np.nan); y[T <= h] = 1.0; y[(T > h) & (obs >= h)] = 0.0
    return y


# ------------------------------------------------------------------ models
def fit(X, T, obs, u, seeds):
    from sklearn.ensemble import HistGradientBoostingClassifier
    pos = T <= max(H); w = np.where(pos, 1.0, 1.0 / RHO)
    models = []
    for s in range(seeds):
        sel = pos | ((u >= s * RHO) & (u < (s + 1) * RHO))
        row = []
        for h in H:
            y = labels(T, obs, h); m_ = sel & ~np.isnan(y)
            clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                                 random_state=10 * (s + 1) + h)
            clf.fit(X[m_], y[m_], sample_weight=w[m_]); row.append(clf)
        models.append(row)
    return models


def predict(models, X, chunk=500_000):
    out = np.zeros((len(X), len(H)), np.float32)
    for s0 in range(0, len(X), chunk):
        Z = X[s0:s0 + chunk]
        out[s0:s0 + chunk] = np.mean([[m.predict_proba(Z)[:, 1] for m in row] for row in models], axis=0).T
    return np.maximum.accumulate(out, axis=1)


def aucs(G, T, obs):
    r = {}
    for j, h in enumerate(H):
        y = labels(T, obs, h); m_ = ~np.isnan(y)
        r[f"h{h}"] = float(roc_auc_score(y[m_], G[m_, j])) if len(np.unique(y[m_])) == 2 else None
    return r


def run_model(m, rng):
    t0 = time.time(); tg = tag(m)
    fo = os.path.join(BB, f"p4bb4_{tg}.npz")
    if os.path.exists(fo): log(f"{m}: already done"); return
    log(f"== {m} ==")
    FA, LA, smart, rate = pass1(m)
    fa, fb = feature_lists(smart)
    log(f"  SMART columns used: {len(smart)}; features A {len(fa)}, B {len(fb)}; training failures/day (v3b window) {rate:.3f} ({time.time() - t0:.0f}s)")
    # pass 2: stored training rows (all positives within 28 d + RHO_STORE negatives)
    Xs, Ts, Os, Ds, Us = [], [], [], [], []
    r_fail, r_days = 0, set()                    # v3b definition: rows failing the next day / days, 2025-07-01 .. TRAIN_END
    for rows, F in windows(m, QUARTERS[:8], fb, TRAIN_START, TRAIN_END):          # Q1_2024 .. Q4_2025
        T, obs = outcome(rows, FA, LA)
        u = rng.random(len(rows))
        dd = rows["day"].values; w = (dd >= RATE_START) & (dd <= TRAIN_END)
        r_fail += int(np.sum((T == 1) & w)); r_days.update(np.unique(dd[w]).tolist())
        keep = (T >= 1) & ((T <= max(H)) | (u < RHO_STORE))
        Xs.append(F[keep]); Ts.append(T[keep]); Os.append(obs[keep])
        Ds.append(rows["day"].values[keep]); Us.append(np.where(T[keep] <= max(H), 1.0, u[keep]))
    X = np.concatenate(Xs); T = np.concatenate(Ts); obs = np.concatenate(Os); day = np.concatenate(Ds); u = np.concatenate(Us)
    del Xs
    rate = r_fail / max(len(r_days), 1)
    log(f"  training failures/day (v3b definition) {rate:.3f}; stored training rows: {len(X):,} ({int((T <= 28).sum()):,} within 28 d of a failure) ({time.time() - t0:.0f}s)")
    ia = [fb.index(c) for c in fa]
    # selection on training data only
    fitm, valm = day <= FIT_END, (day >= VAL_START) & (day <= TRAIN_END)
    sel = {}
    for name, idx in (("A", ia), ("B", list(range(len(fb))))):
        mods = fit(X[fitm][:, idx], T[fitm], obs[fitm], u[fitm], SEEDS_SEL)
        sel[name] = aucs(predict(mods, X[valm][:, idx]), T[valm], obs[valm])
    mean = {k: float(np.mean([x for x in v.values() if x is not None])) for k, v in sel.items()}
    choice = "B" if mean["B"] > mean["A"] + 0.005 else "A"
    log(f"  validation AUC (fit <= {FIT_END}, validate {VAL_START}..{TRAIN_END}): A {sel['A']} mean {mean['A']:.4f}; "
        f"B {sel['B']} mean {mean['B']:.4f} -> chosen {choice} ({time.time() - t0:.0f}s)")
    idx = ia if choice == "A" else list(range(len(fb)))
    feats = fa if choice == "A" else fb
    Xi = X if len(idx) == X.shape[1] else X[:, idx]
    models = fit(Xi, T, obs, u, SEEDS_FINAL)
    Gtr = predict(models, Xi)
    neg = (T > 28) & (obs >= 28)
    thr = float(np.quantile(Gtr[neg, 3], 0.99)) if neg.any() else 1.0
    n_pos = {f"h{h}": int(np.nansum(labels(T, obs, h) == 1)) for h in H}
    del X, Xi, Gtr
    log(f"  final model ({SEEDS_FINAL} seeds x 4 horizons) fitted ({time.time() - t0:.0f}s)")
    # pass 3: evaluation rows
    E = dict(day=[], sid=[], G=[], T=[], obs=[], s5=[])
    for rows, F in windows(m, QUARTERS[6:], fb, EVAL_START, 1 << 14):              # Q3_2025 (context) .. Q2_2026
        T_, o_ = outcome(rows, FA, LA); k = T_ >= 1
        E["G"].append(predict(models, F[k] if len(idx) == F.shape[1] else F[np.ix_(k, idx)]))
        del F
        E["day"].append(rows["day"].values[k] - EVAL_START); E["sid"].append(rows["sid"].values[k])
        E["T"].append(T_[k]); E["obs"].append(o_[k])
        E["s5"].append(rows["smart_5_raw"].values[k].astype(np.float32) if "smart_5_raw" in rows else np.full(int(k.sum()), np.nan, np.float32))
    E = {k: np.concatenate(v) for k, v in E.items()}
    order = np.lexsort((E["day"], E["sid"]))
    E = {k: v[order] for k, v in E.items()}
    ids, uniq = pd.factorize(E["sid"])
    auc = aucs(E["G"], E["T"], E["obs"])
    np.savez_compressed(fo + ".tmp.npz", day=E["day"].astype(np.int16), drive=ids.astype(np.int32), G=E["G"].astype(np.float32),
                        T=np.minimum(E["T"], 9999).astype(np.int16), obs=np.minimum(E["obs"], 9999).astype(np.int16), smart5=E["s5"])
    json.dump(dict(model=m, features=feats, feature_set=choice, selection_auc=sel, selection_mean=mean,
                   train=dict(stored_rows=int(len(T)), n_pos=n_pos, window=["2024-02-01", "2025-10-15"]),
                   thr_fpr1_G28=thr, auc_eval=auc, train_fail_per_day=rate, n_eval_rows=int(len(ids)),
                   n_eval_drives=int(len(uniq)), eval_days=int(len(np.unique(E["day"]))), eval_failures=int((E["T"] == 1).sum()),
                   seconds=time.time() - t0), open(os.path.join(BB, f"p4bb4_{tg}_info.json"), "w"), indent=1)
    os.replace(fo + ".tmp.npz", fo)
    log(f"  {m}: feature set {choice}; eval AUC { {k: round(v, 3) for k, v in auc.items()} }; FPR-1% thr {thr:.4f}; "
        f"{len(ids):,} eval rows, {int((E['T'] == 1).sum())} failures; saved ({time.time() - t0:.0f}s)")


def main():
    rng = np.random.default_rng(2027); t0 = time.time()
    todo = [m for m in MODELS if os.path.exists(os.path.join(BB, f"panel4_{QUARTERS[-1]}_{tag(m)}.pkl.gz"))]
    for m in todo: run_model(m, rng)
    log(f"data step done in {time.time() - t0:.0f}s")
    return True


if __name__ == "__main__":
    ok = False
    try: ok = main()
    except Exception:
        import traceback; traceback.print_exc()
    sys.exit(0 if ok else 1)

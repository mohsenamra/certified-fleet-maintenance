# -*- coding: utf-8 -*-
"""
Paper 4, real-fleet case (Backblaze hard drives), step 1: features, labels, failure-risk predictor, predictions.

Protocol (leakage-free, forward in time; Lyu et al. 2021, Han et al. 2023):
  * one fleet per drive model (the 4 largest homogeneous fleets kept by data/get_backblaze.py);
  * TRAIN rows: decision days 2025-07-01 .. TRAIN_END (labels look at most 28 days ahead, i.e. <= 2025-11-28);
  * PREDICTION rows: every decision day from EVAL_START = 2025-12-01 (gap of 3 days after the last training label);
  * features are causal (values on the decision day, changes over the last 7 / 30 days, age);
  * a drive that fails on day t is not part of the fleet at decision day t; drives that leave the data without a
    failure flag are right-censored (no label beyond their last day).
Predictor: one gradient-boosted classifier per horizon h in {7, 14, 21, 28} days, G_h(x) = P(fail within h days);
negatives subsampled (rate RHO, weight 1/RHO) so the probabilities refer to the full fleet; made monotone in h.
Output per model: data/backblaze/p4bb_<model>.npz with, for every prediction row: day (days since EVAL_START), drive id,
G (4 horizons), T (days to failure, 9999 = no failure in the data), obs (days still observed after the decision day),
smart5 (for the RAIDShield-style baseline). Plus p4bb_<model>_info.json (training sizes, FPR-1% threshold, AUCs).
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
BB = os.path.join(HERE, "..", "data", "backblaze")
QUARTERS = ["Q3_2025", "Q4_2025", "Q1_2026", "Q2_2026"]
TRAIN_END = pd.Timestamp("2025-10-31")
EVAL_START = pd.Timestamp("2025-12-01")
H = [7, 14, 21, 28]
RHO = 0.01
COUNTERS = ["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw", "smart_199_raw", "smart_193_raw"]
OTHER = ["smart_9_raw", "smart_12_raw", "smart_194_raw"]


def log(*a): print(*a, flush=True)


def models_in_panels():
    s = json.load(open(os.path.join(BB, f"summary_{QUARTERS[0]}.json")))
    return list(s["models"].keys())


def load_model(model):
    parts = []
    for q in QUARTERS:
        f = os.path.join(BB, f"panel_{q}.pkl.gz")
        if not os.path.exists(f): raise FileNotFoundError(f)
        d = pd.read_pickle(f)
        d = d[d["model"] == model]
        parts.append(d.drop(columns=["model", "capacity_bytes"]))
        del d
    df = pd.concat(parts, ignore_index=True)
    df["serial_number"] = df["serial_number"].astype(str)
    df = df.sort_values(["serial_number", "date"]).reset_index(drop=True)
    return df


def build(df):
    """Adds features and outcome columns (T = days to failure, obs = days observed after)."""
    cols = [c for c in COUNTERS + OTHER if c in df.columns and df[c].notna().mean() > 0.5]
    g = df.groupby("serial_number", sort=False)
    feats = {}
    for c in cols:
        feats[c] = df[c]
        if c in COUNTERS:
            feats[c + "_d7"] = df[c] - g[c].shift(7)
            feats[c + "_d30"] = df[c] - g[c].shift(30)
    feats["age_days"] = df["smart_9_raw"] / 24.0
    F = pd.DataFrame(feats)
    fail_day = df.loc[df["failure"] == 1].groupby("serial_number")["date"].min()
    last_day = g["date"].max()
    fd = df["serial_number"].map(fail_day); ld = df["serial_number"].map(last_day)
    T = (fd - df["date"]).dt.days
    out = pd.DataFrame({"date": df["date"], "serial": df["serial_number"],
                        "T": T.fillna(9999).astype(int), "obs": (ld - df["date"]).dt.days.astype(int)})
    keep = out["T"] >= 1                     # failed today -> not in the fleet at this decision day
    return F[keep].reset_index(drop=True), out[keep].reset_index(drop=True), cols


def labels(out, h):
    y = np.full(len(out), np.nan)
    y[(out["T"] <= h).values] = 1.0
    y[((out["T"] > h) & (out["obs"] >= h)).values] = 0.0
    return y


def usable_columns(F, out):
    """Feature columns with data in the training period (e.g. 30-day changes need 30 days of history)."""
    tr = (out["date"] <= TRAIN_END).values
    return [c for c in F.columns if F.loc[tr, c].notna().any()]


def train(F, out, rng):
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr = (out["date"] <= TRAIN_END).values
    pos_any = (out["T"] <= max(H)).values
    sel = tr & (pos_any | (rng.random(len(out)) < RHO))
    w_all = np.where(pos_any, 1.0, 1.0 / RHO)
    models, info = [], {}
    for h in H:
        y = labels(out, h)
        m_ = sel & ~np.isnan(y)
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                             random_state=h)
        clf.fit(F.values[m_], y[m_], sample_weight=w_all[m_])
        models.append(clf)
        info[f"h{h}"] = dict(n_train=int(m_.sum()), n_pos=int(np.nansum(y[m_])))
    return models, info


def predict(models, F):
    G = np.column_stack([m.predict_proba(F.values)[:, 1] for m in models]).astype(np.float32)
    return np.maximum.accumulate(G, axis=1)          # monotone in the horizon


def main():
    t0 = time.time(); rng = np.random.default_rng(2026)
    models_ = models_in_panels()
    log("models:", models_)
    for model in models_:
        tag = model.replace(" ", "_")
        fo = os.path.join(BB, f"p4bb_{tag}.npz")
        if os.path.exists(fo): log(f"{model}: already done"); continue
        t1 = time.time()
        df = load_model(model); log(f"{model}: {len(df):,} drive-days, {df['serial_number'].nunique():,} drives ({time.time() - t1:.0f}s)")
        F, out, cols = build(df)
        del df
        F = F[usable_columns(F, out)]
        models, info = train(F, out, rng)
        log(f"  trained 4 horizons ({time.time() - t1:.0f}s): {info}")
        G_tr = predict(models, F[(out["date"] <= TRAIN_END).values])
        out_tr = out[(out["date"] <= TRAIN_END)]
        neg = ((out_tr["T"] > 28) & (out_tr["obs"] >= 28)).values
        thr_fpr1 = float(np.quantile(G_tr[neg, 3], 0.99)) if neg.any() else 1.0
        ev = (out["date"] >= EVAL_START).values
        G = predict(models, F[ev]); o = out[ev]
        auc = {}
        for j, h in enumerate(H):
            y = labels(o, h); m_ = ~np.isnan(y)
            auc[f"h{h}"] = float(roc_auc_score(y[m_], G[m_, j])) if len(np.unique(y[m_])) == 2 else None
        ids, uniq = pd.factorize(o["serial"])
        np.savez_compressed(fo, day=(o["date"] - EVAL_START).dt.days.values.astype(np.int16), drive=ids.astype(np.int32),
                            G=G, T=np.minimum(o["T"].values, 9999).astype(np.int16), obs=np.minimum(o["obs"].values, 9999).astype(np.int16),
                            smart5=F.loc[ev, "smart_5_raw"].values.astype(np.float32))
        trd = (out["date"] <= TRAIN_END)
        fpd = float(((out["T"] == 1) & trd).sum() / max(out.loc[trd, "date"].nunique(), 1))
        json.dump(dict(model=model, features=list(F.columns), train=info, thr_fpr1_G28=thr_fpr1, auc_eval=auc, train_fail_per_day=fpd,
                       n_eval_rows=int(ev.sum()), n_eval_drives=int(len(uniq)), eval_days=int(o["date"].nunique()),
                       eval_failures=int(((o["T"] == 1)).sum()), seconds=time.time() - t1),
                  open(os.path.join(BB, f"p4bb_{tag}_info.json"), "w"), indent=1)
        log(f"  {model}: eval AUC {auc}; FPR-1% threshold on G28 = {thr_fpr1:.4f}; saved ({time.time() - t1:.0f}s)")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ok = False
    try:
        main(); ok = True
    except Exception:
        import traceback; traceback.print_exc()
    sys.exit(0 if ok else 1)

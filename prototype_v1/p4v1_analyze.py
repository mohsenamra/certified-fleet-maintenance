# -*- coding: utf-8 -*-
"""
Paper 4 prototype v1, step 3: analysis (no new solves).
Offline: every rule is evaluated over many random calibration/test splits.
  realised risk  = test-set mean of the loss at the chosen margin
  >alpha         = share of splits whose realised risk exceeds alpha (target for LTT: <= delta)
  regret*        = test cost minus the cost of the best ADDITIVE margin that meets alpha on that test set (hindsight);
                   negative = cheaper than every valid additive margin (possible for sigma-scaled margins)
Online: long-run failure rate per engine-period, cost per period, mean margin.
Output: results/summary_v1.txt and results/summary_v1.json
"""
import os, json
import numpy as np
import p4v1_lib as L

OUT = os.path.join(L.HERE, "results")
R_ID, R_SHIFT, DELTA = 1000, 500, 0.1
lines, summ = [], {}
def log(s=""): print(s, flush=True); lines.append(s)


def load(name):
    d = dict(np.load(os.path.join(OUT, f"v1_{name}.npz")))
    K = int(d["K"]); d["Kn"] = K
    for fam in ("add", "sc", "zhu", "mitici"):
        d[f"{fam}_mean"] = d[f"{fam}_fail"] / K
    d["sc_s"] = np.maximum(d["sig"], L.SIG_FLOOR)
    return d


def choose(dc, target, alpha):
    """Margins chosen from calibration data dc for every rule. Returns dict rule -> (family, index or None for fixed baselines)."""
    lc = "mean" if target == "mean" else "any"
    ga, gs = dc["grid_add"], dc["grid_sc"]
    res_add = (dc["rhat"] - dc["r"]).ravel(); res_sc = ((dc["rhat"] - dc["r"]) / dc["sc_s"]).ravel()
    K = dc["Kn"]
    rules = {"point (margin 0)": ("add", 0), "fixed margin 20": ("add", int(np.searchsorted(ga, 20))),
             "Zhu-style rhat-3sigma": ("zhu", None), "Mitici-style rhat-2.33sigma": ("mitici", None)}
    for fam, g, res in (("add", ga, res_add), ("sc", gs, res_sc)):
        tag = "additive" if fam == "add" else "sigma-scaled"
        Lc = dc[f"{fam}_{lc}"]
        rules[f"validation-tuned {tag}"] = (fam, L.empirical(Lc, alpha))
        rules[f"per-engine CP {tag} (alpha)"] = (fam, L.engine_cp(res, alpha, g))
        if target == "any":
            rules[f"per-engine CP {tag} (alpha/K)"] = (fam, L.engine_cp(res, alpha / K, g))
            j = L.ltt(Lc, alpha, DELTA); rules[f"LTT {tag}"] = (fam, j)
        else:
            j = L.crc(Lc, alpha); rules[f"CRC {tag}"] = (fam, j)
        rules[f"{'LTT' if target == 'any' else 'CRC'} {tag} cheapest-valid"] = (fam, L.cheapest_valid(j, dc[f"{fam}_cost"]))
    return rules


def outcome(dt, fam, j, target):
    lc = "mean" if target == "mean" else "any"
    if j is None: return dt[f"{fam}_{lc}"].mean(), dt[f"{fam}_cost"].mean(), np.nan
    g = dt["grid_add"] if fam == "add" else dt["grid_sc"]
    return dt[f"{fam}_{lc}"][:, j].mean(), dt[f"{fam}_cost"][:, j].mean(), g[j]


def best_valid_cost(dt, target, alpha):
    lc = "mean" if target == "mean" else "any"
    risk = dt[f"add_{lc}"].mean(0); cost = dt["add_cost"].mean(0)
    ok = np.where(risk <= alpha)[0]
    return cost[ok].min() if len(ok) else cost[-1]


def sub(d, idx):
    out = {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == d["rhat"].shape[0] else v) for k, v in d.items()}
    return out


def evaluate(dcal_full, dtest_full, n, R, target, alpha, rng, same_pool):
    acc = {}
    N = dcal_full["rhat"].shape[0]
    for _ in range(R):
        if same_pool:
            p = rng.permutation(N); dc, dt = sub(dcal_full, p[:n]), sub(dcal_full, p[n:])
        else:
            dc = sub(dcal_full, rng.choice(N, n, replace=False)); dt = dtest_full
        cstar = best_valid_cost(dt, target, alpha)
        for name, (fam, j) in choose(dc, target, alpha).items():
            rk, co, lam = outcome(dt, fam, j, target)
            a = acc.setdefault(name, dict(risk=[], cost=[], lam=[], reg=[]))
            a["risk"].append(rk); a["cost"].append(co); a["lam"].append(lam); a["reg"].append(co - cstar)
    res = {}
    for name, a in acc.items():
        rk = np.array(a["risk"])
        res[name] = dict(risk=float(rk.mean()), above=float(np.mean(rk > alpha + 1e-12)), lam=float(np.nanmean(a["lam"])) if not np.all(np.isnan(a["lam"])) else None,
                         cost=float(np.mean(a["cost"])), regret=float(np.mean(a["reg"])))
    return res


def table(title, res, alpha, target):
    log(""); log(title + (f"  (LTT delta={DELTA})" if target == "any" else ""))
    log(f"  {'rule':44s} {'risk':>7s} {'>alpha':>7s} {'margin':>7s} {'cost':>7s} {'regret*':>7s}")
    for name, r in res.items():
        lam = "  fixed" if r["lam"] is None else f"{r['lam']:7.2f}"
        flag = "  <-- violates" if r["above"] > (DELTA + 0.02 if target == "any" else 0.5) else ""
        log(f"  {name:44s} {r['risk']:7.4f} {r['above']:7.3f} {lam} {r['cost']:7.0f} {r['regret']:7.0f}{flag}")


def mono(d):
    out = {}
    for fam in ("add", "sc"):
        out[fam] = (int(np.sum(np.diff(d[f"{fam}_fail"], axis=1) > 0)), int(np.sum(np.diff(d[f"{fam}_any"], axis=1) > 0)), d[f"{fam}_fail"].size)
    return out


def main():
    rng = np.random.default_rng(11)
    info = json.load(open(os.path.join(OUT, "v1_info.json")))
    log("== Paper 4 prototype v1 =="); log(f"predictor RMSE (capped RUL): {info['rmse']}")
    data = {n: load(n) for n in ("ID", "S13_src", "S13_tgt", "S24_src", "S24_tgt", "SENS_K10", "SENS_K40")}
    log(""); log("Monotonicity violations (increases of the loss when the margin grows): (mean-loss, any-loss, of pairs)")
    for n, d in data.items(): log(f"  {n:9s} {mono(d)}")
    log(""); log("Per-engine residual rhat - r: share optimistic (rhat > r) / mean")
    for n, d in data.items(): log(f"  {n:9s} {np.mean(d['rhat'] > d['r']):.2f} / {np.mean(d['rhat'] - d['r']):.1f}")
    TARGETS = (("mean", (0.02, 0.05)), ("any", (0.1, 0.2)))
    # A. in-distribution
    for target, alphas in TARGETS:
        for a in alphas:
            for n in (50, 100):
                key = f"ID | {target} | alpha={a} | n={n}"
                res = evaluate(data["ID"], None, n, R_ID, target, a, rng, True); summ[key] = res; table(key, res, a, target)
    # B. shift
    for s in ("S13", "S24"):
        for target, alphas in TARGETS:
            a = alphas[0]
            key = f"{s} source->source | {target} | alpha={a} | n=100"
            res = evaluate(data[f"{s}_src"], None, 100, R_SHIFT, target, a, rng, True); summ[key] = res; table(key, res, a, target)
            key = f"{s} NAIVE source->target (shift) | {target} | alpha={a} | n=100"
            res = evaluate(data[f"{s}_src"], data[f"{s}_tgt"], 100, R_SHIFT, target, a, rng, False); summ[key] = res; table(key, res, a, target)
            for nt in (25, 50):
                key = f"{s} RECALIBRATED on target | {target} | alpha={a} | n={nt}"
                res = evaluate(data[f"{s}_tgt"], None, nt, R_SHIFT, target, a, rng, True); summ[key] = res; table(key, res, a, target)
    # C. fleet size / capacity
    for nm in ("SENS_K10", "SENS_K40"):
        for target, a in (("any", 0.1), ("mean", 0.02)):
            key = f"{nm} | {target} | alpha={a} | n=100"
            res = evaluate(data[nm], None, 100, R_SHIFT, target, a, rng, True); summ[key] = res; table(key, res, a, target)
    # D. online
    f = os.path.join(OUT, "v1_online.npz")
    if os.path.exists(f):
        o = np.load(f); oi = json.load(open(os.path.join(OUT, "v1_online_info.json")))
        K, T, TS = oi["K"], oi["T"], oi["T_SHIFT"]
        keys = [k.split("|") for k in o["keys"]]
        log(""); log(f"== Online rolling-horizon runs (K={K}, T={T} periods, shift at period {TS}; static CRC margins {oi['static']}) ==")
        log("   failure rate = failures per engine-period (whole run | after shift+20 periods); mean over seeds; "
            "'>alpha' = share of seeds whose whole-run rate exceeds alpha")
        groups = {}
        for i, (st, mode, rn, al, sd) in enumerate(keys): groups.setdefault((st, mode, rn, al), []).append(i)
        osum = {}
        for (st, mode, rn, al), idx in sorted(groups.items()):
            Fi, Ci, Mi, Li = o["F"][idx], o["C"][idx], o["M"][idx], o["LAM"][idx]
            rate = Fi.sum(1) / (K * T); post = Fi[:, TS + 20:].sum(1) / (K * (T - TS - 20))
            al = float(al)
            row = dict(rate=float(rate.mean()), post=float(post.mean()), above=float(np.mean(rate > al)) if al > 0 else None,
                       cost=float(Ci.mean()), maint=float(Mi.mean()), lam=float(Li.mean()))
            osum[f"{st}|{mode}|{rn}|{al}"] = row
            tag = f"{rn}" + (f" (alpha={al})" if al > 0 else "")
            log(f"  {st} {mode:10s} {tag:28s} rate {row['rate']:.4f} | post-shift {row['post']:.4f}"
                + ("" if row["above"] is None else f"  >alpha {row['above']:.1f}") +
                f"  cost/period {row['cost']:7.0f}  maint/period {row['maint']:.2f}  mean margin {row['lam']:.2f}")
        summ["online"] = osum
        log(f"  theory: long-run rate <= alpha + (lam_max + eta)/(eta*T); additive: lam_max = {L.RUL_CAP + 1}, "
            f"bound slack = {(L.RUL_CAP + 1 + oi['eta_add']) / (oi['eta_add'] * T):.4f}; scaled: lam_max = {L.RUL_CAP / L.SIG_FLOOR + 1:.1f}, "
            f"slack = {(L.RUL_CAP / L.SIG_FLOOR + 1 + oi['eta_sc']) / (oi['eta_sc'] * T):.4f}")
    json.dump(summ, open(os.path.join(OUT, "summary_v1.json"), "w"), indent=1)
    open(os.path.join(OUT, "summary_v1.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\nanalysis done -> results/summary_v1.txt")


if __name__ == "__main__":
    main()

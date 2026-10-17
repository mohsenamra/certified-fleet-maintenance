# -*- coding: utf-8 -*-
"""
Paper 4 prototype v2, step 3: analysis of the exact-path records (and of the online runs if present).
Offline: R random calibration/test splits per setting. Every rule picks a margin from the calibration scenarios
(or per instance, for the model-based rules) and is evaluated EXACTLY on the test scenarios' paths.
  risk     = test mean of the loss at the chosen margin;  >alpha = share of splits with risk > alpha
  cost     = test mean plan cost;  regret* = cost - cost of the best constant ADDITIVE margin valid on the test set
Rules (mean target = failure fraction; any target = P(any failure), LTT delta = 0.1):
  point, fixed 20, Zhu-style, Mitici-style, validation-tuned, per-unit CP (alpha [, alpha/K]), joint max-score CP,
  CRC/LTT exact path (+ cheapest-valid), CRC/LTT on the v1 grid, scenario-max (Campi-Garatti d=1),
  model-based chance constraint with Gaussian RUL: Poisson-binomial P(any) (Okumusoglu-style), Markov E[N] (Fallahi-style).
Output: results/summary_v2.txt, results/summary_v2.json
"""
import os, json, pickle, gzip
import numpy as np
import p4v2_lib as L2

V1 = L2.V1
OUT = os.path.join(L2.HERE, "results")
DELTA = 0.1
R_REPS = {"ID": 500, "default": 300}
GRID = {"add": np.arange(0, 101, 2.0), "sc": np.round(np.arange(0, 20.01, 0.4), 2), "s05": np.round(np.arange(0, 40.01, 0.8), 2)}
HIND = {"add": np.arange(0, 150.5, 0.5)}
lines, summ = [], {}


def log(s=""):
    print(s, flush=True); lines.append(s)


def load(name):
    f = os.path.join(OUT, f"v2_{name}.pkl.gz")
    if not os.path.exists(f): return None
    with gzip.open(f, "rb") as fh: return pickle.load(fh)


def paths_of(S, fam): return [s[fam] for s in S]


def data_of(S): return [(s["rh"], s["sg"], s["r"]) for s in S]


def test_eval(test_paths, lam, col, K=None):
    if not np.isfinite(lam):   # outsource all: no failures; cost = cost of the last interval
        v = np.array([p[1][-1] for p in test_paths])
    else:
        v = np.array([L2.at(p, lam) for p in test_paths])
    loss = v[:, col] / (K if K else 1.0)
    return float(loss.mean()), float(v[:, 2].mean())


def grid_rule(cal_paths, fam, target, alpha, K):
    g = GRID[fam]
    col = 0 if target == "mean" else 1
    M = L2.loss_matrix(cal_paths, col, g + L2.EPS, K if target == "mean" else None, envelope=False)
    j = V1.crc(M, alpha) if target == "mean" else V1.ltt(M, alpha, DELTA)
    return float(g[j] + L2.EPS)


def hindsight_cost(test_paths, target, alpha, K):
    col = 0 if target == "mean" else 1
    g = HIND["add"] + L2.EPS
    Lm = L2.loss_matrix(test_paths, col, g, K if target == "mean" else None, envelope=False).mean(0)
    Cm = L2.loss_matrix(test_paths, 2, g, envelope=False).mean(0)
    ok = np.where(Lm <= alpha)[0]
    return float(Cm[ok].min()) if len(ok) else float(Cm[-1])


def choose(Sc, target, alpha, K, fams):
    """rule -> ('fixed', fam, lambda) or ('inst', fam, col, thr) or ('raw', name)."""
    rules = {"point (lambda 0)": ("fixed", "add", 0.0), "fixed additive 20": ("fixed", "add", 20.0 + L2.EPS),
             "Zhu-style rhat-3sigma": ("raw", "zhu"), "Mitici-style rhat-2.33sigma": ("raw", "mitici")}
    dat = data_of(Sc)
    for fam in fams:
        cp = paths_of(Sc, fam)
        if target == "mean":
            rules[f"[{fam}] validation-tuned"] = ("fixed", fam, L2.empirical_path(cp, alpha, 0, K))
            rules[f"[{fam}] per-unit CP (alpha)"] = ("fixed", fam, L2.unit_cp(dat, alpha, fam))
            lam = L2.crc_path(cp, alpha, K)
            rules[f"[{fam}] CRC exact path"] = ("fixed", fam, lam)
            rules[f"[{fam}] CRC exact cheapest-valid"] = ("fixed", fam, L2.cheapest_valid_path(cp, lam))
            if fam in GRID: rules[f"[{fam}] CRC on v1 grid"] = ("fixed", fam, grid_rule(cp, fam, target, alpha, K))
            rules[f"[{fam}] model Gaussian E[N]/K <= alpha (per instance)"] = ("inst", fam, 4, alpha * K)
        else:
            rules[f"[{fam}] validation-tuned"] = ("fixed", fam, L2.empirical_path(cp, alpha, 1))
            rules[f"[{fam}] per-unit CP (alpha)"] = ("fixed", fam, L2.unit_cp(dat, alpha, fam))
            rules[f"[{fam}] per-unit CP Bonferroni (alpha/K)"] = ("fixed", fam, L2.unit_cp(dat, alpha / K, fam))
            rules[f"[{fam}] joint max-score CP"] = ("fixed", fam, L2.joint_cp(dat, alpha, fam))
            lam = L2.ltt_path(cp, alpha, DELTA)
            rules[f"[{fam}] LTT exact path"] = ("fixed", fam, lam)
            rules[f"[{fam}] LTT exact cheapest-valid"] = ("fixed", fam, L2.cheapest_valid_path(cp, lam))
            if fam in GRID: rules[f"[{fam}] LTT on v1 grid"] = ("fixed", fam, grid_rule(cp, fam, target, alpha, K))
            rules[f"[{fam}] scenario-max (Campi-Garatti d=1)"] = ("fixed", fam, L2.scenario_max(cp))
            rules[f"[{fam}] model Gaussian Poisson-binomial P(any) <= alpha (per instance)"] = ("inst", fam, 3, alpha)
            rules[f"[{fam}] model Markov E[N] <= alpha (per instance)"] = ("inst", fam, 4, alpha)
    return rules


def evaluate(S, n, R, target, alpha, rng, fams, St=None):
    K = S[0]["rh"].shape[0]; col = 0 if target == "mean" else 1
    acc = {}
    for _ in range(R):
        if St is None:
            p = rng.permutation(len(S)); Sc = [S[i] for i in p[:n]]; Ste = [S[i] for i in p[n:]]
        else:
            Sc = [S[i] for i in rng.choice(len(S), n, replace=False)]; Ste = St
        cstar = hindsight_cost(paths_of(Ste, "add"), target, alpha, K)
        for name, rule in choose(Sc, target, alpha, K, fams).items():
            if rule[0] == "fixed":
                rk, co = test_eval(paths_of(Ste, rule[1]), rule[2], col, K if target == "mean" else None); lam = rule[2]
            elif rule[0] == "raw":
                v = np.array([s[rule[1]] for s in Ste], float)
                rk = float((v[:, 0] / K).mean() if target == "mean" else v[:, 1].mean()); co = float(v[:, 2].mean()); lam = np.nan
            else:
                _, fam, mc, thr = rule
                lams = [L2.model_rule(s[fam], thr, mc) for s in Ste]
                v = np.array([L2.at(s[fam], l) for s, l in zip(Ste, lams)])
                rk = float((v[:, 0] / K).mean() if target == "mean" else v[:, 1].mean()); co = float(v[:, 2].mean()); lam = float(np.mean(lams))
            a = acc.setdefault(name, dict(risk=[], cost=[], lam=[], reg=[], kind=rule[0]))
            a["risk"].append(rk); a["cost"].append(co); a["lam"].append(lam if np.isfinite(lam) else np.nan); a["reg"].append(co - cstar)
    res = {}
    for name, a in acc.items():
        rk = np.array(a["risk"]); lm = np.array(a["lam"], float)
        res[name] = dict(risk=float(rk.mean()), above=float(np.mean(rk > alpha + 1e-12)),
                         lam=None if np.all(np.isnan(lm)) else float(np.nanmean(lm)),
                         inf_share=float(np.mean(np.isnan(lm))) if not np.all(np.isnan(lm)) else None,
                         cost=float(np.mean(a["cost"])), regret=float(np.mean(a["reg"])), kind=a["kind"])
    return res


def table(title, res, alpha, target):
    log(""); log(title + (f"  (LTT delta={DELTA})" if target == "any" else ""))
    log(f"  {'rule':66s} {'risk':>7s} {'>alpha':>7s} {'margin':>8s} {'cost':>7s} {'regret*':>8s}")
    for name, r in res.items():
        lam = "   fixed" if r["kind"] == "raw" else ("     inf" if r["lam"] is None else f"{r['lam']:8.2f}")
        if r["kind"] != "raw" and r.get("inf_share") and r["lam"] is not None: lam += f" (inf {r['inf_share']:.0%})"
        flag = "  <-- violates" if r["above"] > (DELTA + 0.02 if target == "any" else 0.5) else ""
        log(f"  {name:66s} {r['risk']:7.4f} {r['above']:7.3f} {lam} {r['cost']:7.0f} {r['regret']:8.0f}{flag}")


def offline():
    rng = np.random.default_rng(11)
    log("== Paper 4 prototype v2: exact lambda-path ==")
    data = {}
    for n in ("ID", "S13_src", "S13_tgt", "S24_src", "S24_tgt", "SENS_K10", "SENS_K40", "CAP_C1", "CAP_C2", "CAP_C3", "CAP_C20"):
        d = load(n)
        if d is not None: data[n] = d
    if not data: log("no v2 records found - run p4v2_offline.py first"); return
    log(""); log("A. Exact path size and monotonicity (share of scenarios whose realised loss INCREASES somewhere as lambda grows)")
    log(f"  {'setting':10s} {'K':>3s} {'C':>3s} {'family':6s} {'plans/scen':>10s} {'grid solves':>11s} {'non-mono n_fail':>15s} {'non-mono any':>12s} {'#increases/scen':>15s}")
    mono = {}
    for n, d in data.items():
        for fam in d["fams"]:
            ps = paths_of(d["scen"], fam); nm = np.array([L2.nonmono(p) for p in ps])
            row = dict(plans=float(np.mean([p[2] for p in ps])), share_mean=float(np.mean(nm[:, 0] > 0)), share_any=float(np.mean(nm[:, 1] > 0)),
                       inc=float(nm[:, 0].mean()))
            mono[f"{n}|{fam}"] = row
            log(f"  {n:10s} {d['K']:3d} {d['C']:3d} {fam:6s} {row['plans']:10.1f} {len(GRID.get(fam, [])):11d} {row['share_mean']:15.3f} {row['share_any']:12.3f} {row['inc']:15.2f}")
    summ["monotonicity"] = mono
    fams = ["add", "sc", "s05"]
    TARGETS = (("mean", (0.02, 0.05)), ("any", (0.1, 0.2)))
    if "ID" in data:
        S = data["ID"]["scen"]
        for target, alphas in TARGETS:
            for a in alphas:
                for n in (50, 100):
                    key = f"ID | {target} | alpha={a} | n={n}"
                    res = evaluate(S, n, R_REPS["ID"], target, a, rng, fams); summ[key] = res; table(key, res, a, target)
    for s in ("S13", "S24"):
        if f"{s}_src" not in data or f"{s}_tgt" not in data: continue
        Ss, St = data[f"{s}_src"]["scen"], data[f"{s}_tgt"]["scen"]
        for target, alphas in TARGETS:
            a = alphas[0]
            key = f"{s} source->source | {target} | alpha={a} | n=100"
            res = evaluate(Ss, 100, R_REPS["default"], target, a, rng, fams); summ[key] = res; table(key, res, a, target)
            key = f"{s} NAIVE source->target (shift) | {target} | alpha={a} | n=100"
            res = evaluate(Ss, 100, R_REPS["default"], target, a, rng, fams, St=St); summ[key] = res; table(key, res, a, target)
            for nt in (25, 50):
                key = f"{s} RECALIBRATED on target | {target} | alpha={a} | n={nt}"
                res = evaluate(St, nt, R_REPS["default"], target, a, rng, fams); summ[key] = res; table(key, res, a, target)
    for nm_ in ("SENS_K10", "SENS_K40", "CAP_C1", "CAP_C2", "CAP_C3", "CAP_C20"):
        if nm_ not in data: continue
        f_ = data[nm_]["fams"]
        for target, a in (("any", 0.1), ("mean", 0.02)):
            key = f"{nm_} (K={data[nm_]['K']}, C={data[nm_]['C']}) | {target} | alpha={a} | n=100"
            res = evaluate(data[nm_]["scen"], 100, R_REPS["default"], target, a, rng, f_); summ[key] = res; table(key, res, a, target)


def online():
    f = os.path.join(OUT, "v2_online.pkl.gz")
    if not os.path.exists(f): return
    with gzip.open(f, "rb") as fh: o = pickle.load(fh)
    cfg = o["cfg"]; K, T, TS = cfg["K"], cfg["T"], cfg["T_SHIFT"]
    log(""); log(f"== Online rolling-horizon runs (K={K}, T={T}, shift at {TS}; static CRC margins (s05) {o['static']}) ==")
    log("   rate = failures per engine-period (whole run | after shift+20); cost = per period; mean over seeds; >alpha = share of seeds above alpha")
    groups = {}
    for k, v in o["runs"].items(): groups.setdefault(k[:-1], []).append(v)
    osum = {}
    for key in sorted(groups):
        st, mode, rn = key
        R_ = groups[key]
        F = np.array([r[0] for r in R_]); C = np.array([r[2] for r in R_]); LAM = np.array([r[3] for r in R_])
        rate = F.sum(1) / (K * T); post = F[:, TS + 20:].sum(1) / (K * (T - TS - 20))
        osum["|".join(key)] = dict(rate=float(rate.mean()), post=float(post.mean()), cost=float(C.mean()), lam=float(LAM.mean()),
                                   rate_seeds=rate.tolist())
    for (st, mode) in sorted({(k.split("|")[0], k.split("|")[1]) for k in osum}):
        log(""); log(f"  {st} {mode}")
        fixed = {k: v for k, v in osum.items() if k.startswith(f"{st}|{mode}|fixed_s05")}
        for alpha in cfg["alphas"]:
            ok = [(v["cost"], k) for k, v in fixed.items() if v["rate"] <= alpha]
            best = min(ok) if ok else None
            log(f"    hindsight best fixed s05 margin with rate <= {alpha}: " + (f"{best[1].split('|')[-1]} cost {best[0]:.0f}" if best else "none"))
            summ.setdefault("online_hindsight", {})[f"{st}|{mode}|{alpha}"] = best
        for k, v in osum.items():
            if not k.startswith(f"{st}|{mode}|"): continue
            rn = k.split("|")[2]
            al = None
            for alpha in cfg["alphas"]:
                if rn.endswith(f"a{alpha}"): al = alpha
            above = "" if al is None else f"  >alpha {np.mean(np.array(v['rate_seeds']) > al):.1f}"
            log(f"    {rn:40s} rate {v['rate']:.4f} | post {v['post']:.4f}{above}  cost {v['cost']:6.0f}  mean margin {v['lam']:6.2f}")
    summ["online"] = osum


def main():
    offline(); online()
    json.dump(summ, open(os.path.join(OUT, "summary_v2.json"), "w"), indent=1, default=str)
    open(os.path.join(OUT, "summary_v2.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\nanalysis done -> results/summary_v2.txt")


if __name__ == "__main__":
    main()

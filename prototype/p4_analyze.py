# -*- coding: utf-8 -*-
"""
Paper 4 prototype, step 2: calibration rules evaluated over many random calibration/test splits (no new solves).
Targets: (i) mean fraction of the fleet failing before planned maintenance <= alpha  (CRC)
         (ii) P(at least one failure in the fleet plan) <= alpha with prob. >= 1 - delta  (LTT)
Output: results/summary_v0.json and results/summary_v0.txt
"""
import os, json
import numpy as np
import p4_lib as L

R = 1000; DELTA = 0.1; NS = (25, 50, 100)
OUT = os.path.join(L.HERE, "results")
d = np.load(os.path.join(OUT, "records_v0.npz")); grid = d["grid"]; K = d["A_res"].shape[1]
lines = []; summ = {}
def log(s=""): print(s); lines.append(s)


def mono_violations(M):
    return int(np.sum(np.diff(M, axis=1) > 1e-12))


def run(cal_tag, test_tag, n, target, alpha, rng):
    """Evaluate every rule; cal/test drawn from pool cal_tag (disjoint split) or calibrate on A, test on all of B."""
    Fc, Ac, Cc, Rc = d[f"{cal_tag}_fail"] / K, d[f"{cal_tag}_any"], d[f"{cal_tag}_cost"], d[f"{cal_tag}_res"]
    Ft, At, Ct, Ot = d[f"{test_tag}_fail"] / K, d[f"{test_tag}_any"], d[f"{test_tag}_cost"], d[f"{test_tag}_oracle"]
    N = Fc.shape[0]; out = {}
    for _ in range(R):
        p = rng.permutation(N); cal = p[:n]; te = p[n:] if cal_tag == test_tag else np.arange(Ft.shape[0])
        Lc = Fc[cal] if target == "mean" else Ac[cal]
        rules = {"point (lambda=0)": 0, "fixed margin 10": int(np.searchsorted(grid, 10)), "fixed margin 20": int(np.searchsorted(grid, 20)),
                 "fixed margin 30": int(np.searchsorted(grid, 30)), "validation-tuned (no correction)": L.empirical(Lc, alpha)}
        if target == "mean":
            rules["per-engine conformal (alpha)"] = L.engine_cp(Rc[cal].ravel(), alpha, grid)
            j = L.crc(Lc, alpha); rules["CRC (decision-level)"] = j; rules["CRC cheapest-valid"] = L.cheapest_valid(j, Cc[cal])
        else:
            rules["per-engine conformal (alpha)"] = L.engine_cp(Rc[cal].ravel(), alpha, grid)
            rules["per-engine conformal Bonferroni (alpha/K)"] = L.engine_cp(Rc[cal].ravel(), alpha / K, grid)
            j = L.ltt(Lc, alpha, DELTA); rules["LTT (decision-level)"] = j; rules["LTT cheapest-valid"] = L.cheapest_valid(j, Cc[cal])
        Lt = Ft[te] if target == "mean" else At[te]
        for name, j in rules.items():
            o = out.setdefault(name, dict(risk=[], lam=[], cost=[], gap=[]))
            o["risk"].append(Lt[:, j].mean()); o["lam"].append(grid[j]); o["cost"].append(Ct[te, j].mean())
            o["gap"].append((Ct[te, j].mean() - Ot[te].mean()) / Ot[te].mean())
    res = {}
    for name, o in out.items():
        rk = np.array(o["risk"])
        res[name] = dict(realized_risk=float(rk.mean()), frac_splits_above=float(np.mean(rk > alpha + 1e-12)),
                         lam_mean=float(np.mean(o["lam"])), cost_mean=float(np.mean(o["cost"])), cost_gap_vs_oracle=float(np.mean(o["gap"])))
    return res


def main():
    rng = np.random.default_rng(7)
    log("== Paper 4 prototype v0 (C-MAPSS FD001+FD003, GBM predictor, fleet MILP K=%d) ==" % K)
    for t in ("A", "B"):
        log(f"pool {t}: monotonicity violations  mean-loss {mono_violations(d[t + '_fail'])}, any-loss {mono_violations(d[t + '_any'])}"
            f"  (of {d[t + '_fail'].size} scenario-lambda pairs)")
        log(f"pool {t}: per-engine residual rhat-r  mean {d[t + '_res'].mean():.1f}, P(rhat > r) {np.mean(d[t + '_res'] > 0):.2f}; oracle cost {d[t + '_oracle'].mean():.0f}")
    for setting in (("A", "A"), ("A", "B")):
        for target, alphas in (("mean", (0.01, 0.02, 0.05)), ("any", (0.1, 0.2))):
            for alpha in alphas:
                for n in NS:
                    key = f"{setting[0]}->{setting[1]} | {target} | alpha={alpha} | n={n}"
                    res = run(setting[0], setting[1], n, target, alpha, rng); summ[key] = res
                    log(""); log(key + ("  (LTT delta=%.1f)" % DELTA if target == "any" else ""))
                    log(f"  {'rule':42s} {'risk':>7s} {'>alpha':>7s} {'lambda':>7s} {'cost':>8s} {'vs oracle':>9s}")
                    for name, r in res.items():
                        log(f"  {name:42s} {r['realized_risk']:7.4f} {r['frac_splits_above']:7.3f} {r['lam_mean']:7.1f} {r['cost_mean']:8.0f} {100 * r['cost_gap_vs_oracle']:8.1f}%")
    json.dump(summ, open(os.path.join(OUT, "summary_v0.json"), "w"), indent=1)
    open(os.path.join(OUT, "summary_v0.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\nanalysis done -> results/summary_v0.txt")


if __name__ == "__main__":
    main()

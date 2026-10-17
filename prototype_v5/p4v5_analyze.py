# -*- coding: utf-8 -*-
"""
Paper 4 prototype v5, step 2: analysis of the rich-planner records with EXACTLY the v2 protocol and rules
(p4v2_analyze.evaluate / choose / table are reused unchanged; protocol P2 in PROTOCOLS.md).
Output: results/summary_v5.txt, results/summary_v5.json
"""
import os, sys, json, pickle, gzip
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "prototype_v2"))
import p4v5_lib as L5  # noqa: E402,F401  (path setup)
import p4v2_analyze as A  # noqa: E402

SMOKE = os.environ.get("P4_SMOKE") == "1"
OUT = os.path.join(HERE, "results_smoke" if SMOKE else "results")
R_ID, R_DEF = (20, 10) if SMOKE else (500, 300)
NS, NMAIN, NTS = ((5, 8), 8, (4, 6)) if SMOKE else ((50, 100), 100, (25, 50))
FAMS = ["add", "sc", "s05"]
log = A.log


def load(name):
    f = os.path.join(OUT, f"v5_{name}.pkl.gz")
    if not os.path.exists(f): return None
    with gzip.open(f, "rb") as fh: return pickle.load(fh)


def main():
    rng = np.random.default_rng(12)
    log("== Paper 4 prototype v5: rich planner (multi-period shop visits, spare engines, setup cost), exact lambda-path ==")
    data = {n: d for n in ("RICH_ID", "RICH_TIGHT", "RICH_S13_src", "RICH_S13_tgt", "RICH_S24_src", "RICH_S24_tgt")
            if (d := load(n)) is not None}
    ex = os.path.join(OUT, "v5_exactness.json")
    if os.path.exists(ex):
        e = json.load(open(ex)); A.summ["exactness"] = e
        log(""); log("Exactness (direct MILP solves at random margins that match the path): " +
                     ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in e.items()))
    log(""); log("A. Path size and monotonicity")
    mono = {}
    for n, d in data.items():
        for fam in d["fams"]:
            ps = A.paths_of(d["scen"], fam); nm = np.array([L5.L2.nonmono(p) for p in ps])
            row = dict(plans=float(np.mean([p[2] for p in ps])), share_mean=float(np.mean(nm[:, 0] > 0)),
                       share_any=float(np.mean(nm[:, 1] > 0)), sec=float(np.mean([s["sec"] for s in d["scen"]])))
            mono[f"{n}|{fam}"] = row
            log(f"  {n:13s} C={d['C']} S0={d['S0']} {fam:4s} plans/scen {row['plans']:6.1f}  non-mono n_fail {row['share_mean']:.3f}"
                f"  any {row['share_any']:.3f}  path sec/scen {row['sec']:.1f}")
    A.summ["monotonicity"] = mono
    TARGETS = (("mean", (0.02, 0.05)), ("any", (0.1, 0.2)))
    if "RICH_ID" in data:
        S = data["RICH_ID"]["scen"]
        for target, alphas in TARGETS:
            for a in alphas:
                for n in NS:
                    key = f"RICH_ID | {target} | alpha={a} | n={n}"
                    res = A.evaluate(S, n, R_ID, target, a, rng, FAMS); A.summ[key] = res; A.table(key, res, a, target)
    if "RICH_TIGHT" in data:
        for target, alphas in TARGETS:
            a = alphas[0]
            key = f"RICH_TIGHT | {target} | alpha={a} | n=100"
            res = A.evaluate(data["RICH_TIGHT"]["scen"], NMAIN, R_DEF, target, a, rng, FAMS); A.summ[key] = res; A.table(key, res, a, target)
    for s in ("S13", "S24"):
        if f"RICH_{s}_src" not in data or f"RICH_{s}_tgt" not in data: continue
        Ss, St = data[f"RICH_{s}_src"]["scen"], data[f"RICH_{s}_tgt"]["scen"]
        for target, alphas in TARGETS:
            a = alphas[0]
            key = f"RICH {s} source->source | {target} | alpha={a} | n=100"
            res = A.evaluate(Ss, NMAIN, R_DEF, target, a, rng, FAMS); A.summ[key] = res; A.table(key, res, a, target)
            key = f"RICH {s} NAIVE source->target (shift) | {target} | alpha={a} | n=100"
            res = A.evaluate(Ss, NMAIN, R_DEF, target, a, rng, FAMS, St=St); A.summ[key] = res; A.table(key, res, a, target)
            for nt in NTS:
                key = f"RICH {s} RECALIBRATED on target | {target} | alpha={a} | n={nt}"
                res = A.evaluate(St, nt, R_DEF, target, a, rng, FAMS); A.summ[key] = res; A.table(key, res, a, target)
    open(os.path.join(OUT, "summary_v5.txt"), "w", encoding="utf-8").write("\n".join(A.lines) + "\n")
    json.dump(A.summ, open(os.path.join(OUT, "summary_v5.json"), "w"), indent=1)
    log(""); log("written summary_v5.txt / summary_v5.json")


if __name__ == "__main__":
    main()

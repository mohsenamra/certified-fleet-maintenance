# -*- coding: utf-8 -*-
"""
Paper 4 prototype v6, step 2: N-CMAPSS analysis. Same protocol and rules as v2/v5 (p4v2_analyze unchanged) plus two
direct baselines from the literature (PROTOCOLS.md, P3):
  * "de Pater-style cost-tuned margin": the additive margin that minimises the mean calibration cost, as when the
    safety buffer of an alarm-based schedule is tuned for cost on past data (de Pater et al., 2022);
  * "Mitici 2023 renewal-reward": per-engine replacement times from the renewal-reward criterion under the predictive
    distribution (Mitici et al., 2023), planned with the same planner.
Output: results/summary_v6.txt, results/summary_v6.json
"""
import os, sys, json, pickle, gzip
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "prototype_v2"))
import p4v6_lib as L6  # noqa: E402,F401
import p4v2_analyze as A  # noqa: E402

L2 = L6.L2
SMOKE = os.environ.get("P4_SMOKE") == "1"
OUT = os.path.join(HERE, "results_smoke" if SMOKE else "results")
R_ID, R_DEF = (10, 6) if SMOKE else (300, 300)
NS, NMAIN, NTS = ((5, 8), 8, (4, 6)) if SMOKE else ((50, 100), 100, (25, 50))
FAMS = ["add", "sc", "s05"]
log = A.log
_orig_choose = A.choose


def choose6(Sc, target, alpha, K, fams):
    rules = _orig_choose(Sc, target, alpha, K, fams)
    cp = A.paths_of(Sc, "add"); cand = L2.candidates(cp)
    C = L2.loss_matrix(cp, 2, cand, envelope=False).mean(0)
    rules["de Pater-style cost-tuned margin [add]"] = ("fixed", "add", float(cand[int(np.argmin(C))] + L2.EPS))
    rules["Mitici 2023 renewal-reward deadlines"] = ("raw", "renewal")
    return rules


A.choose = choose6


def load(name):
    f = os.path.join(OUT, f"v6_{name}.pkl.gz")
    if not os.path.exists(f): return None
    with gzip.open(f, "rb") as fh: return pickle.load(fh)


def main():
    rng = np.random.default_rng(13)
    log("== Paper 4 prototype v6: N-CMAPSS (ten sub-data sets), rich planner, exact lambda-path ==")
    info = os.path.join(OUT, "v6_info.json")
    if os.path.exists(info):
        I = json.load(open(info)); A.summ["info"] = {k: I[k] for k in ("units", "subsets", "rmse")}
        log(f"units {I['units']}; predictor RMSE (flights, RUL capped at {L6.RUL_CAP}): {I['rmse']}")
    ex = os.path.join(OUT, "v6_exactness.json")
    if os.path.exists(ex):
        e = json.load(open(ex)); A.summ["exactness"] = e
        log("Exactness: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in e.items()))
    data = {n: d for n in ("NC_ID", "NC_SHIFT_src", "NC_SHIFT_tgt") if (d := load(n)) is not None}
    log(""); log("A. Path size and monotonicity")
    mono = {}
    for n, d in data.items():
        for fam in d["fams"]:
            ps = A.paths_of(d["scen"], fam); nm = np.array([L2.nonmono(p) for p in ps])
            row = dict(plans=float(np.mean([p[2] for p in ps])), share_mean=float(np.mean(nm[:, 0] > 0)), share_any=float(np.mean(nm[:, 1] > 0)))
            mono[f"{n}|{fam}"] = row
            log(f"  {n:13s} {fam:4s} plans/scen {row['plans']:6.1f}  non-mono n_fail {row['share_mean']:.3f}  any {row['share_any']:.3f}")
    A.summ["monotonicity"] = mono
    TARGETS = (("mean", (0.02, 0.05)), ("any", (0.1, 0.2)))
    if "NC_ID" in data:
        for target, alphas in TARGETS:
            for a in alphas:
                for n in NS:
                    key = f"NC_ID | {target} | alpha={a} | n={n}"
                    res = A.evaluate(data["NC_ID"]["scen"], n, R_ID, target, a, rng, FAMS); A.summ[key] = res; A.table(key, res, a, target)
    if "NC_SHIFT_src" in data and "NC_SHIFT_tgt" in data:
        Ss, St = data["NC_SHIFT_src"]["scen"], data["NC_SHIFT_tgt"]["scen"]
        for target, alphas in TARGETS:
            a = alphas[0]
            key = f"NC source->source | {target} | alpha={a} | n={NMAIN}"
            res = A.evaluate(Ss, NMAIN, R_DEF, target, a, rng, FAMS); A.summ[key] = res; A.table(key, res, a, target)
            key = f"NC NAIVE source->target (new fault modes) | {target} | alpha={a} | n={NMAIN}"
            res = A.evaluate(Ss, NMAIN, R_DEF, target, a, rng, FAMS, St=St); A.summ[key] = res; A.table(key, res, a, target)
            for nt in NTS:
                key = f"NC RECALIBRATED on target | {target} | alpha={a} | n={nt}"
                res = A.evaluate(St, nt, R_DEF, target, a, rng, FAMS); A.summ[key] = res; A.table(key, res, a, target)
    open(os.path.join(OUT, "summary_v6.txt"), "w", encoding="utf-8").write("\n".join(A.lines) + "\n")
    json.dump(A.summ, open(os.path.join(OUT, "summary_v6.json"), "w"), indent=1)
    log(""); log("written summary_v6.txt / summary_v6.json")


if __name__ == "__main__":
    main()

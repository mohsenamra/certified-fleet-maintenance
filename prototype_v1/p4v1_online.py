# -*- coding: utf-8 -*-
"""
Paper 4 prototype v1, step 2: rolling-horizon (online) experiments.
A fleet of K engines is planned every period with the MILP; only the first period is executed; maintained and failed
engines are replaced by new engines from a pool. Settings (per predictor S13 / S24):
  stationary : replacements always from the source pool
  shift      : from period T_SHIFT on, replacements come from the target pool (new fault mode)
Rules: point (lambda = 0), static CRC margin calibrated offline on the source pool (horizon-level loss, n = 100),
       Zhu-style rhat - 3 sigma, Mitici-style rhat - 2.326 sigma, and the adaptive margin (additive and sigma-scaled).
Target of the adaptive rule: long-run fraction of engine-periods with a failure <= alpha.
Output: results/v1_online.npz + results/v1_online_info.json. Needs the offline step first.
"""
import os, time, json, pickle
import numpy as np
from multiprocessing import Pool
import p4v1_lib as L

OUT = os.path.join(L.HERE, "results")
CFG = dict(K=20, P=6, Lp=10, C=2, T=1000, T_SHIFT=300, seeds=5, alphas=[0.0025, 0.005],
           eta_add=100.0, eta_sc=30.0, lam0_add=10.0, lam0_sc=2.0, crc_alpha=0.02, crc_n=100, workers=4)
P = pickle.load(open(os.path.join(OUT, "v1_predictors.pkl"), "rb"))
LEN = pickle.load(open(os.path.join(OUT, "v1_lengths.pkl"), "rb"))


def static_crc(setting):
    d = np.load(os.path.join(OUT, f"v1_{setting}_src.npz"))
    Lm = d["add_fail"][:CFG["crc_n"]] / d["K"]
    return float(d["grid_add"][L.crc(Lm, CFG["crc_alpha"])])


def run_one(args):
    setting, mode, rule_name, alpha, seed = args
    d = P[setting]; rng = np.random.default_rng([seed, sum(map(ord, setting + mode + rule_name)), int(alpha * 1e5)])
    src, tgt = d["pools"]["src"], d["pools"]["tgt"]
    pool_fn = (lambda t: src) if mode == "stationary" else (lambda t: src if t < CFG["T_SHIFT"] else tgt)
    m = L.FleetMILP(CFG["K"], CFG["P"], CFG["Lp"], CFG["C"])
    sim = L.FleetSim(m, d["preds"], LEN, rng)
    if rule_name == "point": rule = L.FixedRule(0.0, "add")
    elif rule_name == "static_CRC": rule = L.FixedRule(STATIC[setting], "add")
    elif rule_name == "zhu": rule = L.FixedRule(3.0, "raw")
    elif rule_name == "mitici": rule = L.FixedRule(2.326, "raw")
    elif rule_name == "adaptive_add": rule = L.AdaptiveRule(CFG["lam0_add"], "add", alpha, CFG["eta_add"], None)
    elif rule_name == "adaptive_sc": rule = L.AdaptiveRule(CFG["lam0_sc"], "sc", alpha, CFG["eta_sc"], None)
    F, M, C, LAM = sim.run(pool_fn, CFG["T"], rule)
    return (setting, mode, rule_name, alpha, seed), (F, M, C, LAM)


STATIC = {}


def main():
    t0 = time.time()
    for s in ("S13", "S24"): STATIC[s] = static_crc(s)
    print("static CRC margins (offline, source pool):", STATIC, flush=True)
    tasks = []
    for setting in ("S13", "S24"):
        for mode in ("stationary", "shift"):
            for seed in range(CFG["seeds"]):
                for rn in ("point", "static_CRC", "zhu", "mitici"):
                    tasks.append((setting, mode, rn, 0.0, seed))
                for a in CFG["alphas"]:
                    for rn in ("adaptive_add", "adaptive_sc"):
                        tasks.append((setting, mode, rn, a, seed))
    print(f"{len(tasks)} simulation runs of {CFG['T']} periods", flush=True)
    res = {}
    with Pool(CFG["workers"], initializer=_init, initargs=(STATIC,)) as pw:
        for i, (k, v) in enumerate(pw.imap_unordered(run_one, tasks, chunksize=2)):
            res[k] = v
            if (i + 1) % 20 == 0: print(f"  {i + 1}/{len(tasks)} runs ({time.time() - t0:.0f}s)", flush=True)
    keys = sorted(res)
    np.savez_compressed(os.path.join(OUT, "v1_online.npz"),
                        keys=np.array(["|".join(map(str, k)) for k in keys]),
                        F=np.array([res[k][0] for k in keys]), M=np.array([res[k][1] for k in keys]),
                        C=np.array([res[k][2] for k in keys]), LAM=np.array([res[k][3] for k in keys]))
    json.dump(dict(CFG, static=STATIC, seconds=time.time() - t0), open(os.path.join(OUT, "v1_online_info.json"), "w"), indent=1)
    print(f"online runs done in {time.time() - t0:.0f}s", flush=True)


def _init(static):
    STATIC.update(static)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Paper 4 prototype v1, step 1 (offline records; no calibration here).
Settings
  ID   : predictor trained on part of FD002+FD004, scenarios from held-out FD002+FD004 engines (one pool, random cal/test splits)
  S13  : predictor trained on FD001; source pool = held-out FD001, target pool = FD003 (new fault mode: shift)
  S24  : predictor trained on FD002; source pool = held-out FD002, target pool = FD004 (new fault mode, 6 regimes: shift)
  SENS : ID pool with fleet size K = 10 (C = 1) and K = 40 (C = 4)
For every scenario and every margin on two grids (additive, sigma-scaled) the fleet MILP is solved and the realised
outcome recorded, plus the Zhu-style (rhat - 3 sigma) and Mitici-style (rhat - 2.326 sigma) baselines.
Output: results/v1_<setting>.npz, results/v1_predictors.pkl, results/v1_info.json. Resumable per pool.
"""
import os, sys, time, json, pickle
import numpy as np
from multiprocessing import Pool
import p4v1_lib as L

OUT = os.path.join(L.HERE, "results"); os.makedirs(OUT, exist_ok=True)
CFG = dict(seed=2026, K=20, P=6, Lp=10, C=2, n_scen=400, n_sens=200, M=5,
           grid_add=list(np.arange(0, 101, 2.0)), grid_sc=list(np.round(np.arange(0, 20.01, 0.4), 2)),
           zhu_omega=3.0, mitici_z=2.326, workers=4)
GA, GS = np.array(CFG["grid_add"]), np.array(CFG["grid_sc"])


def work(args):
    rh, sg, r, K, C = args
    m = L.FleetMILP(K, CFG["P"], CFG["Lp"], C)
    res = {}
    for fam, grid in (("add", GA), ("sc", GS)):
        F = np.zeros(len(grid), int); A = np.zeros(len(grid), int); Co = np.zeros(len(grid))
        for j, lam in enumerate(grid):
            F[j], A[j], Co[j] = m.evaluate(m.solve(rh, L.deadlines(rh, sg, lam, fam)), r)
        res[fam] = (F, A, Co)
    res["zhu"] = m.evaluate(m.solve(rh, rh - CFG["zhu_omega"] * sg), r)
    res["mitici"] = m.evaluate(m.solve(rh, rh - CFG["mitici_z"] * sg), r)
    return res


def record(name, scen, K, C, pool_workers):
    f = os.path.join(OUT, f"v1_{name}.npz")
    if os.path.exists(f):
        print(f"  {name}: already done", flush=True); return
    t0 = time.time()
    out = pool_workers.map(work, [(rh, sg, r, K, C) for rh, sg, r in scen], chunksize=4)
    d = {"rhat": np.array([s[0] for s in scen]), "sig": np.array([s[1] for s in scen]), "r": np.array([s[2] for s in scen])}
    for fam in ("add", "sc"):
        d[f"{fam}_fail"] = np.array([o[fam][0] for o in out]); d[f"{fam}_any"] = np.array([o[fam][1] for o in out])
        d[f"{fam}_cost"] = np.array([o[fam][2] for o in out])
    for b in ("zhu", "mitici"):
        d[f"{b}_fail"] = np.array([o[b][0] for o in out]); d[f"{b}_any"] = np.array([o[b][1] for o in out])
        d[f"{b}_cost"] = np.array([o[b][2] for o in out])
    np.savez_compressed(f, grid_add=GA, grid_sc=GS, K=K, C=C, **d)
    print(f"  {name}: {len(scen)} scenarios, K={K}, C={C} ({time.time() - t0:.0f}s)", flush=True)


def build_predictors(eng, rng):
    """Returns dict setting -> (predictor, pools) and saves predictions for every pool engine."""
    reg = L.Regimes().fit({k: v for k, v in eng.items() if k[0] == 2})   # the 6 regimes, learned from FD002 settings only
    by = {fd: sorted(k for k in eng if k[0] == fd) for fd in (1, 2, 3, 4)}
    perm = {fd: [by[fd][i] for i in rng.permutation(len(by[fd]))] for fd in by}
    split = {
        "ID": (perm[2][:160] + perm[4][:150], {"pool": perm[2][160:] + perm[4][150:]}),
        "S13": (perm[1][:60], {"src": perm[1][60:], "tgt": perm[3]}),
        "S24": (perm[2][:160], {"src": perm[2][160:], "tgt": perm[4]}),
    }
    P = {}
    for name, (tr, pools) in split.items():
        t0 = time.time()
        pr = L.EnsemblePredictor(CFG["M"], seed=1).fit({k: eng[k] for k in tr}, reg)
        preds = {k: pr.predict_all(eng[k]) for ks in pools.values() for k in ks}
        err = {pn: float(np.sqrt(np.mean(np.concatenate([(preds[k][0] - np.minimum(len(eng[k]) - 1 - np.arange(len(eng[k])), L.RUL_CAP)) ** 2 for k in ks]))))
               for pn, ks in pools.items()}
        P[name] = dict(preds=preds, pools=pools, rmse=err, train=tr)
        print(f"  predictor {name}: trained on {len(tr)} engines ({time.time() - t0:.0f}s); RMSE vs capped RUL: {err}", flush=True)
    return P


def main():
    t0 = time.time(); rng = np.random.default_rng(CFG["seed"])
    eng = {}
    for fd in (1, 2, 3, 4): eng.update(L.load_fd(fd))
    print(f"engines loaded: {len(eng)}", flush=True)
    pf = os.path.join(OUT, "v1_predictors.pkl")
    if os.path.exists(pf):
        P = pickle.load(open(pf, "rb")); print("  predictors loaded from v1_predictors.pkl", flush=True)
    else:
        P = build_predictors(eng, rng); pickle.dump(P, open(pf, "wb"))
    lengths = {k: len(v) for k, v in eng.items()}
    pickle.dump(lengths, open(os.path.join(OUT, "v1_lengths.pkl"), "wb"))
    with Pool(CFG["workers"]) as pw:
        for name, pool_name in (("ID", "pool"), ("S13_src", "src"), ("S13_tgt", "tgt"), ("S24_src", "src"), ("S24_tgt", "tgt")):
            setting = name.split("_")[0]; d = P[setting]
            srng = np.random.default_rng([CFG["seed"], sum(map(ord, name))])
            scen = L.make_scenarios(d["pools"][pool_name], d["preds"], CFG["n_scen"], CFG["K"], srng)
            record(name, scen, CFG["K"], CFG["C"], pw)
        d = P["ID"]
        for K, C in ((10, 1), (40, 4)):
            srng = np.random.default_rng(CFG["seed"] + K)
            scen = L.make_scenarios(d["pools"]["pool"], d["preds"], CFG["n_sens"], K, srng)
            record(f"SENS_K{K}", scen, K, C, pw)
    json.dump(dict(CFG, rmse={k: v["rmse"] for k, v in P.items()}, seconds=time.time() - t0),
              open(os.path.join(OUT, "v1_info.json"), "w"), indent=1, default=float)
    print(f"offline records done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

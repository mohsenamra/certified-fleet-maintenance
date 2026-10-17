# -*- coding: utf-8 -*-
"""
Paper 4 prototype v2, step 1: EXACT lambda-path records (no calibration here).
Reuses the v1 predictors (prototype_v1/results/v1_predictors.pkl) so predictions are identical to v1.
For every fleet scenario and every margin family (add, sc = max(sigma,2), s05 = max(sigma,0.5)) the full piecewise-constant
loss path over lambda >= 0 is computed exactly (one MILP per distinct plan), plus the fixed Zhu (rhat-3sigma) and
Mitici (rhat-2.326sigma) plans.
Settings: ID, S13_src/tgt, S24_src/tgt (as v1, 400 scenarios, K=20, C=2), SENS_K10 (C=1), SENS_K40 (C=4),
          CAP_C1/C2/C3/C20 (ID pool, K=20, capacity 1, 2, 3, 20 = non-binding; 200 scenarios; family add).
Output: results/v2_<name>.pkl.gz ; resumable per setting.
"""
import os, time, json, pickle, gzip
import numpy as np
from multiprocessing import Pool
import p4v2_lib as L2

V1 = L2.V1
OUT = os.path.join(L2.HERE, "results"); os.makedirs(OUT, exist_ok=True)
PRED = os.path.join(L2.HERE, "..", "prototype_v1", "results", "v1_predictors.pkl")
CFG = dict(seed=2026, K=20, P=6, Lp=10, C=2, n_scen=400, n_sens=200, n_cap=200, families=["add", "sc", "s05"],
           zhu_omega=3.0, mitici_z=2.326, workers=4)


def work(args):
    rh, sg, r, K, C, fams = args
    m = V1.FleetMILP(K, CFG["P"], CFG["Lp"], C); ps = L2.PathSolver(m)
    res = {"rh": rh, "sg": sg, "r": r}
    t0 = time.time()
    for fam in fams:
        b, V, n = ps.path(rh, sg, r, fam)
        res[fam] = (b, V.astype(np.float32), n)
    res["zhu"] = m.evaluate(m.solve(rh, rh - CFG["zhu_omega"] * sg), r)
    res["mitici"] = m.evaluate(m.solve(rh, rh - CFG["mitici_z"] * sg), r)
    res["sec"] = time.time() - t0
    return res


def record(name, scen, K, C, fams, pw):
    f = os.path.join(OUT, f"v2_{name}.pkl.gz")
    if os.path.exists(f):
        print(f"  {name}: already done", flush=True); return
    t0 = time.time()
    out = []
    for i, o in enumerate(pw.imap(work, [(rh, sg, r, K, C, fams) for rh, sg, r in scen], chunksize=2)):
        out.append(o)
        if (i + 1) % 50 == 0: print(f"    {name}: {i + 1}/{len(scen)} ({time.time() - t0:.0f}s)", flush=True)
    with gzip.open(f + ".tmp", "wb") as fh:
        pickle.dump(dict(K=K, C=C, P=CFG["P"], L=CFG["Lp"], fams=fams, scen=out), fh)
    os.replace(f + ".tmp", f)
    ns = {fam: float(np.mean([o[fam][2] for o in out])) for fam in fams}
    print(f"  {name}: {len(scen)} scenarios, K={K}, C={C}; distinct plans per scenario {ns} ({time.time() - t0:.0f}s)", flush=True)


def main():
    t0 = time.time()
    P = pickle.load(open(PRED, "rb"))
    jobs = []
    for name, pool_name in (("ID", "pool"), ("S13_src", "src"), ("S13_tgt", "tgt"), ("S24_src", "src"), ("S24_tgt", "tgt")):
        d = P[name.split("_")[0]]
        srng = np.random.default_rng([CFG["seed"], sum(map(ord, name))])     # same seeds as v1 -> same scenarios
        jobs.append((name, V1.make_scenarios(d["pools"][pool_name], d["preds"], CFG["n_scen"], CFG["K"], srng), CFG["K"], CFG["C"], CFG["families"]))
    d = P["ID"]
    for K, C in ((10, 1), (40, 4)):
        srng = np.random.default_rng(CFG["seed"] + K)
        jobs.append((f"SENS_K{K}", V1.make_scenarios(d["pools"]["pool"], d["preds"], CFG["n_sens"], K, srng), K, C, CFG["families"]))
    srng = np.random.default_rng(CFG["seed"] + 777)
    cap_scen = V1.make_scenarios(d["pools"]["pool"], d["preds"], CFG["n_cap"], CFG["K"], srng)
    for C in (1, 2, 3, 20):
        jobs.append((f"CAP_C{C}", cap_scen, CFG["K"], C, ["add"]))
    with Pool(CFG["workers"]) as pw:
        for name, scen, K, C, fams in jobs:
            record(name, scen, K, C, fams, pw)
    json.dump(dict(CFG, seconds=time.time() - t0), open(os.path.join(OUT, "v2_offline_info.json"), "w"), indent=1)
    print(f"v2 offline records done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

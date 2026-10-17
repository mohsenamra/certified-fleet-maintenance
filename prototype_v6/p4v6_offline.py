# -*- coding: utf-8 -*-
"""
Paper 4 prototype v6, step 1: N-CMAPSS predictors and exact lambda-path records (protocol P3 in PROTOCOLS.md).
Predictors (results/v6_predictors.pkl):
  NC_ID    all units of the ten sub-data sets; 5-fold cross-fitting grouped by unit (stratified by sub-data set):
           every unit is predicted by an ensemble trained on the other folds.
  NC_SHIFT source = DS01, DS02, DS03, DS07 (turbine faults): 5-fold cross-fitted predictions;
           target = DS04, DS05, DS06 (fan / compressor faults, unseen): ensemble trained on all source units.
Planner: p4v5_lib.RichFleetMILP with K 10, P 6 periods of L 5 flights, C 1 shop line, D 2, D_out 3, S0 2 spares,
         c_waste 2 per flight, c_out 300, c_fail 1000, c_setup 100, c_lease 40.
Records per scenario: exact paths for families add / sc / s05, fixed Zhu (rhat - 3 sigma), Mitici-quantile
(rhat - 2.326 sigma) and Mitici 2023 renewal-reward deadline plans.
Settings: NC_ID 1000 scenarios; NC_SHIFT_src 400; NC_SHIFT_tgt 400. Exactness: 100 direct solves per setting.
Output: results/v6_<name>.pkl.gz, results/v6_predictors.pkl, results/v6_exactness.json, results/v6_info.json
"""
import os, time, json, pickle, gzip
import numpy as np
from multiprocessing import Pool
import p4v6_lib as L6

V1, L2, L5 = L6.V1, L6.L2, L6.L5
SMOKE = os.environ.get("P4_SMOKE") == "1"
OUT = os.path.join(L6.HERE, "results_smoke" if SMOKE else "results"); os.makedirs(OUT, exist_ok=True)
CFG = dict(seed=2028, K=10, P=6, Lp=5, C=1, D=2, D_out=3, S0=2, c_waste=2.0, c_out=300.0, c_fail=1000.0,
           c_setup=100.0, c_lease=40.0, M=5, folds=5, min_hist=10, families=["add", "sc", "s05"],
           zhu_omega=3.0, mitici_z=2.326, renewal_cp=300.0, renewal_cc=1000.0,
           n={"NC_ID": 1000, "NC_SHIFT_src": 400, "NC_SHIFT_tgt": 400}, n_check=100,
           source=["DS01-005", "DS02-006", "DS03-012", "DS07"], target=["DS04", "DS05", "DS06"],
           workers=int(os.environ.get("P4_WORKERS", "4")))


def milp():
    return L5.RichFleetMILP(CFG["K"], CFG["P"], CFG["Lp"], CFG["C"], CFG["D"], CFG["D_out"], CFG["S0"],
                            c_waste=CFG["c_waste"], c_out=CFG["c_out"], c_fail=CFG["c_fail"],
                            c_setup=CFG["c_setup"], c_lease=CFG["c_lease"])


def crossfit(U, keys, rng, tag):
    """Grouped K-fold cross-fitting stratified by sub-data set."""
    fold = {}
    for sub in sorted({U[k]["subset"] for k in keys}):
        ks = [k for k in keys if U[k]["subset"] == sub]; ks = [ks[i] for i in rng.permutation(len(ks))]
        for i, k in enumerate(ks): fold[k] = i % CFG["folds"]
    preds = {}
    for f in range(CFG["folds"]):
        tr = [k for k in keys if fold[k] != f]; te = [k for k in keys if fold[k] == f]
        t0 = time.time()
        pr = L6.NCPredictor(CFG["M"], seed=f).fit({k: U[k] for k in tr})
        for k in te: preds[k] = pr.predict(U[k])
        print(f"    {tag} fold {f + 1}/{CFG['folds']}: train {len(tr)} units, predict {len(te)} ({time.time() - t0:.0f}s)", flush=True)
    return preds


def rmse(U, preds, keys):
    e = np.concatenate([preds[k][0] - np.minimum(U[k]["rul"], L6.RUL_CAP) for k in keys])
    return float(np.sqrt(np.mean(e ** 2)))


def build_predictors(U):
    rng = np.random.default_rng(CFG["seed"])
    allk = sorted(U); src = sorted(k for k in U if U[k]["subset"] in CFG["source"])
    tgt = sorted(k for k in U if U[k]["subset"] in CFG["target"])
    P = {}
    p = crossfit(U, allk, rng, "NC_ID")
    P["NC_ID"] = dict(preds=p, pools={"pool": allk}, rmse={"pool": rmse(U, p, allk)})
    p = crossfit(U, src, rng, "NC_SHIFT source")
    pr = L6.NCPredictor(CFG["M"], seed=99).fit({k: U[k] for k in src})
    for k in tgt: p[k] = pr.predict(U[k])
    P["NC_SHIFT"] = dict(preds=p, pools={"src": src, "tgt": tgt}, rmse={"src": rmse(U, p, src), "tgt": rmse(U, p, tgt)})
    for k, v in P.items(): print(f"  predictor {k}: RMSE vs capped RUL {v['rmse']}", flush=True)
    return P


def work(args):
    rh, sg, r, fams = args
    m = milp(); ps = L2.PathSolver(m)
    res = {"rh": rh, "sg": sg, "r": r}; t0 = time.time()
    for fam in fams:
        b, V, n = ps.path(rh, sg, r, fam); res[fam] = (b, V.astype(np.float32), n)
    res["zhu"] = m.evaluate(m.solve(rh, rh - CFG["zhu_omega"] * sg), r)
    res["mitici"] = m.evaluate(m.solve(rh, rh - CFG["mitici_z"] * sg), r)
    res["renewal"] = m.evaluate(m.solve(rh, L6.renewal_deadline(rh, sg, CFG["renewal_cp"], CFG["renewal_cc"])), r)
    res["sec"] = time.time() - t0
    return res


def check(args):
    rec, fam, lam = args
    m = milp(); a = m.solve(rec["rh"], L2.deadlines(rec["rh"], rec["sg"], lam, fam))
    return bool(np.allclose(np.array(m.evaluate(a, rec["r"]), float), np.array(L2.at(rec[fam], lam)[:3], float), atol=1e-3))


def main():
    t0 = time.time()
    U = L6.load_units()
    subs = sorted({u["subset"] for u in U.values()})
    print(f"N-CMAPSS: {len(U)} units from {len(subs)} sub-data sets {subs}", flush=True)
    pf = os.path.join(OUT, "v6_predictors.pkl")
    if os.path.exists(pf): P = pickle.load(open(pf, "rb")); print("  predictors loaded", flush=True)
    else: P = build_predictors(U); pickle.dump(P, open(pf, "wb"))
    ex_file = os.path.join(OUT, "v6_exactness.json")
    exact = json.load(open(ex_file)) if os.path.exists(ex_file) else {}
    with Pool(CFG["workers"]) as pw:
        for name, (setting, pool_name) in {"NC_ID": ("NC_ID", "pool"), "NC_SHIFT_src": ("NC_SHIFT", "src"),
                                           "NC_SHIFT_tgt": ("NC_SHIFT", "tgt")}.items():
            n = 16 if SMOKE else CFG["n"][name]
            f = os.path.join(OUT, f"v6_{name}.pkl.gz")
            if not os.path.exists(f):
                d = P[setting]; srng = np.random.default_rng([CFG["seed"], sum(map(ord, name))])
                scen = V1.make_scenarios(d["pools"][pool_name], d["preds"], n, CFG["K"], srng, min_hist=CFG["min_hist"])
                t1, out = time.time(), []
                for i, o in enumerate(pw.imap(work, [(rh, sg, r, CFG["families"]) for rh, sg, r in scen], chunksize=2)):
                    out.append(o)
                    if (i + 1) % 50 == 0: print(f"    {name}: {i + 1}/{n} ({time.time() - t1:.0f}s)", flush=True)
                with gzip.open(f + ".tmp", "wb") as fh:
                    pickle.dump(dict(K=CFG["K"], C=CFG["C"], S0=CFG["S0"], P=CFG["P"], L=CFG["Lp"], fams=CFG["families"], scen=out), fh)
                os.replace(f + ".tmp", f)
                print(f"  {name}: {n} scenarios; plans/scenario {np.mean([o['add'][2] for o in out]):.1f} ({time.time() - t1:.0f}s)", flush=True)
            else:
                print(f"  {name}: already done", flush=True)
            if name not in exact:
                with gzip.open(f, "rb") as fh: d6 = pickle.load(fh)
                crng = np.random.default_rng([CFG["seed"], 99, sum(map(ord, name))]); tasks = []
                for _ in range(20 if SMOKE else CFG["n_check"]):
                    rec = d6["scen"][int(crng.integers(len(d6["scen"])))]; fam = CFG["families"][int(crng.integers(3))]
                    b = rec[fam][0]; tasks.append((rec, fam, float(crng.uniform(0, b[-1] * 1.1 if len(b) else 1.0))))
                ok = pw.map(check, tasks); exact[name] = [int(sum(ok)), len(ok)]
                json.dump(exact, open(ex_file, "w"), indent=1)
                print(f"  {name}: exactness {sum(ok)}/{len(ok)}", flush=True)
    info = dict(CFG, units=len(U), subsets=subs, rmse={k: v["rmse"] for k, v in P.items()}, seconds=time.time() - t0)
    json.dump(info, open(os.path.join(OUT, "v6_info.json"), "w"), indent=1)
    print(f"v6 offline done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

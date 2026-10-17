# -*- coding: utf-8 -*-
"""
Paper 4 prototype v5, step 1: exact lambda-path records with the RICHER planner (p4v5_lib.RichFleetMILP).
Predictions are the v1 predictors (identical to v1/v2); scenarios use NEW seeds (protocol P2 in PROTOCOLS.md).
Settings (K = 20, P = 6 periods of L = 10 cycles, D = 2, D_out = 3, c_setup 100, c_lease 40):
  RICH_ID      ID pool, 1000 scenarios, C = 2, S0 = 3
  RICH_TIGHT   ID pool,  400 scenarios, C = 1, S0 = 1   (tight shop and spares)
  RICH_S13_src/tgt, RICH_S24_src/tgt   400 scenarios each, C = 2, S0 = 3 (new fault mode)
Families add, sc, s05; fixed Zhu (rhat - 3 sigma) and Mitici (rhat - 2.326 sigma) plans.
Exactness check: per setting 200 random (scenario, family, lambda) direct MILP solves compared with the path.
Output: results/v5_<name>.pkl.gz, results/v5_exactness.json ; resumable per setting.
"""
import os, time, json, pickle, gzip
import numpy as np
from multiprocessing import Pool
import p4v5_lib as L5

V1, L2 = L5.V1, L5.L2
SMOKE = os.environ.get("P4_SMOKE") == "1"
OUT = os.path.join(L5.HERE, "results_smoke" if SMOKE else "results"); os.makedirs(OUT, exist_ok=True)
PRED = os.path.join(L5.HERE, "..", "prototype_v1", "results", "v1_predictors.pkl")
CFG = dict(seed=2027, K=20, P=6, Lp=10, D=2, D_out=3, c_setup=100.0, c_lease=40.0, families=["add", "sc", "s05"],
           zhu_omega=3.0, mitici_z=2.326, workers=int(os.environ.get("P4_WORKERS", "4")), n_check=200,
           settings={"RICH_ID": ("ID", "pool", 1000, 2, 3), "RICH_TIGHT": ("ID", "pool", 400, 1, 1),
                     "RICH_S13_src": ("S13", "src", 400, 2, 3), "RICH_S13_tgt": ("S13", "tgt", 400, 2, 3),
                     "RICH_S24_src": ("S24", "src", 400, 2, 3), "RICH_S24_tgt": ("S24", "tgt", 400, 2, 3)})


def milp(C, S0):
    return L5.RichFleetMILP(CFG["K"], CFG["P"], CFG["Lp"], C, CFG["D"], CFG["D_out"], S0,
                            c_setup=CFG["c_setup"], c_lease=CFG["c_lease"])


def work(args):
    rh, sg, r, C, S0, fams = args
    m = milp(C, S0); ps = L2.PathSolver(m)
    res = {"rh": rh, "sg": sg, "r": r}
    t0 = time.time()
    for fam in fams:
        b, V, n = ps.path(rh, sg, r, fam)
        res[fam] = (b, V.astype(np.float32), n)
    res["zhu"] = m.evaluate(m.solve(rh, rh - CFG["zhu_omega"] * sg), r)
    res["mitici"] = m.evaluate(m.solve(rh, rh - CFG["mitici_z"] * sg), r)
    res["sec"] = time.time() - t0
    return res


def check(args):
    """Direct solve at a random lambda vs the path value (n_fail, any, cost)."""
    rec, fam, lam, C, S0 = args
    m = milp(C, S0)
    a = m.solve(rec["rh"], L2.deadlines(rec["rh"], rec["sg"], lam, fam))
    direct = np.array(m.evaluate(a, rec["r"]), float)
    path = np.array(L2.at(rec[fam], lam)[:3], float)
    return bool(np.allclose(direct, path, atol=1e-3))


def main():
    t0 = time.time()
    P = pickle.load(open(PRED, "rb"))
    ex_file = os.path.join(OUT, "v5_exactness.json")
    exact = json.load(open(ex_file)) if os.path.exists(ex_file) else {}
    with Pool(CFG["workers"]) as pw:
        for name, (grp, pool_name, n, C, S0) in CFG["settings"].items():
            if SMOKE: n = 12
            f = os.path.join(OUT, f"v5_{name}.pkl.gz")
            if os.path.exists(f):
                print(f"  {name}: already done", flush=True)
            else:
                d = P[grp]
                srng = np.random.default_rng([CFG["seed"], sum(map(ord, name))])
                scen = V1.make_scenarios(d["pools"][pool_name], d["preds"], n, CFG["K"], srng)
                t1, out = time.time(), []
                for i, o in enumerate(pw.imap(work, [(rh, sg, r, C, S0, CFG["families"]) for rh, sg, r in scen], chunksize=2)):
                    out.append(o)
                    if (i + 1) % 50 == 0: print(f"    {name}: {i + 1}/{n} ({time.time() - t1:.0f}s)", flush=True)
                with gzip.open(f + ".tmp", "wb") as fh:
                    pickle.dump(dict(K=CFG["K"], C=C, S0=S0, P=CFG["P"], L=CFG["Lp"], fams=CFG["families"], scen=out), fh)
                os.replace(f + ".tmp", f)
                ns = {fam: round(float(np.mean([o[fam][2] for o in out])), 1) for fam in CFG["families"]}
                print(f"  {name}: {n} scenarios, C={C}, S0={S0}; distinct plans per scenario {ns} ({time.time() - t1:.0f}s)", flush=True)
            if name not in exact:
                with gzip.open(f, "rb") as fh: d5 = pickle.load(fh)
                crng = np.random.default_rng([CFG["seed"], 99, sum(map(ord, name))])
                tasks = []
                for _ in range(CFG["n_check"] if not SMOKE else 20):
                    rec = d5["scen"][int(crng.integers(len(d5["scen"])))]
                    fam = CFG["families"][int(crng.integers(3))]
                    b = rec[fam][0]
                    lam = float(crng.uniform(0, (b[-1] * 1.1) if len(b) else 1.0))
                    tasks.append((rec, fam, lam, C, S0))
                ok = pw.map(check, tasks)
                exact[name] = [int(sum(ok)), len(ok)]
                json.dump(exact, open(ex_file, "w"), indent=1)
                print(f"  {name}: exactness {sum(ok)}/{len(ok)} direct solves match the path", flush=True)
    json.dump(dict(CFG, seconds=time.time() - t0), open(os.path.join(OUT, "v5_offline_info.json"), "w"), indent=1)
    print(f"v5 offline records done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

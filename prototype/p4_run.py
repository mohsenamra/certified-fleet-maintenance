# -*- coding: utf-8 -*-
"""
Paper 4 prototype, step 1: train the RUL predictor and record the decision outcome of the fleet MILP
for every fleet scenario and every safety margin lambda (no calibration here, see p4_analyze.py).
Output: results/records_v0.npz  (re-analysis needs no new solves).
"""
import os, sys, time, json
import numpy as np
import p4_lib as L

OUT = os.path.join(L.HERE, "results"); os.makedirs(OUT, exist_ok=True)
CFG = dict(fds=[1, 3], n_train=80, seed=2026, K=20, P=6, Lp=10, C=2, n_scen=400, grid=list(range(0, 101, 2)))


def main():
    t0 = time.time(); rng = np.random.default_rng(CFG["seed"])
    eng = {}
    for fd in CFG["fds"]: eng.update(L.load_fd(fd))
    keys = sorted(eng); perm = rng.permutation(len(keys))
    tr = [keys[i] for i in perm[:CFG["n_train"]]]; rest = [keys[i] for i in perm[CFG["n_train"]:]]
    poolA, poolB = rest[:len(rest) // 2], rest[len(rest) // 2:]
    print(f"engines: {len(keys)} (train {len(tr)}, pool A {len(poolA)}, pool B {len(poolB)})", flush=True)

    pred = L.Predictor().fit({k: eng[k] for k in tr})
    preds = {k: pred.predict_all(eng[k]) for k in poolA + poolB}
    err = np.concatenate([preds[k] - np.minimum(len(eng[k]) - 1 - np.arange(len(eng[k])), L.RUL_CAP) for k in poolA + poolB])
    print(f"predictor fitted ({time.time() - t0:.0f}s); RMSE vs capped RUL on held-out engines: {np.sqrt(np.mean(err ** 2)):.2f}", flush=True)

    model = L.FleetMILP(CFG["K"], CFG["P"], CFG["Lp"], CFG["C"]); grid = np.array(CFG["grid"], float)
    rec = {}
    for tag, pool in (("A", poolA), ("B", poolB)):
        scen = L.make_scenarios(pool, preds, CFG["n_scen"], CFG["K"], rng)
        N, G = len(scen), len(grid)
        F = np.zeros((N, G), int); ANY = np.zeros((N, G), int); COST = np.zeros((N, G)); OUTS = np.zeros((N, G), int)
        RES = np.zeros((N, CFG["K"])); ORA = np.zeros(N); RH = np.zeros((N, CFG["K"])); RT = np.zeros((N, CFG["K"]))
        for s, (_, rh, r) in enumerate(scen):
            RES[s] = rh - r; RH[s] = rh; RT[s] = r
            for j, lam in enumerate(grid):
                f, an, c, no, _ = model.evaluate(model.solve(rh, lam), r)
                F[s, j], ANY[s, j], COST[s, j], OUTS[s, j] = f, an, c, no
            ORA[s] = model.evaluate(model.solve(r, 0.0), r)[2]          # perfect-information plan
            if (s + 1) % 50 == 0: print(f"  pool {tag}: {s + 1}/{N} scenarios ({time.time() - t0:.0f}s)", flush=True)
        rec.update({f"{tag}_fail": F, f"{tag}_any": ANY, f"{tag}_cost": COST, f"{tag}_out": OUTS, f"{tag}_res": RES,
                    f"{tag}_oracle": ORA, f"{tag}_rhat": RH, f"{tag}_r": RT})
    np.savez_compressed(os.path.join(OUT, "records_v0.npz"), grid=grid, **rec)
    json.dump(dict(CFG, rmse=float(np.sqrt(np.mean(err ** 2))), seconds=time.time() - t0), open(os.path.join(OUT, "run_info_v0.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s -> results/records_v0.npz", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Paper 4 prototype v2, step 2b: COST-AWARE conformal controller (online).
Play lambda'_t = max(lambda_t, lambda_cost), update lambda_t with the loss observed at lambda'_t:
    lambda_{t+1} = max(0, lambda_t + eta (err_t - alpha)).
lambda_cost = margin that minimises the mean plan cost on the offline source calibration scenarios (exact path).
Guarantee is unchanged: if lambda_t >= lambda_safe then lambda'_t >= lambda_safe -> err_t = 0, so lambda_t <= lambda_safe + eta
and the telescoping bound (1/T) sum err_t <= alpha + (lambda_safe + eta)/(eta T) holds (CDT Thm 1 argument).
Output: results/v2_online_ca.pkl.gz (same format as v2_online.pkl.gz)
"""
import os, time, pickle, gzip
import numpy as np
from multiprocessing import Pool
import p4v2_lib as L2
import p4v2_online as O

V1 = L2.V1
FAMS = ["s05", "sc"]; ETA = 40.0
LCOST = {}


def cost_min_lambda(setting, fam, n=100, lam_cap=30.0):
    with gzip.open(os.path.join(O.OUT, f"v2_{setting}_src.pkl.gz"), "rb") as fh: d = pickle.load(fh)
    P = [s[fam] for s in d["scen"][:n]]
    cand = L2.candidates(P); cand = cand[cand < lam_cap]
    C = L2.loss_matrix(P, 2, cand + L2.EPS, envelope=False).mean(0)
    return float(cand[int(np.argmin(C))] + L2.EPS)


class CostAware(O.Adaptive):
    def __init__(self, lam_cost, family, alpha, eta):
        super().__init__(0.0, family, alpha, eta); self.lam_cost = lam_cost; self.lam_conf = 0.0
    def deadline(self, rh, sg):
        self.lam = max(self.lam_conf, self.lam_cost)
        return L2.deadlines(rh, sg, self.lam, self.family)
    def update(self, e_mean, e_any):
        self.lam_conf = float(max(0.0, self.lam_conf + self.eta * (e_mean - self.alpha)))


def names():
    out = []
    for fam in FAMS:
        out.append(f"fixedcost_{fam}")
        for a in O.CFG["alphas"]: out.append(f"costaware_{fam}_eta{ETA:g}_a{a}")
        out.append(f"adaptive_sc_cold_eta{ETA:g}_a0.0025"); out.append(f"adaptive_sc_cold_eta{ETA:g}_a0.005")
    return sorted(set(out))


def _init(lc):
    O._init({}); LCOST.update(lc)


def run_one(args):
    setting, mode, rn, seed = args
    d = O.P[setting]
    rng = np.random.default_rng([seed, sum(map(ord, setting + mode + rn))])
    src, tgt = d["pools"]["src"], d["pools"]["tgt"]
    pool_fn = (lambda t: src) if mode == "stationary" else (lambda t: src if t < O.CFG["T_SHIFT"] else tgt)
    m = V1.FleetMILP(O.CFG["K"], O.CFG["P"], O.CFG["Lp"], O.CFG["C"])
    sim = V1.FleetSim(m, d["preds"], O.LEN, rng)
    p = rn.split("_")
    if p[0] == "fixedcost": rule = O.Fixed(LCOST[(setting, p[1])], p[1])
    elif p[0] == "costaware": rule = CostAware(LCOST[(setting, p[1])], p[1], float(p[3][1:]), float(p[2][3:]))
    else: rule = O.Adaptive(0.0, p[1], float(p[4][1:]), float(p[3][3:]))
    F, M, C, LAM = sim.run(pool_fn, O.CFG["T"], rule)
    return (setting, mode, rn, seed), (F.astype(np.int16), M.astype(np.int16), C.astype(np.float32), LAM.astype(np.float32))


def main(out_dir=None):
    t0 = time.time(); out_dir = out_dir or O.OUT
    f = os.path.join(out_dir, "v2_online_ca.pkl.gz")
    if os.path.exists(f): print("cost-aware online: already done"); return
    lc = {(s, fam): cost_min_lambda(s, fam) for s in ("S13", "S24") for fam in FAMS}
    print("lambda_cost (offline cost-minimising margins):", lc, flush=True)
    tasks = [(s, mode, rn, seed) for s in ("S13", "S24") for mode in ("stationary", "shift") for rn in names() for seed in range(O.CFG["seeds"])]
    print(f"{len(tasks)} runs", flush=True)
    runs = {}
    with Pool(O.CFG["workers"], initializer=_init, initargs=(lc,)) as pw:
        for i, (k, v) in enumerate(pw.imap_unordered(run_one, tasks, chunksize=1)):
            runs[k] = v
            if (i + 1) % 10 == 0: print(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    with gzip.open(f + ".tmp", "wb") as fh:
        pickle.dump(dict(cfg=O.CFG, static={str(k): v for k, v in lc.items()}, runs=runs, seconds=time.time() - t0), fh)
    os.replace(f + ".tmp", f)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

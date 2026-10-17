# -*- coding: utf-8 -*-
"""
Paper 4 prototype v2, step 2: online (rolling-horizon) runs with cost-aware variants of the conformal controller.
Same simulator, predictors and settings as v1 (S13 / S24, stationary vs shift at period T_SHIFT, K=20, C=2).
Margin family s05: deadline = rhat - lambda * max(sigma, 0.5) (close to the Zhu / Mitici sigma rules, but with a floor so
that a large lambda forces 'outsource all' = the safe action of Conformal Decision Theory).
Rules
  baselines   : point (lambda 0), Zhu (rhat - 3 sigma), Mitici (rhat - 2.326 sigma), static CRC (s05, calibrated offline
                on the source pool with the exact path, horizon-level mean loss, alpha_h = 0.02, n = 100)
  fixed grid  : s05 with lambda in FIXED_GRID  -> cost/risk frontier and the hindsight-best fixed margin
  adaptive    : lambda_{t+1} = max(0, lambda_t + eta (err_t - alpha)), err_t = failures / K in period t
                variants: cold start (lambda_0 = 0) vs warm start (lambda_0 = static CRC), eta in ETAS, alpha in ALPHAS
                (+ v1 reference: additive family, eta = 100, cold)
Guarantee (CDT Thm 1 / Rolling RC): long-run rate <= alpha + (lam_max + eta) / (eta T), lam_max = 125 / 0.5 = 250 (s05).
Output: results/v2_online.pkl.gz
"""
import os, time, pickle, gzip
import numpy as np
from multiprocessing import Pool
import p4v2_lib as L2

V1 = L2.V1
OUT = os.path.join(L2.HERE, "results")
V1RES = os.path.join(L2.HERE, "..", "prototype_v1", "results")
CFG = dict(K=20, P=6, Lp=10, C=2, T=1000, T_SHIFT=300, seeds=5, alphas=[0.0025, 0.005], etas=[10.0, 40.0],
           fixed_grid=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0], crc_alpha=0.02, crc_n=100, workers=4)
STATIC = {}


class Fixed:
    def __init__(self, lam, family): self.lam, self.family = lam, family
    def deadline(self, rh, sg):
        if self.family == "raw": return rh - self.lam * sg
        return L2.deadlines(rh, sg, self.lam, self.family)
    def update(self, e_mean, e_any): pass


class Adaptive(Fixed):
    def __init__(self, lam0, family, alpha, eta):
        super().__init__(lam0, family); self.alpha, self.eta = alpha, eta
    def update(self, e_mean, e_any):
        self.lam = float(max(0.0, self.lam + self.eta * (e_mean - self.alpha)))


def static_crc(setting):
    f = os.path.join(OUT, f"v2_{setting}_src.pkl.gz")
    with gzip.open(f, "rb") as fh: d = pickle.load(fh)
    paths = [s["s05"] for s in d["scen"][:CFG["crc_n"]]]
    return L2.crc_path(paths, CFG["crc_alpha"], d["K"])


def make_rule(name, setting):
    if name == "point": return Fixed(0.0, "add")
    if name == "zhu": return Fixed(3.0, "raw")
    if name == "mitici": return Fixed(2.326, "raw")
    if name == "static_CRC": return Fixed(STATIC[setting], "s05")
    if name.startswith("fixed_s05_"): return Fixed(float(name.split("_")[-1]), "s05")
    # adaptive_<family>_<cold|warm>_eta<eta>_a<alpha>
    _, fam, start, eta, a = name.split("_")
    lam0 = 0.0 if start == "cold" else min(STATIC[setting], 20.0)   # finite warm start
    return Adaptive(lam0, fam, float(a[1:]), float(eta[3:]))


def rule_names():
    names = ["point", "zhu", "mitici", "static_CRC"] + [f"fixed_s05_{g}" for g in CFG["fixed_grid"]]
    for a in CFG["alphas"]:
        for start in ("cold", "warm"):
            for eta in CFG["etas"]:
                names.append(f"adaptive_s05_{start}_eta{eta:g}_a{a}")
        names.append(f"adaptive_add_cold_eta100_a{a}")
    return names


P = None; LEN = None


def _init(static):
    global P, LEN
    STATIC.update(static)
    P = pickle.load(open(os.path.join(V1RES, "v1_predictors.pkl"), "rb"))
    LEN = pickle.load(open(os.path.join(V1RES, "v1_lengths.pkl"), "rb"))


def run_one(args):
    setting, mode, rn, seed = args
    d = P[setting]
    rng = np.random.default_rng([seed, sum(map(ord, setting + mode + rn))])
    src, tgt = d["pools"]["src"], d["pools"]["tgt"]
    pool_fn = (lambda t: src) if mode == "stationary" else (lambda t: src if t < CFG["T_SHIFT"] else tgt)
    m = V1.FleetMILP(CFG["K"], CFG["P"], CFG["Lp"], CFG["C"])
    sim = V1.FleetSim(m, d["preds"], LEN, rng)
    rule = make_rule(rn, setting)
    F, M, C, LAM = sim.run(pool_fn, CFG["T"], rule)
    return (setting, mode, rn, seed), (F.astype(np.int16), M.astype(np.int16), C.astype(np.float32), LAM.astype(np.float32))


def main():
    t0 = time.time()
    f = os.path.join(OUT, "v2_online.pkl.gz")
    if os.path.exists(f): print("online: already done"); return
    static = {s: static_crc(s) for s in ("S13", "S24")}
    print("static CRC margins (s05, exact path, source pool):", static, flush=True)
    tasks = [(s, mode, rn, seed) for s in ("S13", "S24") for mode in ("stationary", "shift") for rn in rule_names() for seed in range(CFG["seeds"])]
    print(f"{len(tasks)} runs of {CFG['T']} periods", flush=True)
    runs = {}
    with Pool(CFG["workers"], initializer=_init, initargs=(static,)) as pw:
        for i, (k, v) in enumerate(pw.imap_unordered(run_one, tasks, chunksize=2)):
            runs[k] = v
            if (i + 1) % 40 == 0: print(f"  {i + 1}/{len(tasks)} runs ({time.time() - t0:.0f}s)", flush=True)
    with gzip.open(f + ".tmp", "wb") as fh:
        pickle.dump(dict(cfg=CFG, static=static, runs=runs, seconds=time.time() - t0), fh)
    os.replace(f + ".tmp", f)
    print(f"online runs done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

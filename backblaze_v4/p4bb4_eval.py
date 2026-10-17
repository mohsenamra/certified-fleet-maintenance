# -*- coding: utf-8 -*-
"""
Paper 4, Backblaze v4 step 2: the v3b + v3c evaluation, unchanged, on the v4 predictor (p4bb4_*), plus the
extra analyses specified in advance (PROTOCOLS.md, P1):
  * EXACTNESS + TIMING on real instances: exact lambda-path vs direct solves at random lambdas; time per path vs grid;
  * LTT vs calibration size n in {30, 60, 90, 120} (C = 2, alpha 0.4 / 0.5, 100 random splits each);
  * COST-RATIO sensitivity c_fail in {10, 30, 100} (C = 2, alpha 0.4 / 0.5): costs recomputed from stored counts,
    compared with the cost-optimal fixed plan in hindsight (oracle on the test days, no guarantee).
Reuses backblaze_v3/p4bb3_eval.py (E) and backblaze_v3/p4bb3c_ltt.py (L) without modifying them.
Output: results/bb4_<model>.txt/.json per model (resumable) and results/summary_bb4.txt (all models).
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backblaze_v3"))
import p4bb3_eval as E
import p4bb3c_ltt as L

OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
NS = [30, 60, 90, 120]
CFAIL = [10.0, 30.0, 100.0]
lines = []


def log(s=""):
    print(s, flush=True); lines.append(s)


E.log = log; L.log = log                      # one report
if os.environ.get("P4_SMOKE"):                 # quick code check only (few splits, one capacity); never used for results
    E.N_RANDOM, E.CAPS, E.ALPHAS, NS = 3, [2], [0.4], [30, 60]
    OUT = os.path.join(HERE, "results_smoke"); os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- data + shared exact paths
def load4(tg):
    d = np.load(os.path.join(E.BB, f"p4bb4_{tg}.npz")); info = json.load(open(os.path.join(E.BB, f"p4bb4_{tg}_info.json")))
    day = d["day"]; order = np.argsort(day, kind="stable"); days = np.unique(day)
    cut = np.searchsorted(day[order], days, side="left").tolist() + [len(day)]
    rows = {int(t): order[cut[i]:cut[i + 1]] for i, t in enumerate(days)}
    return {k: d[k] for k in d.files}, info, rows


_orig_path = E.Instance.path
def _path_memo(self, C):
    if not hasattr(self, "_pc"): self._pc = {}
    if C not in self._pc: self._pc[C] = _orig_path(self, C)
    return self._pc[C]
E.Instance.path = _path_memo

_cache = {}
_orig_inst = E.instances
def _inst_cached(d, rows, last_full):
    k = (id(d), last_full)
    if k not in _cache: _cache.clear(); _cache[k] = _orig_inst(d, rows, last_full)
    return _cache[k]
E.instances = _inst_cached


def split_reps(days, n, rng, k=E.N_RANDOM):
    reps = []
    for _ in range(k):
        p = rng.permutation(len(days)); reps.append(([days[i] for i in p[:n]], [days[i] for i in p[n:]]))
    return reps


# ---------------------------------------------------------------- extra analyses
def exactness_timing(inst, days, rng, C=2, n_days=20, n_lam=25):
    sel = [days[i] for i in rng.choice(len(days), min(n_days, len(days)), replace=False)]
    ok = tot = 0; tp = tg = 0.0; nb = []
    for t in sel:
        I = inst[t]
        t0 = time.time(); b, V = _orig_path(I, C); tp += time.time() - t0; nb.append(len(b))
        lams = rng.uniform(0, E.LAM_MAX - 1e-3, n_lam)
        t0 = time.time()
        for lam in lams:
            direct = np.array(I.at_lambda(lam, C), np.float32)
            tot += 1; ok += int(np.allclose(direct, E.at((b, V), lam), rtol=1e-5, atol=1e-4))
        tg += time.time() - t0
    r = dict(days=len(sel), breakpoints_mean=float(np.mean(nb)), path_seconds_mean=tp / len(sel),
             direct_solve_seconds_mean=tg / (len(sel) * n_lam), exact_matches=ok, checks=tot)
    log(f"  EXACTNESS/TIMING (C={C}, {len(sel)} days): exact path = direct solve in {ok}/{tot} random-lambda checks; "
        f"{r['breakpoints_mean']:.0f} breakpoints/day; path {r['path_seconds_mean']:.2f}s vs one direct solve "
        f"{r['direct_solve_seconds_mean'] * 1000:.1f} ms (a 100-point grid = {100 * r['direct_solve_seconds_mean']:.2f}s, not exact)")
    return r


def ltt_vs_n(paths, days, rng, alphas=(0.4, 0.5)):
    res = {}
    log("  LTT vs calibration size (C = 2, 100 random splits): n | alpha | CRC miss / >a | LTT abstain / cert.miss / cert>a / cost")
    for n in NS:
        if n >= len(days) - 20: continue
        reps = split_reps(days, n, rng)
        for a in alphas:
            acc = {"CRC": [], "LTT": []}
            for cal, te in reps:
                cp = [paths[t] for t in cal]
                for name, lam in (("CRC", E.crc(cp, a)), ("LTT", L.ltt(cp, a))):
                    v = np.array([E.at(paths[t], E.deploy(lam)) for t in te], float)
                    acc[name].append((v[:, 0].mean(), v[:, 2].mean(), not np.isfinite(lam)))
            c = np.array(acc["CRC"], float); l = np.array(acc["LTT"], float); ce = l[:, 2] == 0
            r = dict(crc_miss=float(c[:, 0].mean()), crc_above=float((c[:, 0] > a + 1e-12).mean()), crc_cost=float(c[:, 1].mean()),
                     ltt_abstain=float(l[:, 2].mean()), ltt_cert_miss=float(l[ce, 0].mean()) if ce.any() else None,
                     ltt_cert_above=float((l[ce, 0] > a + 1e-12).mean()) if ce.any() else None, ltt_cost=float(l[:, 1].mean()))
            res[f"n={n}|alpha={a}"] = r
            f = lambda x, p=3: "  -  " if x is None else f"{x:.{p}f}"
            log(f"    n {n:3d} | {a} | {r['crc_miss']:.3f} / {r['crc_above']:.2f} | {r['ltt_abstain']:.2f} / {f(r['ltt_cert_miss'])} / "
                f"{f(r['ltt_cert_above'], 2)} / {r['ltt_cost']:.0f}")
    return res


def cost_sensitivity(paths, days, rng, alphas=(0.4, 0.5)):
    res = {}
    reps = split_reps(days, E.CAL_DAYS, rng)
    cand = E.domain_candidates([paths[t] for t in days])
    log("  COST-RATIO sensitivity (C = 2, 100 random splits; cost of in-house 1, expedited 3, failure c_fail): "
        "rule cost / premium over the hindsight cost-optimal fixed plan (miss)")
    for cf in CFAIL:
        cvec = lambda V: E.COST["c_in"] * V[:, 6] + E.COST["c_out"] * V[:, 7] + cf * V[:, 4]
        for a in alphas:
            acc = {k: [] for k in ("CRC", "LTT", "validation-tuned", "point", "oracle")}
            for cal, te in reps:
                cp = [paths[t] for t in cal]
                Cte = np.zeros(len(cand)); Mte = np.zeros(len(cand))
                for t in te:
                    b, V = paths[t]; ix = np.searchsorted(b, cand, side="right"); Cte += cvec(V)[ix]; Mte += V[ix, 0]
                Cte /= len(te); Mte /= len(te); io = int(np.argmin(Cte))
                acc["oracle"].append((Cte[io], Mte[io]))
                for name, lam in (("CRC", E.crc(cp, a)), ("LTT", L.ltt(cp, a)), ("validation-tuned", E.empirical(cp, a)),
                                  ("point", E.lam_of_tau(0.5))):
                    vs = [E.at(paths[t], E.deploy(lam)) for t in te]
                    acc[name].append((float(np.mean([E.COST["c_in"] * v[6] + E.COST["c_out"] * v[7] + cf * v[4] for v in vs])),
                                      float(np.mean([v[0] for v in vs]))))
            o = np.array(acc["oracle"]); row = {}
            for k, v in acc.items():
                v = np.array(v); row[k] = dict(cost=float(v[:, 0].mean()), miss=float(v[:, 1].mean()),
                                              premium=float(v[:, 0].mean() / o[:, 0].mean() - 1))
            res[f"c_fail={cf:g}|alpha={a}"] = row
            log(f"    c_fail {cf:5.0f} | alpha {a}: " + "  ".join(f"{k} {r['cost']:.0f} / {100 * r['premium']:+.0f}% ({r['miss']:.2f})" for k, r in row.items()))
    return res


# ---------------------------------------------------------------- main
def main():
    t00 = time.time()
    tags = sorted(f[6:-4] for f in os.listdir(E.BB) if f.startswith("p4bb4_") and f.endswith(".npz") and ".tmp" not in f)
    if not tags: log("no p4bb4_*.npz - run p4bb4_data.py first"); return False
    for tg in tags:
        fj = os.path.join(OUT, f"bb4_{tg}.json")
        if os.path.exists(fj): print(f"{tg}: already evaluated"); continue
        lines.clear(); t0 = time.time()
        d, info, rows = load4(tg)
        log(f"\n######## {info['model']} - v4 predictor (feature set {info['feature_set']}; eval AUC "
            f"{ {k: round(v, 3) for k, v in info['auc_eval'].items()} }; validation means {info['selection_mean']})")
        res = {"info": {k: info[k] for k in ("feature_set", "selection_mean", "auc_eval", "thr_fpr1_G28", "train_fail_per_day")}}
        log("\n--- v3b evaluation (same rules, splits protocol and online rules) ---")
        res["v3b"] = E.offline(tg, d, info, rows, np.random.default_rng(7))
        res["v3b"]["online"] = E.online(tg, d, info, rows)
        log("\n--- v3c: CRC vs LTT-HB certification ---")
        res["v3c"] = L.offline(tg, d, info, rows, np.random.default_rng(11))
        res["v3c"]["online"] = L.online(tg, d, info, rows)
        log("\n--- v4 extra analyses ---")
        last_full = max(rows) - E.P_DAYS
        inst = E.instances(d, rows, last_full); days = sorted(inst)
        paths = {t: inst[t].path(2) for t in days}
        rng = np.random.default_rng(21)
        res["exactness_timing"] = exactness_timing(inst, days, rng)
        res["ltt_vs_n"] = ltt_vs_n(paths, days, rng)
        res["cost_sensitivity"] = cost_sensitivity(paths, days, rng)
        log(f"\n{info['model']} done in {time.time() - t0:.0f}s")
        open(os.path.join(OUT, f"bb4_{tg}.txt"), "w", encoding="utf-8").write("\n".join(lines))
        json.dump(res, open(fj + ".tmp", "w"), indent=1, default=str); os.replace(fj + ".tmp", fj)
        _cache.clear()
    parts = [open(os.path.join(OUT, f"bb4_{tg}.txt"), encoding="utf-8").read() for tg in tags if os.path.exists(os.path.join(OUT, f"bb4_{tg}.txt"))]
    open(os.path.join(OUT, "summary_bb4.txt"), "w", encoding="utf-8").write(
        "== Paper 4: Backblaze v4 (stronger predictor, protocol P1 in PROTOCOLS.md) ==\n" + "\n".join(parts))
    print(f"evaluation done in {time.time() - t00:.0f}s", flush=True)
    return True


if __name__ == "__main__":
    ok = False
    try: ok = main()
    except Exception:
        import traceback; traceback.print_exc()
    sys.exit(0 if ok else 1)

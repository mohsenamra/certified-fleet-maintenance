# -*- coding: utf-8 -*-
"""
Paper 4, Backblaze v3c: HIGH-PROBABILITY CERTIFICATION (Learn-then-Test) next to CRC.

Why: in v3b, when alpha is close to the floor, the few calibration splits on which CRC certifies the target were
over-optimistic (risk above alpha on the test days). CRC controls the MEAN risk over calibration draws; it says nothing
about the risk GIVEN that the rule certified. LTT (Angelopoulos, Bates et al. 2021/2025) with Hoeffding-Bentkus p-values
and fixed-sequence testing (from the most conservative margin downward) gives P(R(lambda_hat) > alpha) <= delta, so a
certificate is wrong with probability at most delta, including near the floor; if even the most conservative in-domain
plan cannot be certified the rule abstains.

Reuses backblaze_v3/p4bb3_eval.py (same data p4bb3b_*, same planning model, same exact paths). Does NOT modify v3b
results. Rules compared on the same splits: CRC (mean guarantee), LTT-HB delta=0.1, validation-tuned.
Online: static LTT calibrated on the first 60 days (as static CRC in v3b), scored on days 60..end.
Output: results/summary_bb3c.txt, results/summary_bb3c.json
"""
import os, sys, json, time, math
import numpy as np
import p4bb3_eval as E

DELTA = 0.1
OUTF = os.path.join(E.OUT, "summary_bb3c")
lines, summ = [], {}


def log(s=""):
    print(s, flush=True); lines.append(s)


# ------------------------------------------------------------------ LTT with Hoeffding-Bentkus p-values
def hb_pvalue(R, n, alpha):
    """Hoeffding-Bentkus p-value for H0: risk > alpha, losses in [0, 1], empirical risk R (array) from n samples."""
    R = np.asarray(R, float)
    a = np.minimum(R, alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        h1 = np.where(a > 0, a * np.log(a / alpha), 0.0) + (1 - a) * np.log((1 - a) / (1 - alpha))
    p_h = np.exp(-n * h1)
    i = np.arange(n + 1)
    logpmf = (math.lgamma(n + 1) - np.array([math.lgamma(k + 1) + math.lgamma(n - k + 1) for k in i])
              + i * math.log(alpha) + (n - i) * math.log(1 - alpha))
    cdf = np.minimum(np.cumsum(np.exp(logpmf)), 1.0)
    k = np.clip(np.ceil(n * R - 1e-9).astype(int), 0, n)
    p_b = math.e * cdf[k]
    return np.minimum(np.minimum(p_h, p_b), 1.0)


def ltt(paths, alpha, delta=DELTA, col=0):
    """Fixed-sequence LTT over the domain, from the most conservative margin down. np.inf = abstain."""
    cand = E.domain_candidates(paths); n = len(paths)
    R = np.zeros(len(cand))
    for b, V in paths: R += V[np.searchsorted(b, cand, side="right"), col]
    R /= n
    p = hb_pvalue(R, n, alpha)
    rej = p[::-1] <= delta                       # conservative end first
    if not rej[0]: return np.inf
    stop = np.argmin(rej) if not rej.all() else len(rej)
    i = len(cand) - stop                         # smallest index in the rejected run
    return float(cand[i] + 1e-7)


# ------------------------------------------------------------------ offline
RULES = ("CRC (mean guarantee)", f"LTT-HB delta={DELTA} (high-probability)", "validation-tuned")


def offline(tag, d, info, rows, rng):
    maxday = max(rows); last_full = maxday - E.P_DAYS
    inst = E.instances(d, rows, last_full); days = sorted(inst)
    log(f"\n=== {info['model']}: {len(days)} decision days; failures within 28 d per day {np.mean([inst[t].nf for t in days]):.1f}")
    res = {}
    for C in E.CAPS:
        t0 = time.time()
        paths = {t: inst[t].path(C) for t in days}
        floor = float(np.mean([E.at(paths[t], E.deploy(np.inf))[0] for t in days]))
        log(f"  C = {C}/day: paths {time.time() - t0:.0f}s; FLOOR {floor:.3f}")
        res[f"C={C} | floor"] = floor
        for split in ("random", "forward"):
            if split == "random":
                reps = []
                for _ in range(E.N_RANDOM):
                    p = rng.permutation(len(days)); reps.append(([days[i] for i in p[:E.CAL_DAYS]], [days[i] for i in p[E.CAL_DAYS:]]))
            else:
                reps = [(days[s:s + E.CAL_DAYS], days[s + E.CAL_DAYS:s + E.CAL_DAYS + E.FWD_TEST])
                        for s in range(0, len(days) - E.CAL_DAYS - E.FWD_TEST + 1, E.FWD_TEST)]
            for alpha in E.ALPHAS:
                acc = {r: dict(risk=[], cost=[], lam=[], abst=[]) for r in RULES}
                for cal, te in reps:
                    cp = [paths[t] for t in cal]
                    for name, lam in zip(RULES, (E.crc(cp, alpha), ltt(cp, alpha), E.empirical(cp, alpha))):
                        v = np.array([E.at(paths[t], E.deploy(lam)) for t in te], float)
                        a = acc[name]; a["risk"].append(v[:, 0].mean()); a["cost"].append(v[:, 2].mean())
                        a["lam"].append(E.deploy(lam)); a["abst"].append(not np.isfinite(lam))
                key = f"C={C} | {split} | alpha={alpha}"
                out = {}
                log(f"    {key} ({len(reps)} splits)   rule: miss / >alpha / abstain / certified miss / certified >alpha / cost / tau")
                for name, a in acc.items():
                    rk, ab = np.array(a["risk"]), np.array(a["abst"], bool)
                    s = dict(miss=float(rk.mean()), above=float(np.mean(rk > alpha + 1e-12)), abstain=float(ab.mean()),
                             cert_miss=float(rk[~ab].mean()) if (~ab).any() else None,
                             cert_above=float(np.mean(rk[~ab] > alpha + 1e-12)) if (~ab).any() else None,
                             cost=float(np.mean(a["cost"])), tau=float(np.exp(-np.mean(a["lam"]))))
                    out[name] = s
                    f = lambda x, p=3: "  -  " if x is None else f"{x:.{p}f}"
                    log(f"      {name:36s} {s['miss']:.3f} / {s['above']:.2f} / {s['abstain']:.2f} / {f(s['cert_miss'])} / "
                        f"{f(s['cert_above'], 2)} / {s['cost']:7.0f} / {s['tau']:.4f}")
                res[key] = out
    return res


def online(tag, d, info, rows):
    days_all = sorted(rows); C = 2; cal_days = days_all[:E.CAL_DAYS]
    fail_rate = info["train_fail_per_day"]
    inst = {t: E.Instance(d["G"][rows[t]], d["T"][rows[t]], d["smart5"][rows[t]], d["drive"][rows[t]]) for t in cal_days}
    cp = [inst[t].path(C) for t in cal_days]
    out = {}
    log(f"  ONLINE {info['model']} (C = 2, scored days {E.CAL_DAYS}..{days_all[-1]}): static LTT-HB on the first 60 days")
    for alpha in E.ALPHAS:
        a_d = alpha * fail_rate
        lam = ltt(cp, alpha)
        s = E.simulate(d, rows, days_all, E.CAL_DAYS, ("fixed", E.deploy(lam)), a_d, C)
        s["abstain"] = not np.isfinite(lam)
        out[str(alpha)] = s
        log(f"    alpha {alpha}: {'ABSTAINS' if s['abstain'] else f'tau {np.exp(-lam):.4f}'}  missed/day {s['missed_per_day']:.3f} "
            f"({'meets' if s['missed_per_day'] <= a_d else 'ABOVE'} {a_d:.3f})  miss frac {s['miss_frac']:.3f}  cost {s['cost']:.0f}")
    return out


def selftest():
    """LTT sanity: p-values monotone in R; certified risk below alpha on a synthetic monotone path."""
    p = hb_pvalue(np.linspace(0, 0.6, 7), 60, 0.4)
    assert np.all(np.diff(p) >= -1e-12) and p[0] < 1e-6 and p[-1] == 1.0, p
    rng = np.random.default_rng(0)
    b = np.linspace(0.1, E.LAM_MAX - 0.01, 50)
    paths = []
    for _ in range(60):
        r = np.clip(0.9 * np.exp(-np.r_[b, E.LAM_MAX] / 2) + rng.normal(0, 0.05, 51), 0, 1)
        V = np.zeros((51, 9), np.float32); V[:, 0] = r; paths.append((b, V))
    lam = ltt(paths, 0.4)
    assert np.isfinite(lam) and lam > E.crc(paths, 0.4) - 1e-6, (lam, E.crc(paths, 0.4))
    assert not np.isfinite(ltt(paths, 0.01))       # far below the floor -> abstain
    log(f"self-test ok (synthetic: LTT lambda {lam:.3f} >= CRC lambda {E.crc(paths, 0.4):.3f}; alpha 0.01 abstains)")


def main():
    t0 = time.time(); rng = np.random.default_rng(11)
    selftest()
    tags = sorted(f[7:-4] for f in os.listdir(E.BB) if f.startswith("p4bb3b_") and f.endswith(".npz"))
    if not tags: log("no p4bb3b_*.npz found"); return False
    log(f"== Paper 4: Backblaze v3c - CRC vs LTT-HB (delta = {DELTA}) certification; alphas {E.ALPHAS}; capacities {E.CAPS} ==")
    for tag in tags:
        d, info, rows = E.load(tag)
        summ[tag] = offline(tag, d, info, rows, rng)
        summ[tag]["online"] = online(tag, d, info, rows)
        json.dump(summ, open(OUTF + ".json", "w"), indent=1, default=str)
        open(OUTF + ".txt", "w", encoding="utf-8").write("\n".join(lines))
    log(f"\ndone in {time.time() - t0:.0f}s")
    open(OUTF + ".txt", "w", encoding="utf-8").write("\n".join(lines))
    return True


if __name__ == "__main__":
    ok = False
    try: ok = main()
    except Exception:
        import traceback; traceback.print_exc()
    sys.exit(0 if ok else 1)

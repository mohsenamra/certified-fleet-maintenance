# -*- coding: utf-8 -*-
"""
Paper 4, real-fleet case (Backblaze), v3 step 2: proactive drive replacement with a calibrated risk margin.
Same planning model as v2 (see backblaze_v2/p4bb_eval.py): one instance per drive model and decision day, horizon 28 d,
C in-house replacements per day, expedited replacement always feasible, deadline day from the first horizon whose
predicted failure probability G_kj >= tau = exp(-lambda); EDF greedy = exact MILP solution; exact lambda-path.
v3 changes:
  * DOMAIN + ABSTENTION: lambda in [0, LAM_MAX) (tau > TAU_MIN). If no lambda in the domain meets the CRC condition the
    rule ABSTAINS (reports 'target not certifiable') and deploys the most conservative in-domain plan; v2 jumped to the
    theoretical safe action 'replace the whole fleet'. Abstention shares are reported; risk statistics are given for all
    splits and for the certified (non-abstaining) splits.
  * FLOOR / FRONTIER: per model, the mean miss fraction and replacements of static plans on a tau grid, and of daily
    re-planning (online) with fixed tau, so alpha can be judged against what the predictor makes achievable.
  * ONLINE over the whole evaluation period: the first 60 days calibrate the static rules (CRC, lambda_cost); every rule
    is scored on days 60..end; adaptive rules start updating on day 0 (warm-up not scored).
  * COST-AWARE controller (as in C-MAPSS): play max(lambda_t, lambda_cost), lambda_cost = margin with the lowest mean
    static-plan cost on the 60 calibration days; the risk bound is unchanged.
  * alpha in {0.3, 0.4, 0.5}; 100 random-day splits; forward splits 60 -> 30 days.
Output: results/summary_bb3.txt, results/summary_bb3.json
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BB = os.path.join(HERE, "..", "data", "backblaze")
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
P_DAYS, TAU_MIN = 28, 1e-3
LAM_MAX = -np.log(TAU_MIN)
Q_OF_J = np.array([1, 8, 15, 22, P_DAYS + 1])            # j = 0..3 -> deadline day; 4 -> defer
COST = dict(c_in=1.0, c_out=3.0, c_fail=30.0)
CAPS = [1, 2, 4]
ALPHAS = [0.3, 0.4, 0.5]
TAU_GRID = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
ETA = 0.5
CAL_DAYS, FWD_TEST, N_RANDOM = 60, 30, 100
EPS = 1e-9
lines, summ = [], {}


def log(s=""):
    print(s, flush=True); lines.append(s)


# ------------------------------------------------------------------ one planning instance
class Instance:
    def __init__(self, G, T, smart5, drive):
        self.G, self.T, self.s5, self.drive = G, T.astype(int), smart5, drive
        self.nf = int(np.sum(self.T <= P_DAYS))
        self.risky = np.where(G[:, 3] >= TAU_MIN)[0]               # only these can change state inside the domain
        lowmask = np.ones(len(G), bool); lowmask[self.risky] = False
        self.low_missed = int(np.sum(self.T[lowmask] <= P_DAYS))   # always missed inside the domain
        self.low_Efail = float(G[lowmask, 3].sum())
        self.Efail = float(G[:, 3].sum())
        self.order_key = np.lexsort((self.drive[self.risky], -G[self.risky, 3]))   # risk desc, then id

    def q_of_tau(self, tau, idx=None):
        Gr = self.G[self.risky] if idx is None else self.G[idx]
        hit = Gr >= tau - EPS
        j = np.where(hit.any(1), hit.argmax(1), 4)
        return Q_OF_J[j]

    def plan(self, q, C, units):
        """Earliest-deadline-first, earliest free day. Returns day (1..P) for in-house, 0 expedited, -1 deferred."""
        day = np.full(len(q), -1)
        o = np.lexsort((np.arange(len(q)), q))        # q asc, keep given (risk) order inside a q group
        S = 0
        for qv in (1, 8, 15, 22):
            g = o[q[o] == qv]
            if len(g) == 0: continue
            free = max(C * qv - S, 0); take = min(len(g), free)
            if take: day[g[:take]] = (S + np.arange(take)) // C + 1; S += take
            day[g[take:]] = 0
        return day

    def outcome(self, q, C):
        """Columns: 0 miss_frac, 1 any_missed, 2 cost, 3 model miss frac, 4 n_missed, 5 n_fail, 6 n_inhouse, 7 n_expedited,
           8 false replacements (replaced, no failure within the horizon)."""
        u = self.risky[self.order_key]; qq = q[self.order_key]
        day = self.plan(qq, C, u)
        T = self.T[u]
        missed_r = np.where(day == -1, T <= P_DAYS, np.where(day == 0, False, T < day))
        missed = int(missed_r.sum()) + self.low_missed
        nin, nout = int(np.sum(day > 0)), int(np.sum(day == 0))
        repl = day >= 0
        false_rep = int(np.sum(repl & (T > P_DAYS)))
        # model: P(fail before replacement day) from the predicted cumulative risk (piecewise linear in days)
        Gu = self.G[u]
        e_def = float(Gu[day == -1, 3].sum())
        sched = day > 0
        e_sched = 0.0
        if sched.any():                                  # P(T <= day-1), linear interpolation of the cumulative risk
            h = (day[sched] - 1).astype(float); Gs = np.hstack([np.zeros((sched.sum(), 1)), Gu[sched]])
            k = np.minimum((h // 7).astype(int), 3); fr = (h - 7 * k) / 7.0
            e_sched = float(np.sum(Gs[np.arange(len(k)), k] + fr * (Gs[np.arange(len(k)), k + 1] - Gs[np.arange(len(k)), k])))
        e_missed = e_def + e_sched + self.low_Efail
        cost = COST["c_in"] * nin + COST["c_out"] * nout + COST["c_fail"] * missed
        return [missed / max(self.nf, 1), float(missed > 0), cost, e_missed / max(self.Efail, 1e-9), missed, self.nf, nin, nout, false_rep]

    def safe(self):
        return [0.0, 0.0, COST["c_out"] * len(self.G), 0.0, 0, self.nf, 0, len(self.G), int(np.sum(self.T > P_DAYS))]

    def path(self, C):
        Gr = self.G[self.risky]
        lam = -np.log(np.clip(Gr[Gr >= TAU_MIN], TAU_MIN, 1.0))
        b = np.unique(np.concatenate([lam[(lam > 0) & (lam < LAM_MAX)], [LAM_MAX]]))
        reps = np.concatenate([[b[0] / 2], (b[:-1] + b[1:]) / 2])
        V = np.zeros((len(b) + 1, 9))
        for i, l in enumerate(reps):
            V[i] = self.outcome(self.q_of_tau(np.exp(-l)), C)
        V[-1] = self.safe()
        return b, V.astype(np.float32)

    def at_lambda(self, lam, C):
        if lam > LAM_MAX: return self.safe()
        return self.outcome(self.q_of_tau(np.exp(-lam)), C)

    def rule_q(self, flag_urgent):
        """Plan from a 0/1 urgency flag (RAIDShield-style rule): flagged drives get deadline day 1, others deferred."""
        f = flag_urgent[self.risky]
        return np.where(f, 1, P_DAYS + 1)


def raidshield_outcome(inst, C, thr):
    """Flag = SMART 5 (reallocated sectors) > thr for ANY drive (not only the risky ones)."""
    flag = inst.s5 > thr
    # temporarily treat all flagged drives as 'risky' units
    sub = Instance(inst.G, inst.T, inst.s5, inst.drive)
    sub.risky = np.where(flag | (inst.G[:, 3] >= TAU_MIN))[0]
    low = np.ones(len(inst.G), bool); low[sub.risky] = False
    sub.low_missed = int(np.sum(inst.T[low] <= P_DAYS)); sub.low_Efail = float(inst.G[low, 3].sum())
    sub.order_key = np.lexsort((inst.drive[sub.risky], -inst.s5[sub.risky]))
    q = np.where(flag[sub.risky], 1, P_DAYS + 1)
    return sub.outcome(q, C)


# ------------------------------------------------------------------ calibration on paths
def at(path, lam):
    b, V = path
    return V[np.searchsorted(b, lam, side="right")]


def sufmax(x): return np.maximum.accumulate(x[::-1])[::-1]


def domain_candidates(paths):
    c = np.unique(np.concatenate([[0.0]] + [p[0] for p in paths]))
    return c[c < LAM_MAX]


def crc(paths, alpha, col=0, B=1.0):
    """CRC over the domain [0, LAM_MAX); np.inf = abstain (target not certifiable with this predictor)."""
    cand = domain_candidates(paths); n = len(paths)
    R = np.zeros(len(cand))
    for b, V in paths:
        R += sufmax(V[:, col])[np.searchsorted(b, cand, side="right")]
    R /= n
    ok = np.where((n * R + B) / (n + 1) <= alpha)[0]
    return float(cand[ok[0]] + 1e-7) if len(ok) else np.inf


def empirical(paths, alpha, col=0):
    cand = domain_candidates(paths)
    R = np.zeros(len(cand))
    for b, V in paths: R += V[np.searchsorted(b, cand, side="right"), col]
    R /= len(paths)
    ok = np.where(R <= alpha)[0]
    return float(cand[ok[0]] + 1e-7) if len(ok) else np.inf


def model_lambda(path, alpha):
    b, V = path; ok = np.where(V[:-1, 3] <= alpha)[0]            # never the safe action
    i = ok[0] if len(ok) else len(V) - 2
    return 0.0 if i == 0 else float(b[i - 1] + 1e-7)


def lam_of_tau(tau): return float(-np.log(max(min(tau, 1.0), 1e-12)))


def deploy(lam):
    """Abstention (inf) deploys the most conservative in-domain plan."""
    return LAM_MAX - 1e-6 if not np.isfinite(lam) else min(lam, LAM_MAX - 1e-6)


def cost_min_lambda(paths):
    cand = domain_candidates(paths)
    C = np.zeros(len(cand))
    for b, V in paths: C += V[np.searchsorted(b, cand, side="right"), 2]
    return float(cand[int(np.argmin(C))] + 1e-7)


# ------------------------------------------------------------------ data
def load(tag):
    d = np.load(os.path.join(BB, f"p4bb3b_{tag}.npz")); info = json.load(open(os.path.join(BB, f"p4bb3b_{tag}_info.json")))
    day = d["day"]; order = np.argsort(day, kind="stable")
    days = np.unique(day)
    cut = np.searchsorted(day[order], days, side="left").tolist() + [len(day)]
    rows = {int(t): order[cut[i]:cut[i + 1]] for i, t in enumerate(days)}
    return d, info, rows


def instances(d, rows, last_full):
    return {t: Instance(d["G"][r], d["T"][r], d["smart5"][r], d["drive"][r]) for t, r in rows.items() if t <= last_full}


# ------------------------------------------------------------------ offline
def summarize(vals):
    v = np.array(vals, float)
    return dict(miss=float(v[:, 0].mean()), cost=float(v[:, 2].mean()), inhouse=float(v[:, 6].mean()), exped=float(v[:, 7].mean()),
                false=float(v[:, 8].mean()), nfail=float(v[:, 5].mean()), missed=float(v[:, 4].mean()))


def frontier(inst, paths, days, C):
    """Static-plan frontier on TAU_GRID + floor (most conservative in-domain plan)."""
    rows = []
    for tau in TAU_GRID + [TAU_MIN * 1.0000001]:
        v = np.array([at(paths[t], deploy(lam_of_tau(tau))) for t in days], float)
        rows.append(dict(tau=tau, miss=float(v[:, 0].mean()), repl=float((v[:, 6] + v[:, 7]).mean()), cost=float(v[:, 2].mean())))
    return rows


def offline(tag, d, info, rows, rng):
    maxday = max(rows); last_full = maxday - P_DAYS
    inst = instances(d, rows, last_full)
    days = sorted(inst)
    log(f"\n=== {info['model']}: {len(days)} decision days with a full 28-day outcome; eval AUC "
        f"{ {k: round(v, 3) for k, v in info['auc_eval'].items()} }; failures within 28 d per day: "
        f"{np.mean([inst[t].nf for t in days]):.1f}; drives at risk (G28 >= {TAU_MIN}) per day: "
        f"{np.mean([len(inst[t].risky) for t in days]):.0f} of {np.mean([len(inst[t].G) for t in days]):.0f}")
    res = {}
    for C in CAPS:
        t0 = time.time()
        paths = {t: inst[t].path(C) for t in days}
        lam_fpr = lam_of_tau(info["thr_fpr1_G28"])
        log(f"  C = {C}/day: exact paths computed ({np.mean([len(paths[t][0]) for t in days]):.0f} breakpoints per day, {time.time() - t0:.0f}s)")
        fr = frontier(inst, paths, days, C); res[f"{tag} | C={C} | frontier"] = fr
        log("    static 28-day plans, mean over decision days:  " +
            "  ".join(f"tau {r['tau']:.3g}: miss {r['miss']:.2f} / repl {r['repl']:.0f}" for r in fr))
        log(f"    FLOOR (most conservative in-domain plan, tau = {TAU_MIN}): miss fraction {fr[-1]['miss']:.3f}")
        rs = {thr: {t: raidshield_outcome(inst[t], C, thr) for t in days} for thr in (0, 200)}
        for split in ("random", "forward"):
            if split == "random":
                reps = []
                for _ in range(N_RANDOM):
                    p = rng.permutation(len(days)); reps.append(([days[i] for i in p[:CAL_DAYS]], [days[i] for i in p[CAL_DAYS:]]))
            else:
                reps = [(days[s:s + CAL_DAYS], days[s + CAL_DAYS:s + CAL_DAYS + FWD_TEST]) for s in range(0, len(days) - CAL_DAYS - FWD_TEST + 1, FWD_TEST)]
            for alpha in ALPHAS:
                acc = {}
                def add(name, outs, lam=np.nan, abst=False):
                    a = acc.setdefault(name, dict(o=[], risk=[], lam=[], abst=[])); v = np.array(outs, float)
                    a["o"].append(v); a["risk"].append(v[:, 0].mean()); a["lam"].append(lam); a["abst"].append(abst)
                for cal, te in reps:
                    cp = [paths[t] for t in cal]
                    lam_crc = crc(cp, alpha); lam_emp = empirical(cp, alpha)
                    for name, lam in (("CRC exact path (abstains if not certifiable)", lam_crc), ("validation-tuned", lam_emp),
                                      ("fixed FPR 1% threshold", lam_fpr), ("point (tau = 0.5)", lam_of_tau(0.5))):
                        add(name, [at(paths[t], deploy(lam)) for t in te], deploy(lam), not np.isfinite(lam))
                    lams = [model_lambda(paths[t], alpha) for t in te]
                    add("model-based (predicted miss <= alpha, per day)", [at(paths[t], l) for t, l in zip(te, lams)], float(np.mean(lams)))
                    for thr in (0, 200):
                        add(f"RAIDShield-style SMART5 > {thr}", [rs[thr][t] for t in te])
                key = f"{tag} | C={C} | {split} split | alpha={alpha}"
                out = {}
                log(f"\n  {key}  ({len(reps)} splits; miss = test mean miss fraction; >alpha = share of splits above alpha; "
                    f"abst = share of splits where the rule abstains; 'certified' = miss / >alpha over non-abstaining splits)")
                log(f"    {'rule':48s} {'miss':>6s} {'>alpha':>6s} {'abst':>5s} {'cert.miss':>9s} {'cert>a':>6s} {'tau':>8s} {'cost':>7s} {'in-house':>8s} {'exped.':>7s} {'false':>6s}")
                for name, a in acc.items():
                    v = np.concatenate(a["o"]); rk = np.array(a["risk"]); lm = np.array(a["lam"], float); ab = np.array(a["abst"], bool)
                    s = summarize(v); s["above"] = float(np.mean(rk > alpha + 1e-12)); s["abstain"] = float(ab.mean())
                    s["cert_miss"] = float(rk[~ab].mean()) if (~ab).any() else None
                    s["cert_above"] = float(np.mean(rk[~ab] > alpha + 1e-12)) if (~ab).any() else None
                    s["tau"] = None if np.all(np.isnan(lm)) else float(np.exp(-np.nanmean(lm)))
                    out[name] = s
                    tau = "    rule" if s["tau"] is None else f"{s['tau']:8.4f}"
                    cm = "      -  " if s["cert_miss"] is None else f"{s['cert_miss']:9.3f}"
                    ca = "     -" if s["cert_above"] is None else f"{s['cert_above']:6.2f}"
                    log(f"    {name:48s} {s['miss']:6.3f} {s['above']:6.2f} {s['abstain']:5.2f} {cm} {ca} {tau} {s['cost']:7.0f} {s['inhouse']:8.1f} {s['exped']:7.1f} {s['false']:6.1f}")
                res[key] = out
    return res


# ------------------------------------------------------------------ online (daily re-planning)
def simulate(d, rows, days_all, score_from, rule, a_d, C=2):
    """rule = ('fixed', lam) | ('adapt', lam0, eta) | ('costaware', lam_cost, eta). Returns score dict over days >= score_from."""
    removed = np.zeros(int(d["drive"].max()) + 1, bool)
    lam = rule[1] if rule[0] != "costaware" else 0.0
    miss = fail = prevented = rep_in = rep_out = n = 0; lam_hist = []; mmax = 0
    last_day = days_all[-1]
    for t in days_all:
        rr = rows[t]; rr = rr[~removed[d["drive"][rr]]]
        inst = Instance(d["G"][rr], d["T"][rr], d["smart5"][rr], d["drive"][rr])
        lam_play = max(lam, rule[1]) if rule[0] == "costaware" else lam
        lam_use = deploy(lam_play)
        q = inst.q_of_tau(np.exp(-lam_use))
        u = inst.risky[inst.order_key]
        day = inst.plan(q[inst.order_key], C, u)
        act = u[(day == 0) | (day == 1)]
        removed[d["drive"][rr][act]] = True
        acted = np.zeros(len(rr), bool); acted[act] = True
        tomorrow = inst.T == 1
        m_t = int(np.sum(tomorrow & ~acted))
        if t >= score_from:
            Ta = inst.T[act]
            prevented += int(np.sum((Ta >= 1) & (t + Ta <= last_day)))
            rep_in += int(np.sum(day == 1)); rep_out += int(np.sum(day == 0))
            miss += m_t; fail += int(tomorrow.sum()); n += 1; lam_hist.append(lam_use); mmax = max(mmax, m_t)
        if rule[0] in ("adapt", "costaware"):
            lam = float(min(max(lam + rule[2] * (m_t - a_d), 0.0), LAM_MAX))
    return dict(missed_per_day=miss / max(n, 1), miss_frac=miss / max(miss + prevented, 1), missed=miss, prevented=prevented,
                inhouse=rep_in, expedited=rep_out, days=n, max_daily_missed=mmax,
                cost=COST["c_in"] * rep_in + COST["c_out"] * rep_out + COST["c_fail"] * miss, mean_tau=float(np.exp(-np.mean(lam_hist))))


def online(tag, d, info, rows):
    days_all = sorted(rows); C = 2; cal_days = days_all[:CAL_DAYS]
    fail_rate = info["train_fail_per_day"]
    inst = {t: Instance(d["G"][rows[t]], d["T"][rows[t]], d["smart5"][rows[t]], d["drive"][rows[t]]) for t in cal_days}
    cp = [inst[t].path(C) for t in cal_days]
    lam_cost = cost_min_lambda(cp)
    out = {"lambda_cost": lam_cost}
    log(f"\n  ONLINE {info['model']}: daily re-planning, C = {C}/day, scored on days {CAL_DAYS}..{days_all[-1]} ({len(days_all) - CAL_DAYS} days); "
        f"training failures/day {fail_rate:.2f}; lambda_cost = {lam_cost:.2f} (tau {np.exp(-lam_cost):.4f})")
    log("    fixed tau (frontier of daily re-planning):")
    for tau in TAU_GRID + [TAU_MIN * 1.0000001]:
        r = simulate(d, rows, days_all, CAL_DAYS, ("fixed", lam_of_tau(tau)), 0.0, C)
        out[f"fixed|{tau:.4g}"] = r
        log(f"      tau {tau:<8.4g} missed/day {r['missed_per_day']:.3f}  miss frac {r['miss_frac']:.3f}  in-house {r['inhouse']:5d}  "
            f"expedited {r['expedited']:6d}  cost {r['cost']:8.0f}")
    log(f"      -> online FLOOR (tau = {TAU_MIN}): miss fraction {out[f'fixed|{TAU_MIN * 1.0000001:.4g}']['miss_frac']:.3f}")
    for alpha in ALPHAS:
        a_d = alpha * fail_rate
        lam_s = crc(cp, alpha)
        rules = {"static CRC (first 60 days; abstains -> most conservative)": ("fixed", deploy(lam_s)),
                 "adaptive cold (lambda_0 = 0)": ("adapt", 0.0, ETA),
                 "adaptive warm (lambda_0 = static CRC)": ("adapt", deploy(lam_s), ETA),
                 "cost-aware (max(lambda_t, lambda_cost))": ("costaware", lam_cost, ETA)}
        log(f"    alpha = {alpha}: alpha_d = {a_d:.3f} missed failures/day; static CRC {'ABSTAINS' if not np.isfinite(lam_s) else f'tau {np.exp(-lam_s):.4f}'}")
        for name, r in rules.items():
            s = simulate(d, rows, days_all, CAL_DAYS, r, a_d, C)
            s["bound_slack_per_day"] = (LAM_MAX + ETA * max(s["max_daily_missed"], 1)) / (ETA * s["days"])
            out[f"{alpha}|{name}"] = s
            ok = "meets" if s["missed_per_day"] <= a_d else "ABOVE"
            log(f"      {name:58s} missed/day {s['missed_per_day']:.3f} ({ok} {a_d:.3f})  miss frac {s['miss_frac']:.3f}  in-house {s['inhouse']:5d}  "
                f"expedited {s['expedited']:6d}  cost {s['cost']:8.0f}  mean tau {s['mean_tau']:.4f}  bound slack/day {s['bound_slack_per_day']:.3f}")
    return out


def main():
    t0 = time.time(); rng = np.random.default_rng(7)
    tags = sorted(f[7:-4] for f in os.listdir(BB) if f.startswith("p4bb3b_") and f.endswith(".npz"))
    if not tags: log("no p4bb3b_*.npz found - run p4bb3_data.py first"); return False
    log("== Paper 4: Backblaze v3b (bagged per-horizon predictor) - proactive replacement, exact path, calibrated risk margin, abstention ==")
    log(f"horizon {P_DAYS} days, capacities {CAPS}/day, costs {COST}, tau_min {TAU_MIN}, alphas {ALPHAS}")
    for tag in tags:
        d, info, rows = load(tag)
        summ[tag] = offline(tag, d, info, rows, rng)
        summ[tag]["online"] = online(tag, d, info, rows)
        json.dump(summ, open(os.path.join(OUT, "summary_bb3.json"), "w"), indent=1, default=str)
        open(os.path.join(OUT, "summary_bb3.txt"), "w", encoding="utf-8").write("\n".join(lines))
    log(f"\ndone in {time.time() - t0:.0f}s")
    open(os.path.join(OUT, "summary_bb3.txt"), "w", encoding="utf-8").write("\n".join(lines))
    return True


if __name__ == "__main__":
    ok = False
    try: ok = main()
    except Exception:
        import traceback; traceback.print_exc()
    sys.exit(0 if ok else 1)

# -*- coding: utf-8 -*-
"""
Paper 4, real-fleet case (Backblaze), step 2: proactive drive-replacement planning with a calibrated risk margin.

Planning instance (one per drive model and decision day t): all drives in service on day t, horizon P = 28 days,
in-house technician capacity C replacements per day; options per drive: in-house replacement on day p (1..P),
immediate expedited replacement ('outsource', always feasible), or no action within the horizon (defer).
Margin: lambda >= 0, threshold tau = exp(-lambda). Drive k gets deadline day q_k = 7 j + 1 where j is the first horizon
(7/14/21/28 days) whose predicted failure probability G_kj >= tau; if none, it is deferred. This is exactly the
deadline structure of the C-MAPSS MILP (state = latest allowed day), so the same exact-path argument applies: the plan
changes only at lambda = -log G_kj, <= 4 breakpoints per drive.
Costs are uniform (c_in per in-house replacement, c_out per expedited one), so the planning problem is unit-job
scheduling with deadlines; earliest-deadline-first with earliest free day is optimal for the number of in-house
replacements (ties broken by risk, then drive id -> deterministic), i.e. the MILP is solved exactly by a greedy.
Realised outcome: a drive that fails on day T (1..28) is a MISSED failure unless it was replaced on a day p <= T
(expedited replacements happen on day 1).
Loss for CRC: missed failures / failures within the horizon (miss fraction, in [0,1]).
Domain: lambda in [0, LAM_MAX] (tau >= TAU_MIN); beyond LAM_MAX the safe action 'replace every drive now' (loss 0).

Offline: random-day splits (exchangeable reference) and forward splits (calibrate on 60 days, test on the next 30).
Online: daily rolling horizon from EVAL_START; the plan is recomputed every day and only day-1 actions are executed;
replaced drives leave the data; err_t = missed failures on day t+1; conformal controller
lambda_{t+1} = clip(lambda_t + eta (err_t - alpha_d), 0, LAM_MAX) with alpha_d = alpha_frac x training failures per day.
Output: results/summary_bb.txt, results/summary_bb.json
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
ALPHAS = [0.2, 0.3, 0.5]
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


def crc(paths, alpha, col=0, B=1.0):
    cand = np.unique(np.concatenate([[0.0]] + [p[0] for p in paths])); n = len(paths)
    R = np.zeros(len(cand))
    for b, V in paths:
        R += sufmax(V[:, col])[np.searchsorted(b, cand, side="right")]
    R /= n
    ok = np.where((n * R + B) / (n + 1) <= alpha)[0]
    return float(cand[ok[0]] + 1e-7) if len(ok) else np.inf


def empirical(paths, alpha, col=0):
    cand = np.unique(np.concatenate([[0.0]] + [p[0] for p in paths]))
    R = np.zeros(len(cand))
    for b, V in paths: R += V[np.searchsorted(b, cand, side="right"), col]
    R /= len(paths)
    ok = np.where(R <= alpha)[0]
    return float(cand[ok[0]] + 1e-7) if len(ok) else np.inf


def model_lambda(path, alpha):
    b, V = path; ok = np.where(V[:, 3] <= alpha)[0]
    i = ok[0] if len(ok) else len(V) - 1
    return 0.0 if i == 0 else float(b[i - 1] + 1e-7)


def lam_of_tau(tau): return float(-np.log(max(min(tau, 1.0), 1e-12)))


# ------------------------------------------------------------------ data
def load(tag):
    d = np.load(os.path.join(BB, f"p4bb_{tag}.npz")); info = json.load(open(os.path.join(BB, f"p4bb_{tag}_info.json")))
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


def offline(tag, d, info, rows, rng):
    maxday = max(rows); last_full = maxday - P_DAYS
    inst = instances(d, rows, last_full)
    days = sorted(inst)
    log(f"\n=== {info['model']}: {len(days)} decision days with a full 28-day outcome; eval AUC {info['auc_eval']}; "
        f"failures within 28 d per day: {np.mean([inst[t].nf for t in days]):.1f}; drives at risk (G28 >= {TAU_MIN}) per day: "
        f"{np.mean([len(inst[t].risky) for t in days]):.0f} of {np.mean([len(inst[t].G) for t in days]):.0f}")
    res = {}
    for C in CAPS:
        t0 = time.time()
        paths = {t: inst[t].path(C) for t in days}
        lam_fpr = lam_of_tau(info["thr_fpr1_G28"])
        log(f"  C = {C}/day: exact paths computed ({np.mean([len(paths[t][0]) for t in days]):.0f} breakpoints per day, {time.time() - t0:.0f}s)")
        rs = {thr: [raidshield_outcome(inst[t], C, thr) for t in days] for thr in (0, 200)}
        for split in ("random", "forward"):
            if split == "random":
                reps = []
                for _ in range(200):
                    p = rng.permutation(len(days)); reps.append(([days[i] for i in p[:60]], [days[i] for i in p[60:]]))
            else:
                reps = [(days[s:s + 60], days[s + 60:s + 90]) for s in range(0, len(days) - 89, 30)]
            for alpha in ALPHAS:
                acc = {}
                def add(name, outs, lam=np.nan):
                    a = acc.setdefault(name, dict(o=[], risk=[], lam=[])); v = np.array(outs, float)
                    a["o"].append(v); a["risk"].append(v[:, 0].mean()); a["lam"].append(lam)
                for cal, te in reps:
                    cp = [paths[t] for t in cal]
                    for name, lam in (("CRC exact path", crc(cp, alpha)), ("validation-tuned", empirical(cp, alpha)),
                                      ("fixed FPR 1% threshold", lam_fpr), ("point (tau = 0.5)", lam_of_tau(0.5))):
                        add(name, [at(paths[t], lam) for t in te], lam)
                    lams = [model_lambda(paths[t], alpha) for t in te]
                    add("model-based (predicted miss <= alpha, per day)", [at(paths[t], l) for t, l in zip(te, lams)], float(np.mean(lams)))
                    for thr in (0, 200):
                        add(f"RAIDShield-style SMART5 > {thr}", [rs[thr][days.index(t)] for t in te])
                key = f"{tag} | C={C} | {split} split | alpha={alpha}"
                out = {}
                log(f"\n  {key}  ({len(reps)} splits; risk = test mean miss fraction; >alpha = share of splits above alpha)")
                log(f"    {'rule':48s} {'miss':>6s} {'>alpha':>6s} {'tau':>8s} {'cost':>7s} {'in-house':>8s} {'exped.':>7s} {'false':>6s}")
                for name, a in acc.items():
                    v = np.concatenate(a["o"]); rk = np.array(a["risk"]); lm = np.array(a["lam"], float)
                    s = summarize(v); s["above"] = float(np.mean(rk > alpha + 1e-12))
                    s["tau"] = None if np.all(np.isnan(lm)) else float(np.exp(-np.nanmean(np.where(np.isinf(lm), LAM_MAX + 1, lm))))
                    s["safe_share"] = float(np.mean(np.isinf(lm)))
                    out[name] = s
                    tau = "    rule" if s["tau"] is None else f"{s['tau']:8.4f}"
                    flag = "  <-- violates" if s["above"] > 0.5 else ""
                    log(f"    {name:48s} {s['miss']:6.3f} {s['above']:6.2f} {tau} {s['cost']:7.0f} {s['inhouse']:8.1f} {s['exped']:7.1f} {s['false']:6.1f}{flag}")
                res[key] = out
    return res


# ------------------------------------------------------------------ online
def online(tag, d, info, rows, static_lams):
    days = sorted(rows)[60:]; C = 2                 # the first 60 days were used to calibrate the static rule
    fail_rate = info.get("train_fail_per_day") or np.mean([np.sum(d["T"][rows[t]] == 1) for t in days[:30]])
    out = {}
    log(f"\n  ONLINE {info['model']}: {len(days)} days (after the 60 calibration days), C = {C}/day, training failures/day = {fail_rate:.2f}")
    for alpha in ALPHAS:
        a_d = alpha * fail_rate
        rules = {"static CRC (calibrated on the first 60 days)": ("fixed", static_lams.get(alpha, np.inf)),
                 "fixed FPR 1% threshold": ("fixed", lam_of_tau(info["thr_fpr1_G28"])),
                 "point (tau = 0.5)": ("fixed", lam_of_tau(0.5)),
                 "most conservative in domain (tau = tau_min): eventual-safety floor": ("fixed", LAM_MAX)}
        for eta in (0.5, 2.0):
            rules[f"adaptive cold eta={eta}"] = ("adapt", 0.0, eta)
            rules[f"adaptive warm eta={eta}"] = ("adapt", min(static_lams.get(alpha, LAM_MAX), LAM_MAX), eta)
        log(f"    alpha = {alpha} (target miss fraction) -> alpha_d = {a_d:.3f} missed failures per day")
        for name, r in rules.items():
            removed = np.zeros(int(d["drive"].max()) + 1, bool); lam = r[1]; eta = r[2] if r[0] == "adapt" else 0.0
            miss = fail = prevented = rep_in = rep_out = 0; lam_hist = []
            last_day = days[-1]
            for t in days:
                rr = rows[t]
                rr = rr[~removed[d["drive"][rr]]]
                inst = Instance(d["G"][rr], d["T"][rr], d["smart5"][rr], d["drive"][rr])
                lam_use = min(lam, LAM_MAX) if np.isfinite(lam) else LAM_MAX
                q = inst.q_of_tau(np.exp(-lam_use))
                u = inst.risky[inst.order_key]
                day = inst.plan(q[inst.order_key], C, u)
                act = u[(day == 0) | (day == 1)]
                removed[d["drive"][rr][act]] = True
                Ta = inst.T[act]
                prevented += int(np.sum((Ta >= 1) & (t + Ta <= last_day)))   # would have failed inside the online window
                rep_in += int(np.sum(day == 1)); rep_out += int(np.sum(day == 0))
                tomorrow = inst.T == 1
                f_t = int(tomorrow.sum()); acted = np.zeros(len(rr), bool); acted[act] = True
                m_t = int(np.sum(tomorrow & ~acted))
                miss += m_t; fail += f_t; lam_hist.append(lam_use)
                if r[0] == "adapt":
                    lam = float(min(max(lam + eta * (m_t - a_d), 0.0), LAM_MAX))   # clipped to the domain (eps_safe > 0, see notes)
            s = dict(missed_per_day=miss / len(days), miss_frac=miss / max(miss + prevented, 1), failures=fail, prevented=prevented,
                     inhouse=rep_in, expedited=rep_out,
                     cost=COST["c_in"] * rep_in + COST["c_out"] * rep_out + COST["c_fail"] * miss, mean_tau=float(np.exp(-np.mean(lam_hist))))
            out[f"{alpha}|{name}"] = s
            log(f"      {name:66s} missed/day {s['missed_per_day']:.3f} (target {a_d:.3f})  miss frac {s['miss_frac']:.3f}  "
                f"repl in-house {rep_in:5d} expedited {rep_out:5d}  cost {s['cost']:7.0f}  mean tau {s['mean_tau']:.4f}")
    return out


def main():
    t0 = time.time(); rng = np.random.default_rng(7)
    tags = sorted(f[5:-4] for f in os.listdir(BB) if f.startswith("p4bb_") and f.endswith(".npz"))
    if not tags: log("no p4bb_*.npz found - run p4bb_data.py first"); return False
    log("== Paper 4: Backblaze proactive replacement (exact path, calibrated risk margin) ==")
    log(f"horizon {P_DAYS} days, capacities {CAPS}/day, costs {COST}, tau_min {TAU_MIN}")
    for tag in tags:
        d, info, rows = load(tag)
        summ[tag] = offline(tag, d, info, rows, rng)
        # static margins for the online runs: CRC on the first 60 decision days, C = 2
        first = sorted(rows)[:60]
        inst = {t: Instance(d["G"][rows[t]], d["T"][rows[t]], d["smart5"][rows[t]], d["drive"][rows[t]]) for t in first}
        cp = [inst[t].path(2) for t in first]
        static = {a: crc(cp, a) for a in ALPHAS}
        summ[tag]["online"] = online(tag, d, info, rows, static)
        json.dump(summ, open(os.path.join(OUT, "summary_bb.json"), "w"), indent=1, default=str)
        open(os.path.join(OUT, "summary_bb.txt"), "w", encoding="utf-8").write("\n".join(lines))
    log(f"\ndone in {time.time() - t0:.0f}s")
    open(os.path.join(OUT, "summary_bb.txt"), "w", encoding="utf-8").write("\n".join(lines))
    return True


if __name__ == "__main__":
    ok = False
    try: ok = main()
    except Exception:
        import traceback; traceback.print_exc()
    sys.exit(0 if ok else 1)

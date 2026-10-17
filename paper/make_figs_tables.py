# -*- coding: utf-8 -*-
"""Paper 4: builds the results tables (Markdown) and figures (PDF + PNG) from the stored result files.
Inputs (read-only): prototype_v2/results/{summary_v2.json, v2_online.pkl.gz, v2_online_ca.pkl.gz},
backblaze_v3/results/{summary_bb3.json, summary_bb3c.json}, backblaze_v4/results/bb4_<model>.json.
Usage: python make_figs_tables.py <paper4 root folder> <output folder>"""
import os, sys, json, gzip, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT, OUT = sys.argv[1], sys.argv[2]
FIG = os.path.join(OUT, "figs"); os.makedirs(FIG, exist_ok=True)
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"      # categorical slots (validated palette, light mode)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2, "savefig.dpi": 300})
FLEETS = ["ST16000NM001G", "TOSHIBA_MG07ACA14TA", "TOSHIBA_MG08ACA16TA", "WDC_WUH722222ALE6L4"]
SHORT = {"ST16000NM001G": "ST16", "TOSHIBA_MG07ACA14TA": "MG07", "TOSHIBA_MG08ACA16TA": "MG08", "WDC_WUH722222ALE6L4": "WDC22"}
CRC, LTT, VT = "CRC (mean guarantee)", "LTT-HB delta=0.1 (high-probability)", "validation-tuned"
CRC3 = "CRC exact path (abstains if not certifiable)"; MB3 = "model-based (predicted miss <= alpha, per day)"
md = []


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight")
    plt.close(fig)


def f(x, p=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"


# ---------------------------------------------------------------- C-MAPSS
v2 = json.load(open(os.path.join(ROOT, "prototype_v2", "results", "summary_v2.json")))
ROWS_MEAN = [("Point (λ = 0)", "point (lambda 0)"), ("Zhu-style r̂ − 3σ", "Zhu-style rhat-3sigma"),
             ("Mitici-style r̂ − 2.33σ", "Mitici-style rhat-2.33sigma"), ("Validation-tuned", "[s05] validation-tuned"),
             ("Model-based CC (Markov)", "[s05] model Gaussian E[N]/K <= alpha (per instance)"),
             ("Per-unit CP (α)", "[s05] per-unit CP (alpha)"), ("CRC, grid", "[s05] CRC on v1 grid"),
             ("**CRC, exact path**", "[s05] CRC exact path")]
ROWS_ANY = [("Point (λ = 0)", "point (lambda 0)"), ("Zhu-style r̂ − 3σ", "Zhu-style rhat-3sigma"),
            ("Mitici-style r̂ − 2.33σ", "Mitici-style rhat-2.33sigma"), ("Validation-tuned", "[s05] validation-tuned"),
            ("Model-based CC (Poisson-binomial)", "[s05] model Gaussian Poisson-binomial P(any) <= alpha (per instance)"),
            ("Per-unit CP (α)", "[s05] per-unit CP (alpha)"), ("Per-unit CP (α/K)", "[s05] per-unit CP Bonferroni (alpha/K)"),
            ("Joint max-score CP", "[s05] joint max-score CP"), ("Scenario-max", "[s05] scenario-max (Campi-Garatti d=1)"),
            ("LTT, grid", "[s05] LTT on v1 grid"), ("**LTT, exact path**", "[s05] LTT exact path")]

md.append("## Table 2. C-MAPSS, in-distribution (ID, K = 20, C = 2, n = 100 calibration scenarios, 500 replications; "
          "σ-scaled family s_min = 0.5). Risk = realised test risk; >α = share of replications above α; cost = mean plan cost.\n")
md.append("| Rule | Failure fraction, α = 0.02: risk | >α | cost | Rule | Any failure, α = 0.1: risk | >α | cost |")
md.append("|---|---|---|---|---|---|---|---|")
A = v2["ID | mean | alpha=0.02 | n=100"]; B = v2["ID | any | alpha=0.1 | n=100"]
for i in range(max(len(ROWS_MEAN), len(ROWS_ANY))):
    left = ROWS_MEAN[i] if i < len(ROWS_MEAN) else None; right = ROWS_ANY[i] if i < len(ROWS_ANY) else None
    l = f"{left[0]} | {f(A[left[1]]['risk'], 4)} | {f(A[left[1]]['above'], 2)} | {A[left[1]]['cost']:.0f}" if left else " | | | "
    r = f"{right[0]} | {f(B[right[1]]['risk'], 3)} | {f(B[right[1]]['above'], 2)} | {B[right[1]]['cost']:.0f}" if right else " | | | "
    md.append(f"| {l} | {r} |")

md.append("\n## Table 3. C-MAPSS under a new fault mode (any failure, α = 0.1, LTT δ = 0.1; σ-scaled s_min = 0.5): "
          "naive transfer (calibrated on the source fleet) vs recalibration on n target scenarios.\n")
md.append("| Setting | Rule | Source → source: risk / cost | Naive source → target: risk / cost | Recalibrated n = 50: risk / cost |")
md.append("|---|---|---|---|---|")
for s in ("S13", "S24"):
    for lab, key in (("Zhu-style", "Zhu-style rhat-3sigma"), ("Model-based CC", "[s05] model Gaussian Poisson-binomial P(any) <= alpha (per instance)"),
                     ("Validation-tuned", "[s05] validation-tuned"), ("Joint max-score CP", "[s05] joint max-score CP"),
                     ("**LTT, exact path**", "[s05] LTT exact path")):
        a = v2[f"{s} source->source | any | alpha=0.1 | n=100"][key]; b = v2[f"{s} NAIVE source->target (shift) | any | alpha=0.1 | n=100"][key]
        c = v2[f"{s} RECALIBRATED on target | any | alpha=0.1 | n=50"][key]
        md.append(f"| {s} | {lab} | {f(a['risk'])} / {a['cost']:.0f} | {f(b['risk'])} / {b['cost']:.0f} | {f(c['risk'])} / {c['cost']:.0f} |")

# Figure 1: non-monotone share by capacity
mono = v2["monotonicity"]
caps = [1, 2, 3, 20]
fig, ax = plt.subplots(figsize=(3.4, 2.3))
x = np.arange(len(caps)); w = 0.36
y1 = [mono[f"CAP_C{c}|add"]["share_mean"] for c in caps]; y2 = [mono[f"CAP_C{c}|add"]["share_any"] for c in caps]
b1 = ax.bar(x - w / 2 - 0.01, y1, w, color=S1, label="failure fraction")
b2 = ax.bar(x + w / 2 + 0.01, y2, w, color=S2, label="any failure")
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{100 * r.get_height():.1f}%".replace(".0%", "%"), ha="center", va="bottom", fontsize=7, color=INK2)
ax.set_xticks(x, [f"C = {c}" + ("\n(non-binding)" if c == 20 else "") for c in caps], fontsize=7.5)
ax.set_xlabel("in-house slots per period (K = 20)", fontsize=7.5); ax.set_ylabel("non-monotone loss paths"); ax.set_ylim(0, 0.6)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=7.5)
save(fig, "fig4_monotonicity_capacity")

# Figure 2: risk vs cost (ID)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
for ax, D, alpha, rows, title, xmax in ((axs[0], A, 0.02, ROWS_MEAN, "Failure fraction, α = 0.02", 0.13), (axs[1], B, 0.1, ROWS_ANY, "Any failure, α = 0.1", 0.95)):
    ax.axvspan(alpha, xmax, color="#f3f2ee", zorder=0); ax.axvline(alpha, color=INK2, lw=1, ls="--", zorder=1)
    ax.text(0.99, 0.98, f"shaded: risk > α = {alpha}", transform=ax.transAxes, va="top", ha="right", fontsize=6.8, color=INK2)
    for lab, key in rows:
        if lab in ("CRC, grid", "LTT, grid", "Joint max-score CP"): continue
        if lab == "Per-unit CP (α/K)": lab = "Per-unit CP (α/K) ≈ joint CP"
        r = D[key]; ours = lab.startswith("**")
        ax.scatter(r["risk"], r["cost"], s=46 if ours else 26, color=S1 if ours else (S2 if r["risk"] > alpha else "#8f8e88"),
                   edgecolor="white", linewidth=1.2, zorder=3, marker="D" if ours else "o")
        off = {"CRC, exact path": (-6, 9), "Zhu-style r̂ − 3σ": (5, -10), "Validation-tuned": (6, -3), "Per-unit CP (α)": (6, -3)}.get(lab.strip("*"), (4, 2))
        ax.annotate(lab.strip("*"), (r["risk"], r["cost"]), xytext=off, textcoords="offset points", fontsize=6.3,
                    color=INK if ours else INK2)
    ax.set_xlabel("realised test risk"); ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    ax.set_xlim(-0.003 if alpha < 0.05 else -0.02, xmax)
axs[0].set_ylabel("mean plan cost")
save(fig, "fig1_cmapss_risk_cost")

# Online C-MAPSS (from the run files)
on = pickle.load(gzip.open(os.path.join(ROOT, "prototype_v2", "results", "v2_online.pkl.gz")))
ca = pickle.load(gzip.open(os.path.join(ROOT, "prototype_v2", "results", "v2_online_ca.pkl.gz")))
K = on["cfg"]["K"]


def series(runs, s, mode, rule):
    F = np.array([runs[(s, mode, rule, k)][0] for k in range(5) if (s, mode, rule, k) in runs], float)
    Cst = np.array([runs[(s, mode, rule, k)][2] for k in range(5) if (s, mode, rule, k) in runs], float)
    return F, Cst


ON_RULES = [("Zhu-style r̂ − 3σ", on["runs"], "zhu"), ("Mitici-style", on["runs"], "mitici"), ("Static CRC", on["runs"], "static_CRC"),
            ("Adaptive (cold, η = 40)", on["runs"], "adaptive_s05_cold_eta40_a{a}"), ("**Cost-aware (η = 40)**", ca["runs"], "costaware_s05_eta40_a{a}")]
md.append("\n## Table 7a. C-MAPSS online (K = 20, T = 1,000 periods, 5 seeds; shift = target fleet after period 300; "
          "σ-scaled s_min = 0.5). Rate = failures per engine-period; cost = mean cost per period.\n")
md.append("| Rule | α | S13 stationary: rate / cost | S13 shift: rate / cost | S24 stationary: rate / cost | S24 shift: rate / cost |")
md.append("|---|---|---|---|---|---|")
for a in (0.0025, 0.005):
    for lab, runs, rule in ON_RULES:
        cells = []
        for s in ("S13", "S24"):
            for mode in ("stationary", "shift"):
                F, Cst = series(runs, s, mode, rule.format(a=a))
                cells.append(f"{F.sum(1).mean() / (K * F.shape[1]):.4f} / {Cst.mean():.0f}" if len(F) else "–")
        md.append(f"| {lab} | {a} | " + " | ".join(cells) + " |")
    if a == 0.0025: pass

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
a = 0.0025
cols = {"Zhu-style r̂ − 3σ": "#8f8e88", "Mitici-style": "#b9b8b1", "Static CRC": S2, "Adaptive (cold, η = 40)": S3, "**Cost-aware (η = 40)**": S1}
for lab, runs, rule in ON_RULES:
    F, Cst = series(runs, "S13", "shift", rule.format(a=a))
    t = np.arange(1, F.shape[1] + 1)
    rate = (np.cumsum(F, 1) / (K * t)).mean(0); cost = (np.cumsum(Cst, 1) / t).mean(0)
    axs[0].plot(t, rate, color=cols[lab], label=lab.strip("*"), lw=2.2 if lab.startswith("**") else 1.6)
    axs[1].plot(t, cost, color=cols[lab], lw=2.2 if lab.startswith("**") else 1.6)
for ax in axs: ax.axvline(300, color=INK2, lw=1, ls=":"); ax.set_xlabel("period")
axs[0].axhline(a, color=INK2, lw=1, ls="--"); axs[0].text(1000, a, " α", va="center", fontsize=7, color=INK2)
axs[0].text(310, 0.0150, "shift →", fontsize=7, color=INK2)
axs[0].set_ylabel("cumulative failures per engine-period"); axs[1].set_ylabel("cumulative mean cost per period")
axs[0].set_ylim(0, 0.016); fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 1.06))
fig.subplots_adjust(wspace=0.32)
save(fig, "fig6_cmapss_online_S13_shift")

# ---------------------------------------------------------------- Backblaze
b3 = json.load(open(os.path.join(ROOT, "backblaze_v3", "results", "summary_bb3.json")))
b3c = json.load(open(os.path.join(ROOT, "backblaze_v3", "results", "summary_bb3c.json")))
b4 = {tg: json.load(open(os.path.join(ROOT, "backblaze_v4", "results", f"bb4_{tg}.json"))) for tg in FLEETS}

md.append("\n## Table 4. Backblaze: predictors and floors (miss fraction of the most conservative in-domain plan, C = 1 / 2 / 4).\n")
md.append("| Fleet | Predictor | Feature set | Eval AUC h7 / h14 / h21 / h28 | Floor C = 1 / 2 / 4 |")
md.append("|---|---|---|---|---|")
for tg in FLEETS:
    info3 = json.load(open(os.path.join(ROOT, "data", "backblaze", f"p4bb3b_{tg}_info.json"))) if os.path.exists(os.path.join(ROOT, "data", "backblaze", f"p4bb3b_{tg}_info.json")) else None
    i4 = b4[tg]["info"]
    a3 = " / ".join(f(v, 2) for v in info3["auc_eval"].values()) if info3 else "see Section 5"
    md.append(f"| {SHORT[tg]} | baseline (v3b) | A | {a3} | " + " / ".join(f(b3c[tg][f'C={c} | floor'], 2) for c in (1, 2, 4)) + " |")
    md.append(f"| {SHORT[tg]} | stronger (v4) | {i4['feature_set']} | " + " / ".join(f(v, 2) for v in i4["auc_eval"].values()) + " | "
              + " / ".join(f(b4[tg]['v3c'][f'C={c} | floor'], 2) for c in (1, 2, 4)) + " |")

md.append("\n## Table 5. Backblaze, offline certification (v4 predictor, C = 2, 100 random-day splits, n = 60). Miss = mean test "
          "miss fraction; abst = abstention share; cert >α = share of CERTIFIED splits above α; cost = mean plan cost per day.\n")
md.append("| Fleet | α | CRC: miss / abst / cert >α / cost | LTT (δ = 0.1): miss / abst / cert >α / cost | Validation-tuned: miss / >α | Model-based: miss / >α |")
md.append("|---|---|---|---|---|---|")
for tg in FLEETS:
    for a in (0.3, 0.4, 0.5):
        r = b4[tg]["v3c"][f"C=2 | random | alpha={a}"]; v = b4[tg]["v3b"][f"{tg} | C=2 | random split | alpha={a}"]
        c, l, vt, mb = r[CRC], r[LTT], v["validation-tuned"], v[MB3]
        md.append(f"| {SHORT[tg]} | {a} | {f(c['miss'])} / {f(c['abstain'], 2)} / {f(c['cert_above'], 2)} / {c['cost']:.0f} | "
                  f"{f(l['miss'])} / {f(l['abstain'], 2)} / {f(l['cert_above'], 2)} / {l['cost']:.0f} | {f(vt['miss'])} / {f(vt['above'], 2)} | "
                  f"{f(mb['miss'])} / {f(mb['above'], 2)} |")

# Figure 3: frontier + certified points (v4, C=2)
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=False)
for ax, tg in zip(axs, FLEETS):
    fr = b4[tg]["v3b"][f"{tg} | C=2 | frontier"]
    ax.plot([p["miss"] for p in fr], [p["cost"] for p in fr], color="#8f8e88", lw=1.4, marker="o", ms=2.5, label="fixed margins (frontier)")
    floor = b4[tg]["v3c"]["C=2 | floor"]
    ax.axvline(floor, color=INK2, lw=1, ls=":"); ax.text(floor, 1.0, " floor", transform=ax.get_xaxis_transform(), fontsize=6.5, color=INK2, va="top")
    for a, mk in ((0.3, "o"), (0.4, "s"), (0.5, "D")):
        r = b4[tg]["v3c"][f"C=2 | random | alpha={a}"]
        for rule, col in ((CRC, S1), (LTT, S2)):
            p = r[rule]
            if p["abstain"] > 0.5: continue
            ax.scatter(p["miss"], p["cost"], color=col, marker=mk, s=28, edgecolor="white", linewidth=1, zorder=4)
    ax.set_title(SHORT[tg], fontsize=8.5, loc="left"); ax.set_xlabel("miss fraction"); ax.set_xlim(0.1, 0.9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if str(int(round(v)))[0] in "25" else ""))
    ax.tick_params(axis="y", which="minor", labelsize=6.5)
axs[0].set_ylabel("mean plan cost per day (log)")
fig.subplots_adjust(wspace=0.45)
from matplotlib.lines import Line2D
h = [Line2D([], [], color="#8f8e88", marker="o", ms=3, lw=1.4, label="fixed margins"),
     Line2D([], [], color=S1, marker="o", lw=0, label="CRC"), Line2D([], [], color=S2, marker="o", lw=0, label="LTT"),
     Line2D([], [], color=INK2, marker="o", lw=0, label="α = 0.3"), Line2D([], [], color=INK2, marker="s", lw=0, label="α = 0.4"),
     Line2D([], [], color=INK2, marker="D", lw=0, label="α = 0.5")]
fig.legend(handles=h, loc="upper center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.08))
save(fig, "fig2_backblaze_frontier")

# Figure 4: LTT vs n (alpha 0.4)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
for tg, col in zip(["ST16000NM001G", "TOSHIBA_MG07ACA14TA", "WDC_WUH722222ALE6L4"], (S1, S2, S3)):
    L = b4[tg]["ltt_vs_n"]; ns = [30, 60, 90, 120]
    ab = [L[f"n={n}|alpha=0.4"]["ltt_abstain"] for n in ns]; co = [L[f"n={n}|alpha=0.4"]["ltt_cost"] / L[f"n={n}|alpha=0.4"]["crc_cost"] for n in ns]
    axs[0].plot(ns, ab, color=col, marker="o", ms=4, label=SHORT[tg]); axs[1].plot(ns, co, color=col, marker="o", ms=4)
    axs[0].annotate(SHORT[tg], (ns[0], ab[0]), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK2)
axs[0].set_ylabel("LTT abstention share"); axs[1].set_ylabel("LTT cost / CRC cost")
for ax in axs: ax.set_xlabel("calibration days n"); ax.set_xticks([30, 60, 90, 120])
axs[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); axs[0].legend(fontsize=7)
axs[1].axhline(1, color=INK2, lw=1, ls="--")
save(fig, "fig5_ltt_vs_n")

# Figure 5: price of the guarantee vs failure cost
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for ax, a in zip(axs, (0.4, 0.5)):
    for tg, col in zip(["ST16000NM001G", "TOSHIBA_MG07ACA14TA", "WDC_WUH722222ALE6L4"], (S1, S2, S3)):
        cs = b4[tg]["cost_sensitivity"]; cf = [10, 30, 100]
        y = [100 * cs[f"c_fail={c}|alpha={a}"]["CRC"]["premium"] for c in cf]
        y2 = [100 * cs[f"c_fail={c}|alpha={a}"]["LTT"]["premium"] for c in cf]
        ax.plot(cf, y, color=col, marker="o", ms=4, label=f"{SHORT[tg]} CRC")
        ax.plot(cf, y2, color=col, marker="o", ms=3, ls="--", lw=1.3, label=f"{SHORT[tg]} LTT")
    ax.set_xscale("log"); ax.set_xticks([10, 30, 100], ["10", "30", "100"]); ax.set_xlabel("failure cost / in-house replacement cost")
    ax.set_title(f"α = {a}", fontsize=8.5, loc="left"); ax.set_yscale("log"); ax.set_ylim(0.8, 1000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
axs[0].set_ylabel("cost premium over the hindsight-\nbest fixed plan (log scale)"); axs[1].legend(fontsize=6.5, ncol=2)
save(fig, "fig3_price_vs_failure_cost")

# Table 5b: Backblaze online (v4)
md.append("\n## Table 7b. Backblaze online (v4 predictor, C = 2, days 60–227). Missed failures per day (target α_d) / cost.\n")
md.append("| Fleet | α | α_d | Online floor | Static CRC | Static LTT | Adaptive cold | Adaptive warm | Cost-aware |")
md.append("|---|---|---|---|---|---|---|---|---|")
for tg in FLEETS:
    o = b4[tg]["v3b"]["online"]; ol = b4[tg]["v3c"]["online"]; fr = b4[tg]["info"]["train_fail_per_day"]
    fl = [v for k, v in o.items() if k.startswith("fixed|")][-1]["missed_per_day"]
    for a in (0.3, 0.4, 0.5):
        g = lambda n: o[f"{a}|{n}"]
        cell = lambda s: f"{s['missed_per_day']:.3f} / {s['cost']:.0f}"
        md.append(f"| {SHORT[tg]} | {a} | {a * fr:.3f} | {fl:.3f} | {cell(g('static CRC (first 60 days; abstains -> most conservative)'))} | "
                  f"{cell(ol[str(a)])}{' (abst.)' if ol[str(a)]['abstain'] else ''} | {cell(g('adaptive cold (lambda_0 = 0)'))} | "
                  f"{cell(g('adaptive warm (lambda_0 = static CRC)'))} | {cell(g('cost-aware (max(lambda_t, lambda_cost))'))} |")

# Table 7: exactness + sensitivity summary
md.append("\n## Table 6. Exact path on real instances and price of the guarantee (v4, C = 2).\n")
md.append("| Fleet | Exact = direct solve | Breakpoints / day | Path time (s) | CRC premium c_fail = 10 / 30 / 100 (α = 0.4) | (α = 0.5) |")
md.append("|---|---|---|---|---|---|")
for tg in FLEETS:
    e = b4[tg]["exactness_timing"]; cs = b4[tg]["cost_sensitivity"]
    p = lambda a: " / ".join(f"{100 * cs[f'c_fail={c}|alpha={a}']['CRC']['premium']:+.0f}%" for c in (10, 30, 100))
    md.append(f"| {SHORT[tg]} | {e['exact_matches']}/{e['checks']} | {e['breakpoints_mean']:.0f} | {e['path_seconds_mean']:.2f} | {p(0.4)} | {p(0.5)} |")

open(os.path.join(OUT, "results_tables.md"), "w", encoding="utf-8").write("# Paper 4 – results tables (generated)\n\n" + "\n".join(md) + "\n")
print("ok:", sorted(os.listdir(FIG)))

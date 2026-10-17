# Figure 2 of the paper. Usage: python make_fig2_cmapss.py <prototype_v2/results/summary_v2.json> <output folder>
import sys, os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S1, S2, GR = "#2a78d6", "#eb6834", "#8f8e88"; INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif", "font.size": 8,
  "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
  "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
v2 = json.load(open(sys.argv[1]))
A = v2["ID | mean | alpha=0.02 | n=100"]; B = v2["ID | any | alpha=0.1 | n=100"]
RM = [("CRC, exact path (ours)", "[s05] CRC exact path"), ("Per-unit CP (α)", "[s05] per-unit CP (alpha)"),
      ("Validation-tuned", "[s05] validation-tuned"), ("Zhu-style $\\hat r-3\\sigma$", "Zhu-style rhat-3sigma"),
      ("Mitici-style $\\hat r-2.33\\sigma$", "Mitici-style rhat-2.33sigma"), ("Model-based CC (Markov)", "[s05] model Gaussian E[N]/K <= alpha (per instance)"),
      ("Point ($\\lambda=0$)", "point (lambda 0)")]
RA = [("LTT, exact path (ours)", "[s05] LTT exact path"), ("Joint max-score CP", "[s05] joint max-score CP"),
      ("Scenario-max", "[s05] scenario-max (Campi-Garatti d=1)"), ("Validation-tuned", "[s05] validation-tuned"),
      ("Per-unit CP (α)", "[s05] per-unit CP (alpha)"), ("Zhu-style $\\hat r-3\\sigma$", "Zhu-style rhat-3sigma"),
      ("Mitici-style $\\hat r-2.33\\sigma$", "Mitici-style rhat-2.33sigma"), ("Model-based CC (Poisson-binomial)", "[s05] model Gaussian Poisson-binomial P(any) <= alpha (per instance)"),
      ("Point ($\\lambda=0$)", "point (lambda 0)")]
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.9), gridspec_kw=dict(height_ratios=[len(RM), len(RA)], width_ratios=[1.15, 1], hspace=0.42, wspace=0.08))
for row, (D, rows, alpha, title, xmax) in enumerate(((A, RM, 0.02, "(a) Mean failure fraction, target α = 0.02 (dashed)", 0.145), (B, RA, 0.1, "(b) Probability of any failure, target α = 0.1 (dashed)", 1.0))):
    y = np.arange(len(rows))[::-1]
    risk = [D[k]["risk"] for _, k in rows]; cost = [D[k]["cost"] for _, k in rows]; above = [D[k]["above"] for _, k in rows]
    ours = [l.endswith("(ours)") for l, _ in rows]
    col = [S1 if o else (S2 if r > alpha else GR) for o, r in zip(ours, risk)]
    ax = axs[row, 0]
    ax.axvspan(alpha, xmax, color="#f3f2ee", zorder=0); ax.axvline(alpha, color=INK2, lw=1, ls="--", zorder=1)
    ax.hlines(y, 0, risk, color=GRID, lw=1, zorder=1)
    ax.scatter(risk, y, s=[52 if o else 34 for o in ours], c=col, marker="o", edgecolor="white", linewidth=1.2, zorder=3)
    for yy, r, a, o in zip(y, risk, above, ours):
        ax.text(r + xmax * 0.018, yy, f"{r:.3f}", va="center", fontsize=6.6, color=INK if o else INK2, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9), zorder=4)
    ax.set_yticks(y, [l for l, _ in rows]); ax.set_xlim(0, xmax); ax.grid(axis="y", visible=False)
    for t, o in zip(ax.get_yticklabels(), ours): t.set_fontweight("bold" if o else "normal")
    ax.set_title(title, loc="left", fontsize=8.3, color=INK)
    ax.set_xlabel("realised test risk (shaded: violates the target)", fontsize=7.3)
    bx = axs[row, 1]
    bx.barh(y, cost, color=col, height=0.55, edgecolor="white", linewidth=1.0, zorder=2)
    for yy, cst in zip(y, cost): bx.text(cst + max(cost) * 0.01, yy, f"{cst:,.0f}", va="center", fontsize=6.6, color=INK2)
    bx.set_yticks(y, []); bx.tick_params(axis="y", length=0); bx.set_xlim(0, max(cost) * 1.15); bx.grid(axis="y", visible=False)
    bx.set_xlabel("mean plan cost", fontsize=7.3)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=S1, ms=7, label="calibrated (ours)"), Line2D([], [], marker="o", ls="", color=GR, ms=6, label="meets the target on average"),
     Line2D([], [], marker="o", ls="", color=S2, ms=6, label="violates the target")]
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.005), fontsize=7.3)
fig.savefig(os.path.join(sys.argv[2], "fig1_cmapss_risk_cost.pdf"), bbox_inches="tight")

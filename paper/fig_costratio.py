# Figure 8 of the paper. Usage: python fig_costratio.py <prototype_v5/results/summary_costsens.json> <output folder>
import sys, os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S1, S2 = "#2a78d6", "#eb6834"; INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif", "font.size": 8,
  "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
  "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
R = json.load(open(sys.argv[1]))
ratios = [1, 2, 3.33, 5, 10, 15, 30, 50]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, fam, title in ((axs[0], "add", "(a) additive margins"), (axs[1], "s05", "(b) $\\sigma$-scaled margins")):
    for key, col, lab, mk in (("crc_prem", S1, "CRC, failure fraction ≤ 0.02", "o"), ("ltt_prem", S2, "LTT, P(any failure) ≤ 0.1", "s")):
        y = [100 * R[f"{fam}|{r}"][key] for r in ratios]
        ax.plot(ratios, y, color=col, lw=2, marker=mk, ms=4.5, mec="white", mew=0.8, label=lab)
    for x in (5, 15):
        ax.axvline(x, color=INK2, lw=0.9, ls="--", zorder=0)
    ax.set_xscale("log"); ax.set_xticks(ratios, ["1", "2", "3.3", "5", "10", "15", "30", "50"])
    ax.minorticks_off(); ax.set_xlabel("failure cost / external visit cost", fontsize=7.6)
    ax.set_title(title, loc="left", fontsize=8.3, color=INK)
    ax.text(5, 176, " aero-engine\n studies", fontsize=6.4, color=INK2, va="top"); ax.text(15, 176, " facility\n accounting", fontsize=6.4, color=INK2, va="top")
axs[0].set_ylabel("premium over the hindsight\ncost-optimal plan (%)", fontsize=7.6); axs[0].set_ylim(0, 180)
axs[1].legend(loc="upper right", fontsize=7, frameon=False, bbox_to_anchor=(1.0, 0.86))
fig.savefig(os.path.join(sys.argv[2], "fig8_price_vs_ratio.pdf"), bbox_inches="tight")

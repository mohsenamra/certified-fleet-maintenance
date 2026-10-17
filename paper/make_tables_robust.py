# -*- coding: utf-8 -*-
"""Builds the robustness tables (richer planner, N-CMAPSS) from summary_v5.json and summary_v6.json.
Usage: python make_tables_robust.py <summary_v5.json> <summary_v6.json> <output folder>"""
import sys, os, json
v5, v6, OUT = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
COLS = (("C-MAPSS, RICH-ID", v5, "RICH_ID"), ("C-MAPSS, RICH-TIGHT", v5, "RICH_TIGHT"), ("N-CMAPSS, NC-ID", v6, "NC_ID"))
MEAN = [(r"Point ($\lambda=0$)", "point (lambda 0)"), (r"$\hat r-3\sigma$", "Zhu-style rhat-3sigma"), (r"$\hat r-2.33\sigma$", "Mitici-style rhat-2.33sigma"),
        ("Cost-tuned margin", "de Pater-style cost-tuned margin [add]"), ("Renewal-reward deadlines", "Mitici 2023 renewal-reward deadlines"),
        ("Validation-tuned", "[s05] validation-tuned"), ("Model-based CC (Markov)", "[s05] model Gaussian E[N]/K <= alpha (per instance)"),
        (r"Per-unit CP ($\alpha$)", "[s05] per-unit CP (alpha)"), (r"\textbf{CRC, exact path}", "[s05] CRC exact path")]
ANY = [(r"Point ($\lambda=0$)", "point (lambda 0)"), (r"$\hat r-3\sigma$", "Zhu-style rhat-3sigma"), (r"$\hat r-2.33\sigma$", "Mitici-style rhat-2.33sigma"),
       ("Cost-tuned margin", "de Pater-style cost-tuned margin [add]"), ("Renewal-reward deadlines", "Mitici 2023 renewal-reward deadlines"),
       ("Validation-tuned", "[s05] validation-tuned"), ("Model-based CC (Poisson-binomial)", "[s05] model Gaussian Poisson-binomial P(any) <= alpha (per instance)"),
       (r"Per-unit CP ($\alpha$)", "[s05] per-unit CP (alpha)"), ("Joint max-score CP", "[s05] joint max-score CP"),
       ("Scenario-max", "[s05] scenario-max (Campi-Garatti d=1)"), (r"\textbf{LTT, exact path}", "[s05] LTT exact path")]


def table(label, caption, target, alpha, rows, dig):
    L = [r"\begin{table}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{4pt}", r"\caption{" + caption + r"}\label{" + label + "}",
         r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
         " & " + " & ".join((r"\multicolumn{3}{c@{}}{%s}" if i == 2 else r"\multicolumn{3}{c}{%s}") % c[0] for i, c in enumerate(COLS)) + r"\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
         r"Rule & Risk & $>\alpha$ & Cost & Risk & $>\alpha$ & Cost & Risk & $>\alpha$ & Cost \\", r"\midrule"]
    for lab, key in rows:
        cells = []
        for _, S, pre in COLS:
            d = S[f"{pre} | {target} | alpha={alpha} | n=100"].get(key)
            cells += ["--", "--", "--"] if d is None else [f"{d['risk']:.{dig}f}", f"{d['above']:.2f}", f"{d['cost']:,.0f}"]
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


open(os.path.join(OUT, "tab_rich_mean.tex"), "w").write(table("tab:rich_mean",
    r"Richer planner and N-CMAPSS: mean failure fraction, $\alpha=0.02$ ($n=100$; $\sigma$-scaled margins, $s_{\min}=0.5$). Risk: realised test risk; $>\alpha$: share of splits above the target; cost: mean plan cost. --: not evaluated in this case.",
    "mean", 0.02, MEAN, 4))
open(os.path.join(OUT, "tab_rich_any.tex"), "w").write(table("tab:rich_any",
    r"Richer planner and N-CMAPSS: probability of any failure, $\alpha=0.1$, $\delta=0.1$ ($n=100$; $\sigma$-scaled margins). Columns as in Table~\ref{tab:rich_mean}.",
    "any", 0.1, ANY, 3))
print("written")

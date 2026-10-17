# -*- coding: utf-8 -*-
"""Paper 4: LaTeX result tables from the stored result files. Usage: python make_tex_tables.py <paper4 root> <out dir>"""
import os, sys, json, gzip, pickle
import numpy as np

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
FLEETS = ["ST16000NM001G", "TOSHIBA_MG07ACA14TA", "TOSHIBA_MG08ACA16TA", "WDC_WUH722222ALE6L4"]
SHORT = {"ST16000NM001G": "ST16", "TOSHIBA_MG07ACA14TA": "MG07", "TOSHIBA_MG08ACA16TA": "MG08", "WDC_WUH722222ALE6L4": "WDC22"}
CRC, LTT = "CRC (mean guarantee)", "LTT-HB delta=0.1 (high-probability)"
MB3 = "model-based (predicted miss <= alpha, per day)"
v2 = json.load(open(os.path.join(ROOT, "prototype_v2", "results", "summary_v2.json")))
b3c = json.load(open(os.path.join(ROOT, "backblaze_v3", "results", "summary_bb3c.json")))
b3 = json.load(open(os.path.join(ROOT, "backblaze_v3", "results", "summary_bb3.json")))
b4 = {tg: json.load(open(os.path.join(ROOT, "backblaze_v4", "results", f"bb4_{tg}.json"))) for tg in FLEETS}


def f(x, p=3):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "--"
    return f"{x:.{p}f}"


def c(x): return f"{x:,.0f}"


def write(name, s): open(os.path.join(OUT, name), "w", encoding="utf-8").write(s + "\n")


# ---------------- Table: C-MAPSS ID
A = v2["ID | mean | alpha=0.02 | n=100"]; B = v2["ID | any | alpha=0.1 | n=100"]
RM = [("Point ($\\lambda=0$)", "point (lambda 0)"), ("$\\hat r-3\\sigma$", "Zhu-style rhat-3sigma"), ("$\\hat r-2.33\\sigma$", "Mitici-style rhat-2.33sigma"),
      ("Validation-tuned", "[s05] validation-tuned"), ("Model-based CC (Markov)", "[s05] model Gaussian E[N]/K <= alpha (per instance)"),
      ("Per-unit CP ($\\alpha$)", "[s05] per-unit CP (alpha)"), ("CRC, 50-point grid", "[s05] CRC on v1 grid"), ("\\textbf{CRC, exact path}", "[s05] CRC exact path")]
RA = [("Point ($\\lambda=0$)", "point (lambda 0)"), ("$\\hat r-3\\sigma$", "Zhu-style rhat-3sigma"), ("$\\hat r-2.33\\sigma$", "Mitici-style rhat-2.33sigma"),
      ("Validation-tuned", "[s05] validation-tuned"), ("Model-based CC (Poisson-binomial)", "[s05] model Gaussian Poisson-binomial P(any) <= alpha (per instance)"),
      ("Per-unit CP ($\\alpha$)", "[s05] per-unit CP (alpha)"), ("Per-unit CP ($\\alpha/K$)", "[s05] per-unit CP Bonferroni (alpha/K)"),
      ("Joint max-score CP", "[s05] joint max-score CP"), ("Scenario-max", "[s05] scenario-max (Campi-Garatti d=1)"),
      ("LTT, 50-point grid", "[s05] LTT on v1 grid"), ("\\textbf{LTT, exact path}", "[s05] LTT exact path")]
L = ["\\begin{table}[t]", "\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}",
     "\\caption{C-MAPSS, in-distribution setting (ID; $K=20$, $C=2$, $n=100$ calibration scenarios, 500 replications; $\\sigma$-scaled margins with $s_{\\min}=0.5$). Risk: realised test risk; $>\\alpha$: share of replications above the target; cost: mean plan cost.}\\label{tab:cmapss}",
     "\\begin{tabular}{@{}lrrr@{\\hspace{1.2em}}lrrr@{}}", "\\toprule",
     "\\multicolumn{4}{@{}l}{Failure fraction, $\\alpha=0.02$} & \\multicolumn{4}{l@{}}{Any failure, $\\alpha=0.1$, $\\delta=0.1$} \\\\",
     "\\cmidrule(r){1-4}\\cmidrule(l){5-8}", "Rule & Risk & $>\\alpha$ & Cost & Rule & Risk & $>\\alpha$ & Cost \\\\", "\\midrule"]
for i in range(max(len(RM), len(RA))):
    l = RM[i] if i < len(RM) else None; r = RA[i] if i < len(RA) else None
    ls = f"{l[0]} & {f(A[l[1]]['risk'], 4)} & {f(A[l[1]]['above'], 2)} & {c(A[l[1]]['cost'])}" if l else " & & & "
    rs = f"{r[0]} & {f(B[r[1]]['risk'])} & {f(B[r[1]]['above'], 2)} & {c(B[r[1]]['cost'])}" if r else " & & & "
    L.append(ls + " & " + rs + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("tab_cmapss.tex", "\n".join(L))

# ---------------- Table: shift
L = ["\\begin{table}[t]", "\\centering\\footnotesize",
     "\\caption{C-MAPSS under a new fault mode (any failure, $\\alpha=0.1$, $\\delta=0.1$; $\\sigma$-scaled margins with $s_{\\min}=0.5$). Realised risk / mean plan cost. Fixed rules do not recalibrate.}\\label{tab:shift}",
     "\\begin{tabular}{@{}llccc@{}}", "\\toprule",
     "Setting & Rule & Source $\\to$ source & Naive source $\\to$ target & Recalibrated ($n=50$ target) \\\\", "\\midrule"]
for s in ("S13", "S24"):
    for lab, key in (("$\\hat r-3\\sigma$", "Zhu-style rhat-3sigma"), ("Model-based CC", "[s05] model Gaussian Poisson-binomial P(any) <= alpha (per instance)"),
                     ("Validation-tuned", "[s05] validation-tuned"), ("Joint max-score CP", "[s05] joint max-score CP"), ("\\textbf{LTT, exact path}", "[s05] LTT exact path")):
        a = v2[f"{s} source->source | any | alpha=0.1 | n=100"][key]; b = v2[f"{s} NAIVE source->target (shift) | any | alpha=0.1 | n=100"][key]
        d = v2[f"{s} RECALIBRATED on target | any | alpha=0.1 | n=50"][key]
        L.append(f"{s} & {lab} & {f(a['risk'])} / {c(a['cost'])} & {f(b['risk'])} / {c(b['cost'])} & {f(d['risk'])} / {c(d['cost'])} \\\\")
    if s == "S13": L.append("\\addlinespace")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("tab_shift.tex", "\n".join(L))

# ---------------- Table: predictors and floors
L = ["\\begin{table}[t]", "\\centering\\small",
     "\\caption{Backblaze fleets: the two predictors, their evaluation AUC per horizon and the floor (miss fraction of the most conservative in-domain plan) for $C=1,2,4$ replacements per day. Feature set chosen on training data only (Section~\\ref{sec:design:bb}).}\\label{tab:floors}",
     "\\begin{tabular}{@{}llcccc@{}}", "\\toprule", "Fleet & Predictor & Features & AUC $h=7/14/21/28$ & Floor $C=1/2/4$ & Eval.\\ failures \\\\", "\\midrule"]
for tg in FLEETS:
    i3 = json.load(open(os.path.join(ROOT, "data", "backblaze", f"p4bb3b_{tg}_info.json"))); i4 = b4[tg]["info"]
    fl3 = " / ".join(f(b3c[tg][f"C={k} | floor"], 2) for k in (1, 2, 4)); fl4 = " / ".join(f(b4[tg]["v3c"][f"C={k} | floor"], 2) for k in (1, 2, 4))
    L.append(f"{SHORT[tg]} & baseline (v3b) & A & {' / '.join(f(v, 2) for v in i3['auc_eval'].values())} & {fl3} & {i3['eval_failures']} \\\\")
    L.append(f" & stronger (v4) & {i4['feature_set']} & {' / '.join(f(v, 2) for v in i4['auc_eval'].values())} & {fl4} & \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("tab_floors.tex", "\n".join(L))

# ---------------- Table: Backblaze offline (v4, C=2)
L = ["\\begin{table}[t]", "\\centering\\scriptsize",
     "\\caption{Backblaze, offline certification with the stronger predictor ($C=2$, 100 random-day splits, $n=60$). Miss: mean test miss fraction; abst.: share of splits in which the rule abstains; cert.\\,$>\\alpha$: share of \\emph{certified} splits whose test miss fraction exceeds $\\alpha$; cost: mean plan cost per day.}\\label{tab:bboffline}",
     "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}llrrrrrrrrrrr@{}}", "\\toprule",
     " & & \\multicolumn{4}{c}{CRC (ours)} & \\multicolumn{3}{c}{LTT, $\\delta=0.1$ (ours)} & \\multicolumn{2}{c}{Validation-tuned} & \\multicolumn{2}{c@{}}{Model-based} \\\\",
     "\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}\\cmidrule(l){12-13}",
     "Fleet & $\\alpha$ & Miss & Abst. & Cert.\\,$>\\alpha$ & Cost & Abst. & Cert.\\,$>\\alpha$ & Cost & Miss & $>\\alpha$ & Miss & $>\\alpha$ \\\\", "\\midrule"]
for j, tg in enumerate(FLEETS):
    for a in (0.3, 0.4, 0.5):
        r = b4[tg]["v3c"][f"C=2 | random | alpha={a}"]; v = b4[tg]["v3b"][f"{tg} | C=2 | random split | alpha={a}"]
        cr, lt, vt, mb = r[CRC], r[LTT], v["validation-tuned"], v[MB3]
        L.append(f"{SHORT[tg] if a == 0.3 else ''} & {a} & {f(cr['miss'])} & {f(cr['abstain'], 2)} & {f(cr['cert_above'], 2)} & {c(cr['cost'])} & "
                 f"{f(lt['abstain'], 2)} & {f(lt['cert_above'], 2)} & {c(lt['cost'])} & {f(vt['miss'])} & {f(vt['above'], 2)} & {f(mb['miss'])} & {f(mb['above'], 2)} \\\\")
    if j < 3: L.append("\\addlinespace")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("tab_bboffline.tex", "\n".join(L))

# ---------------- Table: exactness and price
L = ["\\begin{table}[t]", "\\centering\\small\\setlength{\\tabcolsep}{4pt}",
     "\\caption{Exact $\\lambda$-path on real instances and price of the guarantee (stronger predictor, $C=2$). Exact: agreement with direct solves at random margins; premium: cost of the CRC plan over the cost-optimal fixed plan in hindsight, for three failure costs (in units of an in-house replacement).}\\label{tab:exact}",
     "\\begin{tabular}{@{}lcrrrrrrrr@{}}", "\\toprule",
     " & & & & \\multicolumn{3}{c}{Premium, $\\alpha=0.4$} & \\multicolumn{3}{c@{}}{Premium, $\\alpha=0.5$} \\\\",
     "\\cmidrule(lr){5-7}\\cmidrule(l){8-10}",
     "Fleet & Exact & Breakp./day & Path (s) & $c_{\\mathrm{fail}}=10$ & 30 & 100 & 10 & 30 & 100 \\\\", "\\midrule"]
for tg in FLEETS:
    e = b4[tg]["exactness_timing"]; cs = b4[tg]["cost_sensitivity"]
    p = lambda a: " & ".join(f"{100 * cs[f'c_fail={k}|alpha={a}']['CRC']['premium']:+.0f}\\%" for k in (10, 30, 100))
    L.append(f"{SHORT[tg]} & {e['exact_matches']}/{e['checks']} & {e['breakpoints_mean']:.0f} & {e['path_seconds_mean']:.2f} & {p(0.4)} & {p(0.5)} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("tab_exact.tex", "\n".join(L))

# ---------------- Table: online C-MAPSS
on = pickle.load(gzip.open(os.path.join(ROOT, "prototype_v2", "results", "v2_online.pkl.gz")))
ca = pickle.load(gzip.open(os.path.join(ROOT, "prototype_v2", "results", "v2_online_ca.pkl.gz")))
K = on["cfg"]["K"]
RULES = [("$\\hat r-3\\sigma$", on["runs"], "zhu"), ("$\\hat r-2.33\\sigma$", on["runs"], "mitici"), ("Static CRC", on["runs"], "static_CRC"),
         ("Adaptive (cold, $\\eta=40$)", on["runs"], "adaptive_s05_cold_eta40_a{a}"), ("\\textbf{Cost-aware} ($\\eta=40$)", ca["runs"], "costaware_s05_eta40_a{a}")]
L = ["\\begin{table}[t]", "\\centering\\small",
     "\\caption{C-MAPSS online ($K=20$, $T=1{,}000$ periods, 5 seeds; in the shift scenario the fleet is renewed from the target pool after period 300; $\\sigma$-scaled margins with $s_{\\min}=0.5$). Failures per engine-period / mean cost per period.}\\label{tab:online1}",
     "\\begin{tabular}{@{}llcccc@{}}", "\\toprule", "$\\alpha$ & Rule & S13 stationary & S13 shift & S24 stationary & S24 shift \\\\", "\\midrule"]
for a in (0.0025, 0.005):
    for k, (lab, runs, rule) in enumerate(RULES):
        cells = []
        for s in ("S13", "S24"):
            for mode in ("stationary", "shift"):
                F = np.array([runs[(s, mode, rule.format(a=a), q)][0] for q in range(5)], float)
                Cs = np.array([runs[(s, mode, rule.format(a=a), q)][2] for q in range(5)], float)
                cells.append(f"{F.sum(1).mean() / (K * F.shape[1]):.4f} / {Cs.mean():.0f}")
        L.append(f"{a if k == 0 else ''} & {lab} & " + " & ".join(cells) + " \\\\")
    if a == 0.0025: L.append("\\addlinespace")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("tab_online1.tex", "\n".join(L))

# ---------------- Table: online Backblaze
L = ["\\begin{table}[t]", "\\centering\\scriptsize",
     "\\caption{Backblaze online (stronger predictor, $C=2$, days 60--227). Missed failures per day / total cost. $\\alpha_d$: daily target; floor: missed failures per day of the most conservative in-domain margin; (a): static rule abstains and deploys the most conservative plan; values above $\\alpha_d$ in italics.}\\label{tab:online2}",
     "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}llcccccccc@{}}", "\\toprule",
     "Fleet & $\\alpha$ & $\\alpha_d$ & Floor & Static CRC & Static LTT & Adaptive cold & Adaptive warm & Cost-aware \\\\", "\\midrule"]
for j, tg in enumerate(FLEETS):
    o = b4[tg]["v3b"]["online"]; ol = b4[tg]["v3c"]["online"]; fr = b4[tg]["info"]["train_fail_per_day"]
    fl = [v for k, v in o.items() if k.startswith("fixed|")][-1]["missed_per_day"]
    for a in (0.3, 0.4, 0.5):
        ad = a * fr
        def cell(s, ab=False):
            m = f"{s['missed_per_day']:.3f}"; m = f"\\emph{{{m}}}" if s["missed_per_day"] > ad else m
            return f"{m} / {c(s['cost'])}" + (" (a)" if ab else "")
        g = lambda n: o[f"{a}|{n}"]
        sc = g("static CRC (first 60 days; abstains -> most conservative)")
        L.append(f"{SHORT[tg] if a == 0.3 else ''} & {a} & {ad:.3f} & {fl:.3f} & {cell(sc, sc['mean_tau'] < 0.00101)} & {cell(ol[str(a)], ol[str(a)]['abstain'])} & "
                 f"{cell(g('adaptive cold (lambda_0 = 0)'))} & {cell(g('adaptive warm (lambda_0 = static CRC)'))} & {cell(g('cost-aware (max(lambda_t, lambda_cost))'))} \\\\")
    if j < 3: L.append("\\addlinespace")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("tab_online2.tex", "\n".join(L))

# ---------------- Complete results: all capacities, both predictors (CRC and LTT, random splits)
def full(pred):
    rows = []
    for tg in FLEETS:
        for C in (1, 2, 4):
            for a in (0.3, 0.4, 0.5):
                if pred == "v4": r = b4[tg]["v3c"][f"C={C} | random | alpha={a}"]; fl = b4[tg]["v3c"][f"C={C} | floor"]
                else: r = b3c[tg][f"C={C} | random | alpha={a}"]; fl = b3c[tg][f"C={C} | floor"]
                cr, lt = r[CRC], r[LTT]
                rows.append(f"{SHORT[tg] if (C == 1 and a == 0.3) else ''} & {C if a == 0.3 else ''} & {f(fl, 2) if a == 0.3 else ''} & {a} & "
                            f"{f(cr['miss'])} & {f(cr['above'], 2)} & {f(cr['abstain'], 2)} & {f(cr['cert_above'], 2)} & {c(cr['cost'])} & "
                            f"{f(lt['miss'])} & {f(lt['abstain'], 2)} & {f(lt['cert_above'], 2)} & {c(lt['cost'])} \\\\")
            if C != 4: rows.append("\\addlinespace[2pt]")
        rows.append("\\midrule" if tg != FLEETS[-1] else "")
    return rows
for pred, lab, nm in (("v4", "stronger predictor (v4)", "tab_full_v4.tex"), ("v3b", "baseline predictor (v3b)", "tab_full_v3b.tex")):
    L = ["\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3.5pt}",
         f"\\begin{{longtable}}{{@{{}}lccc rrrrr rrrr@{{}}}}",
         f"\\caption{{Complete offline results for the Backblaze fleets, {lab}: all capacities and targets (100 random-day splits, $n=60$). Columns as in Table~\\ref{{tab:bboffline}}; $>\\alpha$: share of all splits above $\\alpha$.}}\\label{{tab:full{pred}}}\\\\",
         "\\toprule", "Fleet & $C$ & Floor & $\\alpha$ & \\multicolumn{5}{c}{CRC} & \\multicolumn{4}{c@{}}{LTT ($\\delta=0.1$)} \\\\",
         "\\cmidrule(lr){5-9}\\cmidrule(l){10-13}", " & & & & Miss & $>\\alpha$ & Abst. & Cert.\\,$>\\alpha$ & Cost & Miss & Abst. & Cert.\\,$>\\alpha$ & Cost \\\\", "\\midrule", "\\endfirsthead",
         "\\toprule", "Fleet & $C$ & Floor & $\\alpha$ & Miss & $>\\alpha$ & Abst. & Cert.\\,$>\\alpha$ & Cost & Miss & Abst. & Cert.\\,$>\\alpha$ & Cost \\\\", "\\midrule", "\\endhead",
         "\\bottomrule", "\\endlastfoot"] + [r for r in full(pred) if r] + ["\\end{longtable}", "\\endgroup"]
    write(nm, "\n".join(L))
print("tables written:", sorted(os.listdir(OUT)))

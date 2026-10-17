# -*- coding: utf-8 -*-
"""
Paper 4, task 3: price of the guarantee as a function of the failure-cost ratio (C-MAPSS, rich planner, RICH_ID records).
The paths store n_fail and the realised cost at c_fail = 1000, so the cost at any other failure cost is exact:
cost(c) = cost(1000) + (c - 1000) * n_fail. Ratios r = c_fail / c_out (c_out = 300, the cost of an external
preventive shop visit), including the ratios used in the literature: 5 (de Pater et al. 2022: 50,000 vs 10,000;
Mitici et al. 2023: 50 vs 10) and 15 (facility accounting in the uncertainty-aware scheduling study: 48,000 vs 3,200).
For each ratio and random split (n = 100 calibration, rest test): CRC exact (mean failure fraction <= 0.02) and LTT
exact (P(any) <= 0.1, delta 0.1), each with the cheapest-valid rule at that cost; compared with the cost-optimal
fixed margin in hindsight on the test set WITHOUT any risk constraint (same family). Premium = cost / hindsight - 1.
Also reported: the risk of the hindsight cost-optimal plan (what the cost function alone would choose).
Output: results/summary_costsens.txt / .json
"""
import os, sys, json, pickle, gzip
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import p4v5_lib as L5  # noqa: E402
L2 = L5.L2
OUT = os.path.join(HERE, "results")
RATIOS = [1, 2, 3.33, 5, 10, 15, 30, 50]
C_OUT, C_BASE, K = 300.0, 1000.0, 20
R, N = 100, 100
HGRID = {"add": np.arange(0, 150.01, 0.5), "s05": np.arange(0, 40.01, 0.1)}


def recost(paths, c):
    return [(b, np.column_stack([V[:, 0], V[:, 1], V[:, 2] + (c - C_BASE) * V[:, 0]] + [V[:, j] for j in range(3, V.shape[1])]), n)
            for b, V, n in paths]


def main():
    with gzip.open(os.path.join(OUT, "v5_RICH_ID.pkl.gz"), "rb") as fh: S = pickle.load(fh)["scen"]
    rng = np.random.default_rng(14); lines, res = [], {}
    splits = [rng.permutation(len(S)) for _ in range(R)]
    for fam in ("add", "s05"):
        base = [s[fam] for s in S]
        for ratio in RATIOS:
            c = ratio * C_OUT; P = recost(base, c)
            acc = {k: [] for k in ("crc_prem", "ltt_prem", "crc_risk", "ltt_any", "h_mean", "h_any", "crc_abst", "ltt_abst")}
            for p in splits:
                cal = [P[i] for i in p[:N]]; te = [P[i] for i in p[N:]]
                cand = HGRID[fam] + L2.EPS
                Cm = L2.loss_matrix(te, 2, cand, envelope=False).mean(0); j = int(np.argmin(Cm)); hcost = Cm[j]
                acc["h_mean"].append(L2.loss_matrix(te, 0, cand[j:j + 1], K, envelope=False).mean())
                acc["h_any"].append(L2.loss_matrix(te, 1, cand[j:j + 1], envelope=False).mean())
                for name, lam in (("crc", L2.crc_path(cal, 0.02, K)), ("ltt", L2.ltt_path(cal, 0.1, 0.1))):
                    lam = L2.cheapest_valid_path(cal, lam)
                    v = np.array([p_[1][-1] if not np.isfinite(lam) else L2.at(p_, lam) for p_ in te])
                    acc[f"{name}_prem"].append(v[:, 2].mean() / hcost - 1)
                    acc[f"{name}_abst"].append(0.0 if np.isfinite(lam) else 1.0)
                    if name == "crc": acc["crc_risk"].append((v[:, 0] / K).mean())
                    else: acc["ltt_any"].append(v[:, 1].mean())
            r_ = {k: float(np.mean(v)) for k, v in acc.items()}
            res[f"{fam}|{ratio}"] = r_
            lines.append(f"[{fam}] c_fail/c_out = {ratio:5.2f} (c_fail {c:6.0f}): premium CRC {100 * r_['crc_prem']:+6.1f}%  "
                         f"LTT {100 * r_['ltt_prem']:+6.1f}% | hindsight cost-optimal plan: failure fraction {r_['h_mean']:.4f}, "
                         f"P(any) {r_['h_any']:.3f} | CRC risk {r_['crc_risk']:.4f}, LTT P(any) {r_['ltt_any']:.3f}")
            print(lines[-1], flush=True)
    open(os.path.join(OUT, "summary_costsens.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(OUT, "summary_costsens.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

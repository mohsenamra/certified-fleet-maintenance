# -*- coding: utf-8 -*-
"""
Paper 4 prototype v5: a RICHER fleet maintenance planner for C-MAPSS, used to show that the calibration layer does not
depend on the simple v1/v2 planner. Same interface as p4v1_lib.FleetMILP (K, P, L, C, solve(rhat, dl), evaluate(a, r)),
so the exact lambda-path machinery of p4v2_lib (PathSolver, CRC/LTT on the path, all baselines) is reused unchanged.

Planning horizon: P periods of L cycles. Decisions per engine k: start an in-house shop visit in period p (allowed only
if (p-1) L <= d_k, the deadline), send it to an external shop now (always allowed), or defer beyond the horizon
(allowed only if d_k >= P L). Compared with v1/v2 the planner now has
  (1) multi-period shop visits: an in-house visit occupies one of C shop lines for D consecutive periods;
  (2) a pool of S0 serviceable spare engines: every removed engine (in-house or external) is replaced by a spare;
      an engine returns to the pool after D periods (in-house) or D_out periods (external); spares missing in a
      period are leased at c_lease per engine-period;
  (3) a fixed cost c_setup for every period in which at least one in-house visit starts (opening a maintenance
      window), so visits are grouped opportunistically;
  (4) as before: lost useful life (rhat - start) at c_waste per cycle, external visit c_out, failure c_fail.
The margin lambda still enters ONLY through the deadlines (allowed periods and deferral); no cost depends on lambda,
so Assumption 2 of the paper holds and the plan is piecewise constant in lambda with the same breakpoints as in v2.

Assignment codes (as v1): a_k = p in 1..P (in-house start), 0 = external now, -1 = defer.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "prototype_v1"))
sys.path.insert(0, os.path.join(HERE, "..", "prototype_v2"))
import p4v1_lib as V1  # noqa: E402
import p4v2_lib as L2  # noqa: E402


class RichFleetMILP:
    def __init__(self, K, P=6, L=10, C=2, D=2, D_out=3, S0=3, c_waste=1.0, c_out=300.0, c_fail=1000.0,
                 c_setup=100.0, c_lease=40.0):
        self.K, self.P, self.L, self.C, self.D, self.D_out, self.S0 = K, P, L, C, D, D_out, S0
        self.c_waste, self.c_out, self.c_fail, self.c_setup, self.c_lease = c_waste, c_out, c_fail, c_setup, c_lease

    # -------------------------------------------------------------- plan-level quantities used by both solve and evaluate
    def _plan_costs(self, a):
        """setup and lease costs of an assignment vector (deterministic, independent of the true lives)."""
        P, D, Do = self.P, self.D, self.D_out
        starts = [int(np.sum(a == p)) for p in range(1, P + 1)]
        setup = self.c_setup * sum(1 for s in starts if s > 0)
        n_ext = int(np.sum(a == 0))
        lease = 0.0
        for p in range(1, P + 1):
            in_shop = sum(starts[q - 1] for q in range(max(1, p - D + 1), p + 1))
            out = in_shop + (n_ext if p <= Do else 0)
            lease += self.c_lease * max(0, out - self.S0)
        return setup, lease

    def solve(self, rhat, dl):
        import highspy
        K, P, L, C, D, Do = len(rhat), self.P, self.L, self.C, self.D, self.D_out
        nx = K * (P + 2)
        ix = lambda i, p: i * (P + 2) + (p - 1); io = lambda i: i * (P + 2) + P; iu = lambda i: i * (P + 2) + P + 1
        iy = lambda p: nx + (p - 1); iz = lambda p: nx + P + (p - 1)
        nv = nx + 2 * P
        c = np.zeros(nv); ub = np.ones(nv); lbv = np.zeros(nv)
        for i in range(K):
            for p in range(1, P + 1):
                s = (p - 1) * L
                c[ix(i, p)] = self.c_waste * max(rhat[i] - s, 0.0)
                if s > dl[i] + 1e-9: ub[ix(i, p)] = 0
            c[io(i)] = self.c_out + self.c_waste * rhat[i]
            if dl[i] < P * L: ub[iu(i)] = 0
        for p in range(1, P + 1):
            c[iy(p)] = self.c_setup
            c[iz(p)] = self.c_lease; ub[iz(p)] = K
        rows, vals, lo, up = [], [], [], []
        def add(idx, v, l, u): rows.append(idx); vals.append(v); lo.append(l); up.append(u)
        for i in range(K):                                          # one decision per engine
            add([ix(i, p) for p in range(1, P + 1)] + [io(i), iu(i)], [1.0] * (P + 2), 1, 1)
        for p in range(1, P + 1):                                   # shop lines: visits started in the last D periods
            idx = [ix(i, q) for i in range(K) for q in range(max(1, p - D + 1), p + 1)]
            add(idx, [1.0] * len(idx), -np.inf, C)
        for p in range(1, P + 1):                                   # starts only in an opened window
            idx = [ix(i, p) for i in range(K)] + [iy(p)]
            add(idx, [1.0] * K + [-float(C)], -np.inf, 0)
        for p in range(1, P + 1):                                   # spare engines: removed and not yet back <= S0 + leased
            idx = [ix(i, q) for i in range(K) for q in range(max(1, p - D + 1), p + 1)]
            if p <= Do: idx += [io(i) for i in range(K)]
            idx += [iz(p)]
            add(idx, [1.0] * (len(idx) - 1) + [-1.0], -np.inf, float(self.S0))
        h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
        h.setOptionValue("mip_rel_gap", 0.0)
        lp = highspy.HighsLp(); lp.num_col_, lp.num_row_ = nv, len(rows)
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, lbv, ub
        lp.row_lower_, lp.row_upper_ = np.array(lo, float), np.array(up, float)
        start, index, value = [0], [], []
        for rw, vv in zip(rows, vals): index += rw; value += vv; start.append(len(index))
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = start, index, value
        lp.integrality_ = [highspy.HighsVarType.kInteger] * nv
        h.passModel(lp); h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal: raise RuntimeError("rich MILP not optimal")
        x = np.round(np.array(h.getSolution().col_value)).astype(int)
        a = np.full(K, -1)
        for i in range(K):
            if x[io(i)]: a[i] = 0
            for p in range(1, P + 1):
                if x[ix(i, p)]: a[i] = p
        return a

    def evaluate(self, a, r):
        """(n_fail, any_fail, cost) with true remaining lives r. A unit fails if its life ends before its removal."""
        P, L = self.P, self.L
        start = np.where(a > 0, (a - 1) * L, np.where(a == 0, 0, P * L)).astype(float)
        fail = r < start
        waste = float(np.sum(np.where((a >= 0) & ~fail, r - start, 0)))
        setup, lease = self._plan_costs(a)
        cost = self.c_waste * waste + self.c_out * np.sum(a == 0) + self.c_fail * np.sum(fail) + setup + lease
        return int(fail.sum()), int(fail.any()), float(cost)

    def plan_objective(self, a, rhat):
        """Planner objective of an assignment (for checks): waste at rhat + external + setup + lease."""
        P, L = self.P, self.L
        start = np.where(a > 0, (a - 1) * L, 0).astype(float)
        waste = float(np.sum(np.where(a >= 0, np.maximum(rhat - start, 0), 0)))
        setup, lease = self._plan_costs(a)
        return self.c_waste * waste + self.c_out * np.sum(a == 0) + setup + lease

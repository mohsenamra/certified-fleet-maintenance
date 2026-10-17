# -*- coding: utf-8 -*-
"""
N-CMAPSS (Arias Chao et al. 2021): flight-level aggregation of all ten sub-data sets.
Reads data/ncmapss/data_set.zip (already downloaded), extracts one .h5 at a time to a temporary file, aggregates the
1 Hz records of every (unit, flight cycle) and deletes the temporary file. DS02 is read from its extracted copy.
Per (unit, cycle) record: mean and std of the 14 measured sensors X_s, mean of the 4 scenario descriptors W
(alt, Mach, TRA, T2), flight class Fc, health-state flag hs (share of the flight in the healthy state), RUL = Y at the
first record of the flight (in flight cycles), number of 1 Hz records.
Output: data/ncmapss/cycles_<subset>.npz (unit ids are made unique: 1000 * subset index + unit).
Memory: arrays are read in blocks of 1,000,000 rows. Safe to re-run (finished subsets are skipped).
"""
import os, re, sys, time, zipfile
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "ncmapss")
ZIP = os.path.join(D, "data_set.zip")
BLOCK = 1_000_000
SUBSETS = ["DS01-005", "DS02-006", "DS03-012", "DS04", "DS05", "DS06", "DS07", "DS08a-009", "DS08c-008", "DS08d-010"]


def aggregate(h5path, sidx):
    f = h5py.File(h5path, "r")
    out = []
    for part in ("dev", "test"):
        n = f["A_" + part].shape[0]
        keys, sums, sq, wsum, cnt, first = {}, [], [], [], [], []
        acc = None
        for s in range(0, n, BLOCK):
            e = min(n, s + BLOCK)
            A = f["A_" + part][s:e]; X = f["X_s_" + part][s:e]; W = f["W_" + part][s:e]; Y = f["Y_" + part][s:e, 0]
            unit = A[:, 0].astype(np.int64); cyc = A[:, 1].astype(np.int64)
            key = unit * 10000 + cyc
            uk, inv = np.unique(key, return_inverse=True)
            m = len(uk)
            bc = lambda w: np.bincount(inv, weights=w, minlength=m)
            S = np.column_stack([bc(X[:, j]) for j in range(14)]); Q = np.column_stack([bc(X[:, j] ** 2) for j in range(14)])
            WS = np.column_stack([bc(W[:, j]) for j in range(4)]); C = np.bincount(inv, minlength=m).astype(float)
            HS = bc(A[:, 3]); FC = bc(A[:, 2])
            firsty = np.full(m, -1.0); idx_first = np.unique(inv, return_index=True)[1]; firsty[inv[idx_first]] = Y[idx_first]
            block = dict(key=uk, S=S, Q=Q, W=WS, C=C, HS=HS, FC=FC, Y=firsty)
            if acc is None: acc = block
            else:                                   # merge flights split across block boundaries
                allk = np.union1d(acc["key"], uk)
                new = {k: (np.zeros((len(allk),) + acc[k].shape[1:]) if k not in ("key", "Y") else None) for k in acc}
                new["key"] = allk; new["Y"] = np.full(len(allk), -1.0)
                for b in (acc, block):
                    pos = np.searchsorted(allk, b["key"])
                    for k in ("S", "Q", "W", "C", "HS", "FC"): new[k][pos] += b[k]
                    take = new["Y"][pos] < 0; new["Y"][pos[take]] = b["Y"][take]
                acc = new
            print(f"\r    {part}: {e:,}/{n:,} rows", end="", flush=True)
        print(flush=True)
        C = acc["C"][:, None]
        mean = acc["S"] / C; std = np.sqrt(np.maximum(acc["Q"] / C - mean ** 2, 0))
        unit = acc["key"] // 10000; cyc = acc["key"] % 10000
        out.append(dict(unit=(1000 * sidx + unit).astype(np.int32), cycle=cyc.astype(np.int16), xs_mean=mean.astype(np.float32),
                        xs_std=std.astype(np.float32), w_mean=(acc["W"] / C).astype(np.float32),
                        fc=np.rint(acc["FC"] / acc["C"]).astype(np.int8), hs=(acc["HS"] / acc["C"]).astype(np.float32),
                        rul=acc["Y"].astype(np.float32), n=acc["C"].astype(np.int32), part=np.full(len(unit), part == "test", bool)))
    f.close()
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


def main():
    zf = zipfile.ZipFile(ZIP)
    names = {re.search(r"N-CMAPSS_(.+)\.h5$", i.filename).group(1): i for i in zf.infolist() if i.filename.endswith(".h5")}
    for sidx, sub in enumerate(SUBSETS, 1):
        outf = os.path.join(D, f"cycles_{sub}.npz")
        if os.path.exists(os.path.join(D, f"skipped_{sub}.txt")):
            print(f"{sub}: skipped earlier (see skipped_{sub}.txt)"); continue
        if os.path.exists(outf):
            print(f"{sub}: already done"); continue
        t0 = time.time()
        local = os.path.join(D, f"N-CMAPSS_{sub}.h5")
        tmp = None
        if not os.path.exists(local):
            tmp = os.path.join(D, "_tmp_extract.h5")
            print(f"{sub}: extracting from the archive ...", flush=True)
            with zf.open(names[sub]) as src, open(tmp, "wb") as dst:
                while True:
                    b = src.read(16 << 20)
                    if not b: break
                    dst.write(b)
            local = tmp
        print(f"{sub}: aggregating flights", flush=True)
        try:
            r = aggregate(local, sidx)
        except OSError as e:                       # e.g. a file that is truncated inside the public archive
            print(f"{sub}: SKIPPED - the file cannot be opened ({e})", flush=True)
            open(os.path.join(D, f"skipped_{sub}.txt"), "w").write(str(e) + "\n")
            if tmp: os.remove(tmp)
            continue
        np.savez_compressed(outf + ".tmp.npz", **r)
        os.replace(outf + ".tmp.npz", outf)
        if tmp: os.remove(tmp)
        u = np.unique(r["unit"])
        print(f"{sub}: {len(u)} units, {len(r['unit'])} flights ({time.time() - t0:.0f}s)", flush=True)
    print("DONE: flight-level data for all sub-data sets.")


if __name__ == "__main__":
    main()

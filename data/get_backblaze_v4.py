# -*- coding: utf-8 -*-
"""
Paper 4, Backblaze v4 data: longer training history + richer SMART attributes for the SAME 4 drive models.

  python get_backblaze_v4.py            download Q1_2024 .. Q2_2025 (6 quarters, about 7-9 GB, resumable) and build
                                        per-model panels for all 10 quarters Q1_2024 .. Q2_2026 (existing zips are reused)

Output per quarter and model: backblaze\\panel4_<Q>_<model>.pkl.gz with columns
  day (int16, days since 2024-01-01), sid (int32 serial id, stable across quarters), failure (int8),
  smart_<k>_raw (float32) and smart_<k>_normalized (float16) for every attribute in RAW that the model reports.
Serial ids: backblaze\\serials4_<model>.json. Nothing from v1-v3 (panel_<Q>.pkl.gz, p4bb*_*.npz) is changed.
Re-runnable: finished downloads and quarter panels are skipped.
"""
import os, sys, json, time, zipfile
import numpy as np, pandas as pd
import get_backblaze as G1                     # reuse the resumable downloader (same folder)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = G1.OUT
QUARTERS = ["Q1_2024", "Q2_2024", "Q3_2024", "Q4_2024", "Q1_2025", "Q2_2025", "Q3_2025", "Q4_2025", "Q1_2026", "Q2_2026"]
MODELS = ["ST16000NM001G", "TOSHIBA MG07ACA14TA", "TOSHIBA MG08ACA16TA", "WDC WUH722222ALE6L4"]
RAW = [1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 18, 22, 23, 24, 71, 90, 187, 188, 190, 191, 192, 193, 194, 196, 197, 198, 199,
       200, 220, 222, 223, 224, 226, 240, 241, 242]
DAY0 = pd.Timestamp("2024-01-01")


def log(*a): print(*a, flush=True)


def tag(m): return m.replace(" ", "_")


def out_path(q, m): return os.path.join(OUT, f"panel4_{q}_{tag(m)}.pkl.gz")


def load_ids(m):
    f = os.path.join(OUT, f"serials4_{tag(m)}.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def save_ids(m, ids):
    f = os.path.join(OUT, f"serials4_{tag(m)}.json")
    json.dump(ids, open(f + ".tmp", "w")); os.replace(f + ".tmp", f)


def process(q, zpath):
    if all(os.path.exists(out_path(q, m)) for m in MODELS): log(f"   {q}: already processed"); return True
    ids = {m: load_ids(m) for m in MODELS}
    parts = {m: [] for m in MODELS}
    with zipfile.ZipFile(zpath) as z:
        days = sorted(n for n in z.namelist() if n.endswith(".csv") and "__MACOSX" not in n)
        log(f"   {q}: {len(days)} daily files")
        t0 = time.time()
        for i, n in enumerate(days):
            with z.open(n) as f: h = pd.read_csv(f, nrows=0).columns
            want = ["date", "serial_number", "model", "failure"] + [c for k in RAW for c in (f"smart_{k}_raw", f"smart_{k}_normalized")]
            use = [c for c in want if c in h]
            with z.open(n) as f:
                d = pd.read_csv(f, usecols=use, dtype={"serial_number": str, "model": str}, low_memory=False)
            d = d[d["model"].isin(MODELS)]
            day = int((pd.Timestamp(d["date"].iloc[0]) - DAY0).days) if len(d) else 0
            for m, g in d.groupby("model"):
                idm = ids[m]
                sid = np.array([idm.setdefault(s, len(idm)) for s in g["serial_number"].values], np.int32)
                cols = {"day": np.full(len(g), day, np.int16), "sid": sid, "failure": g["failure"].values.astype(np.int8)}
                for c in use[4:]:
                    v = pd.to_numeric(g[c], errors="coerce").values
                    cols[c] = v.astype(np.float32) if c.endswith("_raw") else v.astype(np.float16)
                parts[m].append(pd.DataFrame(cols))
            if (i + 1) % 15 == 0: log(f"     {i + 1}/{len(days)} days ({time.time() - t0:.0f}s)")
    for m in MODELS:
        df = pd.concat(parts[m], ignore_index=True); parts[m] = None
        keep = [c for c in df.columns if not c.startswith("smart_") or df[c].notna().mean() > 0.5]
        df = df[keep]
        df.to_pickle(out_path(q, m) + ".tmp", compression="gzip"); os.replace(out_path(q, m) + ".tmp", out_path(q, m))
        save_ids(m, ids[m])
        log(f"   {m}: {df['sid'].nunique()} drives, {int(df['failure'].sum())} failures, {len(keep) - 3} SMART columns")
        del df
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    qs = args or QUARTERS
    for q in qs:
        log(f"== {q} ==")
        if all(os.path.exists(out_path(q, m)) for m in MODELS): log("   already processed"); continue
        zpath = os.path.join(OUT, f"data_{q}.zip")
        if not G1.download(G1.BASE + f"data_{q}.zip", zpath): return False
        if not process(q, zpath): return False
    return True


if __name__ == "__main__":
    ok = False
    try: ok = main()
    except Exception:
        import traceback; traceback.print_exc()
    log("\nDONE." if ok else "\nNot finished - see the messages above (run again to resume).")
    sys.exit(0 if ok else 1)

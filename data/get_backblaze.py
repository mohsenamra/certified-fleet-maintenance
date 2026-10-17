# -*- coding: utf-8 -*-
"""
Paper 4, real-fleet data: Backblaze Drive Stats (daily SMART records of ~300,000 hard drives in a real data centre).
Source: https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data  (free; cite Backblaze; do not resell the data)

  python get_backblaze.py --probe          check that the server is reachable from this network (no download)
  python get_backblaze.py                  download + process the last 4 quarters (2025 Q3 .. 2026 Q2, about 5.7 GB of zips)
  python get_backblaze.py Q2_2026 Q1_2026  only these quarters

Processing is streamed day by day straight from the zip (nothing is extracted, low RAM): only the most common drive
models and a few SMART attributes are kept. Output per quarter: backblaze\\panel_<Q>.pkl.gz (pandas DataFrame) and
backblaze\\summary_<Q>.json. Re-runnable: finished downloads and quarters are skipped. Add --delete-zips to remove
each zip after it has been processed.
"""
import os, sys, json, time, zipfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "backblaze"); os.makedirs(OUT, exist_ok=True)
BASE = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/"
DEFAULT_Q = ["Q3_2025", "Q4_2025", "Q1_2026", "Q2_2026"]
SMART = ["smart_5_raw", "smart_9_raw", "smart_12_raw", "smart_187_raw", "smart_188_raw", "smart_193_raw", "smart_194_raw",
         "smart_197_raw", "smart_198_raw", "smart_199_raw", "smart_241_raw", "smart_242_raw"]
BASECOLS = ["date", "serial_number", "model", "capacity_bytes", "failure"]
N_MODELS = 4          # keep the N most common models (largest homogeneous fleets)


def log(*a): print(*a, flush=True)


def head_size(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception as e:
        log("   not reachable:", repr(e)[:120]); return None


def download(url, dest):
    size = head_size(url)
    if size is None: return False
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have == size: log(f"   already downloaded ({size / 1e9:.2f} GB)"); return True
    log(f"   downloading {size / 1e9:.2f} GB (resuming at {have / 1e9:.2f} GB) ...")
    for attempt in range(10):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have >= size: break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab" if have else "wb") as f:
                if have and r.status != 206: f.seek(0); f.truncate()
                done, last = have, time.time()
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); done += len(b)
                    if time.time() - last > 15: log(f"     {done / 1e9:.2f} / {size / 1e9:.2f} GB"); last = time.time()
        except Exception as e:
            log(f"     interrupted ({repr(e)[:80]}); retry in {2 ** min(attempt, 6)} s"); time.sleep(2 ** min(attempt, 6))
    ok = os.path.getsize(dest) == size
    log("   download complete" if ok else "   download incomplete - run again to resume")
    return ok


def process(q, zpath):
    import pandas as pd, numpy as np
    out = os.path.join(OUT, f"panel_{q}.pkl.gz")
    if os.path.exists(out): log(f"   {q}: already processed"); return True
    with zipfile.ZipFile(zpath) as z:
        days = sorted(n for n in z.namelist() if n.endswith(".csv") and "__MACOSX" not in n)
        log(f"   {q}: {len(days)} daily files")
        with z.open(days[0]) as f:
            hdr = pd.read_csv(f, nrows=0).columns.tolist()
        with z.open(days[0]) as f:
            first = pd.read_csv(f, usecols=["model"])
        models = first["model"].value_counts().index[:N_MODELS].tolist()
        cols = BASECOLS + [c for c in SMART if c in hdr]
        log(f"   models kept: {models}; SMART columns: {[c for c in SMART if c in hdr]}")
        parts = []
        t0 = time.time()
        for i, n in enumerate(days):
            with z.open(n) as f:
                h = pd.read_csv(f, nrows=0).columns
            use = [c for c in cols if c in h]
            with z.open(n) as f:
                d = pd.read_csv(f, usecols=use, dtype={"serial_number": str, "model": str})
            d = d[d["model"].isin(models)]
            for c in use:
                if c.startswith("smart_"): d[c] = d[c].astype("float32")
            d["failure"] = d["failure"].astype("int8")
            parts.append(d)
            if (i + 1) % 15 == 0: log(f"     {i + 1}/{len(days)} days ({time.time() - t0:.0f}s)")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]); df["model"] = df["model"].astype("category"); df["serial_number"] = df["serial_number"].astype("category")
    df.to_pickle(out, compression="gzip")
    summ = {}
    for m, g in df.groupby("model", observed=True):
        summ[m] = dict(drives=int(g["serial_number"].nunique()), drive_days=int(len(g)), failures=int(g["failure"].sum()),
                       capacity_TB=round(float(g["capacity_bytes"].median()) / 1e12, 1),
                       smart_nonmissing={c: round(float(g[c].notna().mean()), 3) for c in g.columns if c.startswith("smart_")})
    json.dump(dict(quarter=q, days=len(days), models=summ), open(os.path.join(OUT, f"summary_{q}.json"), "w"), indent=1)
    for m, s in summ.items():
        log(f"   {m}: {s['drives']} drives, {s['failures']} failures, {s['capacity_TB']} TB")
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--probe" in sys.argv:
        url = BASE + "Drive_Stats_Schema_Current.csv"
        log("Probing", url)
        s = head_size(url)
        log("REACHABLE" if s is not None else "NOT REACHABLE from this network (try a VPN, or download the zips by hand into data\\backblaze)")
        s = head_size(BASE + "data_Q2_2026.zip")
        if s: log(f"data_Q2_2026.zip size: {s / 1e9:.2f} GB")
        return s is not None
    qs = args or DEFAULT_Q
    for q in qs:
        log(f"== {q} ==")
        zpath = os.path.join(OUT, f"data_{q}.zip")
        if not os.path.exists(os.path.join(OUT, f"panel_{q}.pkl.gz")):
            if not download(BASE + f"data_{q}.zip", zpath): return False
            if not process(q, zpath): return False
        else:
            log("   already processed")
        if "--delete-zips" in sys.argv and os.path.exists(zpath): os.remove(zpath); log("   zip deleted")
    return True


if __name__ == "__main__":
    ok = False
    try: ok = main()
    except Exception as e: log("ERROR:", repr(e))
    log("\nDONE." if ok else "\nNot finished - see the messages above.")
    if "--nopause" not in sys.argv: input("Press Enter to close.")

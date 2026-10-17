# -*- coding: utf-8 -*-
"""
Download NASA C-MAPSS (and optionally N-CMAPSS) into data/CMAPSS and data/NCMAPSS.
Standard library only. Re-runnable: finished downloads are kept.
  python get_data.py            -> C-MAPSS only (~12 MB)
  python get_data.py --ncmapss  -> also N-CMAPSS (several GB; needs ~15 GB free disk)
"""
import os, sys, zipfile, shutil, urllib.request, hashlib, io

HERE = os.path.dirname(os.path.abspath(__file__))
CMAPSS_URLS = [
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
    "https://zenodo.org/records/15346912/files/CMAPSSData.zip?download=1",   # CC-BY-4.0 mirror
]
NCMAPSS_URLS = ["https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip"]
NEED = [f"{k}_FD00{i}.txt" for k in ("train", "test", "RUL") for i in range(1, 5)]


def download(urls, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print("  already downloaded:", os.path.basename(dest)); return True
    for u in urls:
        try:
            print("  downloading", u.split("/")[-1][:70], flush=True)
            tmp = dest + ".part"
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=120) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0); got = 0
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); got += len(b)
                    if total: print(f"\r  {got / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            print(); os.replace(tmp, dest); return True
        except Exception as e:
            print("\n  failed:", e)
    return False


def extract_all(zpath, outdir):
    """Extract a zip, also unpacking zips nested inside it (the NASA archive wraps CMAPSSData.zip)."""
    os.makedirs(outdir, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n.endswith("/"): continue
            data = z.read(n); base = os.path.basename(n)
            if base.lower().endswith(".zip"):
                inner = os.path.join(outdir, base); open(inner, "wb").write(data); extract_all(inner, outdir); os.remove(inner)
            elif base:
                open(os.path.join(outdir, base), "wb").write(data)


def main():
    out = os.path.join(HERE, "CMAPSS")
    if all(os.path.exists(os.path.join(out, f)) for f in NEED):
        print("C-MAPSS already present.")
    else:
        print("1) C-MAPSS")
        z = os.path.join(HERE, "cmapss.zip")
        if not download(CMAPSS_URLS, z):
            print("ERROR: could not download C-MAPSS. Download it manually from\n  " + "\n  ".join(CMAPSS_URLS) +
                  "\nand put the file in this folder as cmapss.zip, then run again."); return 1
        extract_all(z, out)
        miss = [f for f in NEED if not os.path.exists(os.path.join(out, f))]
        if miss: print("ERROR: files missing after extraction:", miss); return 1
    md5 = {f: hashlib.md5(open(os.path.join(out, f), "rb").read()).hexdigest()[:10] for f in NEED}
    print("C-MAPSS OK:", ", ".join(f"{k}={v}" for k, v in md5.items() if k.startswith("train")))
    if "--ncmapss" in sys.argv:
        free = shutil.disk_usage(HERE).free / 1e9
        print(f"2) N-CMAPSS (free disk {free:.0f} GB)")
        if free < 15: print("ERROR: need at least 15 GB free disk."); return 1
        z = os.path.join(HERE, "ncmapss.zip")
        if not download(NCMAPSS_URLS, z): print("ERROR: could not download N-CMAPSS."); return 1
        extract_all(z, os.path.join(HERE, "NCMAPSS")); print("N-CMAPSS OK")
    return 0


if __name__ == "__main__":
    rc = 1
    try: rc = main()
    except Exception as e: print("ERROR:", e)
    print("DONE" if rc == 0 else "FAILED")
    if "--nopause" not in sys.argv: input("Press Enter to close.")
    sys.exit(rc)

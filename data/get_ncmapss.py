# -*- coding: utf-8 -*-
"""
Download N-CMAPSS (NASA PCoE data set 17, Arias Chao et al. 2021) and keep only the sub-data set DS02
(the one used by most RUL studies and by the PHM 2021 data challenge reference split).
Uses HTTP range requests to read the remote zip's directory and extract only the needed member, so the multi-GB
archive is not downloaded in full when the server allows ranges. Falls back to a resumable full download.
Output: data/ncmapss/N-CMAPSS_DS02-006.h5 (+ ncmapss_listing.txt with the archive contents).
Standard library only. Safe to re-run (skips work already done).
"""
import os, re, io, sys, time, shutil, zipfile, urllib.request

URL = "https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ncmapss"); os.makedirs(OUT, exist_ok=True)
TARGET = re.compile(r"DS02.*\.h5$", re.I)
LISTING = os.path.join(OUT, "ncmapss_listing.txt")


def head(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", "0")), r.headers.get("Accept-Ranges", "")


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests (with a small read-ahead cache)."""
    def __init__(self, url, size, block=4 << 20):
        self.url, self.size, self.pos, self.block = url, size, 0, block
        self.cache_start, self.cache = -1, b""

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _fetch(self, start, end):
        for attempt in range(8):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception as e:
                print(f"    retry {attempt + 1} ({e})", flush=True); time.sleep(5 * (attempt + 1))
        raise RuntimeError("range request failed repeatedly")

    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b""
        if self.cache_start <= self.pos and self.pos + n <= self.cache_start + len(self.cache):
            i = self.pos - self.cache_start; out = self.cache[i:i + n]
        elif n >= self.block:
            out = self._fetch(self.pos, self.pos + n - 1)
        else:
            self.cache_start = self.pos; self.cache = self._fetch(self.pos, min(self.size, self.pos + self.block) - 1)
            out = self.cache[:n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def extract(zf, info, dest):
    if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
        print(f"  {os.path.basename(dest)} already present ({info.file_size / 1e9:.2f} GB)", flush=True); return
    t0, done = time.time(), 0
    with zf.open(info) as src, open(dest + ".part", "wb") as dst:
        while True:
            chunk = src.read(8 << 20)
            if not chunk: break
            dst.write(chunk); done += len(chunk)
            print(f"\r  extracting {os.path.basename(dest)}: {done / 1e9:.2f} / {info.file_size / 1e9:.2f} GB "
                  f"({done / max(time.time() - t0, 1) / 1e6:.1f} MB/s)", end="", flush=True)
    print(flush=True)
    os.replace(dest + ".part", dest)


def full_download(url, size, dest):
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if size and have >= size: return
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab") as f:
        while True:
            chunk = r.read(8 << 20)
            if not chunk: break
            f.write(chunk); have += len(chunk)
            print(f"\r  downloading: {have / 1e9:.2f} / {size / 1e9:.2f} GB", end="", flush=True)
    print(flush=True)


def handle(zf, label):
    infos = zf.infolist()
    with open(LISTING, "a", encoding="utf-8") as fh:
        fh.write(f"# {label}\n" + "".join(f"{i.filename}\t{i.file_size}\t{i.compress_size}\n" for i in infos))
    for i in infos: print(f"    {i.filename}  {i.file_size / 1e9:.2f} GB", flush=True)
    hit = [i for i in infos if TARGET.search(os.path.basename(i.filename))]
    if hit:
        extract(zf, hit[0], os.path.join(OUT, os.path.basename(hit[0].filename))); return True
    for inner in [i for i in infos if i.filename.lower().endswith(".zip")]:
        local = os.path.join(OUT, os.path.basename(inner.filename))
        extract(zf, inner, local)
        with zipfile.ZipFile(local) as z2:
            if handle(z2, inner.filename): return True
    return False


def main():
    done = [f for f in os.listdir(OUT) if TARGET.search(f)]
    if done:
        print(f"Already present: {done}. Nothing to do."); return
    free = shutil.disk_usage(OUT).free
    size, ranges = head(URL)
    print(f"Archive size {size / 1e9:.2f} GB; range requests: {ranges or 'unknown'}; free disk {free / 1e9:.1f} GB", flush=True)
    open(LISTING, "w").close()
    if ranges.lower() == "bytes":
        try:
            with zipfile.ZipFile(RangeFile(URL, size)) as zf:
                if handle(zf, "remote archive"):
                    print("DONE: DS02 extracted without downloading the whole archive."); return
        except Exception as e:
            print(f"Range-based extraction failed ({e}); falling back to full download.", flush=True)
    if free < 2.2 * size:
        sys.exit(f"Not enough free disk space for a full download ({free / 1e9:.1f} GB free, need about {2.2 * size / 1e9:.1f} GB).")
    local = os.path.join(OUT, "ncmapss_full.zip")
    full_download(URL, size, local)
    with zipfile.ZipFile(local) as zf:
        if not handle(zf, "local archive"): sys.exit("DS02 not found in the archive (see ncmapss_listing.txt).")
    print("DONE: DS02 extracted.")


if __name__ == "__main__":
    main()

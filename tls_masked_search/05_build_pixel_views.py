"""Step 5: difference images (which star dimmed during the dips), an input of the shape model (step 6).

For each labeled signal (03's load(), NT rows included), from the star's SPOC 2-min target pixel files (TPFs):
  in-dip cadences   within half a dip length of a dip centre
  out-of-dip        1-2 dip lengths from the centre, either side (local baseline)
  quality-flagged cadences dropped. Per dip: difference = mean out - mean in (positive where light was
  lost), averaged over dips weighted by in-dip cadences; per-pixel noise from the out-of-dip scatter.
Sectors are combined pixel by pixel (inverse-variance) after cropping SIZE x SIZE around the target's pixel
(target position from the TPF's WCS and RA_OBJ/DEC_OBJ).
Channels (SIZE x SIZE, NaN -> 0):
  0 difference / its peak absolute value (sign kept)
  1 difference in noise units (SNR), compressed: sign * log1p(|SNR|)
  2 out-of-dip image / its max
  3 target marker: Gaussian (sigma MARKER_SIGMA px) at the target's sub-pixel position
offset_px: distance from the target to the brightest difference pixel (largest light loss); > OFF_TARGET px
counts as off-target. Nearby-star (TIC) marker map skipped.

Sectors: the TPFs of the MAX_SECTORS sectors (of the star's step-2 light-curve files) with the most good
in-dip cadences over the star's labeled signals; TPF names are the light curves' with _tp instead of _lc,
streamed from MAST through data/raw/tpf_dataset/ (git-ignored) and deleted once a star is built
(~80 MB per star; keeping them would be hundreds of GB).

Usage (from tls_masked_search/):  python 05_build_pixel_views.py [--name dataset] [--max-sectors 2] [--limit N]
Writes output/{name}_pixel_views.npz (tic, signal, ok, x (N, 4, SIZE, SIZE), offset_px, reason, meta).
--limit N: only N random stars (download-size pilot).
--max-chunks K: as given to step 2 for this dataset (dataset_v2: 4).
--reuse NAME: copy rows of output/NAME_pixel_views.npz for stars whose signals are identical in NAME and that
pick the same sectors there; only the other stars are downloaded.
  e.g. python 05_build_pixel_views.py --name dataset_v2 --max-chunks 4 --reuse dataset   # ~15 min
--name candidates: step 7's undecided-TOI signals (output/candidates_signals.csv) of stars with light curves of
their own (data/raw/candidates/) -> output/candidate_pixel_views.npz; TPFs pass through
data/raw/tpf_candidates/ (~13 min, ~380 GB streamed).
"""
import argparse
import importlib
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import lightkurve as lk
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits

from common import OUT_DIR, ROOT

m2 = importlib.import_module("02_build_dataset")
m3 = importlib.import_module("03_train_trees")
TPF_DIR = ROOT / "data" / "raw" / "tpf_dataset"
MAST = "https://mast.stsci.edu/api/v0.1/Download/file"
SIZE = 11
MIN_IN = 10            # good in-dip cadences needed (20 min of 2-min data)
MAX_SECTORS = 2
MARKER_SIGMA = 1.0
OFF_TARGET = 1.0       # px
WORKERS = 48
REUSE_CHUNKS = None    # --reuse: the MAX_CHUNKS the reused views were built with
CHANNELS = ["diff_over_peak", "diff_snr_log1p", "out_over_max", "target_marker"]


def windows(t, period, t0, dur):
    """Dip number, in-dip mask, out-of-dip mask per cadence (times in days)."""
    cyc = (t - t0) / period
    n = np.round(cyc)
    adt = np.abs(cyc - n) * period
    return n.astype(int), adt < dur / 2, (adt >= dur) & (adt <= 2 * dur)


def pick_sectors(tic, rows, k):
    """The k step-2 light-curve files of this star with the most good in-dip cadences (all if k is None),
    as TPF names. Stars whose light curves are not on disk: the k most recent SPOC 2-min TPFs MAST lists."""
    paths = m2.capped(sorted(str(p) for p in m2.RAW.glob(f"*{tic:016d}*_lc.fits")))
    if not paths:
        found = lk.search_targetpixelfile(f"TIC {tic}", author="SPOC", exptime=120)
        names = sorted({u.split("/")[-1] for u in (found.table["dataURI"] if len(found) else []) if f"{tic:016d}" in u})
        return names[::-1][:k]
    counts = []
    for p in paths:
        d = fits.getdata(p, 1)
        t, good = d["TIME"], (d["QUALITY"] == 0) & np.isfinite(d["TIME"])
        counts.append(sum((good & windows(t, r["period"], r["t0"], r["duration_h"] / 24)[1]).sum() for r in rows))
    order = sorted(range(len(paths)), key=lambda i: -counts[i])
    return [os.path.basename(paths[i]).replace("_lc.fits", "_tp.fits") for i in order[:k] if counts[i] > 0]


def fetch(name, session):
    """TPF path, downloaded from MAST if missing (3 tries). Returns (path, MB downloaded)."""
    path = TPF_DIR / name
    if path.exists():
        return path, 0.0
    for attempt in range(3):
        try:
            r = session.get(MAST, params={"uri": f"mast:TESS/product/{name}"}, timeout=600)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))
    tmp = path.with_name(f"{name}.part{os.getpid()}")
    tmp.write_bytes(r.content)
    tmp.rename(path)
    return path, len(r.content) / 1e6


def crop(img, cx, cy):
    """SIZE x SIZE window centred on pixel (column cx, row cy), NaN outside the stamp."""
    h, out = SIZE // 2, np.full((SIZE, SIZE), np.nan)
    r0, c0 = cy - h, cx - h
    rs = slice(min(max(r0, 0), img.shape[0]), min(max(r0 + SIZE, 0), img.shape[0]))
    cs = slice(min(max(c0, 0), img.shape[1]), min(max(c0 + SIZE, 0), img.shape[1]))
    if rs.stop > rs.start and cs.stop > cs.start:
        out[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0] = img[rs, cs]
    return out


def sector_images(path, rows):
    """Per signal: (difference, its variance, out-of-dip image, in-dip cadences, sub-pixel dx, dy) on the
    target-centred crop, or None if no dip has both in- and out-of-dip data."""
    tpf = lk.read(str(path), quality_bitmask="none")
    t, f = tpf.time.value, tpf.flux.value
    good = (np.asarray(tpf.quality) == 0) & np.isfinite(t) & np.isfinite(f).any((1, 2))
    t, f = t[good], f[good]
    x, y = (float(v) for v in tpf.wcs.world_to_pixel(SkyCoord(tpf.ra, tpf.dec, unit="deg")))
    cx, cy = int(round(x)), int(round(y))
    out = []
    for r in rows:
        n, inn, oot = windows(t, r["period"], r["t0"], r["duration_h"] / 24)
        d = np.zeros(f.shape[1:])
        base = np.zeros(f.shape[1:])
        ss, dof, var_factor, n_in = np.zeros(f.shape[1:]), 0, 0.0, 0
        for e in np.unique(n[inn]):
            i, o = inn & (n == e), oot & (n == e)
            ni, no = int(i.sum()), int(o.sum())
            if no < 2:
                continue
            mo = np.nanmean(f[o], 0)
            d += ni * (mo - np.nanmean(f[i], 0))
            base += ni * mo
            ss += np.nansum((f[o] - mo) ** 2, 0)
            dof += no - 1
            var_factor += ni * ni * (1 / ni + 1 / no)
            n_in += ni
        if n_in == 0:
            out.append(None)
            continue
        var = ss / dof * var_factor / n_in ** 2
        out.append((crop(d / n_in, cx, cy), crop(var, cx, cy), crop(base / n_in, cx, cy), n_in, x - cx, y - cy))
    return out


def channels(parts):
    """Combine sectors (inverse variance per pixel) -> (x (4, SIZE, SIZE), offset_px, reason)."""
    parts = [p for p in parts if p is not None]
    n_in = sum(p[3] for p in parts)
    if n_in < MIN_IN:
        return None, np.nan, f"few in-dip cadences ({n_in})"
    w = np.stack([np.where(np.isfinite(p[1]) & (p[1] > 0) & np.isfinite(p[0]), 1 / p[1], 0) for p in parts])
    wsum = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.nansum(w * np.stack([np.nan_to_num(p[0]) for p in parts]), 0) / wsum
        snr = d * np.sqrt(wsum)
    base = np.nanmean(np.stack([p[2] for p in parts]), 0)
    dx = sum(p[3] * p[4] for p in parts) / n_in
    dy = sum(p[3] * p[5] for p in parts) / n_in
    if not np.isfinite(d).any() or np.nanmax(d) <= 0:
        return None, np.nan, "no light loss in any pixel"
    h = SIZE // 2
    # Raw-difference peak: as a fake-vs-planet score it beat the SNR-map peak, SNR-gated peaks and the
    # significant-light centroid (AUC 0.67 vs 0.64-0.66 on the labeled rows).
    py, px = np.unravel_index(np.nanargmax(d), d.shape)
    offset = float(np.hypot(px - (h + dx), py - (h + dy)))
    yy, xx = np.mgrid[:SIZE, :SIZE]
    x = np.stack([d / np.nanmax(np.abs(d)), np.sign(snr) * np.log1p(np.abs(snr)), base / np.nanmax(base),
                  np.exp(-((xx - h - dx) ** 2 + (yy - h - dy) ** 2) / (2 * MARKER_SIGMA ** 2))])
    return np.nan_to_num(x).astype(np.float32), offset, ""


def build(tic, signal_rows, paths):
    """Pixel views for one star's signals (dicts with signal, period, t0, duration_h) from its TPF paths.
    Returns {signal: (x or None, offset_px, reason)}; x is None when not ok."""
    rows = list(signal_rows)
    if not paths:
        return {r["signal"]: (None, np.nan, "no TPF") for r in rows}
    per_sector = [sector_images(p, rows) for p in paths]
    return {r["signal"]: channels([s[j] for s in per_sector]) for j, r in enumerate(rows)}


def star_job(job):
    """Pick sectors, download, build; or copy `cached` (signal -> result, --reuse) if the reused run picked the
    same sectors. Returns (results, stats)."""
    warnings.filterwarnings("ignore")
    tic, rows, k, cached = job
    stats = {"tic": tic, "mb": 0.0, "seconds": 0.0, "sectors": 0, "files": [], "reused": False}
    paths = []
    try:
        names = pick_sectors(tic, rows, k)
        if cached:
            m2.MAX_CHUNKS, chunks = REUSE_CHUNKS, m2.MAX_CHUNKS
            same = pick_sectors(tic, rows, k) == names
            m2.MAX_CHUNKS = chunks
            if same:
                stats["reused"] = True
                return [(tic, s, *v) for s, v in cached.items()], stats
        start = time.time()
        with requests.Session() as session:
            for name in names:
                p, mb = fetch(name, session)
                paths.append(p)
                stats["mb"] += mb
                stats["files"].append((name, os.path.getsize(p) / 1e6))
        stats["seconds"], stats["sectors"] = time.time() - start, len(paths)
        res = build(tic, rows, paths)
    except Exception as e:  # download or unreadable file: every signal of this star fails
        res = {r["signal"]: (None, np.nan, f"error: {type(e).__name__}: {e}"[:200]) for r in rows}
    finally:
        for p in paths:  # streamed, not kept: ~80 MB per star
            p.unlink(missing_ok=True)
    return [(tic, s, *v) for s, v in res.items()], stats


def reusable(s, name, max_sectors):
    """tic -> {signal: (x or None, offset_px, reason)} from output/<name>_pixel_views.npz, for stars whose
    signal rows (signal, period, t0, duration_h) are identical in `name`'s labeled set; and its MAX_CHUNKS."""
    npz = np.load(OUT_DIR / f"{name}_pixel_views.npz")
    meta = json.loads(str(npz["meta"]))
    assert meta["max_sectors"] == max_sectors, f"{name} pixel views were built with --max-sectors {meta['max_sectors']}"
    cols = ["tic", "signal", "period", "t0", "duration_h"]
    old = m3.load(name, nt=True)[cols].sort_values(cols).reset_index(drop=True)
    new = s[cols].sort_values(cols).reset_index(drop=True)
    same = [t for t, g in new.groupby("tic") if g.reset_index(drop=True).equals(
        old[old.tic == t].reset_index(drop=True))]
    res = {}
    for i in np.flatnonzero(np.isin(npz["tic"], same)):
        res.setdefault(int(npz["tic"][i]), {})[int(npz["signal"][i])] = (
            npz["x"][i] if npz["ok"][i] else None, float(npz["offset_px"][i]), str(npz["reason"][i]))
    return res, meta.get("max_chunks", 2)  # 2: step 2's default, before max_chunks was recorded


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="dataset")
    parser.add_argument("--max-sectors", type=int, default=MAX_SECTORS, help="0 = every step-2 sector")
    parser.add_argument("--max-chunks", type=int, default=m2.MAX_CHUNKS, help="as given to step 2")
    parser.add_argument("--reuse", help="dataset name whose pixel views are copied where identical")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    start = time.time()
    global TPF_DIR, REUSE_CHUNKS
    m2.MAX_CHUNKS = args.max_chunks  # read by capped() in the (forked) workers
    if args.name == "candidates":  # 07's candidate set; like 07, drop stars with no light curve of their own
        m2.RAW, TPF_DIR = ROOT / "data" / "raw" / "candidates", ROOT / "data" / "raw" / "tpf_candidates"
        s = pd.read_csv(OUT_DIR / "candidates_signals.csv")
        s = s[m3.own_data(s.tic.values, m2.RAW)]
    else:
        s = m3.load(args.name, nt=True)
    cached = {}
    if args.reuse:
        cached, REUSE_CHUNKS = reusable(s, args.reuse, args.max_sectors)
        print(f"--reuse {args.reuse}: {len(cached)} of {s.tic.nunique()} stars have identical signals", flush=True)
    TPF_DIR.mkdir(parents=True, exist_ok=True)
    tics = s.tic.unique()
    if args.limit:
        tics = np.random.default_rng(m3.SEED).choice(tics, args.limit, replace=False)
    cols = ["signal", "period", "t0", "duration_h"]
    jobs = [(int(t), s[s.tic == t][cols].to_dict("records"), args.max_sectors or None, cached.get(int(t)))
            for t in tics]

    results, stats = [], []
    with ProcessPoolExecutor(WORKERS) as pool:
        for i, (res, st) in enumerate(pool.map(star_job, jobs, chunksize=1), 1):
            results += res
            stats.append(st)
            if i % 50 == 0 or i == len(jobs):
                mb = sum(x["mb"] for x in stats)
                print(f"{i}/{len(jobs)} stars, {mb / 1e3:.1f} GB downloaded, {time.time() - start:.0f} s", flush=True)

    files = [f for st in stats for f in st["files"]]
    sizes = np.array([f[1] for f in files]) if files else np.zeros(1)
    print(f"\nTPFs: {len(files)} files, {sizes.sum() / 1e3:.1f} GB downloaded, per sector median {np.median(sizes):.0f} MB "
          f"(p90 {np.percentile(sizes, 90):.0f}, max {sizes.max():.0f}); stars with no TPF: "
          f"{sum(st['sectors'] == 0 and not st['reused'] for st in stats)}; reused stars: {sum(st['reused'] for st in stats)}; "
          f"wall {time.time() - start:.0f} s; "
          f"download seconds per star median {np.median([st['seconds'] for st in stats]):.0f}")

    ok = np.array([r[2] is not None for r in results])
    x = np.stack([r[2] if r[2] is not None else np.zeros((len(CHANNELS), SIZE, SIZE), np.float32) for r in results])
    tic = np.array([r[0] for r in results], np.int64)
    signal = np.array([r[1] for r in results], np.int64)
    offset = np.array([r[3] for r in results], np.float32)
    reason = np.array([r[4] for r in results])
    meta = {"channels": CHANNELS, "size": SIZE, "min_in_dip_cadences": MIN_IN, "max_sectors": args.max_sectors,
            "max_chunks": m2.MAX_CHUNKS,
            "marker_sigma_px": MARKER_SIGMA, "off_target_px": OFF_TARGET,
            "built": "SPOC 2-min TPFs, quality==0; in-dip |dt|<dur/2, out 1-2 dur from centre; per-dip "
                     "difference out-in weighted by in-dip cadences; sectors combined inverse-variance on a "
                     "target-centred crop (no rotation between sectors); offset_px = distance of the largest "
                     "positive difference pixel from the target's sub-pixel position"}
    stem = "candidate" if args.name == "candidates" else args.name
    out = OUT_DIR / f"{stem}_pixel_views{'_pilot' if args.limit else ''}.npz"
    np.savez_compressed(out, tic=tic, signal=signal, ok=ok, x=x, offset_px=offset, reason=reason, meta=json.dumps(meta))
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB): x {x.shape}, {ok.sum()} ok, {(~ok).sum()} failed")
    print(pd.Series([r.split(" (")[0].split(":")[0] + (": " + r.split(": ")[1] if r.startswith("error") else "")
                     for r in reason[~ok]]).value_counts().to_string())

    # Sanity check: fakes should more often dim off the target. offset_px uses the raw difference peak; for
    # comparison, the SNR-map peak (target position from the marker channel's centroid).
    lab = pd.DataFrame({"tic": tic, "signal": signal}).merge(s[["tic", "signal", "label"]], how="left").label.values
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    mk = x[:, 3] / np.maximum(x[:, 3].sum((1, 2), keepdims=True), 1e-9)
    ty, tx = (mk * yy).sum((1, 2)), (mk * xx).sum((1, 2))
    snr_peak = x[:, 1].reshape(len(x), -1).argmax(1)
    snr_off = np.where(ok, np.hypot(snr_peak % SIZE - tx, snr_peak // SIZE - ty), np.nan)
    raw_peak = x[:, 0].reshape(len(x), -1).argmax(1)
    border = np.isin(raw_peak // SIZE, [0, SIZE - 1]) | np.isin(raw_peak % SIZE, [0, SIZE - 1])
    outer = x[ok][:, 1][:, np.hypot(yy - SIZE // 2, xx - SIZE // 2) > 3]
    print(f"noise check: median log1p-SNR of pixels > 3 px from the centre {np.median(outer[outer != 0]):.3f} (0 = unbiased)")
    if args.name == "candidates":  # unlabeled: just how many dim off the target
        print(f"candidates: {ok.sum()} ok rows, offset_px median {np.nanmedian(offset[ok]):.2f} px, "
              f"> {OFF_TARGET} px (off-target) in {(offset[ok] > OFF_TARGET).mean():.0%}, > 2 px in {(offset[ok] > 2).mean():.0%}")
        return
    for name, cls in [("planets CP/KP", ["CP", "KP"]), ("fakes FP/FA", ["FP", "FA"])]:
        m = ok & np.isin(lab, cls)
        for what, off in [("raw diff peak (offset_px)", offset), ("SNR peak", snr_off)]:
            print(f"{name}, {what}: {m.sum()} ok rows, median {np.nanmedian(off[m]):.2f} px, "
                  f"> {OFF_TARGET} px in {(off[m] > OFF_TARGET).mean():.0%}, > 2 px in {(off[m] > 2).mean():.0%}")
        print(f"  of rows > {OFF_TARGET} px: {border[m & (offset > OFF_TARGET)].mean():.0%} peak on the crop border")
    for title, m in [("planet", ok & np.isin(lab, ["CP", "KP"]) & (offset < 0.8)),
                     ("off-target fake", ok & np.isin(lab, ["FP", "FA"]) & (offset > 2.5))]:
        if m.any():
            i = np.flatnonzero(m)[0]
            print(f"\nexample {title}: TIC {tic[i]} signal {signal[i]} ({lab[i]}), offset {offset[i]:.2f} px")
            for c in [0, 1]:
                print(f"channel {c} {CHANNELS[c]}:\n{x[i, c].round(2)}")


if __name__ == "__main__":
    main()

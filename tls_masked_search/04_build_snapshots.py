"""Step 4: fixed-size folded views ("snapshots") of every step-2 signal, for the shape model (step 6).

For each signal, the star's capped sectors (the same files step 2 searched) are cleaned with the dips
masked while flattening, folded at the signal's period and dip time, and binned (median per bin):
  global     whole orbit, GLOBAL_BINS bins
  local      +/-2 dip lengths around the dip, LOCAL_BINS bins
  odd, even  the local view of alternating dips
  secondary  the local window centred half an orbit after the dip
  centroid   the local window of the star's position offset (mom_centr1/2, distance from the
             out-of-dip median), as a per-bin significance
Flux views are shifted and scaled so the out-of-dip median is 0 and the local dip minimum is -1 (the
depth itself is already a step-3 feature). Empty bins are interpolated; `ok` is False when a view has
too few filled bins or no dip.

Usage (from tls_masked_search/):  python 04_build_snapshots.py [--name dataset] [--max-chunks 2]   (~ minutes on 96 cores)
Reads output/{name}_signals.csv; writes output/views.npz for "dataset", else output/{name}_views.npz.
--max-chunks must match the step-2 run, so the same sector files are used.
"""
import argparse
import importlib
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import lightkurve as lk
import numpy as np
import pandas as pd

from common import OUT_DIR, clean_and_stitch

m2 = importlib.import_module("02_build_dataset")
capped, RAW = m2.capped, m2.RAW
GLOBAL_BINS, LOCAL_BINS = 201, 61
HALF_WINDOW = 2.0      # local window half-width, in dip lengths
MIN_FILLED = 1 / 3     # a view needs at least this share of bins with data
WORKERS = 96
FLUX_VIEWS = ["global", "local", "odd", "even", "secondary"]


def binned(x, y, lo, hi, n, stat=np.median):
    """`stat` of y in n equal bins of x over [lo, hi); empty bins interpolated. Returns (view, filled share)."""
    idx = np.floor((x - lo) / (hi - lo) * n).astype(int)
    keep = (idx >= 0) & (idx < n)
    idx, y = idx[keep], y[keep]
    out = np.full(n, np.nan)
    for i in np.unique(idx):
        out[i] = stat(y[idx == i])
    filled = np.isfinite(out)
    if not filled.any():
        return np.zeros(n), 0.0
    out = np.interp(np.arange(n), np.flatnonzero(filled), out[filled])
    return out, filled.mean()


def centroid_offsets(lc, mask):
    """Per-point distance (pixels) of the star's position from its out-of-dip trend, NaN if missing."""
    d2 = 0.0
    for col in ["mom_centr1", "mom_centr2"]:
        c = lk.LightCurve(time=lc.time, flux=np.asarray(lc[col].value, float))
        good = np.isfinite(c.flux.value)
        if good.sum() < 1100:
            return np.full(len(lc), np.nan)
        flat, trend = c[good].flatten(window_length=1001, mask=mask[good], return_trend=True)
        off = np.full(len(lc), np.nan)
        off[good] = c[good].flux.value - trend.flux.value
        d2 = d2 + off ** 2
    return np.sqrt(d2)


def signal_views(lcs, period, t0, dur):
    """All six views of one signal, plus an ok flag."""
    parts = [clean_and_stitch([lc], bin_minutes=2, ephemeris=(period, t0, dur)) for lc in lcs]
    t = np.concatenate([np.asarray(p[0]) for p in parts])
    f = np.concatenate([np.ma.filled(p[1], np.nan) for p in parts])
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    cyc = (t - t0) / period
    dt = (cyc - np.round(cyc)) * period              # days from the nearest dip centre
    dt2 = (cyc - np.floor(cyc) - 0.5) * period        # days from the nearest phase-0.5 point
    odd = np.round(cyc).astype(int) % 2 == 1
    w = HALF_WINDOW * dur
    oot = np.abs(dt) > dur
    base = np.median(f[oot]) if oot.any() else np.median(f)

    v, fill = {}, []
    v["global"], q = binned(dt / period, f, -0.5, 0.5, GLOBAL_BINS); fill.append(q)
    for name, x, sel in [("local", dt, slice(None)), ("odd", dt[odd], odd), ("even", dt[~odd], ~odd),
                         ("secondary", dt2, slice(None))]:
        v[name], q = binned(x, f[sel], -w, w, LOCAL_BINS); fill.append(q)
    depth = base - v["local"].min()
    for name in FLUX_VIEWS:
        v[name] = (v[name] - base) / depth if depth > 0 else np.zeros_like(v[name])

    # Centroid: per-bin mean offset minus the out-of-dip median, in units of its standard error.
    ct, cd = [], []
    for lc in lcs:
        lc = lc.remove_nans()
        mask = lc.create_transit_mask(period=period, transit_time=t0, duration=2 * dur)
        ct.append(lc.time.value)
        cd.append(centroid_offsets(lc, mask))
    ct, cd = np.concatenate(ct), np.concatenate(cd)
    good = np.isfinite(ct) & np.isfinite(cd)
    ct, cd = ct[good], cd[good]
    ccyc = (ct - t0) / period
    cdt = (ccyc - np.round(ccyc)) * period
    coot = np.abs(cdt) > dur
    if coot.sum() > 10:
        med, sd = np.median(cd[coot]), np.std(cd[coot])
        n = np.histogram(cdt, bins=LOCAL_BINS, range=(-w, w))[0]
        mean, q = binned(cdt, cd, -w, w, LOCAL_BINS, np.mean)
        v["centroid"] = np.clip((mean - med) / (sd / np.sqrt(np.maximum(n, 1))), -20, 20)
    else:
        v["centroid"], q = np.zeros(LOCAL_BINS), 0.0
    fill.append(q)
    good = depth > 0 and min(fill) >= MIN_FILLED and all(np.isfinite(x).all() for x in v.values())
    return v, bool(good)


def star_views(job):
    """Views for every signal of one star; its files are read once."""
    warnings.filterwarnings("ignore")
    tic, rows = job
    paths = capped(sorted(str(p) for p in RAW.glob(f"*{tic:016d}*")))
    out = []
    try:
        lcs = [lk.read(p, flux_column="pdcsap_flux") for p in paths]
    except Exception as e:  # unreadable file: every signal of this star fails
        lcs, err = None, repr(e)
    for r in rows:
        try:
            if lcs is None:
                raise RuntimeError(err)
            v, ok = signal_views(lcs, r["period"], r["t0"], r["duration_h"] / 24)
        except Exception:
            v = {k: np.zeros(GLOBAL_BINS if k == "global" else LOCAL_BINS) for k in FLUX_VIEWS + ["centroid"]}
            ok = False
        out.append((tic, r["signal"], v, ok))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="dataset")
    parser.add_argument("--max-chunks", type=int, default=m2.MAX_CHUNKS)
    args = parser.parse_args()
    m2.MAX_CHUNKS = args.max_chunks  # read by capped() in the (forked) workers
    start = time.time()
    s = pd.read_csv(OUT_DIR / f"{args.name}_signals.csv")
    jobs = [(int(tic), g[["signal", "period", "t0", "duration_h"]].to_dict("records")) for tic, g in s.groupby("tic")]
    results = []
    with ProcessPoolExecutor(WORKERS) as pool:
        for i, res in enumerate(pool.map(star_views, jobs, chunksize=1), 1):
            results += res
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} stars, {len(results)} signals, {time.time() - start:.0f} s", flush=True)

    names = FLUX_VIEWS + ["centroid"]
    arrays = {k: np.stack([r[2][k] for r in results]).astype(np.float32) for k in names}
    tic = np.array([r[0] for r in results], np.int64)
    signal = np.array([r[1] for r in results], np.int64)
    ok = np.array([r[3] for r in results])
    np.savez_compressed(OUT_DIR / ("views.npz" if args.name == "dataset" else f"{args.name}_views.npz"),
                        tic=tic, signal=signal, ok=ok, **arrays)
    print(f"\n{ok.sum()} rows ok, {(~ok).sum()} failed, of {len(ok)}; {time.time() - start:.0f} s")

    # Sanity check: planets (U-shaped) vs eclipsing-binary-like fakes (V-shaped), and the dip centre.
    lab = pd.DataFrame({"tic": tic, "signal": signal}).merge(s[["tic", "signal", "label"]], how="left").label.values
    c = LOCAL_BINS // 2
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    for name in ["CP", "FP"]:
        m = ok & (lab == name)
        print(f"mean local view, {name} ({m.sum()} rows), centre +/-10 bins:\n  {arrays['local'][m].mean(0)[c - 10:c + 11]}")
    m = ok & np.isin(lab, ["CP", "KP"])
    off = np.abs(arrays["local"][m].argmin(1) - c)
    print(f"planets: local minimum within 3 bins of the centre in {(off <= 3).mean():.0%} of rows "
          f"(median offset {np.median(off):.0f} bins)")


if __name__ == "__main__":
    main()

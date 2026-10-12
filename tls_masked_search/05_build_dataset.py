"""Step 5: build the labeled training set from TESS 2-min data.

Labels: NASA Exoplanet Archive TOI table, expert (TFOPWG) dispositions.
  planet = CP (confirmed) or KP (known planet); false = FP (false positive) or FA (false alarm).
For an equal number of stars from each class (default: as many as the smaller class allows; benchmark
stars excluded):
  1. find every SPOC 2-min sector with lightkurve, download from MAST (data/raw/dataset/, git-ignored)
  2. search each run of up to 4 consecutive sectors (TLS 0.5-30 d, twice: signal 1 hidden for the second);
     merge periods (and half/double aliases) found in several runs, keep the three found in the most runs
     (ties: strongest), refine each period on all sectors.
     (Searching years of gappy data in one go gives wrong dip lengths, which breaks every measurement.)
  3. measure each signal with the repo's own vetting (tess_hunt.search.vet_signal: odd/even depths,
     secondary eclipse, star-position shift during dips, noise-aware SNR, per-dip counts)
  4. label each signal by matching its period (within 1%) to the star's TOIs; unmatched = "none"
Optional (--random-stars N): N "nothing here" stars, drawn from the SPOC 2-min target lists of a few
sectors with no TOI at all (any disposition) and not in the benchmark; same steps, every signal
labeled "NT" (star_class "random").
Reports data availability, download size/time, search time, and how often the labeled TOI is recovered.

Writes output/{name}_stars.csv (one row per star) and output/{name}_signals.csv (one row per signal);
name defaults to "dataset".

Usage (from tls_masked_search/):  python 05_build_dataset.py [--per-class N] [--workers 96]
    [--name dataset] [--max-chunks 2] [--random-stars 0]
Full set (~2,450 stars): ~25 min, mostly MAST downloads (latency-bound), ~15 GB in data/raw/dataset/.
"""
import argparse
import io
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import lightkurve as lk
from astropy.io import fits
import numpy as np
import pandas as pd
import requests

from common import OUT_DIR, ROOT, clean_and_stitch, count_transits_with_data, relation, star_kwargs

sys.path.insert(0, str(ROOT))
from tess_hunt.photometry import preprocess  # noqa: E402
from tess_hunt.search import vet_signal  # noqa: E402

CONFIG = json.loads((ROOT / "config" / "benchmark.json").read_text())
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
LABELS = ROOT / "data" / "dataset" / "toi_labeled.csv"
RAW = ROOT / "data" / "raw" / "dataset"
CLASSES = {"planet": ["CP", "KP"], "false": ["FP", "FA"]}
PERIOD_RANGE = (0.5, 30.0)
CHUNK_SECTORS = 4             # ~110 days of mostly continuous data per search
MAX_CHUNKS = 2               # so at most 8 sectors per star
MIN_HIDE_DAYS = 3 / 24        # hide at least 3 h around each dip, in case a dip length comes out short
TLS_THREADS = 4
SEED = 0
RANDOM_SECTORS = range(1, 92, 10)  # sectors whose 2-min target lists the random stars are drawn from


def toi_table():
    """Every TOI with an expert disposition (cached)."""
    if not LABELS.exists():
        query = ("select tid, toi, tfopwg_disp, pl_orbper, pl_tranmid, pl_trandurh, pl_trandep, "
                 "st_tmag, st_rad, st_teff, st_logg from toi where tfopwg_disp is not null")
        r = requests.get(TAP, params={"query": query, "format": "csv"}, timeout=120)
        r.raise_for_status()
        LABELS.parent.mkdir(parents=True, exist_ok=True)
        LABELS.write_text(r.text)
    return pd.read_csv(LABELS)


def sample_stars(tois, per_class=None):
    """Stars whose labeled TOIs are all one class, `per_class` each (default: the smaller class's size),
    benchmark stars excluded."""
    bench = set(pd.read_csv(ROOT / "data" / "benchmark" / "targets.csv").tic)
    labeled = tois[tois.tfopwg_disp.isin(sum(CLASSES.values(), []))]
    cls = labeled.groupby("tid").tfopwg_disp.agg(lambda d: {c for c, v in CLASSES.items() if set(d) & set(v)})
    single = cls[cls.map(len).eq(1) & ~cls.index.isin(bench)].map(lambda s: next(iter(s)))
    per_class = per_class or int(single.value_counts().min())
    rng = np.random.default_rng(SEED)
    picks = [rng.choice(single[single.eq(c)].index, per_class, replace=False) for c in CLASSES]
    return pd.DataFrame({"tic": np.concatenate(picks), "star_class": np.repeat(list(CLASSES), per_class)})


def random_stars(n):
    """`n` SPOC 2-min targets of RANDOM_SECTORS with no TOI (any disposition) and not in the benchmark."""
    r = requests.get(TAP, params={"query": "select distinct tid from toi", "format": "csv"}, timeout=120)
    r.raise_for_status()
    exclude = set(pd.read_csv(io.StringIO(r.text)).tid) | set(pd.read_csv(ROOT / "data" / "benchmark" / "targets.csv").tic)
    tics = set()
    for sector in RANDOM_SECTORS:
        r = requests.get(f"https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{sector}_lc.sh",
                         timeout=120)
        r.raise_for_status()
        tics |= {int(line.split()[5].split("-")[2]) for line in r.text.splitlines() if line.startswith("curl")}
    pool = np.array(sorted(tics - exclude))
    picks = np.random.default_rng(SEED).choice(pool, n, replace=False)
    print(f"Random stars: {len(pool)} no-TOI 2-min targets in sectors {list(RANDOM_SECTORS)}, picked {n}")
    return pd.DataFrame({"tic": picks, "star_class": "random"})


def download(tic):
    """The star's SPOC 2-min light curves (capped, see `capped`): found with lightkurve, fetched straight
    from MAST (astroquery's progress bar is not thread-safe). Returns (paths, sectors available, MB, seconds)."""
    start = time.time()
    found = lk.search_lightcurve(f"TIC {tic}", author="SPOC", exptime=120)
    uris = {uri.split("/")[-1]: uri for uri in (found.table["dataURI"] if len(found) else [])}
    paths = []
    with requests.Session() as session:
        for name in capped(list(uris)):
            path = RAW / name
            if not path.exists():
                r = session.get("https://mast.stsci.edu/api/v0.1/Download/file", params={"uri": uris[name]},
                                timeout=300)
                r.raise_for_status()
                path.with_suffix(".part").write_bytes(r.content)
                path.with_suffix(".part").rename(path)
            paths.append(str(path))
    return paths, len(uris), sum(os.path.getsize(p) for p in paths) / 1e6, time.time() - start


def star_params(rows):
    """TLS priors from the TOI table: radius, and mass from surface gravity (fallback: Sun-like)."""
    r = rows.st_rad.dropna()
    radius = float(r.iloc[0]) if len(r) else 1.0
    g = rows.st_logg.dropna()
    mass = float(10 ** (g.iloc[0] - 4.438) * radius ** 2) if len(g) else 1.0
    return float(np.clip(mass, 0.1, 5)), float(np.clip(radius, 0.1, 10))


def chunks(paths):
    """Group sector files into runs of consecutive sectors, at most CHUNK_SECTORS each. TLS needs
    evenly sampled data: on years of gappy data its dip length (and so every dip measurement) is wrong."""
    by_sector = sorted(paths, key=lambda p: int(p.split("-s")[1][:4]))
    groups, last = [], None
    for p in by_sector:
        s = int(p.split("-s")[1][:4])
        if groups and s == last + 1 and len(groups[-1]) < CHUNK_SECTORS:
            groups[-1].append(p)
        else:
            groups.append([p])
        last = s
    return groups


def capped(paths):
    """The MAX_CHUNKS biggest runs of consecutive sectors (ties: most recent), at most
    MAX_CHUNKS * CHUNK_SECTORS sectors per star, to bound download size and search time."""
    groups = sorted(chunks(paths), key=lambda g: (len(g), g[-1].split("-s")[1][:4]), reverse=True)
    return [p for g in groups[:MAX_CHUNKS] for p in g]


def stitched(paths):
    """Cleaned, binned light curve, binned one sector at a time: astropy's binning across long gaps
    is very slow (55 s vs 2 s for 5 sectors spread over 2 years)."""
    parts = [clean_and_stitch([lk.read(p, flux_column="pdcsap_flux")], CONFIG["bin_minutes"]) for p in paths]
    t, y, dy = (np.ma.filled(np.ma.concatenate(a), np.nan) for a in zip(*parts))
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy)
    order = np.argsort(t[ok])
    return t[ok][order], y[ok][order], dy[ok][order]


def search_star(job):
    """Search each chunk of consecutive sectors twice; merge the periods found across chunks (and their
    half/double aliases), refine each of the 3 best on all sectors, then measure and label it."""
    warnings.filterwarnings("ignore")
    from transitleastsquares import transit_mask, transitleastsquares
    tic, paths, tois = job["tic"], job["paths"], job["tois"]
    start = time.time()
    mass, radius = star_params(tois)
    star = {**star_kwargs(mass, radius), "use_threads": TLS_THREADS}
    props = {}
    if job.get("star_props"):  # TIC values from the FITS header, the same source for TOI and non-TOI stars
        head = fits.getheader(paths[0])
        props = {c: float(head[k]) if head.get(k) is not None else np.nan
                 for c, k in [("st_tmag", "TESSMAG"), ("st_rad", "RADIUS"), ("st_teff", "TEFF"), ("st_logg", "LOGG")]}

    vet_parts = []
    for p in paths:
        try:
            vet_parts.append(preprocess(p, CONFIG)[0])
        except ValueError:
            pass
    if not vet_parts:
        return {"tic": tic, "sectors": len(paths), **props, "skipped": "preprocess rejected every sector"}, []
    vet_data = {k: np.concatenate([d[k] for d in vet_parts]) for k in ["time", "flux", "error", "centroid1", "centroid2"]}

    found = []  # (period, sde, duration, chunk index) from every chunk
    for ci, group in enumerate(chunks(paths)):
        t, y, dy = stitched(group)
        for _ in range(2):
            pmax = min(PERIOD_RANGE[1], np.ptp(t) / 2)
            if pmax <= PERIOD_RANGE[0] * 2:
                break
            r = transitleastsquares(t, y, dy).power(period_min=PERIOD_RANGE[0], period_max=pmax,
                                                    n_transits_min=2, **star)
            found.append((r.period, r.SDE, r.duration, ci))
            keep = ~transit_mask(t, r.period, max(2 * r.duration, MIN_HIDE_DAYS), r.T0)
            t, y, dy = t[keep], y[keep], dy[keep]

    # Merge periods that agree within 1% (or are half/double) across chunks; rank by how many chunks
    # found them, then by strength; keep three.
    merged = []
    for period, sde, duration, ci in sorted(found, key=lambda f: -f[1]):
        same = next((m for m in merged if relation(period, m["period"])), None)
        if same:
            same["chunks"].add(ci)
        else:
            merged.append({"period": period, "best_chunk_sde": sde, "duration": duration, "chunks": {ci}})
    merged = sorted(merged, key=lambda m: (-len(m["chunks"]), -m["best_chunk_sde"]))[:3]

    t, y, dy = stitched(paths)

    signals = []
    for n, m in enumerate(merged, 1):
        # All sectors pin down period and dip time; the dip length comes from the chunk search.
        r = transitleastsquares(t, y, dy).power(period_min=m["period"] * 0.995, period_max=m["period"] * 1.005,
                                                n_transits_min=2, **star)
        vet = vet_signal(vet_data, r.period, r.T0, m["duration"])
        matches = [(relation(r.period, row.pl_orbper), row) for row in tois.itertuples()
                   if np.isfinite(row.pl_orbper)]
        exact = [row for rel, row in matches if rel == "exact"]
        harmonic = [row for rel, row in matches if rel in ("half_period", "double_period")]
        hit = exact[0] if exact else None
        signals.append({
            "tic": tic, "signal": n, "period": r.period, "t0": r.T0, "duration_h": m["duration"] * 24,
            "best_chunk_sde": m["best_chunk_sde"], "n_chunks_found": len(m["chunks"]), "all_data_sde": r.SDE,
            **{k: v for k, v in vet.items() if k not in ("events", "flags")}, "flags": ";".join(vet["flags"]),
            "label_toi": hit.toi if hit else np.nan, "label": hit.tfopwg_disp if hit else "none",
            "harmonic_of_toi": harmonic[0].toi if harmonic else np.nan,
        })

    labeled = tois[tois.tfopwg_disp.isin(sum(CLASSES.values(), []))]
    dips = [count_transits_with_data(vet_data["time"], row.pl_tranmid - 2457000, row.pl_orbper,
                                     row.pl_trandurh / 48 if np.isfinite(row.pl_trandurh) else 1 / 24)
            for row in labeled.itertuples() if np.isfinite(row.pl_orbper) and np.isfinite(row.pl_tranmid)]
    star_row = {"tic": tic, "sectors": len(paths), "chunks": len(chunks(paths)), **props,
                "span_days": float(np.ptp(vet_data["time"])),
                "labeled_toi_dips_with_data": max(dips, default=0), "search_seconds": time.time() - start,
                "recovered": any(s["label"] != "none" for s in signals)}
    return star_row, signals


def main():
    global MAX_CHUNKS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=None, help="stars per class (default: all the smaller class has)")
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--download-workers", type=int, default=48)  # MAST queries take ~8 s each; latency-bound
    parser.add_argument("--name", default="dataset", help="output prefix: output/{name}_stars.csv, _signals.csv")
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS, help="runs of up to 4 sectors per star")
    parser.add_argument("--random-stars", type=int, default=0, help="add N no-TOI stars, signals labeled NT")
    parser.add_argument("--star-props", action="store_true", help="add TIC st_tmag/st_rad/st_teff/st_logg per star")
    args = parser.parse_args()
    MAX_CHUNKS = args.max_chunks
    warnings.filterwarnings("ignore")
    RAW.mkdir(parents=True, exist_ok=True)

    tois = toi_table()
    print("TOI dispositions:", tois.tfopwg_disp.value_counts(dropna=False).to_dict())
    stars = sample_stars(tois, args.per_class)
    if args.random_stars:
        stars = pd.concat([stars, random_stars(args.random_stars)], ignore_index=True)

    def safe_download(tic):  # retry MAST hiccups; NaN sectors_available = gave up (finished files stay cached)
        for attempt in range(3):
            try:
                return download(tic)
            except Exception as e:
                print(f"  download TIC {tic} attempt {attempt + 1} failed: {e!r}", flush=True)
                time.sleep(10 * 2 ** attempt)
        return [], np.nan, 0.0, 0.0

    start = time.time()
    with ThreadPoolExecutor(args.download_workers) as pool:
        got = list(pool.map(safe_download, stars.tic))
    stars["paths"], stars["sectors_available"], stars["download_mb"], stars["download_seconds"] = zip(*got)
    stars["sectors"] = stars.paths.map(len)
    have = stars[stars.sectors.gt(0)]
    print(f"Download: {time.time() - start:.0f} s wall, {stars.download_mb.sum():.0f} MB, "
          f"{stars.sectors_available.isna().sum()} failed", flush=True)
    start = time.time()

    jobs = [{"tic": s.tic, "paths": s.paths, "tois": tois[tois.tid.eq(s.tic)], "star_props": args.star_props}
            for s in have.itertuples()]
    star_rows, signal_rows, failed = [], [], []

    def save():  # rewritten after every star, so a killed run keeps its partial data
        rows = pd.DataFrame(star_rows or {"tic": []})
        per_star = stars.drop(columns=["paths", "sectors"]).merge(rows, on="tic", how="left")
        per_star.to_csv(OUT_DIR / f"{args.name}_stars.csv", index=False)
        signals = pd.DataFrame(signal_rows or {"tic": []}).merge(stars[["tic", "star_class"]], on="tic")
        signals.loc[signals.star_class.eq("random"), "label"] = "NT"
        signals.to_csv(OUT_DIR / f"{args.name}_signals.csv", index=False)
        return per_star, signals

    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(search_star, j): j["tic"] for j in jobs}
        for i, f in enumerate(as_completed(futures), 1):
            tic = futures[f]
            try:
                s, sig = f.result()
                star_rows.append(s)
                signal_rows.extend(sig)
                note = s.get("skipped") or (f"{s['chunks']} chunks, {s['search_seconds']:.0f} s, "
                                            f"recovered={s['recovered']}")
                print(f"  [{i}/{len(jobs)}] TIC {tic}: {s['sectors']} sectors, {note}", flush=True)
            except Exception as e:
                failed.append((tic, repr(e)))
                print(f"  [{i}/{len(jobs)}] TIC {tic}: FAILED {e!r}", flush=True)
            save()
    print(f"Search: {time.time() - start:.0f} s wall", flush=True)
    per_star, signals = save()

    print(f"\nFailed searches: {len(failed)}", failed[:3])
    for c, g in per_star.groupby("star_class"):
        s = g[g.sectors.gt(0)]
        print(f"\n[{c}] {len(g)} stars, {len(s)} with 2-min data")
        print(f"  sectors used per star: median {s.sectors.median():.0f} (available: median "
              f"{s.sectors_available.median():.0f}, max {s.sectors_available.max():.0f}); "
              f"download {s.download_mb.sum():.0f} MB total")
        print(f"  search time per star: median {s.search_seconds.median():.0f} s, max {s.search_seconds.max():.0f} s")
        print(f"  labeled TOI has >=2 dips on data: {s.labeled_toi_dips_with_data.ge(2).sum()}/{len(s)}")
        print(f"  labeled TOI recovered by one of the 3 signals: {s.recovered.sum()}/{len(s)}")
    print("\nSignal labels:", signals.label.value_counts().to_dict())
    print("\nMedian measurements by signal label:")
    cols = ["best_chunk_sde", "all_data_sde", "depth_ppm", "pink_snr", "odd_even_sigma", "secondary_snr",
            "centroid_snr", "observed_transits"]
    print(signals.groupby("label")[cols].median().round(1).to_string())


if __name__ == "__main__":
    main()

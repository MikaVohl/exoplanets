"""Step 6: why do pipelines A and B both miss some known transiting hosts?

For every transiting catalog planet of each host that neither A nor B recovers (from
output/ablation_catalog.csv), on pipeline-B-cleaned data (lightkurve flatten, 10-min bins):
  - n_tr / n_tr_best: transits with data, at the catalog ephemeris / at the best-scan phase
    (catalog timing can drift, the best-scan count does not depend on it)
  - snr_exp:  catalog depth / per-point noise * sqrt(in-transit points)   (white-noise ideal)
  - snr_eph:  box depth SNR folded at the catalog period and epoch       (what the data shows)
  - snr_best: same, best of 400 trial phases; `floor` is that best-of-400 at four unrelated
    periods, i.e. how strong pure noise/variability looks. Trust snr_best only well above floor.
  - dep_calc_ppm: (Rp/Rs)^2 sanity check on the catalog depth
  - harm: B's found period vs the catalog period (exact, or x 1/3, 1/2, 2, 3 within 1%)
Writes output/miss_table.csv. Raw FITS are fetched into data/raw/benchmark/ if missing.
"""
import csv
import pathlib
import sys
import warnings

import numpy as np
import lightkurve as lk

from common import clean_and_stitch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tess_hunt.photometry import download_target  # noqa: E402

HARMONICS = [("1/3", 1 / 3), ("1/2", 0.5), ("2", 2), ("3", 3)]
FLOOR_PERIOD_FACTORS = (0.73, 1.37, 0.61, 1.61)


def rows(path):
    with open(REPO / path) as f:
        return list(csv.DictReader(f))


def box_snr(t, y, sigma, period, t0, dur):
    ph = ((t - t0) / period + 0.5) % 1 - 0.5
    inn = np.abs(ph * period) < dur / 2
    if inn.sum() < 3:
        return np.nan
    return (np.median(y[~inn]) - y[inn].mean()) / (sigma / np.sqrt(inn.sum()))


def count_transits(t, t0, period, dur):
    ks = np.arange(np.floor((t.min() - t0) / period), np.ceil((t.max() - t0) / period) + 1)
    return int(sum(np.sum(np.abs(t - (t0 + k * period)) < dur / 2) >= 3 for k in ks))


def analyse_planet(t, y, sigma, span, star_radius, p, found_period):
    P = float(p["pl_orbper"])
    dur = float(p["pl_trandur"]) / 24 if p["pl_trandur"] else 3 / 24
    dep_calc = (float(p["pl_rade"]) * 0.00916 / star_radius) ** 2
    dep = float(p["pl_trandep"]) / 100 if p["pl_trandep"] else dep_calc  # pl_trandep is in percent
    t0 = float(p["pl_tranmid"]) - 2457000

    phases = np.linspace(0, 1, 400, endpoint=False)
    scan_at = lambda Q: [box_snr(t, y, sigma, Q, t.min() + f * Q, dur) for f in phases]
    scan = scan_at(P)
    t_best = t.min() + phases[np.nanargmax(scan)] * P
    n_tr = count_transits(t, t0, P, dur)
    n_tr_best = count_transits(t, t_best, P, dur)
    ratio = found_period / P
    harm = "exact" if abs(ratio - 1) < 0.01 else next(
        (f"x{h}" for h, v in HARMONICS if abs(ratio / v - 1) < 0.01), "")
    return dict(
        planet=p["pl_name"], P=round(P, 3), dep_ppm=round(dep * 1e6), dep_calc_ppm=round(dep_calc * 1e6),
        dur_h=round(dur * 24, 2), n_tr=n_tr, n_tr_best=n_tr_best, noise_ppm=round(sigma * 1e6),
        snr_exp=round(float(dep / sigma * np.sqrt(max(n_tr, n_tr_best) * dur / (10 / 1440))), 1),
        snr_eph=round(float(box_snr(t, y, sigma, P, t0, dur)), 1),
        snr_best=round(float(np.nanmax(scan)), 1),
        floor=round(float(np.nanmean([np.nanmax(scan_at(P * m)) for m in FLOOR_PERIOD_FACTORS])), 1),
        in_range=bool(P <= min(12, span / 2)), found_P=round(found_period, 4), harm=harm)


def main():
    warnings.filterwarnings("ignore")
    cat = {r["tic"]: r for r in rows("tls_masked_search/output/ablation_catalog.csv")}
    found = {r["tic"]: r for r in rows("tls_masked_search/output/benchmark_100.csv")}
    targets = {r["tic"]: r for r in rows("data/benchmark/targets.csv")}
    planets = rows("data/benchmark/known_planets.csv")
    misses = [t for t, r in cat.items() if r["group"] == "known_transiting_host"
              and r["hit_A"] == "False" and r["hit_B"] == "False"]
    print(f"{len(misses)} hosts missed by both A and B")

    out = []
    for tic in misses:
        tgt = targets[tic]
        download_target(REPO, tgt)
        lc = lk.read(REPO / "data/raw/benchmark" / tgt["filename"], quality_bitmask="default")
        t, y, _ = (np.ma.filled(a, np.nan) for a in clean_and_stitch([lc], 10))
        ok = np.isfinite(t) & np.isfinite(y)
        t, y = t[ok], y[ok]
        sigma = 1.4826 * np.median(np.abs(y - np.median(y)))
        for p in (p for p in planets if p["tic"] == tic and p["tran_flag"] == "1"):
            row = dict(tic=tic, host=cat[tic]["hostname"],
                       **analyse_planet(t, y, sigma, t.max() - t.min(), float(tgt["radius"]), p,
                                        float(found[tic]["period"])),
                       found_sde=round(float(found[tic]["sde"]), 2))
            out.append(row)
            print(*row.values(), sep="\t", flush=True)

    with open("output/miss_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out[0].keys())
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()

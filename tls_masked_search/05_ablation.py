"""Step 5: ablation. Is this pipeline actually better, and which part makes it so?

Three pipelines share the SAME TLS call, period range and star priors. Only preprocessing differs:
  A  repo detrending        1-day median filter, 0.1 d segment-edge trim   (tess_hunt.photometry)
  B  lightkurve flatten     33 h Savitzky-Golay flatten, wide TLS only
  C  B + masked 2nd pass    mask transits, re-flatten, fine TLS around the peak (the masked two-pass method)
A vs B isolates the detrending. B vs C isolates the masking + fine pass.

Two tests:
  1. Catalog: do A/B/C recover the 60 known transiting hosts? Paired exact (McNemar) tests say
     whether differences could be chance. False alarms are counted on the 40 comparison stars.
  2. Injection-recovery: inject synthetic transits (known period/depth, no catalog involved) into
     real light curves of comparison stars and see which pipeline gets them back. This is the
     cleanest evidence: the right answer is known by construction.

Usage (from tls_masked_search/):
    python 05_ablation.py --quick     # 6 stars per test, a few minutes
    python 05_ablation.py             # full run
"""
import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.stats import binomtest
from transitleastsquares.transit_model import light_curve

from common import MIN_SDE, OUT_DIR, RAW, ROOT, relation, star_kwargs, tls_wide, two_pass

sys.path.insert(0, str(ROOT))
from tess_hunt.photometry import preprocess  # noqa: E402  (repo's own detrending)

CONFIG = json.loads((ROOT / "config" / "benchmark.json").read_text())
SEED = 42
INJECT_DEPTHS = [0.00025, 0.0005, 0.001, 0.002, 0.004]   # fractional depth, 250-4000 ppm
CONDITIONS = {"A": "repo detrend", "B": "lk flatten", "C": "lk flatten + mask + fine"}


def run_conditions(path, mass, radius):
    """Run A, B, C on one FITS file. Returns {cond: (period, sde)}, None where a pipeline failed."""
    star = star_kwargs(mass, radius)
    out = {}
    try:
        d, _ = preprocess(path, CONFIG)
        r, _ = tls_wide(d["time"], d["flux"], d["error"], star, CONFIG)
        out["A"] = (float(r.period), float(r.SDE))
    except Exception:
        out["A"] = None
    try:
        wide, fine = two_pass(lk.read(path, flux_column="pdcsap_flux"), star, CONFIG)
        out["B"] = (float(wide.period), float(wide.SDE))
        out["C"] = (float((fine or wide).period), float(wide.SDE))
    except Exception:
        out["B"] = out["C"] = None
    return out


def recovered(result, periods):
    """SDE passes and the period matches (exactly, within 1%) one of `periods`."""
    return bool(result and result[1] >= MIN_SDE and any(relation(result[0], p) == "exact" for p in periods))


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired booleans. Returns (a_only, b_only, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return only_a, only_b, p


def compare_table(df, cols, label):
    print(f"\n{label}")
    print(f"  {'':<8}" + "".join(f"{c + ' ' + CONDITIONS[c][:12]:<22}" for c in cols))
    for pair in [("A", "B"), ("B", "C"), ("A", "C")]:
        a_only, b_only, p = mcnemar(df[f"hit_{pair[0]}"], df[f"hit_{pair[1]}"])
        verdict = "significant" if p < 0.05 else "not significant"
        print(f"  {pair[0]} vs {pair[1]}: {pair[0]}-only {a_only}, {pair[1]}-only {b_only}, p = {p:.3f} ({verdict})")


# ---------------------------------------------------------------- test 1: catalog
def catalog_test(targets, known, limit):
    hosts = targets[targets.catalog_group.eq("known_transiting_host")]
    comps = targets[targets.catalog_group.eq("no_cataloged_transiting_planet")]
    if limit:
        hosts, comps = hosts.head(limit), comps.head(limit)
    rows = []
    for i, t in enumerate(pd.concat([hosts, comps]).to_dict("records"), 1):
        res = run_conditions(RAW / t["filename"], float(t["mass"]), float(t["radius"]))
        planets = known[known.tic.eq(t["tic"]) & known.tran_flag.eq(1)
                        & known.pl_orbper.between(CONFIG["period_min_days"], CONFIG["period_max_days"])].pl_orbper
        row = {"tic": t["tic"], "hostname": t["hostname"], "group": t["catalog_group"], "radius_bin": t["radius_bin"],
               "n_planets": len(planets)}
        for c in CONDITIONS:
            row[f"hit_{c}"] = recovered(res[c], planets)
            row[f"alarm_{c}"] = bool(res[c] and res[c][1] >= MIN_SDE)
        rows.append(row)
        print(f"  catalog {i}/{len(hosts) + len(comps)} {t['hostname']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ablation_catalog.csv", index=False)

    h = df[df.group.eq("known_transiting_host") & df.n_planets.gt(0)]
    print("\n=== TEST 1: catalog recovery (known transiting hosts) ===")
    for c in CONDITIONS:
        print(f"  {c} {CONDITIONS[c]:<26} {h[f'hit_{c}'].sum()}/{len(h)} hosts recovered")
    compare_table(h, CONDITIONS, "Paired tests on hosts (differences between pipelines):")
    k = df[df.group.eq("no_cataloged_transiting_planet")]
    print("\n  Comparison stars with SDE >= 7 (not necessarily false: known RV planet hosts):")
    for c in CONDITIONS:
        print(f"  {c} {CONDITIONS[c]:<26} {k[f'alarm_{c}'].sum()}/{len(k)}")
    return df


# ---------------------------------------------------------------- test 2: injection
def inject(path, out_path, mass, radius, period, t0, depth):
    """Copy a TESS light-curve FITS with a synthetic limb-darkened transit multiplied into PDCSAP_FLUX."""
    a_over_rs = 215.03 * (mass * (period / 365.25) ** 2) ** (1 / 3) / radius
    inc = np.degrees(np.arccos(0.3 / a_over_rs))            # impact parameter 0.3
    with fits.open(path) as hdul:
        time = np.asarray(hdul[1].data["TIME"], float)
        good = np.isfinite(time)
        model = np.ones_like(time)
        model[good] = light_curve(time[good], t0, period, np.sqrt(depth), a_over_rs, inc,
                                  0.0, 90.0, [0.3, 0.2], "quadratic")
        hdul[1].data["PDCSAP_FLUX"] = hdul[1].data["PDCSAP_FLUX"] * model
        hdul.writeto(out_path, overwrite=True)
    return float(1 - model[good].min())


def injection_test(targets, catalog_df, limit):
    comps = targets[targets.catalog_group.eq("no_cataloged_transiting_planet")]
    # Every comparison star is used. A pre-existing signal can't count as a recovery because the
    # period must match the injected one. `quiet` marks stars where no pipeline alarmed un-injected,
    # so results can be split into calm vs variable stars.
    quiet_tics = set(catalog_df[catalog_df.group.eq("no_cataloged_transiting_planet")
                                & ~catalog_df[[f"alarm_{c}" for c in CONDITIONS]].any(axis=1)].tic)
    if limit:
        comps = comps.head(limit)
    rng = np.random.default_rng(SEED)
    rows, tmp = [], Path(tempfile.mkdtemp(prefix="inject_"))
    trials = [(t, d) for t in comps.to_dict("records") for d in INJECT_DEPTHS]
    for i, (t, depth) in enumerate(trials, 1):
        with fits.open(RAW / t["filename"]) as h:
            time = np.asarray(h[1].data["TIME"], float)
        time = time[np.isfinite(time)]
        span = np.ptp(time)
        period = float(np.exp(rng.uniform(np.log(0.8), np.log(min(10.0, span / 3)))))
        t0 = float(time.min() + rng.uniform(0, period))
        path = tmp / "injected.fits"
        true_depth = inject(RAW / t["filename"], path, float(t["mass"]), float(t["radius"]), period, t0, depth)
        res = run_conditions(path, float(t["mass"]), float(t["radius"]))
        row = {"tic": t["tic"], "hostname": t["hostname"], "depth": depth, "true_depth_ppm": true_depth * 1e6,
               "period": period, "quiet": t["tic"] in quiet_tics}
        for c in CONDITIONS:
            row[f"hit_{c}"] = recovered(res[c], [period])
            row[f"found_period_{c}"] = res[c][0] if res[c] else np.nan
        rows.append(row)
        print(f"  injection {i}/{len(trials)} {t['hostname']} P={period:.2f} depth={depth * 1e6:.0f}ppm "
              f"-> " + " ".join(f"{c}={'Y' if row[f'hit_{c}'] else 'n'}" for c in CONDITIONS), flush=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / "ablation_injection.csv", index=False)
    df = pd.DataFrame(rows)

    print(f"\n=== TEST 2: injection-recovery ({df.tic.nunique()} quiet stars x {len(INJECT_DEPTHS)} depths = {len(df)} trials) ===")
    print(f"  {'depth':<12}" + "".join(f"{c:<16}" for c in CONDITIONS))
    for depth, g in df.groupby("depth"):
        print(f"  {depth * 1e6:>6.0f} ppm  " + "".join(f"{g[f'hit_{c}'].sum():>3}/{len(g):<12}" for c in CONDITIONS))
    print("  " + "-" * 12 + "".join("-" * 16 for _ in CONDITIONS))
    for c in CONDITIONS:
        k = int(df[f"hit_{c}"].sum())
        lo, hi = binomtest(k, len(df)).proportion_ci(0.95)
        print(f"  {c} {CONDITIONS[c]:<26} overall {k}/{len(df)} = {k / len(df):.0%}  (95% CI {lo:.0%}-{hi:.0%})")
    compare_table(df, CONDITIONS, "Paired tests on injected transits:")
    for name, g in [("calm stars (no un-injected alarm)", df[df.quiet]), ("variable stars (some pipeline alarmed)", df[~df.quiet])]:
        print(f"\n  {name}: {g.tic.nunique()} stars, {len(g)} trials")
        for c in CONDITIONS:
            print(f"    {c} {CONDITIONS[c]:<26} {int(g[f'hit_{c}'].sum())}/{len(g)}")
        if len(g):
            compare_table(g, CONDITIONS, "    paired:")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--injection-only", action="store_true", help="reuse output/ablation_catalog.csv")
    parser.add_argument("--quick", action="store_true", help="6 stars per group, for a smoke test")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    OUT_DIR.mkdir(exist_ok=True)
    targets = pd.read_csv(ROOT / "data" / "benchmark" / "targets.csv")
    known = pd.read_csv(ROOT / "data" / "benchmark" / "known_planets.csv")
    limit = 6 if args.quick else None
    catalog_df = (pd.read_csv(OUT_DIR / "ablation_catalog.csv") if args.injection_only
                  else catalog_test(targets, known, limit))
    injection_test(targets, catalog_df, limit)


if __name__ == "__main__":
    main()

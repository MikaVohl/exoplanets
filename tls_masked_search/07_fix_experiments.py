"""Step 7: test candidate fixes for the misses found in step 6, one change at a time.

Every variant starts from pipeline B (lightkurve 33 h flatten, wide TLS, top signal) and changes ONE thing:
  B       baseline
  B2      + second search after hiding the first signal's transits (a hit if either signal matches)
  short   flatten window 12 h instead of 33 h (follows fast-changing stars better, may eat long dips)
  wide    longest searched period = half the time span, instead of min(12 d, half the span)
  B2wide  B2 + wide: is the wider range still worth it once the second search is in? (compared with B2)
The score cutoff (SDE) needs no rerun: it is swept afterwards over the stored scores.
Known planets count up to 12 d, or up to half the time span for the wide variants.

Deciding evidence is synthetic, because the fixes were picked by looking at the real misses:
  inj_standard  200 fake planets, same draws as step 5 (B must reproduce 148/200)
  inj_long      fake planets with periods 8 d .. half the span, at least 2 dips with data (judges `wide`)
  inj_two       a strong fake planet plus a weaker one; scored on the weak one (judges `B2`)
  catalog       comparison-star alarms (40 stars); known-host hits (60) are supporting evidence only

Usage (from tls_masked_search/):  python 07_fix_experiments.py [--quick] [--workers N]
"""
import argparse
import importlib
import json
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd

from common import MIN_SDE, OUT_DIR, RAW, ROOT, clean_and_stitch, hide_transits, relation, star_kwargs, tls_wide

ablation = importlib.import_module("05_ablation")  # reuse its fake-planet injector and paired test

CONFIG = json.loads((ROOT / "config" / "benchmark.json").read_text())
VARIANTS = ["B", "B2", "short", "wide", "B2wide"]
WIDE = {"wide", "B2wide"}
SHORT_WINDOW = 361            # 2-min points ~ 12 h
TLS_THREADS = 4
CUTOFFS = [6.0, 6.5, 7.0]


def search(path, mass, radius):
    """Run every variant on one FITS file. Returns ({variant: [(period, sde), ...]}, {variant: error})."""
    star = {**star_kwargs(mass, radius), "use_threads": TLS_THREADS}
    lc = lk.read(path, flux_column="pdcsap_flux")
    wide_config = {**CONFIG, "period_max_days": np.inf}

    def signals(config, n, window=1001):
        found, light = [], lc
        for _ in range(n):
            r, _ = tls_wide(*clean_and_stitch([light], CONFIG["bin_minutes"], window_length=window), star, config)
            found.append((float(r.period), float(r.SDE)))
            light = hide_transits(light, r)
        return found

    runs = {"B": lambda: signals(CONFIG, 1), "B2": lambda: signals(CONFIG, 2),
            "short": lambda: signals(CONFIG, 1, SHORT_WINDOW), "wide": lambda: signals(wide_config, 1),
            "B2wide": lambda: signals(wide_config, 2)}
    out, errors = {}, {}
    for v, run in runs.items():
        try:
            out[v] = run()
        except Exception as e:
            out[v], errors[v] = [], repr(e)
    return out, errors


def best_match_sde(signals, periods):
    """Highest SDE among signals whose period matches one of `periods` within 1% (nan if none)."""
    s = [sde for p, sde in signals if any(relation(p, q) == "exact" for q in periods)]
    return max(s) if s else np.nan


def run_job(job):
    """One light curve: optionally inject fake planets, search, score. Runs in a worker process."""
    warnings.filterwarnings("ignore")
    star, kind, fakes = job["star"], job["kind"], job.get("fakes", [])
    path = RAW / star["filename"]
    tmp = Path(tempfile.mkdtemp(prefix="fix_"))
    for i, (period, t0, depth) in enumerate(fakes):
        out_path = tmp / f"inj{i}.fits"
        ablation.inject(path, out_path, float(star["mass"]), float(star["radius"]), period, t0, depth)
        path = out_path
    res, errors = search(path, float(star["mass"]), float(star["radius"]))
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    row = {"kind": kind, "tic": star["tic"], "hostname": star["hostname"], **job.get("meta", {})}
    for v in VARIANTS:
        targets = job.get("targets_wide", job["targets"]) if v in WIDE else job["targets"]  # periods that count
        row[f"match_sde_{v}"] = best_match_sde(res[v], targets)
        row[f"top_sde_{v}"] = max((s for _, s in res[v]), default=np.nan)
        row[f"periods_{v}"] = ";".join(f"{p:.5f}" for p, _ in res[v])
        row[f"err_{v}"] = errors.get(v, "")
    return row


def data_times(star):
    """(all finite timestamps, as step 5 uses for its draws; timestamps with usable flux)."""
    from astropy.io import fits
    with fits.open(RAW / star["filename"]) as h:
        time = np.asarray(h[1].data["TIME"], float)
        flux = np.asarray(h[1].data["PDCSAP_FLUX"], float)
    return time[np.isfinite(time)], time[np.isfinite(time) & np.isfinite(flux)]


def n_dips_with_data(times, period, t0, half_window=1 / 24):
    ks = np.arange(0, np.ceil((times.max() - t0) / period) + 1)
    return int(sum(np.any(np.abs(times - (t0 + k * period)) < half_window) for k in ks))


def build_jobs(targets, known, limit):
    hosts = targets[targets.catalog_group.eq("known_transiting_host")]
    comps = targets[targets.catalog_group.eq("no_cataloged_transiting_planet")]
    if limit:
        hosts, comps = hosts.head(limit), comps.head(limit)
    jobs = []
    for star in pd.concat([hosts, comps]).to_dict("records"):
        transiting = known[known.tic.eq(star["tic"]) & known.tran_flag.eq(1)
                           & known.pl_orbper.ge(CONFIG["period_min_days"])].pl_orbper
        planets = transiting[transiting.le(CONFIG["period_max_days"])]
        half_span = np.ptp(data_times(star)[1]) / 2
        jobs.append({"star": star, "kind": "catalog", "targets": list(planets),
                     "targets_wide": list(transiting[transiting.le(half_span)]),
                     "meta": {"group": star["catalog_group"], "n_planets": len(planets)}})

    # Same random draws, in the same order, as step 5's injection test.
    rng = np.random.default_rng(ablation.SEED)
    for star in comps.to_dict("records"):
        times, _ = data_times(star)
        span = np.ptp(times)
        for depth in ablation.INJECT_DEPTHS:
            period = float(np.exp(rng.uniform(np.log(0.8), np.log(min(10.0, span / 3)))))
            t0 = float(times.min() + rng.uniform(0, period))
            jobs.append({"star": star, "kind": "inj_standard", "fakes": [(period, t0, depth)], "targets": [period],
                         "meta": {"depth": depth, "period": period}})

    rng = np.random.default_rng(ablation.SEED + 1)
    for star in comps.to_dict("records"):
        times, usable = data_times(star)
        span = np.ptp(times)
        for depth in [0.0005, 0.001, 0.002]:
            while True:  # long period, but at least 2 dips must land on data
                period = float(np.exp(rng.uniform(np.log(8.0), np.log(span / 2))))
                t0 = float(times.min() + rng.uniform(0, period))
                if n_dips_with_data(usable, period, t0) >= 2:
                    break
            jobs.append({"star": star, "kind": "inj_long", "fakes": [(period, t0, depth)], "targets": [period],
                         "meta": {"depth": depth, "period": period}})

    rng = np.random.default_rng(ablation.SEED + 2)
    for star in comps.to_dict("records"):
        times, _ = data_times(star)
        span = np.ptp(times)
        draw = lambda: float(np.exp(rng.uniform(np.log(0.8), np.log(min(10.0, span / 3)))))
        for depth in [0.0005, 0.001, 0.002]:
            p_strong = draw()
            while True:  # keep the weak planet's period away from simple multiples of the strong one
                p_weak = draw()
                if all(abs(p_weak / (p_strong * m) - 1) > 0.05 for m in (1 / 3, 1 / 2, 1, 2, 3)):
                    break
            fakes = [(p_strong, float(times.min() + rng.uniform(0, p_strong)), 0.004),
                     (p_weak, float(times.min() + rng.uniform(0, p_weak)), depth)]
            jobs.append({"star": star, "kind": "inj_two", "fakes": fakes, "targets": [p_weak],
                         "meta": {"depth": depth, "period": p_weak, "strong_period": p_strong}})
    return jobs


def hits(df, v, cut=MIN_SDE):
    return df[f"match_sde_{v}"].ge(cut)


def report(df, quiet_tics):
    def paired(g, v, base="B", cut=MIN_SDE, base_cut=MIN_SDE):
        a, b = hits(g, base, base_cut), hits(g, v, cut)
        base_only, v_only, p = ablation.mcnemar(a, b)
        return f"{b.sum():>3}/{len(g):<4} (+{v_only} / -{base_only}, p={p:.2f})"

    cat = df[df.kind.eq("catalog")]
    hosts = cat[cat.group.eq("known_transiting_host") & cat.n_planets.gt(0)]
    comps = cat[cat.group.eq("no_cataloged_transiting_planet")]
    tests = {name: df[df.kind.eq(name)] for name in ["inj_standard", "inj_long", "inj_two"]}
    std = tests["inj_standard"]
    tests["  calm stars"] = std[std.tic.isin(quiet_tics)]
    tests["  active stars"] = std[~std.tic.isin(quiet_tics)]

    print("\nEach cell: found / tried (+gained vs B / -lost vs B, p = chance the difference is luck)")
    print(f"{'':<16}" + "".join(f"{v:<30}" for v in VARIANTS))
    for name, g in tests.items():
        print(f"{name:<16}" + "".join(f"{paired(g, v):<30}" for v in VARIANTS))
    print(f"{'known hosts':<16}" + "".join(f"{paired(hosts, v):<30}" for v in VARIANTS))
    print(f"{'false alarms':<16}" + "".join(
        f"{int(comps[f'top_sde_{v}'].ge(MIN_SDE).sum()):>3}/{len(comps):<26}" for v in VARIANTS))
    print("  (false alarms = comparison stars where any reported signal scores >= 7)")

    print("\nB2wide vs B2 (is the wider range worth it on top of the second search?):")
    for name, g in list(tests.items()) + [("known hosts", hosts)]:
        print(f"  {name:<16}{paired(g, 'B2wide', base='B2')}")
    print(f"  {'false alarms':<16}B2 {int(comps['top_sde_B2'].ge(MIN_SDE).sum())}/{len(comps)}"
          f" -> B2wide {int(comps['top_sde_B2wide'].ge(MIN_SDE).sum())}/{len(comps)}")

    print("\nScore cutoff sweep on B (B at cutoff 7 is the reference):")
    for cut in CUTOFFS:
        cells = [paired(g, "B", cut=cut) for g in [std, hosts]]
        alarms = int(comps["top_sde_B"].ge(cut).sum())
        print(f"  cutoff {cut}: fake planets {cells[0]}   known hosts {cells[1]}   false alarms {alarms}/{len(comps)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="3 stars per group, for a smoke test")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    targets = pd.read_csv(ROOT / "data" / "benchmark" / "targets.csv")
    known = pd.read_csv(ROOT / "data" / "benchmark" / "known_planets.csv")
    catalog = pd.read_csv(OUT_DIR / "ablation_catalog.csv")  # step 5's calm/active split
    quiet_tics = set(catalog[catalog.group.eq("no_cataloged_transiting_planet")
                             & ~catalog[[f"alarm_{c}" for c in "ABC"]].any(axis=1)].tic)

    jobs = build_jobs(targets, known, 3 if args.quick else None)
    print(f"{len(jobs)} light curves to search", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, row in enumerate(pool.map(run_job, jobs), 1):
            rows.append(row)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / ("fix_experiments_quick.csv" if args.quick else "fix_experiments.csv"), index=False)
    failed = {v: int(df[f"err_{v}"].astype(bool).sum()) for v in VARIANTS}
    print("Failed searches per variant:", failed)
    if failed["B"]:
        raise SystemExit(f"Baseline B failed on {failed['B']} light curves, e.g. "
                         f"{df.loc[df.err_B.astype(bool), 'err_B'].iloc[0]}")
    report(df, quiet_tics)


if __name__ == "__main__":
    main()

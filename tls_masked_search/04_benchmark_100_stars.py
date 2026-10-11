"""Step 4: run the masked two-pass TLS search on the repo's
100-star benchmark sample and compare recovery with the repo's existing BLS+TLS benchmark.

Same stars, same raw files, same catalog check as reports/benchmark. Only the pipeline differs:
  - flattening: lightkurve flatten (33 h window), re-run with the transits masked
  - search: TLS only, wide pass then fine pass
  - second signal: the first signal's transits are hidden and the search runs again
    (it finds the weaker planet in two-planet systems)
Catalog periods are used only AFTER each search finishes, to score it.

Usage (from tls_masked_search/; needs `python ../scripts/run_benchmark.py download` first):
    python 04_benchmark_100_stars.py --limit 3     # smoke test
    python 04_benchmark_100_stars.py               # all 100 stars (about an hour)
    python 04_benchmark_100_stars.py --tics 26826078 4070275 --events run.jsonl --out output/live
                                                   # two stars, recording events for the visualizer,
                                                   # CSVs kept out of output/ (not the full results)
"""
import argparse
import json
import traceback
from collections import Counter
from pathlib import Path

import lightkurve as lk
import pandas as pd

from common import (CONTEXT, MIN_SDE, OUT_DIR, RAW, ROOT, emit, events_to, hide_transits, relation,
                    star_kwargs, two_pass)

BENCH = ROOT / "config" / "benchmark.json"

def search_star(target, config):
    """Two-pass TLS on one star, then again with the first signal hidden. Returns a dict of results;
    signal 1 keeps the original column names, signal 2 columns are prefixed `s2_`."""
    lc = lk.read(RAW / target["filename"], flux_column="pdcsap_flux")
    emit("loaded", tic=target["tic"], n_points=len(lc), t_start=round(float(lc.time.value.min()), 4),
         t_end=round(float(lc.time.value.max()), 4))
    star = star_kwargs(float(target["mass"]), float(target["radius"]))
    out = {}
    for signal, prefix in [(1, ""), (2, "s2_")]:
        CONTEXT.update(tic=target["tic"], signal=signal)
        wide, fine = two_pass(lc, star, config)
        final = fine if fine is not None else wide
        out.update({f"{prefix}pass1_period": float(wide.period), f"{prefix}pass1_sde": float(wide.SDE),
                    f"{prefix}pass1_depth": float(1 - wide.depth), f"{prefix}pass1_duration_h": float(wide.duration * 24),
                    f"{prefix}period": float(final.period), f"{prefix}sde": float(wide.SDE),
                    f"{prefix}depth": float(1 - final.depth), f"{prefix}duration_h": float(final.duration * 24),
                    f"{prefix}refined": fine is not None})
        lc = hide_transits(lc, wide)
        if signal == 1:
            emit("hide", tic=target["tic"], signal=1, period=round(float(wide.period), 6),
                 t0=round(float(wide.T0), 4), duration_d=round(float(wide.duration), 6))
    return out


def target_info(target):
    """Where and what a star is, for the event log."""
    return {"tic": target["tic"], "hostname": target["hostname"], "ra": target["ra"], "dec": target["dec"],
            "sector": target["sector"], "tmag": target["tmag"], "teff": target["teff"], "radius": target["radius"],
            "mass": target["mass"], "group": target["catalog_group"], "radius_bin": target["radius_bin"]}


def outcome(row):
    """One word for how the star turned out (see the Mission Control contract)."""
    matches = [row.get(p + "match") for p in ["", "s2_"]]
    if row["status"] != "ok":
        return "error"
    if row["catalog_group"] == "known_transiting_host":
        return "recovered" if "exact" in matches else "harmonic" if any(matches) else "missed"
    return "alarm" if row["detected"] else "quiet"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only the first N stars")
    parser.add_argument("--tics", nargs="+", help="only these TIC IDs (space or comma separated)")
    parser.add_argument("--events", help="record what the pipeline does to this JSONL file (for the visualizer)")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="folder for the CSVs (default: output/)")
    args = parser.parse_args()

    config = json.loads(BENCH.read_text())
    targets = pd.read_csv(ROOT / "data" / "benchmark" / "targets.csv")
    known = pd.read_csv(ROOT / "data" / "benchmark" / "known_planets.csv")
    if args.tics:
        tics = {int(x) for arg in args.tics for x in arg.split(",") if x}
        if tics - set(targets.tic):
            parser.error(f"not in targets.csv: {sorted(tics - set(targets.tic))}")
        targets = targets[targets.tic.isin(tics)]
    if args.limit:
        targets = targets.head(args.limit)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.events:
        events_to(args.events)
    emit("run_start", config={"min_sde": MIN_SDE, **{k: config[k] for k in ["period_min_days", "period_max_days",
                                                                          "bin_minutes"]}},
         targets=[target_info(t) for t in targets.to_dict("records")])

    rows = []
    for i, target in enumerate(targets.to_dict("records"), 1):
        emit("star_start", **target_info(target), index=i, total=len(targets))
        row = {"tic": target["tic"], "hostname": target["hostname"],
               "catalog_group": target["catalog_group"], "radius_bin": target["radius_bin"]}
        try:
            row.update(search_star(target, config))
            row["status"] = "ok"
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
            print(traceback.format_exc(limit=2))
        # Scoring against the catalog happens only now, after the search is finished.
        planets = known[known.tic.eq(target["tic"]) & known.tran_flag.eq(1)
                        & known.pl_orbper.between(config["period_min_days"], config["period_max_days"])]
        row["n_catalog_planets"] = len(planets)
        for prefix in ["", "s2_"]:
            row[f"{prefix}match"] = None
            if row["status"] == "ok" and row[f"{prefix}sde"] >= MIN_SDE:
                matches = [relation(row[f"{prefix}period"], p) for p in planets.pl_orbper]
                row[f"{prefix}match"] = "exact" if "exact" in matches else next((m for m in matches if m), None)
        row["detected"] = bool(row["status"] == "ok" and max(row["sde"], row["s2_sde"]) >= MIN_SDE)
        rows.append(row)
        emit("verdict", tic=target["tic"], status=row["status"], **({"error": row["error"]} if "error" in row else {}),
             signals=[{"period": round(row[p + "period"], 6), "sde": round(row[p + "sde"], 3), "match": row[p + "match"]}
                      for p in ["", "s2_"] if p + "period" in row],
             catalog_periods=list(planets.pl_orbper), detected=row["detected"], outcome=outcome(row))
        print(f"{i}/{len(targets)} {row['hostname']}: {row['status']} "
              + " | ".join(f"P={row.get(p + 'period', float('nan')):.4f} SDE={row.get(p + 'sde', float('nan')):.1f} "
                           f"match={row[p + 'match']}" for p in ["", "s2_"]), flush=True)
        pd.DataFrame(rows).to_csv(args.out / "benchmark_100.csv", index=False)

    df = pd.DataFrame(rows)
    counts = Counter(outcome(r) for r in rows)
    hosts = int(df.catalog_group.eq("known_transiting_host").sum())
    emit("run_end", summary={"hosts": hosts, "recovered": counts["recovered"], "harmonic": counts["harmonic"],
                             "missed": counts["missed"], "comparison": len(df) - hosts, "alarms": counts["alarm"],
                             "errors": counts["error"]})
    summarize(df, config, args.out)


def repo_baseline(tic):
    """The repo benchmark's first-iteration TLS peak for a star (SDE, period), or None."""
    path = ROOT / "reports" / "benchmark" / "results" / f"tic{tic}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    if r.get("status") != "ok":
        return None
    best = r["signals"][0]["tls"]["best"]
    return best["period"], best["sde"]


def summarize(df, config, out):
    known = pd.read_csv(ROOT / "data" / "benchmark" / "known_planets.csv")
    hosts = df[df.catalog_group.eq("known_transiting_host") & df.n_catalog_planets.gt(0)].copy()
    # Same criterion for the repo's TLS signal 1: SDE >= 7 and period matches a catalog planet
    base_match = []
    for tic in hosts.tic:
        b = repo_baseline(tic)
        planets = known[known.tic.eq(tic) & known.tran_flag.eq(1)
                        & known.pl_orbper.between(config["period_min_days"], config["period_max_days"])]
        base_match.append(bool(b and b[1] >= MIN_SDE and any(relation(b[0], p) == "exact" for p in planets.pl_orbper)))
    hosts["repo_tls_exact"] = base_match
    hosts["ours_exact_s1"] = hosts.match.eq("exact")
    hosts["ours_exact"] = hosts.ours_exact_s1 | hosts.s2_match.eq("exact")

    print("\n=== Recovery of known transiting hosts (exact period within 1%, SDE >= 7) ===")
    print(f"{'group':<8}{'ours, either signal':>22}{'ours, signal 1':>16}{'repo TLS signal 1':>20}")
    for name, g in [("all", hosts)] + list(hosts.groupby("radius_bin")):
        print(f"{name:<8}{g.ours_exact.sum():>16}/{len(g):<5}{g.ours_exact_s1.sum():>10}/{len(g):<5}"
              f"{g.repo_tls_exact.sum():>14}/{len(g)}")
    comp = df[df.catalog_group.eq("no_cataloged_transiting_planet")]
    print(f"\nComparison stars (no cataloged transiting planet) with any signal SDE >= 7: "
          f"{comp.detected.sum()}/{len(comp)} (signal 1 only: {comp.sde.ge(MIN_SDE).sum()})")
    print(f"Errors: {(df.status != 'ok').sum()}")
    hosts.to_csv(out / "benchmark_100_hosts.csv", index=False)
    print(f"\nSaved {out / 'benchmark_100.csv'} and benchmark_100_hosts.csv")


if __name__ == "__main__":
    main()

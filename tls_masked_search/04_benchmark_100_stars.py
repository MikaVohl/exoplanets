"""Step 4: run the WASP-18 method (TLS, mask transits, re-flatten, fine TLS) on the repo's
100-star benchmark sample and compare recovery with the repo's existing BLS+TLS benchmark.

Same stars, same raw files, same catalog check as reports/benchmark. Only the pipeline differs:
  - flattening: lightkurve flatten (33 h window), re-run with the transits masked
  - search: TLS only, wide pass then fine pass
Catalog periods are used only AFTER each search finishes, to score it.

Usage (from tls_masked_search/; needs `python ../scripts/run_benchmark.py download` first):
    python 04_benchmark_100_stars.py --limit 3     # smoke test
    python 04_benchmark_100_stars.py               # all 100 stars (about an hour)
"""
import argparse
import json
import traceback

import lightkurve as lk
import pandas as pd

from common import MIN_SDE, OUT_DIR, RAW, ROOT, relation, star_kwargs, two_pass

BENCH = ROOT / "config" / "benchmark.json"

def search_star(target, config):
    """Two-pass TLS on one star. Returns a dict of results."""
    lc = lk.read(RAW / target["filename"], flux_column="pdcsap_flux")
    star = star_kwargs(float(target["mass"]), float(target["radius"]))
    wide, fine = two_pass(lc, star, config)
    out = {"pass1_period": float(wide.period), "pass1_sde": float(wide.SDE),
           "pass1_depth": float(1 - wide.depth), "pass1_duration_h": float(wide.duration * 24)}
    final = fine if fine is not None else wide
    out.update(period=float(final.period), sde=out["pass1_sde"], depth=float(1 - final.depth),
               duration_h=float(final.duration * 24), refined=fine is not None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only the first N stars")
    args = parser.parse_args()

    config = json.loads(BENCH.read_text())
    targets = pd.read_csv(ROOT / "data" / "benchmark" / "targets.csv")
    known = pd.read_csv(ROOT / "data" / "benchmark" / "known_planets.csv")
    if args.limit:
        targets = targets.head(args.limit)
    OUT_DIR.mkdir(exist_ok=True)

    rows = []
    for i, target in enumerate(targets.to_dict("records"), 1):
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
        row["match"] = None
        if row["status"] == "ok" and row["sde"] >= MIN_SDE:
            matches = [relation(row["period"], p) for p in planets.pl_orbper]
            row["match"] = "exact" if "exact" in matches else next((m for m in matches if m), None)
        row["detected"] = bool(row["status"] == "ok" and row["sde"] >= MIN_SDE)
        rows.append(row)
        print(f"{i}/{len(targets)} {row['hostname']}: {row['status']} "
              f"P={row.get('period', float('nan')):.4f} SDE={row.get('sde', float('nan')):.1f} match={row['match']}",
              flush=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / "benchmark_100.csv", index=False)

    df = pd.DataFrame(rows)
    summarize(df, config)


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


def summarize(df, config):
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
    hosts["ours_exact"] = hosts.match.eq("exact")

    print("\n=== Recovery of known transiting hosts (exact period within 1%, SDE >= 7, first signal) ===")
    print(f"{'group':<8}{'ours (masked 2-pass TLS)':>26}{'repo TLS signal 1':>20}")
    for name, g in [("all", hosts)] + list(hosts.groupby("radius_bin")):
        print(f"{name:<8}{g.ours_exact.sum():>20}/{len(g):<5}{g.repo_tls_exact.sum():>14}/{len(g)}")
    comp = df[df.catalog_group.eq("no_cataloged_transiting_planet")]
    print(f"\nComparison stars (no cataloged transiting planet) with SDE >= 7: "
          f"ours {comp.detected.sum()}/{len(comp)}")
    print(f"Errors: {(df.status != 'ok').sum()}")
    hosts.to_csv(OUT_DIR / "benchmark_100_hosts.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'benchmark_100.csv'} and benchmark_100_hosts.csv")


if __name__ == "__main__":
    main()

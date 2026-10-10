"""Catalog snapshots and deterministic, availability-aware host selection."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.table import Table
from astroquery.mast import Observations

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
QUERY = ("select pl_name,hostname,tic_id,tran_flag,pl_orbper,pl_tranmid,pl_trandur,"
         "pl_trandep,pl_rade,pl_bmasse,st_rad,st_mass,st_teff,sy_tmag,ra,dec,"
         "sy_pnum,discoverymethod,disc_year from pscomppars where tic_id is not null")


def catalog_frames(path):
    planets = pd.read_csv(path)
    planets["tic"] = planets.tic_id.str.extract(r"(\d+)").astype("int64")
    stars = planets.groupby("tic").agg(
        hostname=("hostname", "first"), transit=("tran_flag", "max"),
        tmag=("sy_tmag", "first"), radius=("st_rad", "first"),
        mass=("st_mass", "first"), teff=("st_teff", "first"),
        ra=("ra", "first"), dec=("dec", "first"),
        known_planet_count=("pl_name", "size"),
    )
    return planets, stars


def snapshots(root, refresh=False):
    folder = root / "data" / "catalogs"
    folder.mkdir(parents=True, exist_ok=True)
    planet_path = folder / "confirmed_planets.csv"
    if refresh or not planet_path.exists():
        response = requests.get(TAP, params={"query": QUERY, "format": "csv"}, timeout=120)
        response.raise_for_status()
        if not response.text.startswith("pl_name,"):
            raise ValueError("Unexpected Exoplanet Archive response")
        planet_path.write_text(response.text)
    planets, stars = catalog_frames(planet_path)
    stars = stars[stars.tmag.between(4, 11.5) & stars.radius.between(.15, 2.5)
                  & stars.mass.between(.08, 2.5)]
    short = planets[planets.tran_flag.eq(1) & planets.pl_orbper.between(.5, 12)
                    & planets.pl_rade.notna()]
    positives = stars[stars.transit.eq(1) & stars.index.isin(short.tic)].sample(frac=1, random_state=42).head(250)
    comparisons = stars[stars.transit.eq(0)].sample(frac=1, random_state=43).head(200)
    candidates = pd.concat([positives, comparisons]).reset_index()
    candidates.to_csv(folder / "host_candidates.csv", index=False)
    mast_path = folder / "mast_observations.ecsv"
    if refresh or not mast_path.exists():
        print(f"Querying MAST for {len(candidates)} cataloged planet hosts", flush=True)
        Observations.TIMEOUT = 120
        observations = Observations.query_criteria(
            obs_collection="TESS", provenance_name="SPOC", dataproduct_type="timeseries",
            t_exptime=[119, 121], target_name=[str(x) for x in candidates.tic],
        )
        observations.write(mast_path, format="ascii.ecsv", overwrite=True)
    return planets, stars, Table.read(mast_path, format="ascii.ecsv")


def select_targets(root, config, refresh=False):
    planets, stars, observations = snapshots(root, refresh)
    # Only public, single-target, two-minute SPOC light-curve observations.
    rows = []
    for row in observations:
        uri = str(row["dataURL"])
        if not uri.endswith("_lc.fits") or str(row["dataRights"]).upper() != "PUBLIC":
            continue
        rows.append({"tic": int(row["target_name"]), "sector": int(row["sequence_number"]),
                     "source_uri": uri, "filename": Path(uri).name,
                     "obs_id": str(row["obs_id"]), "mjd_start": float(row["t_min"]),
                     "mjd_end": float(row["t_max"])})
    available = pd.DataFrame(rows).sort_values(["tic", "mjd_end", "filename"])
    latest = available.drop_duplicates("tic", keep="last").set_index("tic")
    eligible = stars.join(latest, how="inner")
    short = planets[planets.tran_flag.eq(1) & planets.pl_orbper.between(
        config["period_min_days"], config["period_max_days"]) & planets.pl_rade.notna()]
    # Stratify by the largest cataloged short-period transiting planet per host.
    # This is a selection label only; it is never passed to a search algorithm.
    sizes = short.groupby("tic").pl_rade.max()
    eligible["selection_planet_radius_earth"] = sizes
    eligible["radius_bin"] = pd.cut(sizes, [0, 3, 8, np.inf], labels=["small", "medium", "giant"])
    selected = []
    for i, label in enumerate(["small", "medium", "giant"]):
        pool = eligible[eligible.transit.eq(1) & eligible.radius_bin.eq(label)]
        count = config["transiting_hosts_per_radius_bin"]
        if len(pool) < count:
            raise ValueError(f"Only {len(pool)} available hosts in {label} bin; need {count}")
        sample = pool.sort_index().sample(n=count, random_state=config["seed"] + i).copy()
        sample["catalog_group"] = "known_transiting_host"
        selected.append(sample)
    comparisons = eligible[eligible.transit.eq(0)].sort_index().sample(
        n=config["comparison_hosts"], random_state=config["seed"] + 3).copy()
    comparisons["catalog_group"] = "no_cataloged_transiting_planet"
    comparisons["radius_bin"] = "comparison"
    selected.append(comparisons)
    targets = pd.concat(selected).reset_index().sort_values(["catalog_group", "tic"])
    assert len(targets) == config["transiting_hosts"] + config["comparison_hosts"]
    assert targets.tic.is_unique
    folder = root / "data" / "benchmark"
    folder.mkdir(parents=True, exist_ok=True)
    targets.to_csv(folder / "targets.csv", index=False)
    planets[planets.tic.isin(targets.tic)].to_csv(folder / "known_planets.csv", index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "planet_catalog": "NASA Exoplanet Archive pscomppars (confirmed planets)",
        "tap_endpoint": TAP, "tap_query": QUERY,
        "candidate_hosts": 450, "available_candidate_hosts": len(eligible),
        "selected_hosts": len(targets), "seed": config["seed"],
        "catalog_groups": targets.catalog_group.value_counts().to_dict(),
        "selection": "Tmag 4–11.5, radius 0.15–2.5 solar, mass 0.08–2.5 solar; public two-minute SPOC coverage. 20 hosts each with largest short-period planet ≤3, 3–8, >8 Earth radii; 40 hosts with no cataloged transiting planet. Latest available sector by archive observation end time. No selection on search outcomes.",
        "limitations": ["Comparison hosts are not proven transit-free.",
                         "Brightness/radius distributions are not matched between groups.",
                         "Availability and known-planet selection introduce selection bias.",
                         "Planet-table stellar parameters may combine heterogeneous references."],
        "snapshot_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [root / "data/catalogs/confirmed_planets.csv",
                                      root / "data/catalogs/mast_observations.ecsv"]},
    }
    (folder / "selection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Selected {len(targets)} stars: {targets.catalog_group.value_counts().to_dict()}", flush=True)
    return targets

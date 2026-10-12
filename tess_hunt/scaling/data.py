"""Expand the real labeled cohort using batched MAST product inventories."""
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from astroquery.mast import ObservationsClass

from tess_hunt.learning.dataset import ROOT, load_config, select_stars, save_json, prepare, catalog_snapshot

LC_PATTERN = re.compile(r"^tess\d+-s(\d{4})-(\d{16})-\d+-s_lc\.fits$")


def inventory_batch(tics):
    """Exact TIC queries avoid resolving names into nearby, unrelated targets."""
    client = ObservationsClass()
    observations = client.query_criteria(obs_collection="TESS", dataproduct_type="timeseries",
                                         target_name=[str(tic) for tic in tics])
    grouped = {tic: [] for tic in tics}
    if len(observations):
        # SPOC short cadence observations; fast cadence ends with '-a_fast'.
        observations = observations[[str(value).endswith("-s") for value in observations["obs_id"]]]
    if len(observations):
        products = client.get_product_list(observations)
        for row in products:
            name = str(row["productFilename"])
            match = LC_PATTERN.match(name)
            if match is None:
                continue
            sector, tic = int(match[1]), int(match[2])
            if tic in grouped:
                grouped[tic].append({"uri": str(row["dataURI"]), "filename": name, "sector": sector})
    directory = ROOT / "data/raw/learning"
    directory.mkdir(parents=True, exist_ok=True)
    for tic, products in grouped.items():
        # Duplicate records sometimes reference the same product.
        products = list({p["filename"]: p for p in products}.values())
        save_json(directory / f"tic{tic}_products.json", products)
    return {"requested_stars": len(tics), "stars_with_products": sum(bool(p) for p in grouped.values())}


def expand_dataset(run, source, per_class=512, workers=4, download_workers=16):
    run.mkdir(parents=True, exist_ok=True)
    config = load_config(source / "config.json")
    # The same catalog and preprocessing make sample-size changes interpretable.
    for name in ("catalog.csv", "catalog_source.json"):
        if not (run / name).exists():
            shutil.copyfile(source / name, run / name)
    catalog = catalog_snapshot(run)
    selected = select_stars(catalog, config, per_class)
    missing = [s["tic"] for s in selected if not (ROOT / "data/raw/learning" / f"tic{s['tic']}_products.json").exists()]
    batches = [missing[start:start + 40] for start in range(0, len(missing), 40)]
    summaries = []
    with ThreadPoolExecutor(4) as pool:
        futures = {pool.submit(inventory_batch, tics): tics for tics in batches}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                summary = future.result()
                summaries.append(summary)
                print(f"MAST inventory batch {i}/{len(batches)}: {summary}", flush=True)
            except Exception as exc:
                # prepare falls back to the existing per-star query/retry path.
                summaries.append({"requested_stars": len(futures[future]), "error": repr(exc)})
                print(f"MAST inventory batch failed, per-star fallback: {exc}", flush=True)
    save_json(run / "inventory_batches.json", summaries)
    # Reuse earlier per-star numerical artifacts only if prepare's complete key agrees.
    cached = run / "stars"
    cached.mkdir(exist_ok=True)
    for path in (source / "stars").glob("*"):
        if not (cached / path.name).exists():
            shutil.copyfile(path, cached / path.name)
    return prepare(run, config, per_class, 1, download_workers, workers)

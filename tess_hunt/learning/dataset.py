"""Reproducible MAST acquisition and cached blind searches of labeled hosts."""
import hashlib
import importlib.metadata
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd
import requests

from scripts.download_tess import validate
from .labels import match_signal, partition
from .photometry import read_raw, clean, make_views
from .search import find_signals

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash():
    # Reports do not change numerical artifacts; pipeline sources do.
    paths = [p for p in sorted(Path(__file__).parent.glob("*.py")) if p.name != "report.py"]
    paths += [ROOT / "tess_hunt/search.py", ROOT / "tess_hunt/photometry.py"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "astropy", "lightkurve", "transitleastsquares", "scipy", "torch", "scikit-learn", "batman-package")}


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config["search_method"] not in ("tls", "bls"):
        raise ValueError("search_method must be tls or bls")
    if not 0 < config["train_fraction"] < config["train_fraction"] + config["validation_fraction"] < 1:
        raise ValueError("Invalid train/validation/test fractions")
    if set(config["positive_dispositions"]) - {"CP", "KP"} or set(config["negative_dispositions"]) - {"FP", "FA"}:
        raise ValueError("Only reliable CP/KP positives and FP/FA negatives are supported")
    if not config["positive_dispositions"] or not config["negative_dispositions"]:
        raise ValueError("Both label classes must be configured")
    if not 0 < config["period_min_days"] < config["period_max_days"]:
        raise ValueError("Invalid period range")
    if not 0 <= config["target_validation_fpr"] < 1:
        raise ValueError("Invalid validation false-positive rate")
    for key in ("maximum_signals", "bin_minutes", "flatten_window_days", "global_bins", "local_bins", "event_bins",
                "maximum_events", "width", "epochs", "patience", "batch_size", "torch_threads"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    return config


def catalog_snapshot(run, refresh=False):
    path = run / "catalog.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    cached = ROOT / "data/dataset/toi_labeled.csv"
    if cached.exists() and not refresh:
        path.write_bytes(cached.read_bytes())
        save_json(run / "catalog_source.json", {"source": str(cached.relative_to(ROOT)),
                                                "sha256": sha256(cached),
                                                "note": "Existing snapshot; original acquisition date not recorded"})
    else:
        query = ("select tid,toi,tfopwg_disp,pl_orbper,pl_tranmid,pl_trandurh,pl_trandep,"
                 "st_tmag,st_rad,st_teff,st_logg,toi_created,rowupdate from toi")
        response = requests.get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                                params={"query": query, "format": "csv"}, timeout=120)
        response.raise_for_status()
        path.write_text(response.text)
        save_json(run / "catalog_source.json", {"source": response.url, "sha256": sha256(path),
                                                "retrieved_unix": time.time()})
    return pd.read_csv(path)


def select_stars(catalog, config, per_class):
    """Select confirmed and false-signal hosts; mixed hosts are allowed.

    Host cohort affects sampling only. It never labels another signal on that host.
    Reserve the existing benchmark for independent discovery runs.
    """
    reserved_path = ROOT / "data/benchmark/targets.csv"
    reserved = set(pd.read_csv(reserved_path).tic) if reserved_path.exists() else set()
    valid = catalog[catalog.pl_orbper.between(config["period_min_days"], config["period_max_days"])
                    & catalog.pl_tranmid.notna()]
    rng = np.random.default_rng(config["seed"])
    selected, used = [], set(reserved)
    for cohort, dispositions in [("positive", config["positive_dispositions"]),
                                  ("negative", config["negative_dispositions"])]:
        tics = sorted(set(valid[valid.tfopwg_disp.isin(dispositions)].tid.astype(int)) - used)
        if not tics:
            raise ValueError(f"No eligible {cohort} hosts")
        chosen = rng.permutation(tics)[:per_class]
        selected.extend({"tic": int(tic), "cohort": cohort, "split": partition(tic, config)} for tic in chosen)
        used.update(chosen)
    return selected


def acquire_star(selection, maximum_sectors):
    tic = selection["tic"]
    # lightkurve search cache plus our product list makes repeat runs offline.
    raw_dir = ROOT / "data/raw/learning"
    raw_dir.mkdir(parents=True, exist_ok=True)
    inventory = raw_dir / f"tic{tic}_products.json"
    if inventory.exists():
        products = json.loads(inventory.read_text())
    else:
        found = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="SPOC", exptime=120)
        products = [{"uri": str(row["dataURI"]), "filename": str(row["productFilename"]),
                     "sector": int(row["sequence_number"])} for row in found.table]
        save_json(inventory, products)
    products = sorted(products, key=lambda p: (p["sector"], p["filename"]), reverse=True)
    # One version per sector.
    picked, sectors = [], set()
    for product in products:
        if product["sector"] not in sectors:
            picked.append(product)
            sectors.add(product["sector"])
        if len(picked) >= maximum_sectors:
            break
    result = {**selection, "sectors_available": len({p["sector"] for p in products}), "files": []}
    with requests.Session() as session:
        for product in picked:
            path = raw_dir / Path(product["filename"]).name
            if not path.exists():
                temporary = path.with_suffix(".fits.part")
                try:
                    with session.get("https://mast.stsci.edu/api/v0.1/Download/file",
                                     params={"uri": product["uri"]}, stream=True, timeout=(20, 120)) as response:
                        response.raise_for_status()
                        with temporary.open("wb") as output:
                            for chunk in response.iter_content(1024 * 1024):
                                output.write(chunk)
                    validate(temporary, tic, product["sector"])
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            metadata = validate(path, tic, product["sector"])
            result["files"].append({**metadata, "path": str(path.relative_to(ROOT)),
                                    "source_uri": product["uri"], "sha256": sha256(path)})
    return result


def search_job(job):
    """Per sector search avoids a long, gappy baseline; outputs include every signal."""
    entry, config, catalog, run, fingerprint = job
    cache = run / "stars" / f"tic{entry['tic']}"
    meta = cache.with_suffix(".json")
    arrays_path = cache.with_suffix(".npz")
    key = hashlib.sha256((fingerprint + json.dumps(entry, sort_keys=True)).encode()).hexdigest()
    if meta.exists() and arrays_path.exists():
        saved = json.loads(meta.read_text())
        if saved["cache_key"] == key:
            return saved
    rows, views, recovery = [], [], []
    star_catalog = catalog[catalog.tid.eq(entry["tic"])]
    for product in entry["files"]:
        raw = read_raw(ROOT / product["path"])
        data = clean(raw, config)
        signals = find_signals(data, config)  # no catalog periods, epochs, dispositions or stellar priors
        for signal in signals:
            row = {**signal, "tic": entry["tic"], "sector": raw["sector"], "split": entry["split"],
                   "source_path": product["path"], "source_sha256": product["sha256"],
                   **match_signal(signal, star_catalog, data["time"], config)}
            rows.append(row)
            views.append(make_views(data, signal, config))
        for item in star_catalog.to_dict("records"):
            period, epoch = item.get("pl_orbper", np.nan), item.get("pl_tranmid", np.nan)
            if item["tfopwg_disp"] not in config["positive_dispositions"] + config["negative_dispositions"]:
                continue
            eligible = bool(np.isfinite(period) and config["period_min_days"] <= period <=
                            min(config["period_max_days"], np.ptp(data["time"]) / 2))
            observed = 0
            if eligible and np.isfinite(epoch):
                from tess_hunt.search import phase_offset
                duration = item.get("pl_trandurh", np.nan) / 24
                duration = duration if np.isfinite(duration) and duration > 0 else .1
                inside = np.abs(phase_offset(data["time"], period, epoch - 2457000)) < duration / 2
                observed = len(np.unique(np.rint((data["time"][inside] - (epoch - 2457000)) / period)))
            recovery.append({"tic": entry["tic"], "sector": raw["sector"], "split": entry["split"],
                             "toi": str(item["toi"]), "label": item["tfopwg_disp"],
                             "in_search_range": eligible, "observed_catalog_events": observed,
                             "recovered": any(r.get("label_toi") == str(item["toi"]) for r in rows
                                              if r["sector"] == raw["sector"])})
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not views:
        raise ValueError("No candidates found")
    np.savez_compressed(arrays_path, **{k: np.stack([v[k] for v in views]) for k in views[0]})
    saved = {"cache_key": key, "rows": rows, "recovery": recovery}
    # Vetting intentionally returns NaN for unavailable diagnostics. CSV keeps them missing.
    meta.write_text(json.dumps(saved, indent=2) + "\n")
    return saved


def prepare(run, config, per_class, maximum_sectors, download_workers, workers, refresh=False):
    run.mkdir(parents=True, exist_ok=True)
    config_path = run / "config.json"
    if config_path.exists() and load_config(config_path) != config:
        raise ValueError("Run configuration differs. Use a new --run directory for a new experiment.")
    save_json(config_path, config)
    catalog = catalog_snapshot(run, refresh)
    selection = select_stars(catalog, config, per_class)
    save_json(run / "selection.json", selection)
    entries, failures = [], []
    def download_with_retry(item):
        for attempt in range(3):
            try:
                return acquire_star(item, maximum_sectors)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(download_workers) as pool:
        futures = {pool.submit(download_with_retry, item): item for item in selection}
        for i, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                entry = future.result()
                entries.append(entry)
                print(f"download {i}/{len(selection)} TIC {item['tic']}: {len(entry['files'])} sectors", flush=True)
            except Exception as exc:
                failures.append({"tic": item["tic"], "stage": "download", "error": repr(exc)})
                print(f"download TIC {item['tic']} failed: {exc}", flush=True)
    entries.sort(key=lambda e: e["tic"])
    save_json(run / "manifest.json", entries)
    fingerprint = hashlib.sha256((json.dumps(config, sort_keys=True) + sha256(run / "catalog.csv") + code_hash()
                                  + json.dumps(versions(), sort_keys=True)).encode()).hexdigest()
    completed = []
    jobs = [(e, config, catalog, run, fingerprint) for e in entries if e["files"]]
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(search_job, job): job[0]["tic"] for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            tic = futures[future]
            try:
                completed.append((tic, future.result()))
                print(f"search {i}/{len(jobs)} TIC {tic}: {len(completed[-1][1]['rows'])} signals", flush=True)
            except Exception as exc:
                failures.append({"tic": tic, "stage": "search", "error": repr(exc)})
                print(f"search TIC {tic} failed: {exc}", flush=True)
    save_json(run / "failures.json", failures)
    if not completed:
        raise RuntimeError("No successful searches; see failures.json")
    completed.sort(key=lambda pair: pair[0])
    rows, recovery, parts = [], [], []
    for tic, saved in completed:
        rows.extend(saved["rows"])
        recovery.extend(saved["recovery"])
        with np.load(run / "stars" / f"tic{tic}.npz") as arrays:
            parts.append({k: arrays[k] for k in arrays.files})
    df = pd.DataFrame(rows)
    df.insert(0, "candidate_id", [f"{r.tic}-s{r.sector}-{r.signal}" for r in df.itertuples()])
    df.to_csv(run / "candidates.csv", index=False)
    pd.DataFrame(recovery).to_csv(run / "catalog_recovery.csv", index=False)
    np.savez_compressed(run / "views.npz", **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    save_json(run / "dataset_provenance.json", {"fingerprint": fingerprint, "code_hash": code_hash(),
                                              "catalog_sha256": sha256(run / "catalog.csv"),
                                              "candidate_sha256": sha256(run / "candidates.csv"),
                                              "views_sha256": sha256(run / "views.npz"), "versions": versions(),
                                              "photometry": "SPOC PDCSAP 120-second; one search per sector",
                                              "successful_stars": len(completed), "failed_stages": len(failures)})
    print(df.groupby(["split", "label"]).size().to_string(), flush=True)
    return df

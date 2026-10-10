"""Resumable benchmark orchestration; catalog evaluation follows blind searches."""

import hashlib
import json
import platform
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .photometry import download_target, preprocess
from .search import bls_search, tls_search, conclude, phase_offset, evaluate_catalog


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(content), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def download_sample(root, targets, workers=4):
    folder = root / "data" / "benchmark"
    entries, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_target, root, t): t for t in targets.to_dict("records")}
        for index, future in enumerate(as_completed(futures), 1):
            target = futures[future]
            try:
                entries.append(future.result())
            except Exception as error:
                failures.append({"tic": int(target["tic"]), "hostname": target["hostname"], "error": str(error)})
                print(f"Download failed: {target['hostname']}: {error}", flush=True)
            write_json(folder / "download_manifest.json", {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "files": sorted(entries, key=lambda e: e["tic_id"]), "failures": failures})
            if index % 5 == 0 or index == len(futures):
                print(f"Downloads checked: {index}/{len(futures)}; {len(entries)} successful, {len(failures)} failed", flush=True)
    return entries, failures


def fingerprint(root, config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for name in ["photometry.py", "search.py", "benchmark.py"]:
        digest.update((root / "tess_hunt" / name).read_bytes())
    for package in ["astropy", "transitleastsquares", "numpy", "scipy"]:
        digest.update(f"{package}={version(package)}".encode())
    return digest.hexdigest()


def search_target(root_text, target, config, run_fingerprint, force=False):
    root = Path(root_text)
    tic = int(target["tic"])
    result_path = root / "reports" / "benchmark" / "results" / f"tic{tic}.json"
    file_path = root / "data" / "raw" / "benchmark" / target["filename"]
    file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest() if file_path.exists() else None
    if not force and result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached.get("fingerprint") == run_fingerprint and cached.get("source_sha256") == file_hash and cached.get("status") == "ok":
            return cached
    result = {"tic": tic, "hostname": target["hostname"], "sector": int(target["sector"]),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "fingerprint": run_fingerprint, "source_sha256": file_hash,
              "source_uri": target["source_uri"], "config": config,
              "versions": {p: version(p) for p in ["astropy", "transitleastsquares", "numpy", "scipy", "lightkurve"]},
              "python": platform.python_version()}
    try:
        data, metadata = preprocess(file_path, config)
        result["preprocessing"] = metadata
        processed = root / "data" / "processed" / "benchmark" / f"tic{tic}.npz"
        processed.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(processed, **data)
        spectra_folder = root / "reports" / "benchmark" / "spectra"
        spectra_folder.mkdir(parents=True, exist_ok=True)
        signals = []
        working = {k: v.copy() for k, v in data.items() if not k.startswith("unbinned")}
        # Stellar mass/radius are permissible physical priors. No planet periods,
        # epochs, catalog labels, or planet radii enter either search.
        stellar = {"mass": float(target["mass"]), "radius": float(target["radius"])}
        for iteration in range(1, config["maximum_signals_per_star"] + 1):
            bls, bls_spectrum = bls_search(working, stellar, config)
            tls, tls_spectrum = tls_search(working, stellar, config)
            decision = conclude(bls, tls, config)
            signal = {"iteration": iteration, "bls": bls, "tls": tls, "decision": decision}
            signals.append(signal)
            np.savez_compressed(spectra_folder / f"tic{tic}_signal{iteration}.npz",
                                bls_period=bls_spectrum["periods"], bls_power=bls_spectrum["score"],
                                tls_period=tls_spectrum["periods"], tls_power=tls_spectrum["power"],
                                tls_model_phase=tls_spectrum["model_phase"], tls_model_flux=tls_spectrum["model_flux"])
            if not (decision["bls_supported"] or decision["tls_supported"]):
                break
            mask = abs(phase_offset(working["time"], decision["candidate_period"],
                                    decision["candidate_epoch_btjd"])) > .75 * decision["candidate_duration_days"]
            if mask.sum() < 300:
                break
            working = {k: v[mask] for k, v in working.items()}
        result["status"] = "ok"
        result["signals"] = signals
        ranked = sorted(signals, key=lambda s: {
            "supported_transit_like_signal": 0, "signal_requires_review": 1,
            "no_significant_signal_in_search_range": 2}[s["decision"]["conclusion"]])
        result["conclusion"] = ranked[0]["decision"]["conclusion"]
        result["reason"] = ranked[0]["decision"]["reason"]
    except Exception as error:
        result["status"] = "insufficient_data" if isinstance(error, ValueError) and str(error).startswith(("Insufficient", "No usable")) else "error"
        result["conclusion"] = result["status"]
        result["reason"] = str(error)
        result["traceback"] = traceback.format_exc()
    # Catalog comparisons happen AFTER the independent search results are fixed.
    result["catalog_group"] = target["catalog_group"]
    result["radius_bin"] = target["radius_bin"]
    result["stellar_priors"] = {"mass_solar": target["mass"], "radius_solar": target["radius"]}
    known = pd.read_csv(root / "data" / "benchmark" / "known_planets.csv")
    result["known_planets"] = json_safe(known[known.tic.eq(tic)].to_dict("records"))
    if result["status"] == "ok":
        result["catalog_evaluation"] = evaluate_catalog(result["signals"], result["known_planets"], config)
    write_json(result_path, result)
    return json_safe(result)


def run_searches(root, targets, config, workers=2, force=False):
    run_id = fingerprint(root, config)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(search_target, str(root), t, config, run_id, force): t
                   for t in targets.to_dict("records")}
        for index, future in enumerate(as_completed(futures), 1):
            target = futures[future]
            try:
                result = future.result()
            except Exception as error:
                raise RuntimeError(f"Worker failed for {target['hostname']}; completed per-star files are preserved") from error
            results.append(result)
            if result["status"] == "ok":
                first = result["signals"][0]
                print(f"Search {index}/{len(futures)} {result['hostname']}: {result['conclusion']}; "
                      f"BLS P={first['bls']['best']['period']:.4f}, TLS P={first['tls']['best']['period']:.4f}", flush=True)
            else:
                print(f"Search {index}/{len(futures)} {result['hostname']}: {result['status']}: {result['reason']}", flush=True)
    write_json(root / "reports" / "benchmark" / "run_metadata.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": run_id,
        "config": config, "target_count": len(targets), "results_count": len(results),
        "workers": workers, "tls_threads_per_worker": config["tls_threads_per_worker"]})
    return results

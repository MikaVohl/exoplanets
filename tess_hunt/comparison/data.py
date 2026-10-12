"""Observational views; catalog labels never enter a representation function."""
import hashlib
import importlib
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.io import fits
from lightkurve.utils import TessQualityFlags

from tess_hunt.learning.dataset import ROOT, code_hash, save_json, sha256, versions
from tess_hunt.learning.photometry import read_raw, clean, make_views, binned_view
from tess_hunt.photometry import robust_sigma
from tess_hunt.search import phase_offset, vet_signal

OBSERVATIONAL_FIELDS = ("period", "t0", "duration_days", "search_sde", "depth_ppm", "depth_snr",
                        "pink_snr", "observed_transits", "significant_events", "odd_even_sigma", "secondary_snr")
LEGACY_FEATURES = ("period", "duration_h", "depth_ppm", "sde_per_transit", "depth_snr_per_transit",
                   "pink_snr_per_transit", "odd_even_sigma", "secondary_snr", "centroid_snr",
                   "transits_per_sector", "significant_event_fraction", "chunk_fraction")


def representation_hash():
    files = [Path(__file__), ROOT / "tls_masked_search/07_build_views.py",
             ROOT / "tls_masked_search/common.py", ROOT / "tls_masked_search/05_build_dataset.py"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def legacy_module():
    directory = str(ROOT / "tls_masked_search")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module("07_build_views")


def detrend(raw, signal, config, *, mask_transits=True, window_factor=1.0):
    """Same quality/gap rules as learning.clean, with a candidate-only transit mask.

    Centroids have pixel units and require subtraction, rather than flux division.
    """
    t, y, dy = (np.asarray(raw[k]) for k in ("time", "flux", "error"))
    keep = (np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (y > 0) & (dy > 0)
            & ((raw["quality"] & TessQualityFlags.DEFAULT_BITMASK) == 0))
    t, y, dy = (a[keep] for a in (t, y, dy))
    order = np.argsort(t)
    t, y, dy = (a[order] for a in (t, y, dy))
    unique = np.r_[True, np.diff(t) > 0]
    t, y, dy = (a[unique] for a in (t, y, dy))
    if len(t) < config["minimum_points"]:
        raise ValueError("Insufficient usable photometry")
    edges = np.r_[0, np.flatnonzero(np.diff(t) > config["split_gap_days"]) + 1, len(t)]
    parts = []
    for start, end in zip(edges[:-1], edges[1:]):
        st, sy, se = (a[start:end] for a in (t, y, dy))
        if len(st) < 100:
            continue
        window = max(5, int(config["flatten_window_days"] * window_factor / np.median(np.diff(st))) | 1)
        window = min(window, len(st) if len(st) % 2 else len(st)-1)
        mask = (np.abs(phase_offset(st, signal["period"], signal["t0"])) < signal["duration_days"]
                if mask_transits else None)
        flat = lk.LightCurve(time=st, flux=sy, flux_err=se).normalize().flatten(window_length=window, mask=mask)
        fy, fe = (np.asarray(a.value, float) for a in (flat.flux, flat.flux_err))
        noise = max(robust_sigma(fy), float(np.median(fe)), 1e-8)
        valid = np.isfinite(fy) & np.isfinite(fe) & (fe > 0) & (fy > 0) & (fy < 1 + 8*noise)
        parts.append(np.column_stack([st[valid], fy[valid], fe[valid]]))
    if not parts:
        raise ValueError("No observing segments")
    t, y, dy = np.concatenate(parts).T
    adjacent = np.diff(t) < np.median(np.diff(t))*1.5
    scatter = robust_sigma(np.diff(y)[adjacent]) / np.sqrt(2)
    dy *= max(1., scatter / np.median(dy))
    return {"time": t, "flux": y, "error": dy}


def residual_series(t, values, errors, quality, signal, config):
    """Subtract smooth centroid motion separately across gaps; retain measured errors."""
    good = (np.isfinite(t) & np.isfinite(values) & np.isfinite(errors) & (errors > 0)
            & ((quality & TessQualityFlags.DEFAULT_BITMASK) == 0))
    t, values, errors = (a[good] for a in (t, values, errors))
    order = np.argsort(t)
    t, values, errors = (a[order] for a in (t, values, errors))
    if len(t) < 100:
        return {"time": np.array([]), "flux": np.array([]), "error": np.array([])}
    edges = np.r_[0, np.flatnonzero(np.diff(t) > config["split_gap_days"]) + 1, len(t)]
    parts = []
    for start, end in zip(edges[:-1], edges[1:]):
        st, sy, se = (a[start:end] for a in (t, values, errors))
        if len(st) < 100:
            continue
        window = max(5, int(config["flatten_window_days"] / np.median(np.diff(st))) | 1)
        window = min(window, len(st) if len(st) % 2 else len(st)-1)
        mask = np.abs(phase_offset(st, signal["period"], signal["t0"])) < signal["duration_days"]
        # Positive offset avoids LightCurve.flatten dividing by a near-zero centroid.
        offset = sy - np.median(sy) + 100.
        _, trend = lk.LightCurve(time=st, flux=offset).flatten(window_length=window, mask=mask, return_trend=True)
        residual = offset - np.asarray(trend.flux.value)
        noise = max(robust_sigma(residual), float(np.median(se)), 1e-8)
        se = se * max(1., noise / np.median(se))
        parts.append(np.column_stack([st, residual, se]))
    if not parts:
        return {"time": np.array([]), "flux": np.array([]), "error": np.array([])}
    t, y, dy = np.concatenate(parts).T
    return {"time": t, "flux": y, "error": dy}


def fold_series(data, signal, config, *, center=None, selection=None, global_view=False):
    """Zero-filled missing bins carry an explicit occupancy channel."""
    t, y, dy = (data[k] for k in ("time", "flux", "error"))
    if selection is not None:
        t, y, dy = (a[selection] for a in (t, y, dy))
    bins = config["global_bins"] if global_view else config["local_bins"]
    if len(t) == 0:
        return np.zeros((3, bins), np.float32)
    phase = phase_offset(t, signal["period"], signal["t0"] if center is None else center)
    outside = np.abs(phase) > signal["duration_days"]
    base = np.median(y[outside]) if outside.any() else np.median(y)
    scale = max(robust_sigma(y[outside]) if outside.any() else robust_sigma(y), np.median(dy), 1e-8)
    if global_view:
        return binned_view(phase/signal["period"], y-base, dy, -.5, .5, bins, scale)
    width = config["local_half_width_durations"]
    return binned_view(phase/signal["duration_days"], y-base, dy, -width, width, bins, scale)


def rich_views(data, sap_slow, pdc_slow, centroids, signal, config):
    cycles = np.rint((data["time"]-signal["t0"])/signal["period"]).astype(int)
    # Common photometric scale for odd/even/secondary preserves their relative depths.
    t, y, dy = (data[k] for k in ("time", "flux", "error"))
    phase = phase_offset(t, signal["period"], signal["t0"])
    outside = np.abs(phase) > signal["duration_days"]
    base = np.median(y[outside]) if outside.any() else np.median(y)
    scale = max(robust_sigma(y[outside]) if outside.any() else robust_sigma(y), np.median(dy), 1e-8)
    width, bins = config["local_half_width_durations"], config["local_bins"]
    local = []
    for parity in (1, 0):
        chosen = cycles % 2 == parity
        local.append(binned_view(phase[chosen]/signal["duration_days"], y[chosen]-base, dy[chosen], -width, width, bins, scale))
    secondary = phase_offset(t, signal["period"], signal["t0"]+signal["period"]/2)
    local.append(binned_view(secondary/signal["duration_days"], y-base, dy, -width, width, bins, scale))
    local.extend(fold_series(c, signal, config) for c in centroids)
    return {"context_local": np.concatenate(local),
            "context_global": np.concatenate([fold_series(sap_slow, signal, config, global_view=True),
                                                fold_series(pdc_slow, signal, config, global_view=True)])}


def candidate_views(path, signal, config):
    """Only a file and blind-search measurements; no label, TIC or catalog features."""
    raw = read_raw(path)
    original = clean(raw, config)
    masked = detrend(raw, signal, config)
    with fits.open(path) as hdus:
        table = hdus[1].data
        sap = {**raw, "flux": np.array(table["SAP_FLUX"], float), "error": np.array(table["SAP_FLUX_ERR"], float)}
        centroid_input = [(np.array(table[f"MOM_CENTR{i}"], float), np.array(table[f"MOM_CENTR{i}_ERR"], float)) for i in (1, 2)]
    sap_slow, pdc_slow = (detrend(r, signal, config, window_factor=3) for r in (sap, raw))
    centroids = [residual_series(raw["time"], y, dy, raw["quality"], signal, config) for y, dy in centroid_input]
    views = {f"masked_{k}": value for k, value in make_views(masked, signal, config).items()}
    views.update(rich_views(original, sap_slow, pdc_slow, centroids, signal, config))
    views.update({f"masked_{k}": value for k, value in rich_views(masked, sap_slow, pdc_slow, centroids, signal, config).items()})
    legacy = legacy_module()
    lc = lk.read(path, flux_column="pdcsap_flux")
    lv, ok = legacy.signal_views([lc], signal["period"], signal["t0"], signal["duration_days"])
    views["legacy_global"] = np.clip(lv["global"][None], -10, 10).astype(np.float32)
    views["legacy_local"] = np.clip(np.stack([lv[k] for k in ("local", "odd", "even", "secondary", "centroid")]), -10, 10).astype(np.float32)
    # The legacy feature definitions, adapted to exactly one sector/chunk.
    aligned = dict(original)
    available = False
    for i, c in enumerate(centroids, 1):
        ct, cy = c["time"], c["flux"]
        positions = np.searchsorted(ct, original["time"])
        present = positions < len(ct)
        if len(ct):
            clipped = np.minimum(positions, len(ct)-1)
            present &= np.isclose(ct[clipped], original["time"], rtol=0, atol=1e-8)
            value = np.full(len(original["time"]), np.nan)
            value[present] = cy[positions[present]]
            aligned[f"centroid{i}"] = value
            phase = phase_offset(original["time"], signal["period"], signal["t0"])
            available |= bool((present & (abs(phase) < signal["duration_days"]/2)).sum() >= 5
                              and (present & (abs(phase) > signal["duration_days"])).sum() > 20)
    measured = vet_signal(aligned, signal["period"], signal["t0"], signal["duration_days"])
    count = max(signal["observed_transits"], 1)
    f = dict(signal, duration_h=signal["duration_days"]*24,
             sde_per_transit=signal["search_sde"]/np.sqrt(count),
             depth_snr_per_transit=signal["depth_snr"]/np.sqrt(count),
             pink_snr_per_transit=signal["pink_snr"]/np.sqrt(count),
             centroid_snr=measured.get("centroid_snr", np.nan) if available else np.nan,
             transits_per_sector=signal["observed_transits"],
             significant_event_fraction=signal["significant_events"]/count, chunk_fraction=1.)
    views["legacy_features"] = np.array([f[k] for k in LEGACY_FEATURES], np.float32)
    original_scalar = make_views(original, signal, config)["scalar"]
    return views, {"legacy_ok": ok, "centroid_available": available,
                   "original_depth_snr_view": float(np.sinh(original_scalar[3])),
                   "masked_depth_snr_view": float(np.sinh(views["masked_scalar"][3]))}


def source_check(source):
    provenance = json.loads((source/"dataset_provenance.json").read_text())
    if provenance["code_hash"] != code_hash() or provenance["versions"] != versions():
        raise ValueError("Source code/environment no longer matches its prepared data")
    for name, key in (("candidates.csv", "candidate_sha256"), ("views.npz", "views_sha256")):
        if sha256(source/name) != provenance[key]:
            raise ValueError(f"Source {name} has changed")
    return provenance


def prepare_job(job):
    row, config = job
    warnings.filterwarnings("ignore", category=UserWarning)
    path = ROOT/row["source_path"]
    if sha256(path) != row["source_sha256"]:
        raise ValueError("Source FITS checksum changed")
    # This is the explicit boundary that keeps dispositions and catalog ephemerides out.
    return candidate_views(path, {k: row[k] for k in OBSERVATIONAL_FIELDS}, config)


def prepare(source, output, workers=6):
    provenance = source_check(source)
    original = pd.read_csv(source/"candidates.csv")
    if original.groupby("tic").split.nunique().max() != 1:
        raise ValueError("A star crosses partitions")
    selected = original.y.ge(0) & original.split.isin(["train", "validation"])
    df = original[selected].copy().reset_index(drop=True)
    if not df.label.isin(["CP", "FP", "FA"]).all():
        raise ValueError("This experiment requires CP positives and FP/FA negatives")
    config = json.loads((source/"config.json").read_text())
    metadata = {"source": str(source), "source_candidate_sha256": sha256(source/"candidates.csv"),
                "source_views_sha256": sha256(source/"views.npz"), "core_code_hash": code_hash(),
                "representation_code_hash": representation_hash(), "versions": versions(),
                "config": config, "candidate_ids": df.candidate_id.tolist(), "test_evaluated": False,
                "photometry": "Same single SPOC 120-second sector and blind-search ephemeris for all models",
                "legacy_features": list(LEGACY_FEATURES)}
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"provenance.json").exists():
        existing = json.loads((output/"provenance.json").read_text())
        if existing["fingerprint"] != fingerprint:
            raise ValueError("Preparation changed; use a new output directory")
        if (output/"views.npz").exists() and (output/"candidates.csv").exists():
            if sha256(output/"views.npz") != existing["views_sha256"] or sha256(output/"candidates.csv") != existing["candidate_sha256"]:
                raise ValueError("Prepared comparison artifacts were modified")
            return
    cache = output/"cache"
    cache.mkdir(exist_ok=True)
    completed = {}
    pending = []
    for i, row in enumerate(df.to_dict("records")):
        name = row["candidate_id"]
        if (cache/f"{name}.npz").exists() and (cache/f"{name}.json").exists():
            info = json.loads((cache/f"{name}.json").read_text())
            if info["fingerprint"] == fingerprint:
                completed[i] = info
                continue
        pending.append((i, row))
    with ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(prepare_job, (row, config)): (i, row) for i, row in pending}
        for future in as_completed(futures):
            i, row = futures[future]
            views, diagnostics = future.result()  # fail loudly; never alter model cohorts after failures
            np.savez_compressed(cache/f"{row['candidate_id']}.npz", **views)
            info = {"fingerprint": fingerprint, **diagnostics}
            save_json(cache/f"{row['candidate_id']}.json", info)
            completed[i] = info
            if len(completed) % 25 == 0 or len(completed) == len(df):
                print(f"observational views {len(completed)}/{len(df)}", flush=True)
    with np.load(source/"views.npz") as original_views:
        arrays = {k: original_views[k][selected.to_numpy()] for k in original_views.files}
    parts = []
    for i, row in enumerate(df.itertuples()):
        with np.load(cache/f"{row.candidate_id}.npz") as views:
            parts.append({k: views[k] for k in views.files})
    arrays.update({k: np.stack([p[k] for p in parts]) for k in parts[0]})
    for key, value in arrays.items():
        if key != "legacy_features" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite observational view: {key}")
    df.to_csv(output/"candidates.csv", index=False)
    np.savez_compressed(output/"views.npz", **arrays)
    diagnostics = df[["candidate_id", "tic", "split", "label"]].copy()
    for key in ("legacy_ok", "centroid_available", "original_depth_snr_view", "masked_depth_snr_view"):
        diagnostics[key] = [completed[i][key] for i in range(len(df))]
    diagnostics.to_csv(output/"view_diagnostics.csv", index=False)
    save_json(output/"provenance.json", {**metadata, "fingerprint": fingerprint,
                                        "candidate_sha256": sha256(output/"candidates.csv"),
                                        "views_sha256": sha256(output/"views.npz")})
    print(df.groupby(["split", "label"]).size().to_string(), flush=True)

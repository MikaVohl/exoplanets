"""Download, validate, detrend and bin individual stellar light curves."""

import hashlib
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from lightkurve.utils import TessQualityFlags
from requests.adapters import HTTPAdapter
from scipy.ndimage import median_filter
from urllib3.util.retry import Retry

from scripts.download_tess import validate


def download_target(root, target):
    tic, sector = int(target["tic"]), int(target["sector"])
    path = root / "data" / "raw" / "benchmark" / target["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".fits.part")
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        try:
            with session.get("https://mast.stsci.edu/api/v0.1/Download/file",
                             params={"uri": target["source_uri"]}, stream=True,
                             timeout=(20, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            validate(temporary, tic, sector)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
            session.close()
    entry = validate(path, tic, sector)
    entry.update({"path": str(path.relative_to(root)), "source_uri": target["source_uri"],
                  "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return entry


def robust_sigma(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def odd_window(window_days, cadence, n):
    size = max(3, int(round(window_days / cadence)) | 1)
    return min(size, n if n % 2 else n - 1)


def preprocess(path, config):
    with fits.open(path) as h:
        d = h[1].data
        t = np.asarray(d["TIME"], dtype=float)
        f = np.asarray(d["PDCSAP_FLUX"], dtype=float)
        e = np.asarray(d["PDCSAP_FLUX_ERR"], dtype=float)
        q = np.asarray(d["QUALITY"], dtype=int)
        c1 = np.asarray(d["MOM_CENTR1"], dtype=float)
        c2 = np.asarray(d["MOM_CENTR2"], dtype=float)
        original = len(t)
    keep = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (f > 0) & (e > 0)
    keep &= (q & TessQualityFlags.DEFAULT_BITMASK) == 0
    t, f, e, c1, c2 = (x[keep] for x in [t, f, e, c1, c2])
    order = np.argsort(t)
    t, f, e, c1, c2 = (x[order] for x in [t, f, e, c1, c2])
    unique = np.r_[True, np.diff(t) > 0]
    t, f, e, c1, c2 = (x[unique] for x in [t, f, e, c1, c2])
    initial_kept = len(t)
    if initial_kept < config["minimum_points"]:
        raise ValueError(f"Insufficient usable photometry: {initial_kept} cadences")
    boundaries = np.r_[0, np.flatnonzero(np.diff(t) > config["split_gap_days"]) + 1, len(t)]
    arrays = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        st, sf, se, sc1, sc2 = (x[start:end].copy() for x in [t, f, e, c1, c2])
        if len(st) < 100:
            continue
        inner = (st >= st[0] + config["trim_segment_edge_days"]) & (st <= st[-1] - config["trim_segment_edge_days"])
        st, sf, se, sc1, sc2 = (x[inner] for x in [st, sf, se, sc1, sc2])
        if len(st) < 100:
            continue
        cadence = np.median(np.diff(st))
        width = odd_window(config["trend_window_days"], cadence, len(st))
        trend = median_filter(sf, size=width, mode="nearest")
        normalized, errors = sf / trend, se / trend
        # Only clip upward excursions; retain downward points for transit vetting.
        sigma = max(robust_sigma(normalized), float(np.median(errors)))
        valid = np.isfinite(normalized) & (normalized > 0) & (normalized <= 1 + 8 * sigma)
        detrended_centroids = []
        for centroid in [sc1, sc2]:
            finite = np.isfinite(centroid)
            if finite.sum() > 10:
                interpolated = np.interp(st, st[finite], centroid[finite])
                residual = centroid - median_filter(interpolated, size=width, mode="nearest")
            else:
                residual = np.full(len(st), np.nan)
            detrended_centroids.append(residual)
        arrays.append(np.column_stack([st[valid], normalized[valid], errors[valid],
                                       sc1[valid], sc2[valid],
                                       detrended_centroids[0][valid], detrended_centroids[1][valid]]))
    if not arrays:
        raise ValueError("No usable contiguous observing segments")
    unbinned = np.concatenate(arrays)
    if len(unbinned) < config["minimum_points"] or np.ptp(unbinned[:, 0]) < config["minimum_span_days"]:
        raise ValueError("Insufficient observing coverage after preprocessing")
    # Fixed bins contain only observed samples; no interpolation across gaps.
    labels = np.floor((unbinned[:, 0] - unbinned[0, 0]) / (config["bin_minutes"] / 1440)).astype(int)
    binned = []
    for label in np.unique(labels):
        a = unbinned[labels == label]
        if len(a) < 2:
            continue
        weights = 1 / a[:, 2] ** 2
        row = [np.average(a[:, 0], weights=weights), np.average(a[:, 1], weights=weights),
               1 / np.sqrt(weights.sum())]
        row.extend(float(np.nanmean(a[:, j])) if np.isfinite(a[:, j]).any() else np.nan for j in [5, 6])
        binned.append(row)
    binned = np.asarray(binned)
    t, y, dy, c1, c2 = binned.T
    adjacent = np.diff(t) < 1.5 * config["bin_minutes"] / 1440
    scatter = robust_sigma(np.diff(y)[adjacent]) / np.sqrt(2)
    # Do not let underestimated formal uncertainties alone produce large SNR.
    scale = max(1.0, scatter / np.median(dy))
    dy = dy * scale
    metadata = {"original_points": original, "quality_kept_points": initial_kept,
                "detrended_points": len(unbinned), "search_points": len(t),
                "span_days": float(np.ptp(t)), "error_scale": float(scale),
                "point_scatter_ppm": float(scatter * 1e6),
                "quality_bitmask": int(TessQualityFlags.DEFAULT_BITMASK)}
    return {"time": t, "flux": y, "error": dy, "centroid1": c1, "centroid2": c2,
            "unbinned_time": unbinned[:, 0], "unbinned_flux": unbinned[:, 1],
            "unbinned_error": unbinned[:, 2]}, metadata

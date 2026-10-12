"""TESS preprocessing and gap-aware model views shared by training and inference."""
import numpy as np
import lightkurve as lk
from astropy.io import fits
from lightkurve.utils import TessQualityFlags

from tess_hunt.photometry import robust_sigma
from tess_hunt.search import phase_offset


def read_raw(path):
    with fits.open(path) as hdus:
        d, h = hdus[1].data, hdus[0].header
        if not np.isclose(hdus[1].header["TIMEDEL"] * 86400, 120, atol=1):
            raise ValueError("This experiment requires SPOC 120-second photometry")
        # Own arrays: injections must not change the original FITS bytes.
        return {"time": np.array(d["TIME"], dtype=float),
                "flux": np.array(d["PDCSAP_FLUX"], dtype=float),
                "error": np.array(d["PDCSAP_FLUX_ERR"], dtype=float),
                "quality": np.array(d["QUALITY"], dtype=int),
                "tic": int(h["TICID"]), "sector": int(h["SECTOR"])}


def clean(raw, config):
    t, y, dy = (raw[k] for k in ("time", "flux", "error"))
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
        window = max(5, int(config["flatten_window_days"] / np.median(np.diff(st))) | 1)
        window = min(window, len(st) if len(st) % 2 else len(st) - 1)
        lc = lk.LightCurve(time=st, flux=sy, flux_err=se).normalize().flatten(window_length=window)
        ft, fy, fe = (np.asarray(a.value, dtype=float) for a in (lc.time, lc.flux, lc.flux_err))
        noise = max(robust_sigma(fy), float(np.median(fe)), 1e-8)
        valid = np.isfinite(fy) & np.isfinite(fe) & (fe > 0) & (fy > 0) & (fy < 1 + 8 * noise)
        parts.append(np.column_stack([ft[valid], fy[valid], fe[valid]]))
    if not parts:
        raise ValueError("No usable observing segments")
    t, y, dy = np.concatenate(parts).T
    cadence = np.median(np.diff(t))
    adjacent = np.diff(t) < cadence * 1.5
    scatter = robust_sigma(np.diff(y)[adjacent]) / np.sqrt(2)
    dy *= max(1.0, scatter / np.median(dy))
    return {"time": t, "flux": y, "error": dy}


def bin_search(data, minutes):
    """Weighted bins contain observed samples only; never interpolate over gaps."""
    t, y, dy = (data[k] for k in ("time", "flux", "error"))
    labels = np.floor((t - t.min()) / (minutes / 1440)).astype(int)
    _, inverse = np.unique(labels, return_inverse=True)
    weights = 1 / dy**2
    weight = np.bincount(inverse, weights=weights)
    return {"time": np.bincount(inverse, weights=t * weights) / weight,
            "flux": np.bincount(inverse, weights=y * weights) / weight,
            "error": 1 / np.sqrt(weight)}


def binned_view(x, y, dy, left, right, bins, scale):
    """Channels: standardized flux, uncertainty, observed-bin mask. Empty = zero."""
    index = np.floor((x - left) / (right - left) * bins).astype(int)
    valid = (index >= 0) & (index < bins)
    index, values, error = index[valid], y[valid], dy[valid]
    weights = 1 / error**2
    weight = np.bincount(index, weights=weights, minlength=bins)
    occupied = weight > 0
    view = np.zeros((3, bins), dtype=np.float32)
    view[0, occupied] = np.clip(np.bincount(index, weights=values * weights, minlength=bins)[occupied]
                                / weight[occupied] / scale, -100, 100)
    view[1, occupied] = np.log1p(1 / np.sqrt(weight[occupied]) / scale)
    view[2, occupied] = 1
    return view


def make_views(data, signal, config):
    t, y, dy = (data[k] for k in ("time", "flux", "error"))
    period, epoch, duration = (signal[k] for k in ("period", "t0", "duration_days"))
    if not (period > 0 and 0 < duration < period and np.isfinite(epoch)):
        raise ValueError("Invalid candidate ephemeris")
    phase = phase_offset(t, period, epoch)
    outside = np.abs(phase) > duration
    base = float(np.median(y[outside])) if outside.any() else float(np.median(y))
    residual = y - base
    scale = max(robust_sigma(residual[outside]), float(np.median(dy)), 1e-8)
    global_view = binned_view(phase / period, residual, dy, -.5, .5, config["global_bins"], scale)
    width = config["local_half_width_durations"]
    local_view = binned_view(phase / duration, residual, dy, -width, width, config["local_bins"], scale)
    cycles = np.rint((t - epoch) / period).astype(int)
    available = np.unique(cycles[np.abs(phase) <= width * duration])
    # Evenly cover the observing baseline, avoiding a brightest-event selection bias.
    if len(available) > config["maximum_events"]:
        available = available[np.linspace(0, len(available) - 1, config["maximum_events"]).astype(int)]
    events = np.zeros((config["maximum_events"], 3, config["event_bins"]), dtype=np.float32)
    event_mask = np.zeros(config["maximum_events"], dtype=np.float32)
    for i, cycle in enumerate(available):
        chosen = cycles == cycle
        events[i] = binned_view((t[chosen] - (epoch + cycle * period)) / duration,
                               residual[chosen], dy[chosen], -width, width, config["event_bins"], scale)
        event_mask[i] = float(events[i, 2].sum() >= 3)
    inside = np.abs(phase) < duration / 2
    depth = -float(np.median(residual[inside])) if inside.any() else 0
    # Direct measurements only, no TIC, disposition, sector ID, or catalog planet features.
    scalar = np.array([np.log(period), np.log(duration), np.log(scale),
                       np.arcsinh(depth / scale)], dtype=np.float32)
    return {"global": global_view, "local": local_view, "events": events,
            "event_mask": event_mask, "scalar": scalar}

"""Shared settings and helpers for the transit search, the benchmark and the dataset build."""
import json
import math
import time
from pathlib import Path

import lightkurve as lk
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent             # repo root
OUT_DIR = HERE / "output"       # CSVs written by the numbered steps

# Benchmark settings: same TLS threshold and period tolerance as the repo benchmark.
RAW = ROOT / "data" / "raw" / "benchmark"
MIN_SDE = 7.0
MATCH_TOL = 0.01


def clean_and_stitch(lcs, bin_minutes=10, ephemeris=None):
    """Clean each light curve, stitch them, bin them; return (time, flux, flux_err) arrays.

    Each sector is cleaned on its own because each has its own brightness level and drift.

    `ephemeris=(period, t0, duration)` (days) makes flattening ignore the transits: the
    smoothing curve is fit only to out-of-transit points. Without it, the curve dips into
    each transit and absorbs part of it, making the transit look shallower and shorter.
    """
    cleaned = []
    for lc in lcs:
        lc = lc.remove_nans().normalize()
        # Flatten removes slow drifts. The window must be much longer than the ~2 hr transit
        # (~65 points) or it would erase the dip. 1001 points ~ 33 hours.
        mask = None
        if ephemeris is not None:
            period, t0, duration = ephemeris
            # Mask twice the measured duration: a first-pass duration is biased short.
            mask = lc.create_transit_mask(period=period, transit_time=t0, duration=2 * duration)
        lc = lc.flatten(window_length=1001, mask=mask)
        # Clip upward spikes only: downward points are the transit signal we want to keep.
        lc = lc.remove_outliers(sigma=6, sigma_lower=np.inf)
        cleaned.append(lc)
    # Binning speeds TLS up a lot, and 10-min points still resolve a ~2 hr transit.
    lc = lk.LightCurveCollection(cleaned).stitch().bin(time_bin_size=bin_minutes / 1440).remove_nans()
    return (lc.time.value.astype(float), lc.flux.value.astype(float),
            lc.flux_err.value.astype(float))


def relation(p, known):
    """'exact', 'half_period' or 'double_period' if p matches `known` within MATCH_TOL, else None."""
    for ratio, label in [(1.0, "exact"), (0.5, "half_period"), (2.0, "double_period")]:
        if abs(p / (known * ratio) - 1) < MATCH_TOL:
            return label
    return None


def count_transits_with_data(t, t0, period, half_window, min_points=1):
    """How many of a planet's dips (times t0 + k*period) have at least `min_points` data points within
    `half_window` days of the dip centre."""
    ks = np.arange(np.floor((t.min() - t0) / period), np.ceil((t.max() - t0) / period) + 1)
    return int(sum(np.sum(np.abs(t - (t0 + k * period)) < half_window) >= min_points for k in ks))


def star_kwargs(mass, radius):
    """TLS keyword arguments for a star: +/-20% mass/radius priors, generic limb darkening."""
    import os
    return dict(R_star=radius, R_star_min=0.8 * radius, R_star_max=1.2 * radius,
                M_star=mass, M_star_min=0.8 * mass, M_star_max=1.2 * mass,
                u=[0.3, 0.2], use_threads=os.cpu_count(), show_progress_bar=False, verbose=False)


def hide_transits(lc, result):
    """The light curve with a TLS result's transits removed (2x duration), for a search for a second signal."""
    return lc[~lc.create_transit_mask(period=result.period, transit_time=result.T0, duration=2 * result.duration)]


def tls_wide(t, y, dy, star, config):
    """Wide TLS search over the benchmark period range (capped at half the time span)."""
    from transitleastsquares import transitleastsquares
    pmax = min(config["period_max_days"], np.ptp(t) / 2)
    return transitleastsquares(t, y, dy).power(
        period_min=config["period_min_days"], period_max=pmax, n_transits_min=2, **star), pmax


def two_pass(lc, star, config):
    """The masked two-pass method on one light curve: wide TLS, then (if SDE passes) mask the transits,
    re-flatten and re-search a +/-0.5% window. Returns (wide_result, fine_result_or_None)."""
    from transitleastsquares import transitleastsquares
    t, y, dy = clean_and_stitch([lc], config["bin_minutes"])
    if _events:
        emit("cleaned", **CONTEXT, **{"pass": "wide"}, **_curve(t, y))
    wide, pmax = tls_wide(t, y, dy, star, config)
    if _events:
        emit("search", **CONTEXT, **{"pass": "wide"}, period_range=[config["period_min_days"], _sig(pmax)],
             **_spectrum(wide), **_numbers(wide))
    fine = None
    if not wide.SDE < MIN_SDE:
        t, y, dy = clean_and_stitch([lc], config["bin_minutes"],
                                    ephemeris=(wide.period, wide.T0, wide.duration))
        lo, hi = wide.period * 0.995, min(wide.period * 1.005, pmax)
        if _events:
            emit("cleaned", **CONTEXT, **{"pass": "masked"}, **_curve(t, y),
                 mask={"period": _sig(wide.period), "t0": round(wide.T0, 4), "duration_d": _sig(wide.duration)})
        fine = transitleastsquares(t, y, dy).power(
            period_min=lo, period_max=hi, n_transits_min=2, oversampling_factor=20, **star)
        if _events:
            emit("search", **CONTEXT, **{"pass": "fine"}, period_range=[_sig(lo), _sig(hi)],
                 **_spectrum(fine), **_numbers(fine))
    if _events:
        final = fine if fine is not None else wide
        # Detection uses the wide pass's SDE (the fine window is too narrow for a meaningful SDE).
        emit("signal_done", **CONTEXT, **dict(_numbers(final), sde=_sig(wide.SDE)), refined=fine is not None,
             folded=_folded(final))
    return wide, fine


# --- Mission Control: a log of what the pipeline is doing, one JSON object per line, read by the
# TESS visualizer (live via live.py, or replayed from a file). Off unless 01 is run with --events.
_events = None    # the open log file while recording, else None (then emit() does nothing)
CONTEXT = {}      # tic and signal (1 or 2) of the search in progress, added to the events two_pass emits


def events_to(path):
    """Start recording events to `path` (overwritten)."""
    global _events
    _events = open(path, "w")


def emit(type, **fields):
    """Record one event, stamped with the current time, if recording is on."""
    if _events:
        event = _jsonable({"type": type, "ts": round(time.time(), 3), **fields})
        _events.write(json.dumps(event, separators=(",", ":")) + "\n")
        _events.flush()


def _jsonable(o):
    """numpy values -> plain Python, NaN/inf -> None (JSON has no NaN)."""
    if isinstance(o, np.ndarray):
        o = o.tolist()
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    return None if isinstance(o, float) and not math.isfinite(o) else o


def _sig(x):
    """Round a number, or each number in an array, to 6 significant figures."""
    return float(f"{x:.6g}") if np.ndim(x) == 0 else [float(f"{v:.6g}") for v in x]


def _numbers(r):
    """The headline numbers of a TLS result (depth as the fractional dip)."""
    return dict(period=_sig(r.period), t0=round(float(r.T0), 4), sde=_sig(r.SDE), depth=_sig(1 - r.depth),
                duration_h=_sig(r.duration * 24))


def _curve(t, y, n=4000):
    """A light curve, thinned to at most n points."""
    s = slice(None, None, max(1, -(-len(t) // n)))
    return dict(t=np.round(t[s], 4), y=np.round(y[s], 6))


def _spectrum(r, n=2000):
    """TLS SDE spectrum, at most n points, keeping the highest point of each bucket so peaks survive."""
    p, s = np.atleast_1d(r.periods), np.atleast_1d(r.power)
    k = max(1, -(-len(p) // n))
    i = [j + int(np.argmax(np.nan_to_num(s[j:j + k]))) for j in range(0, len(p), k)]
    return dict(periods=_sig(p[i]), power=_sig(s[i]))


def _folded(r, n=1500):
    """TLS phase-folded data (thinned to at most n points) and model. The model is flat (1) away from
    the dip, so only the dip, its edges and the two ends are kept."""
    ph, fy = np.atleast_1d(r.folded_phase), np.atleast_1d(r.folded_y)
    mp, mm = np.atleast_1d(r.model_folded_phase), np.atleast_1d(r.model_folded_model)
    s = slice(None, None, max(1, -(-len(ph) // n)))
    dip = mm < 1
    keep = dip | np.roll(dip, 1) | np.roll(dip, -1)
    keep[[0, -1]] = True
    return dict(phase=np.round(ph[s], 6), flux=np.round(fy[s], 6),
                model_phase=np.round(mp[keep], 6), model_flux=np.round(mm[keep], 6))

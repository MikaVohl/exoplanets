"""Shared settings and helpers for the transit search, the benchmark and the dataset build."""
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
    wide, pmax = tls_wide(t, y, dy, star, config)
    if wide.SDE < MIN_SDE:
        return wide, None
    t, y, dy = clean_and_stitch([lc], config["bin_minutes"],
                                ephemeris=(wide.period, wide.T0, wide.duration))
    fine = transitleastsquares(t, y, dy).power(
        period_min=wide.period * 0.995, period_max=min(wide.period * 1.005, pmax),
        n_transits_min=2, oversampling_factor=20, **star)
    return wide, fine

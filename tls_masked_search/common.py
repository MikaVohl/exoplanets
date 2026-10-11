"""Shared settings and helpers for the WASP-18 b pipeline (imported by the numbered steps)."""
from pathlib import Path

import lightkurve as lk
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent             # repo root
DATA_DIR = HERE / "data"        # cached MAST downloads (git-ignored)
OUT_DIR = HERE / "output"       # results.json + plots written by the steps

TARGET = "WASP-18"
TIC_ID = 100100827
SEARCH_SECTORS = [2, 3]         # ~50 contiguous days: plenty to find a ~1 day planet quickly

# Published values we compare against (ranges because papers disagree with each other).
# Period: Cortes-Zuleta+2020 via the NASA Exoplanet Archive.
PUBLISHED_PERIOD = 0.94145224           # days
PUBLISHED_DEPTH = (0.0088, 0.0104)      # fraction of flux
PUBLISHED_DURATION_H = (2.1, 2.2)       # hours

# "Close enough" tolerances. Period is pinned down tightly; depth and duration vary by
# method (box vs. limb-darkened fit, detrending, binning), so they get a looser margin.
PERIOD_TOL_DAYS = 1e-4
# Benchmark-sample settings (steps 4-5): same TLS threshold and period tolerance as the repo benchmark.
RAW = ROOT / "data" / "raw" / "benchmark"
MIN_SDE = 7.0
MATCH_TOL = 0.01
RANGE_TOL_FRAC = 0.15                   # allowed overshoot beyond the published range


def download(sectors=None):
    """Download (or reuse cached) SPOC 2-min light curves. `sectors=None` means all of them."""
    kwargs = {"sector": sectors} if sectors else {}
    search = lk.search_lightcurve(TARGET, author="SPOC", exptime=120, **kwargs)
    return search.download_all(download_dir=str(DATA_DIR))


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


def load_clean(sectors=None, bin_minutes=10, ephemeris=None):
    """Download (or reuse cached) WASP-18 sectors, then clean_and_stitch them."""
    return clean_and_stitch(download(sectors), bin_minutes, ephemeris)


def in_range(value, lo, hi, frac=RANGE_TOL_FRAC):
    """True if `value` is inside [lo, hi], widened by `frac` on each side."""
    return lo * (1 - frac) <= value <= hi * (1 + frac)


def relation(p, known):
    """'exact', 'half_period' or 'double_period' if p matches `known` within MATCH_TOL, else None."""
    for ratio, label in [(1.0, "exact"), (0.5, "half_period"), (2.0, "double_period")]:
        if abs(p / (known * ratio) - 1) < MATCH_TOL:
            return label
    return None


def star_kwargs(mass, radius):
    """TLS keyword arguments for a star: +/-20% mass/radius priors, generic limb darkening."""
    import os
    return dict(R_star=radius, R_star_min=0.8 * radius, R_star_max=1.2 * radius,
                M_star=mass, M_star_min=0.8 * mass, M_star_max=1.2 * mass,
                u=[0.3, 0.2], use_threads=os.cpu_count(), show_progress_bar=False, verbose=False)


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

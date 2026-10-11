"""Blind transit search with TLS (realistic transit shape) on WASP-18, compared with published values."""
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from transitleastsquares import transitleastsquares, catalog_info

PUBLISHED_PERIOD = 0.94145224
PUBLISHED_DEPTH = (0.0088, 0.0104)
PUBLISHED_DURATION_H = (2.1, 2.2)

# --- 1. Load + clean sectors 2-3, same recipe as 02_find_planet.py ---
lcs = lk.search_lightcurve("WASP-18", author="SPOC", exptime=120, sector=[2, 3]).download_all(download_dir="data")
cleaned = []
for lc in lcs:
    lc = lc.remove_nans().normalize().flatten(window_length=1001)
    lc = lc.remove_outliers(sigma=6, sigma_lower=np.inf)
    cleaned.append(lc)
# Bin to 10 min: TLS is slow, and 10-min points still resolve a ~2 hr transit
lc = lk.LightCurveCollection(cleaned).stitch().bin(time_bin_size=10 / 1440).remove_nans()
t, y, dy = lc.time.value.astype(float), lc.flux.value.astype(float), lc.flux_err.value.astype(float)
print(f"{len(t)} points")

# --- 2. Star properties from the TESS Input Catalog: TLS uses them to build the transit shape ---
# (limb darkening = star is dimmer at its edges; radius/mass bound the plausible transit durations)
ab, mass, mass_err_lo, mass_err_hi, radius, radius_err_lo, radius_err_hi = catalog_info(TIC_ID=100100827)
# The catalog returns +/- uncertainties; TLS wants absolute lower/upper bounds
mass_min, mass_max = mass - mass_err_lo, mass + mass_err_hi
radius_min, radius_max = radius - radius_err_lo, radius + radius_err_hi

# --- 3. Search: TLS compares the data to a realistic transit shape at each trial period ---
star = dict(u=ab, M_star=mass, M_star_min=mass_min, M_star_max=mass_max,
            R_star=radius, R_star_min=radius_min, R_star_max=radius_max,
            use_threads=8, show_progress_bar=False)
r = transitleastsquares(t, y, dy).power(period_min=0.5, period_max=10, **star)
coarse = r  # keep for the periodogram plot
print(f"Pass 1 (wide search) period: {r.period:.6f} d")

# Pass 2: the wide grid is coarse, so a small period error smears the folded transit.
# Re-search a narrow window around the peak with a much finer grid.
r = transitleastsquares(t, y, dy).power(
    period_min=r.period - 0.003, period_max=r.period + 0.003, oversampling_factor=20, **star)

dip = 1 - r.depth   # TLS reports mean in-transit flux (e.g. 0.99), so the dip is 1 - that
print(f"\nPeriod  : {r.period:.6f} +/- {r.period_uncertainty:.6f} d")
print(f"Depth   : {dip * 100:.2f}%")
print(f"Duration: {r.duration * 24:.2f} h")
print(f"SDE     : {coarse.SDE:.1f}  (wide-search signal strength vs noise; >9 is generally a solid detection)")
print(f"Transits: {r.transit_count} ({r.distinct_transit_count} with data)")
print(f"Odd/even mismatch: {r.odd_even_mismatch:.2f} sigma (should be small, < ~3)")

print("\n=== Comparison with published values ===")
print(f"Period  : found {r.period:.6f} d | published {PUBLISHED_PERIOD:.6f} d | diff {abs(r.period - PUBLISHED_PERIOD) * 86400:.1f} s")
print(f"Depth   : found {dip * 100:.2f}% | published {PUBLISHED_DEPTH[0] * 100:.2f}-{PUBLISHED_DEPTH[1] * 100:.2f}%")
print(f"Duration: found {r.duration * 24:.2f} h | published {PUBLISHED_DURATION_H[0]}-{PUBLISHED_DURATION_H[1]} h")

# --- 4. Plots ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(coarse.periods, coarse.power, lw=0.7)
axes[0].axvline(r.period, color="r", alpha=0.4)
axes[0].set_xlabel("Period [d]"); axes[0].set_ylabel("SDE")
axes[0].set_title("TLS periodogram - tall peak = likely period")

axes[1].scatter(r.folded_phase, r.folded_y, s=2, alpha=0.3, label="folded data")
axes[1].plot(r.model_folded_phase, r.model_folded_model, color="r", label="TLS transit model")
axes[1].set_xlim(0.4, 0.6)
axes[1].set_xlabel("Phase"); axes[1].set_ylabel("Normalized flux")
axes[1].set_title(f"Folded at {r.period:.6f} d (zoomed on transit)")
axes[1].legend()
fig.tight_layout()
fig.savefig("wasp18b_tls.png", dpi=120)
print("Saved wasp18b_tls.png")

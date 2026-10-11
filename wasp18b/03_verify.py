"""Verify the BLS result: refine the period over all sectors, run sanity checks, compare to published."""
import numpy as np
import lightkurve as lk

PUBLISHED_PERIOD = 0.94145224   # days, Cortes-Zuleta+2020 (NASA Exoplanet Archive); many papers agree to ~1e-7
PUBLISHED_DEPTH = (0.0088, 0.0104)   # range across papers (fraction)
PUBLISHED_DURATION_H = (2.1, 2.2)    # range across papers (hours)

# --- 1. Load ALL sectors (2018-2026) and clean them the same way as 02_find_planet.py ---
lcs = lk.search_lightcurve("WASP-18", author="SPOC", exptime=120).download_all(download_dir="data")
cleaned = []
for lc in lcs:
    lc = lc.remove_nans().normalize().flatten(window_length=1001)
    lc = lc.remove_outliers(sigma=6, sigma_lower=np.inf)
    cleaned.append(lc)
lc = lk.LightCurveCollection(cleaned).stitch().bin(time_bin_size=10 / 1440)  # 10-min bins = fewer points, faster
lc = lc.remove_nans()
print(f"{len(lc)} binned points, baseline {lc.time.value.max() - lc.time.value.min():.0f} days")

# --- 2. Refine the period: a long baseline pins the period down much more tightly ---
# Each extra orbit adds a tiny shift if the period is slightly off, so 8 years of data
# can distinguish periods that differ by ~1e-6 days. Search a narrow window around the coarse answer.
coarse = 0.94150
periods = np.arange(coarse - 0.0005, coarse + 0.0005, 2e-7)
durations = np.linspace(0.06, 0.12, 7)          # 1.4 h to 2.9 h
pg = lc.to_periodogram(method="bls", period=periods, duration=durations)
P = pg.period_at_max_power.value
t0 = pg.transit_time_at_max_power.value
dur = pg.duration_at_max_power.value
print(f"\nRefined period: {P:.7f} d")

# --- 3. Sanity checks that the period is the TRUE one, not a multiple/fraction of it ---
t = lc.time.value
f = lc.flux.value
def depth_at(period, in_phase=0.0):
    """Mean dip depth when folding at `period`, measured at the transit centre."""
    ph = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    inside = np.abs(ph - in_phase) < dur / 4
    return 1 - f[inside].mean()

# Check A: half the period. If P/2 were real, there'd be a dip at the midpoint too (there isn't).
mid = depth_at(P, 0.5 * P)
print(f"Depth at mid-orbit (should be ~0, no second planet/eclipse): {mid * 100:.3f}%")

# Check B: odd vs even transits. If true period were 2P, odd/even dips could differ.
n = np.round((t - t0) / P)
ph = t - t0 - n * P
inside = np.abs(ph) < dur / 4
odd = 1 - f[inside & (n % 2 == 1)].mean()
even = 1 - f[inside & (n % 2 == 0)].mean()
print(f"Odd transits depth : {odd * 100:.3f}%")
print(f"Even transits depth: {even * 100:.3f}%  (should match odd)")

# --- 4. Compare with published ---
print("\n=== Comparison with published values ===")
print(f"Period  : found {P:.7f} d | published {PUBLISHED_PERIOD:.7f} d | diff {abs(P - PUBLISHED_PERIOD) * 86400:.2f} s")
print(f"Depth   : found {float(pg.depth_at_max_power.value) * 100:.2f}% | published {PUBLISHED_DEPTH[0] * 100:.2f}-{PUBLISHED_DEPTH[1] * 100:.2f}%")
print(f"Duration: found {dur * 24:.2f} h | published {PUBLISHED_DURATION_H[0]}-{PUBLISHED_DURATION_H[1]} h")

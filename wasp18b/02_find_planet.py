"""Blind transit search on WASP-18: clean the data, find the period with BLS, fold, plot."""
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

# --- 1. Load: sectors 2 and 3 only (~50 contiguous days). Using all 8 years of sectors would
# need an extremely fine period grid and be very slow; 50 days is plenty for a ~1 day planet. ---
lcs = lk.search_lightcurve("WASP-18", author="SPOC", exptime=120, sector=[2, 3]).download_all(download_dir="data")

# --- 2. Clean each sector separately (each has its own brightness level and drift) ---
cleaned = []
for lc in lcs:
    lc = lc.remove_nans()                       # drop missing points
    lc = lc.normalize()                         # scale so typical brightness = 1.0
    # Flatten removes slow drifts. window_length must be much longer than the transit
    # (~2 hr = ~65 points) or it would erase the dip. 1001 pts ~ 33 hrs.
    lc = lc.flatten(window_length=1001)
    lc = lc.remove_outliers(sigma=6, sigma_lower=np.inf)  # clip upward spikes only; dips are real signal
    cleaned.append(lc)

# Stitch all sectors into one long light curve
lc = lk.LightCurveCollection(cleaned).stitch()
print(f"{len(lc)} points across {len(cleaned)} sectors (2, 3)")

# --- 3. Blind search: BLS tries many periods and scores how well a box-shaped dip fits ---
# Search 0.5-10 days in small steps so we don't step over the true peak.
periods = np.arange(0.5, 10, 0.0001)
durations = [0.05, 0.08, 0.1, 0.15]             # candidate transit lengths in days
pg = lc.to_periodogram(method="bls", period=periods, duration=durations)

best_period = pg.period_at_max_power
t0 = pg.transit_time_at_max_power
depth = pg.depth_at_max_power
duration = pg.duration_at_max_power
print(f"Best period : {best_period:.5f}")
print(f"Transit time: {t0.value:.5f}")
print(f"Depth       : {depth:.5f}  ({float(depth.value) * 100:.2f}%)")
print(f"Duration    : {duration.to('hour'):.2f}")

# --- 4. Fold at the found period and bin to see the dip clearly ---
folded = lc.fold(period=best_period, epoch_time=t0)
binned = folded.bin(time_bin_size=0.005)        # 0.005 d ~ 7 min bins

# --- 5. Plots ---
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

lc.scatter(ax=axes[0], s=1, label="cleaned")
axes[0].set_title("Cleaned light curve (sectors 2-3)")

pg.plot(ax=axes[1])
axes[1].axvline(best_period.value, color="r", alpha=0.4, label="best period")
axes[1].set_title("BLS periodogram - tall peak = likely period")
axes[1].legend()

folded.scatter(ax=axes[2], s=1, alpha=0.2, label="folded")
binned.scatter(ax=axes[2], s=15, color="r", label="binned")
axes[2].set_title(f"Folded at {best_period.value:.5f} d")
axes[2].legend()

fig.tight_layout()
fig.savefig("wasp18b_result.png", dpi=120)
print("Saved wasp18b_result.png")

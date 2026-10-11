"""Step 3: check the step-2 result is the TRUE signal, then compare with published values.

1. Refine the PERIOD using all sectors (a years-long baseline pins it down far better).
   Only the period is taken from this: TLS assumes evenly sampled data, and 8 years with
   ~30 days of coverage per sector breaks its duration/depth estimates (it reports ~0.16 h).
   Depth and duration come from the contiguous sectors 2-3 fit in step 2.
2. Sanity checks: no dip at mid-orbit (so P isn't half the real period), odd/even depths match
   (so P isn't a fraction of the real period).
3. Compare period, depth and duration with the literature, within the tolerances in common.py.
"""
import json
import os

import numpy as np
from transitleastsquares import catalog_info, transitleastsquares

from common import (OUT_DIR, PERIOD_TOL_DAYS, PUBLISHED_DEPTH, PUBLISHED_DURATION_H,
                    PUBLISHED_PERIOD, RANGE_TOL_FRAC, TIC_ID, in_range, load_clean)


def main():
    found = json.loads((OUT_DIR / "results.json").read_text())
    dip, dur = found["depth"], found["duration_h"] / 24

    # --- 1. Refine the period over all sectors ---
    t, y, dy = load_clean(sectors=None)
    print(f"{len(t)} binned points, baseline {t.max() - t.min():.0f} days")
    ab, mass, mass_lo, mass_hi, radius, radius_lo, radius_hi = catalog_info(TIC_ID=TIC_ID)
    P0 = found["period_d"]
    r = transitleastsquares(t, y, dy).power(
        period_min=P0 - 0.0005, period_max=P0 + 0.0005, oversampling_factor=5,
        u=ab, M_star=mass, M_star_min=mass - mass_lo, M_star_max=mass + mass_hi,
        R_star=radius, R_star_min=radius - radius_lo, R_star_max=radius + radius_hi,
        use_threads=os.cpu_count(), show_progress_bar=False)
    P, t0 = r.period, r.T0
    print(f"Refined period: {P:.7f} d")

    # --- 2. Sanity checks ---
    def depth_at(phase_offset):
        """Mean dip depth near `phase_offset` days from transit centre when folded at P."""
        ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P
        return 1 - y[np.abs(ph - phase_offset) < dur / 4].mean()

    mid = depth_at(0.5 * P)
    print(f"\nDepth at mid-orbit      : {mid * 100:.3f}%  (should be ~0)")

    n = np.round((t - t0) / P)
    inside = np.abs(t - t0 - n * P) < dur / 4
    odd = 1 - y[inside & (n % 2 == 1)].mean()
    even = 1 - y[inside & (n % 2 == 0)].mean()
    print(f"Odd / even transit depth: {odd * 100:.3f}% / {even * 100:.3f}%  (should match)")

    # --- 3. Compare with published ---
    checks = {
        "Period": abs(P - PUBLISHED_PERIOD) < PERIOD_TOL_DAYS,
        "Depth": in_range(dip, *PUBLISHED_DEPTH),
        "Duration": in_range(dur * 24, *PUBLISHED_DURATION_H),
        "No mid-orbit dip": mid < 0.1 * dip,
        "Odd/even match": abs(odd - even) < 0.1 * dip,
    }
    print("\n=== Comparison with published values ===")
    print(f"Period  : found {P:.7f} d | published {PUBLISHED_PERIOD:.7f} d | diff {abs(P - PUBLISHED_PERIOD) * 86400:.1f} s")
    print(f"Depth   : found {dip * 100:.2f}% | published {PUBLISHED_DEPTH[0] * 100:.2f}-{PUBLISHED_DEPTH[1] * 100:.2f}%")
    print(f"Duration: found {dur * 24:.2f} h | published {PUBLISHED_DURATION_H[0]}-{PUBLISHED_DURATION_H[1]} h")
    print(f"(depth/duration pass if within {RANGE_TOL_FRAC:.0%} of the published range)\n")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


if __name__ == "__main__":
    main()

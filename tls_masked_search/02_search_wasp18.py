"""Step 2: blind transit search with TLS on sectors 2-3. Writes output/results.json + plot.

TLS (Transit Least Squares) slides a realistic, limb-darkened transit shape over the data
at many trial periods and scores the fit. The score is the SDE (signal detection efficiency):
how far the best period stands out from the rest of the periodogram.

Two passes, each fixing a different problem:
  Pass 1  wide search, 0.5-10 d, on standard-flattened data -> rough period, t0, duration.
  Pass 2  (a) re-flatten with the transits masked, so the baseline isn't dragged into the dips;
          (b) re-search a narrow window with a finer period grid, so a small period error
              doesn't smear the folded transit.
"""
import json
import os

import matplotlib.pyplot as plt
from transitleastsquares import catalog_info, transitleastsquares

from common import OUT_DIR, SEARCH_SECTORS, TIC_ID, load_clean


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # --- Star properties from the TESS Input Catalog ---
    # Limb darkening (stars are dimmer at their edges) shapes the transit; radius and mass
    # bound which transit durations are physically plausible.
    ab, mass, mass_lo, mass_hi, radius, radius_lo, radius_hi = catalog_info(TIC_ID=TIC_ID)
    star = dict(u=ab,
                M_star=mass, M_star_min=mass - mass_lo, M_star_max=mass + mass_hi,
                R_star=radius, R_star_min=radius - radius_lo, R_star_max=radius + radius_hi,
                use_threads=os.cpu_count(), show_progress_bar=False)

    # --- Pass 1: wide blind search on standard-flattened data ---
    t, y, dy = load_clean(SEARCH_SECTORS)
    print(f"{len(t)} points from sectors {SEARCH_SECTORS}")
    wide = transitleastsquares(t, y, dy).power(period_min=0.5, period_max=10, **star)
    print(f"Pass 1 (wide search): period {wide.period:.6f} d, duration {wide.duration * 24:.2f} h, "
          f"SDE {wide.SDE:.1f}")

    # --- Pass 2: mask the transits, re-flatten, then search finely around the peak ---
    t, y, dy = load_clean(SEARCH_SECTORS, ephemeris=(wide.period, wide.T0, wide.duration))
    fine = transitleastsquares(t, y, dy).power(
        period_min=wide.period - 0.003, period_max=wide.period + 0.003,
        oversampling_factor=20, **star)

    # TLS's own `duration` is conservative. The width of the fitted model where it drops below 1
    # is the first-to-last-contact duration (T14), which is what papers quote.
    in_transit = fine.model_folded_model < 0.99999
    t14_h = float((fine.model_folded_phase[in_transit].max() - fine.model_folded_phase[in_transit].min())
                  * fine.period * 24)

    results = {
        "period_d": float(fine.period),
        "period_uncertainty_d": float(fine.period_uncertainty),
        "t0_bjd": float(fine.T0),
        "duration_h": t14_h,
        "tls_duration_h": float(fine.duration * 24),
        "depth": float(1 - fine.depth),     # TLS reports mean in-transit flux (e.g. 0.99); dip = 1 - that
        "sde": float(wide.SDE),
        "transit_count": int(fine.transit_count),
        "distinct_transit_count": int(fine.distinct_transit_count),
        "odd_even_mismatch_sigma": float(fine.odd_even_mismatch),
        "pass1_duration_h": float(wide.duration * 24),
        "pass1_depth": float(1 - wide.depth),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\nPeriod  : {results['period_d']:.6f} +/- {results['period_uncertainty_d']:.6f} d")
    print(f"Depth   : {results['depth'] * 100:.2f}%   (pass 1: {results['pass1_depth'] * 100:.2f}%)")
    print(f"Duration: {results['duration_h']:.2f} h  (T14 from model; TLS's own estimate {results['tls_duration_h']:.2f} h)")
    print(f"SDE     : {results['sde']:.1f}  (>9 is generally a solid detection)")
    print(f"Transits: {results['transit_count']} ({results['distinct_transit_count']} with data)")
    print(f"Odd/even mismatch: {results['odd_even_mismatch_sigma']:.2f} sigma (should be < ~3)")

    # --- Plots ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    axes[0].scatter(t, y, s=1)
    axes[0].set_title(f"Cleaned light curve, transits masked during flattening (sectors {SEARCH_SECTORS})")
    axes[0].set_xlabel("Time [BTJD]"); axes[0].set_ylabel("Normalized flux")

    axes[1].plot(wide.periods, wide.power, lw=0.7)
    axes[1].axvline(fine.period, color="r", alpha=0.4)
    axes[1].set_title("TLS periodogram (pass 1): a tall peak is the likely period")
    axes[1].set_xlabel("Period [d]"); axes[1].set_ylabel("SDE")

    axes[2].scatter(fine.folded_phase, fine.folded_y, s=2, alpha=0.3, label="folded data")
    axes[2].plot(fine.model_folded_phase, fine.model_folded_model, color="r", label="TLS transit model")
    axes[2].set_xlim(0.4, 0.6)
    axes[2].set_title(f"Folded at {fine.period:.6f} d (zoomed on the transit)")
    axes[2].set_xlabel("Phase"); axes[2].set_ylabel("Normalized flux")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "search.png", dpi=120)
    print(f"\nSaved {OUT_DIR / 'results.json'} and {OUT_DIR / 'search.png'}")


if __name__ == "__main__":
    main()

# 100-star TESS transit benchmark

Completed searches: 100/100. Cataloged transiting hosts with a supported exact-period match: 36/60.

A recovery means a leading BLS or TLS peak in up to two blind iterations passed the fixed screening thresholds and matched a catalog period within 1%. Epoch offsets are diagnostic only. This is a period-matched benchmark, not a planet confirmation.

| Catalog group | Conclusion | Stars |
| --- | --- | ---: |
| known_transiting_host | no_significant_signal_in_search_range | 10 |
| known_transiting_host | signal_requires_review | 22 |
| known_transiting_host | supported_transit_like_signal | 28 |
| no_cataloged_transiting_planet | no_significant_signal_in_search_range | 24 |
| no_cataloged_transiting_planet | signal_requires_review | 15 |
| no_cataloged_transiting_planet | supported_transit_like_signal | 1 |

## How to read the results

- supported_transit_like_signal: BLS/TLS agree in period and phase, with repeated observed dips and screening checks passed.
- signal_requires_review: one or both methods support a signal, but agreement or false-positive diagnostics require inspection.
- no_significant_signal_in_search_range: no leading peak passed all thresholds. It does not establish absence of planets or transits.
- insufficient_data/error/not_run: not an astrophysical negative result.

## Fixed scope and limitations

Latest available public two-minute SPOC sector for each selected star; gaps retained. Search 0.5–12 days, shortened when necessary to half the usable time span. Gap-aware one-day median detrending, 0.1-day segment edge trim, upward-only 8-sigma clipping, and ten-minute weighted bins. Up to two blind iterations, masking only the discovered leading signal. BLS and TLS search independently over the full allowed range. Catalog periods, epochs and labels are introduced only after both searches.

Screening thresholds are heuristic, not calibrated false-alarm rates. Correlated-noise SNR is a duration-block estimate. Centroid, secondary-eclipse and odd/even diagnostics are warnings, not definitive false-positive identifications. TLS uses generic quadratic limb-darkening coefficients. Detrending/bins may attenuate short or long transits. The strongest two signals can miss additional planets. Sparse transits and catalog-period drift can cause missed matches.

The 40 comparison stars have confirmed planets but no cataloged transiting planets. A flagged comparison signal is not automatically a false positive or a new planet. The groups differ in brightness and are selected from known planet hosts with available TESS data, so performance is not representative of a blind stellar survey. No injection-recovery sensitivity calibration or pixel-level contamination vetting is included.

## Outputs

summary.csv: one row per star. candidate_signals.csv: leading peaks from both methods and each iteration. planet_recovery.csv: post-search known-planet comparison. results/: full JSON with parameters, events, alternative peaks, provenance and versions. plots/: diagnostic panels. spectra/: saved numerical periodograms.

Sources: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html), [MAST TESS](https://archive.stsci.edu/missions-and-data/tess), [Astropy BLS](https://docs.astropy.org/en/stable/timeseries/bls.html), [TLS 2.0](https://transitleastsquares.readthedocs.io/en/latest/).

## Measured recovery by host size group

| Largest cataloged short-period transiting planet per host | Hosts with a supported exact-period match |
| --- | ---: |
| small | 5/20 |
| medium | 12/20 |
| giant | 19/20 |

These are host-level, period-matched recoveries for this selected sample. They do not measure detection completeness; brightness, stellar variability, observing coverage, transit duration and planet size all differ among stars.

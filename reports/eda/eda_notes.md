# TESS starter-data EDA

Target: Pi Mensae (TIC 261136679). Public MAST/SPOC 120-second light curves.

Original cadences: 56,515. Retained: 45,841 (81.1%).
Quality policy: Lightkurve default bitmask 17087, finite/positive PDCSAP flux and uncertainty.

| Sector | Retained rows | Span (days) | Robust scatter (ppm) | Adjacent difference (ppm) | Reported error (ppm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 18,264 | 27.88 | 125.5 | 112.3 | 89.1 |
| 4 | 14,642 | 25.51 | 191.6 | 124.5 | 88.8 |
| 8 | 12,935 | 24.02 | 214.7 | 184.0 | 88.6 |

Spans include observing gaps; they are not uninterrupted exposure times. Scatter is not CDPP or detection significance.

The original FITS files and all-cadence CSV retain missing values and flags. The clean CSV has only minimal quality filtering and per-sector normalization. No detrending, sigma clipping, gap interpolation, or period search has been applied.

Next: inspect anomalies, test per-sector detrending, then run a blind Box Least Squares search and examine individual events and the phase-folded curve.

# Project handoff: TESS exoplanet classification

Written for a new Claude agent picking this project up cold. Read this first, then `README.md`,
`tls_masked_search/README.md`, and `reports/benchmark/README.md`.

## 1. The goal

Build a system that takes **new, unlabeled TESS light curves** and finds/classifies exoplanet
transit candidates, using a transit-search pipeline **plus ML models**. The user (Abdullah) works on
this with a friend (Mika Vohl; the `origin` remote is `git@github.com:MikaVohl/exoplanets.git`).
The user started from NASA's TESS tutorial and is not an astronomy/ML specialist.

**Communication preferences learned in this session**
- The user asked twice for plain-English explanations after I used jargon (SDE, ablation, McNemar,
  detrending). Explain concepts simply first, technical terms second.
- They want **verifiable** evidence that changes are improvements (they asked for an ablation for
  exactly this reason), not claims.
- They asked for a recommendation, not a survey of options.

## 2. Repository map

| Path | What it is |
|---|---|
| `tess_hunt/` | Friend's package: `photometry.py` (download + repo detrending), `search.py` (BLS + TLS + vetting), `benchmark.py`, `catalog.py` (NASA Exoplanet Archive target selection), `report.py` |
| `scripts/run_benchmark.py` | CLI for the repo's **100-star benchmark**: stages `select/download/search/report` |
| `config/benchmark.json` | Benchmark thresholds (TLS min SDE 7, BLS min SDE 6, min depth SNR 7.1, 1% period tolerance, etc.) |
| `data/benchmark/` | `targets.csv` (100 stars), `known_planets.csv` (catalog planets). Raw FITS go in `data/raw/benchmark/` (git-ignored; re-fetch with `python scripts/run_benchmark.py download`) |
| `reports/benchmark/` | The repo's original BLS+TLS benchmark results (`results/tic*.json`, `planet_recovery.csv`, README). Its headline: **36/60** known transiting hosts recovered with BLS+TLS agreement + vetting |
| `notebooks/` | EDA and TOI-880 walkthrough notebooks |
| `tls_masked_search/` | **Our work this session** (details below) |
| `docs/HowToFindAnExoplanet.md` | Background write-up |

Benchmark sample: 100 stars, one TESS sector each, 2-min SPOC cadence. 60 "known transiting hosts"
(20 each of small / medium / giant planets) and 40 "comparison" stars that host confirmed planets
but no cataloged transiting planet. Labels/catalog periods are used only after a search finishes.

## 3. Environment (gotchas)

- Python **3.14**, venv at `.venv/` (`source .venv/bin/activate`; `pip install -r requirements.txt`).
  Linux (Arch). Run scripts from inside `tls_masked_search/` (they import `common.py` from cwd).
- **Python 3.14 uses `forkserver` for multiprocessing.** TLS starts worker processes that re-import
  the main script, so every script that calls TLS **must** wrap its body in
  `if __name__ == "__main__":`. Without it you get "attempt to start a new process before the current
  process has finished its bootstrapping" followed by a `ConnectionResetError`. All scripts here have it.
- TLS 2.0 ships its own batman-compatible model (`transitleastsquares.transit_model.light_curve`); the
  `batman` package is **not** installed and not needed.
- Harmless warnings: lightkurve `oktopus/autograd`, "30% of cadences ignored due to the quality mask",
  and "period_grid defaults to R_star=1 and M_star=1".
- TLS raises the search lower bound (asked 0.5 d, got 0.858 d for WASP-18's priors). Fine for 0.94 d, but
  watch ultra-short-period planets.
- Git commits should end with the attribution line the harness specifies (Co-Authored-By trailer).

## 4. What was built: `tls_masked_search/`

A blind transit search using TLS, with an extra step: **transit-masked re-flattening**.

```
raw light curve
  -> clean: drop NaNs, normalize, lightkurve flatten (window 1001 pts ~ 33 h), clip upward 6-sigma spikes, bin 10 min
  -> pass 1: wide TLS, 0.5-10 d (benchmark: 0.5-12 d, capped at half the time span)   -> period, t0, duration
  -> re-flatten with transits masked (mask width = 2x pass-1 duration)
  -> pass 2: fine TLS in a narrow window (+/-0.5% of period; +/-0.003 d for WASP-18), oversampling 20
```

Why masking: flattening fits a smooth curve that dips along with the transit and absorbs part of it,
so transits look shallower. Masking them when estimating the baseline fixes that.

Files: `common.py` (settings, `clean_and_stitch`, `tls_wide`, `two_pass`, `relation`, `star_kwargs`),
`01_download_wasp18.py`, `02_search_wasp18.py`, `03_verify_wasp18.py`, `04_benchmark_100_stars.py`,
`05_ablation.py`, and `output/` (committed CSVs + `search.png`).

History: the folder was previously `wasp18b/` with BLS and TLS scripts. The user asked to replace all
BLS logic with TLS, tidy it, then rename the folder (done; pushed to `origin/main`, commit `6000533`).

## 5. Results so far

### 5a. WASP-18 b (TIC 100100827; sectors 2-3 for the search, all sectors for period refinement)
All five checks pass. Period 0.9414540 d (published 0.9414522 d, 0.2 s apart); depth 1.08%
(published 0.88-1.04%); duration 1.88 h (published 2.1-2.2 h; passes only because the tolerance is 15%).
Odd/even and mid-orbit checks pass. SDE ~37.7.

Lessons:
- **Exact agreement with published values is not expected.** Published values are ranges across papers,
  and depth/duration depend on method. "Close enough" is defined in `common.py` (`PERIOD_TOL_DAYS=1e-4`,
  `RANGE_TOL_FRAC=0.15`). Period is the real pass/fail check.
- TLS's own `duration` is conservative (1.73 h here). Step 2 now reports the width of the fitted model
  (first-to-last contact, ~1.88 h), which is what papers quote. The folded data suggest ~2.1 h, so the
  reported duration is still slightly low. Not fully resolved.
- **TLS on a multi-year, gappy light curve gives garbage duration/depth** (it reported 0.16 h): it assumes
  evenly spaced data. So step 3 uses all sectors **only for the period**; depth/duration come from the
  contiguous sectors 2-3 fit.
- Masking transits when flattening raised the depth from 0.98% to 1.08%.

### 5b. 100-star benchmark and ablation
Three pipelines share the **same** TLS call, period range, priors (+/-20% mass/radius, generic limb
darkening u=[0.3,0.2]); only preprocessing differs: **A** = repo detrending (1-day median filter, 0.1 d
edge trim), **B** = lightkurve 33 h flatten, **C** = B + masked second pass.

| | A | B | C |
|---|---|---|---|
| Known hosts recovered (of 60) | 33 | 41 | 41 |
| Comparison stars with SDE>=7 (of 40) | 16 | 10 | 10 |
| Injected transits recovered (of 200) | 140 | 148 | 148 |

"Recovered" = top TLS peak has SDE >= 7 and period within 1% of a cataloged transiting planet.

- A reproduces the repo's stored TLS signal-1 result (33/60) exactly, so the comparison harness is faithful.
- **A vs B on real hosts:** 8 hosts only B finds, 0 only A finds (exact paired/McNemar p = 0.008).
- **A vs B on injected transits:** 11 vs 3 (p = 0.057, not significant at 0.05). Difference is at shallow
  depths (250-500 ppm: 47 vs 39) and vanishes >= 1000 ppm.
- **B vs C identical**: the masked second pass adds no detections; it only improves measured depth
  (median +22% on hits).
- The 8 B-only hosts are mostly small/medium planets (median ~2.9 Earth radii vs 9.7 for hosts both find),
  half have long transits (3.4-4.4 h), brighter stars (median Tmag 9.5). Giants are found by both.
  Hypothesis (**not proven**): the 1-day median filter partly erases shallow/long dips.
- An earlier injection run used only "quiet" stars (21 of 40) and showed no difference; that filter
  likely removed the variable stars where detrending matters. The final run uses all 40 stars and
  reports a calm/variable split (variable: B-only 7 vs A-only 1, p = 0.070).

Honest framing: B is better than the repo's detrending at shallow planets; the real-host result is strongly
significant, the injection result only suggestive. It is **not** "better across the board".

### 5c. Caveats on these numbers
- Our 41/60 is **not directly comparable to the repo's reported 36/60**: the repo also requires BLS+TLS
  agreement, a correlated-noise (pink) SNR >= 7.1, observed-transit counts, and vetting flags. Ours is
  SDE >= 7 + period match only (no vetting).
- One sector per star; first signal only (the repo does up to two blind iterations).
- Injected planets: impact parameter 0.3, quadratic limb darkening, period log-uniform 0.8 to min(10, span/3).
- "Comparison star alarms" are not necessarily false positives (they are confirmed RV-planet hosts).
- Match tolerance is 1% (exact only for hit counting; half/double periods count as misses).

## 6. Where the user's head is now (the open discussion)

Last exchanges, summarised:

1. **Role of ML (agreed framing):** TLS stays the detector; ML is a **vetting classifier** that ranks the
   candidates the search produces (planet vs eclipsing binary / starspot / systematics). The repo currently
   uses hand-set thresholds (SDE, pink SNR, odd/even, secondary eclipse, centroid); ML would learn them.
2. **The 100 stars are a validation sample, not a training set** (only 60 positives, one sector each,
   selection-biased toward planet hosts). Do not "recreate a bigger one"; build a labeled dataset.
3. **Should we chase 60/60?** Recommendation given: **no**. Some misses are physically undetectable in one
   sector. The user suspects a gap in the approach. I agreed it is possible but said I had not yet analysed
   the 19 misses. Prior evidence: misses ("neither" group) have median planet radius ~2.5 Earth radii, period
   ~7.9 d (few transits), higher scatter (~606 ppm vs ~364 ppm for B-only hosts).

## 7. Recommended next steps (in order)

1. **Miss analysis (cheap, was about to start).** For each of the 19 hosts neither A nor B recovers, classify:
   - (a) *physically too faint*: expected SNR from catalog depth/duration, star's noise and the number of
     observed transits is below threshold (catalog `pl_trandep`, `pl_trandur`, `pl_orbper`, plus
     `point_scatter_ppm` in `reports/benchmark/results/tic*.json`);
   - (b) *found at a harmonic* (half/double period; our scoring counts it as a miss; the repo has a
     `harmonic_supported_match` column in `reports/benchmark/planet_recovery.csv`);
   - (c) *should have been found but wasn't*: this is the only real pipeline gap. Investigate those
     (period beyond half the sector span, gaps, momentum dumps, transit near sector edge, trimmed edges,
     TLS period grid, SDE just under 7, the 0.858 d lower bound).
   Fix buckets (b) and (c) before moving on. Keep ablation discipline: change one thing, measure paired.
2. **Build a labeled dataset at scale.** NASA TOI catalog dispositions (CP/KP confirmed, PC candidate,
   FP false positive, EB-like) for a few thousand stars + random non-TOI stars as extra negatives. Check
   `tess_hunt/catalog.py` for how the Exoplanet Archive is queried; the TOI table needs a similar fetch
   (ExoFOP/TFOPWG). Consider SPOC TCE/DV tables too.
3. **Run the pipeline on all of them and save every candidate signal** (top peaks with period, depth,
   duration, SDE, odd/even, secondary SNR, centroid shift, transit count, stellar params) **and the
   phase-folded global/local light-curve views.** Include multi-sector stars where possible (the benchmark
   only uses one sector).
4. **Baseline-first modelling.** Gradient-boosted trees on the tabular features; only try a 1D CNN on folded
   views (AstroNet-style) if the simple model plateaus. Must beat the current hand-set thresholds.
5. **Honest evaluation.** Split by **star** (never by signal), hold out later sectors in time, report
   precision/recall and PR curves (false positives vastly outnumber planets), not accuracy.
6. **Completeness via injection-recovery** (`05_ablation.py` already has the machinery) so "found nothing"
   is interpretable.
7. **Apply to new unlabeled sectors**, rank candidates, check known TOIs reappear (sanity), and vet the
   top unknowns by hand before claiming anything.

## 8. Known risks / open questions
- Catalog labels come from other pipelines (SPOC/QLP + human vetting); a model can learn their biases.
  Use time-split evaluation and be cautious about "new discovery" claims.
- Eclipsing binaries are the main false positive: ensure they are well represented as negatives.
- Multi-planet systems: only two blind iterations in the repo; ours is one pass per star.
- Residual WASP-18 b duration mismatch (1.88 h vs 2.1-2.2 h) is not fully explained.
- Generic limb darkening is used in the benchmark; WASP-18 uses TIC-based coefficients.
- Mask width (2x pass-1 duration) and fine-window width (+/-0.5%) were chosen by judgment, not tuned.
- Reproducibility: `output/*.csv` are committed; raw FITS and the WASP-18 cache are not.

## 9. How to reproduce
```bash
source .venv/bin/activate
python scripts/run_benchmark.py download          # raw FITS for the 100 stars (~minutes)
cd tls_masked_search
python 01_download_wasp18.py && python 02_search_wasp18.py && python 03_verify_wasp18.py
python 04_benchmark_100_stars.py                  # ~5 min
python 05_ablation.py                             # ~30 min; --quick for a smoke test; --injection-only reuses the catalog CSV
```
Note: the harness may have removed `tls_masked_search/data/` (WASP-18 cache); step 1 re-fetches it.

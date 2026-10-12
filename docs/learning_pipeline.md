# TESS supervised discovery pipeline

This pipeline searches real SPOC two-minute light curves, learns to rank the signals using confirmed planets and catalog-vetted negatives, and scores unmatched signals without assigning them negative training labels. It is an initial experiment, not a planet-confirmation system.

The executed starter run and diagnostic plots are in [the initial results report](../reports/learning-starter/README.md).

Run these commands from the repository root:

```sh
uv pip install --python .venv/bin/python -r requirements-learning.txt

.venv/bin/python -m tess_hunt.learning --run runs/tess-learning-starter prepare \
  --per-class 128 --sectors 1 --download-workers 8 --workers 2
.venv/bin/python -m tess_hunt.learning --run runs/tess-learning-starter train
.venv/bin/python -m tess_hunt.learning --run runs/tess-learning-starter inject --count 120
.venv/bin/python -m tess_hunt.learning --run runs/tess-learning-starter report

# Search a new TIC using the saved model and frozen validation threshold:
.venv/bin/python -m tess_hunt.learning --run runs/tess-learning-starter search --tic 261136679

# Or search already downloaded files (shell expands the wildcard):
.venv/bin/python -m tess_hunt.learning --run runs/tess-learning-starter search \
  --files data/raw/benchmark/*.fits
```

`--run` goes before the subcommand. CPU is the default; `train --device mps` and `train --device cuda` support available accelerators. `--workers` controls independent CPU search jobs, with one TLS thread per job. An initial 128 hosts per class will download at most 256 sectors; some hosts have no matching SPOC data. Archives can be slow. Successful downloads and per-star searches are cached. The default selects the latest available sector in the saved product inventory.

## Data and candidate generation

Original FITS files remain untouched in `data/raw/learning/`. Downloads are validated for target, sector, two-minute cadence and FITS checksums, and recorded with source URIs and SHA-256 hashes. Each run saves its catalog snapshot, selected stars, configuration, input manifest, and failures.

The initial catalog snapshot reuses `data/dataset/toi_labeled.csv` when available. Its original acquisition time is unknown; that limitation is recorded. `prepare --refresh-catalog` requests a fresh NASA TOI table. Product inventories are cached separately in `data/raw/learning/tic*_products.json`; deleting a TIC's inventory makes the next acquisition query newly released products. A cached inventory is not evidence that the archive has no newer sectors. Catalog absence means absent from this snapshot, not necessarily unknown to the community.

The cleaner uses PDCSAP flux and errors, Lightkurve's default TESS quality mask, separate contiguous observing segments, a cadence-aware 1.4-day flattening window, upward-only outlier clipping, and scatter-adjusted errors. Weighted ten-minute search bins never interpolate over missing observations. The neural network uses the cleaned two-minute samples.

The blind search uses TLS over 0.5–12 days, shortened to half the sector baseline. It searches up to three signals, masking events found by the preceding search. Candidate generation receives generic solar mass/radius priors, with the existing TLS helper's 20% prior bounds. These are a starting assumption, not measured stellar properties; unusual host densities or durations can limit sensitivity. No catalog period, epoch, disposition or planet size enters the search. Vetting measurements accompany candidates but do not impose a manual classification veto. Low-SDE candidates are retained for learning.

Each sector is searched independently. `prepare --sectors 2` uses two sectors of each star, producing separate candidates and views. This first version does **not** combine sectors to uncover signals detectable only in their combined baseline; that is an explicit next search improvement. The search currently requires repeat events and will not recover arbitrary single-transit planets.

## Labels and partitions

The default positive class is **CP**, confirmed planets. **FP** and **FA** are negative classes, preserving their subtype in the output. Optionally add **KP** to `positive_dispositions` in a new configuration. PC, APC, missing catalog matches, half/double aliases, missing epochs, conflicting matches and phase mismatches remain `y = -1` and do not enter supervised training.

Labels apply to **signals**, not stars. A match requires period agreement within 0.5% and predicted transit timing agreement across the observing baseline. Catalog BJD epochs are converted to BTJD. Timing tolerance is 0.75 times the larger catalog/detected duration, capped at 10% of the period. This deliberately conservative check can exclude legitimate planets with stale ephemerides or transit-timing variations. Review excluded matches rather than silently relaxing their labels.

With the CP-only default, a matched KP signal also has `y = -1` because it is outside the chosen training class; its `label = KP` still identifies it as a known planet. The generic `unresolved_disposition` reason means outside the configured supervised label sets, including this KP case. Do not interpret every row with `y = -1` as a potentially new planet.

A deterministic hash of TIC ID and seed assigns 70%/15%/15% train/validation/test partitions. Every sector and signal of the same star stays together, and assignments persist when the dataset grows. The existing 100-star benchmark is reserved from training selection. Nearby stars sharing contaminating signals are not automatically grouped; TIC grouping alone cannot eliminate every contamination-related dependence.

`catalog_recovery.csv` records every reliable catalog signal on the observed hosts, including signals the search missed. Classifier metrics concern recovered labeled signals. Search recovery and end-to-end injection recovery are separate measurements; a high classification score cannot compensate for an undetected signal.

## Learned representations and evaluation

The CNN has three convolutional encoders:

- A 256-bin view of the full folded orbit, including any secondary eclipse.
- A 128-bin close-up spanning ±2.5 candidate durations.
- Up to 16 individual transit windows with 64 bins each. Their encodings are pooled by masked mean and spread, preserving information about inconsistent events. Events are selected evenly over time when there are more than 16.

Every view contains standardized brightness, propagated uncertainty and an observed-bin mask. Missing bins remain zero with mask zero; they are not invented observations. Padded event windows are excluded from pooling. Four direct measurements—log period, log duration, log photometric scatter, and depth relative to scatter—provide amplitude and scale information. Scalar normalization is fitted on training data only. TIC, sector identifiers, host cohort and catalog parameters are excluded from model features.

A gradient-boosted feature baseline trains on the same labeled training rows. Both models select independent thresholds on validation negatives, allowing at most a configured 10% validation false-positive rate, conservatively handling ties. Small validation sets make this threshold coarse. Test labels never influence fitting, early stopping, scalar normalization or threshold selection.

`metrics.json` reports sample sizes, PR-AUC, ROC-AUC, recall, false-positive rate and sample precision for each model, plus depth/period subgroups. The selected labeled sample is not the survey population; its precision is not the expected precision of an archive-wide search. Neither model's score is a calibrated probability of a planet. Repeatedly redesigning the pipeline after inspecting test results would turn that test set into development data; reserve fresh stars for a subsequent final evaluation.

`scores.csv` includes every detected signal. `discovery_candidates.csv` ranks all unlabeled signals and identifies whether their star was used in training. An unmatched signal on a training star is useful for exploration but is not an independent generalization test. `search` applies the saved preprocessing, model and threshold to new observations and cross-matches afterward against the run's catalog snapshot.

## Injection recovery

`inject` draws circular, limb-darkened, exposure-integrated batman transits with periods from 0.8 days to the search ceiling, nominal depths 300–10,000 ppm, random epochs and impact parameters, and a solar host. Nominal depth is radius ratio squared; sampled depth changes with limb darkening and geometry. It multiplies the original PDCSAP flux by the transit model **before** quality filtering, cleaning, flattening, binning, search and scoring. Original files are unchanged.

Only held-out test stars are used. The trained model and validation threshold remain frozen. Unmodified paired controls identify chance matches to signals already present. Recovery requires matching period and event timing; aliases are recorded separately. Summary denominators exclude preexisting matches and injections with fewer than two observed events, while retaining all trials in the CSV. Depth-binned results show whether failure occurs in search or learned selection.

These injections test processing after PDCSAP extraction. They do not measure losses from pixel extraction, SPOC correction, dilution, all stellar geometries, or the astrophysical false-positive population. Control light curves can contain real signals, so their selected candidates are not automatically false alarms.

## Scaling experiments

Copy `config/learning.json`, change the experiment, and use a new run directory:

```sh
.venv/bin/python -m tess_hunt.learning --run runs/tess-learning-larger prepare \
  --config config/learning.json --per-class 512 --workers 4
.venv/bin/python -m tess_hunt.learning --run runs/tess-learning-larger train
```

First grow the real labeled sample, then compare larger encoder `width`, more epochs, and broader candidate searches. Keep search and model changes measurable against the feature baseline and injections. More unlabeled light curves do not become supervised training examples automatically; self-supervised pretraining is a possible later experiment. Ensemble learning, uncertainty calibration, pixel/centroid inputs and combined-sector searches are also future improvements.

Runs and checkpoints are Git-ignored. Preserve the run directory to reproduce its scores. Source/configuration fingerprints invalidate stale searches or models; changes to the pipeline require preparing and training compatible artifacts. The learning dependencies have their own lockfile, leaving the existing environment definition available.

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Primary references: [TESS data products](https://heasarc.gsfc.nasa.gov/docs/tess/data-products.html), [NASA TOI disposition and time definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html), [batman transit modeling](https://lkreidberg.github.io/batman/docs/html/tutorial.html).

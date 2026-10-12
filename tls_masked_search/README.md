# TESS transit search and planet-vs-fake classifier

For the confirmed-label training and discovery experiment, see [the learning pipeline guide](../docs/learning_pipeline.md).
Step 6 now trains only on signals matched to CP/KP or FP/FA labels. Unmatched signals on every host
remain unlabeled. The classifier results below describe the earlier experiment, which included
unmatched signals on false-positive hosts as negatives; rerun step 6 to evaluate the revised labels.
Legacy step-5 matches use period only; the new pipeline additionally checks transit timing.

Finds repeating dips in TESS 2-min light curves with Transit Least Squares (TLS), then scores each
dip with a model trained on NASA's expert-labeled TESS candidates (TOIs).

For an executed walkthrough, open [Inside `tls_masked_search`](../notebooks/03_tls_masked_search_walkthrough.ipynb).
It starts with a census of both datasets, selects TOI-880 from the benchmark, and follows its original
flux through cleaning, wide TLS, transit-masked flattening, local refinement and a second search.
It returns that entry to the aggregate benchmark, then traces an actual classifier-dataset entry
from saved measurements through feature transformations to a freshly computed held-out score,
before returning to aggregate classifier results. It distinguishes the two cohorts and their search paths.
The notebook checks its expanded search against `common.two_pass` and compares with saved results.

```bash
# From the repository root, with the project environment active:
jupyter lab notebooks/03_tls_masked_search_walkthrough.ipynb
```

## The method

```
TESS 2-min light curve
  -> clean: drop bad points, normalize, flatten (33 h window), clip upward spikes, bin to 10 min
  -> TLS search, 0.5-12 d (benchmark) or 0.5-30 d (dataset)   -> period, dip time, length
  -> re-flatten with the dips masked, fine TLS around the peak -> dip not eaten by the smoothing
  -> hide signal 1's dips, search again                        -> second signal
  -> measure each signal (tess_hunt.search.vet_signal): odd/even depths, secondary eclipse,
     star-position shift during dips, noise-aware SNR, dip counts
  -> classifier: probability that the signal is a planet
```

Multi-sector stars are searched in runs of up to 4 consecutive sectors (at most 2 runs per star).
TLS on years of gappy data returns wrong dip lengths, which breaks every dip measurement.

## Files

| File | What it does |
|---|---|
| `common.py` | Shared settings, the cleaning recipe (`clean_and_stitch`) and the two-pass search (`two_pass`). |
| `04_benchmark_100_stars.py` | Runs the search on the repo's 100-star benchmark. Writes `output/benchmark_100*.csv`. |
| `05_build_dataset.py` | Builds the labeled training set from NASA TOI dispositions. Writes `output/dataset_*.csv`. |
| `06_train_classifier.py` | Trains and cross-validates the classifier against the fixed rules. Writes `output/classifier_scores.csv`. |
| `07_build_views.py` | Folds each signal's light curve into fixed-size "snapshots" of the dip shape (whole orbit, zoom, odd, even, halfway point, star position). Writes `output/views.npz`. CPU, ~6 min. |
| `08_train_shape_model.py` | Shape model: a small 1D neural network (PyTorch) on the snapshots plus 06's measurements, same folds as 06. 5 copies per fold; their score spread is saved as `shape_score_spread` ("unsure" flag). Writes `output/shape_scores.csv`. GPU, ~5 min. `--extra-views pixel` adds step 11 (see Results). |
| `09_rank_candidates.py` | Runs steps 5 and 7 on every star with an undecided (PC/APC) TOI, scores each signal with 06/08 trained on all labeled rows, and compares with NASA's ExoMiner++ catalog. Writes `output/candidate*`. |
| `11_build_pixel_views.py` | Difference images (out-of-dip minus in-dip pixels from the SPOC target pixel files): which star dimmed. Downloads TPFs to `data/dataset_tpf/`. Writes `output/dataset_pixel_views.npz`. |
| `live.py` | Runs step 4 and shows it live in the TESS visualizer ("Mission Control"). See below. |

## Run

From the repo root with the project environment active (`pip install -r requirements.txt`):

```bash
cd tls_masked_search
python ../scripts/run_benchmark.py download    # once: raw files for the 100 benchmark stars
python 04_benchmark_100_stars.py                # ~5 min
python 05_build_dataset.py                      # ~25 min, ~15 GB in data/raw/dataset/ (--per-class N for less)
python 06_train_classifier.py                   # ~1 min
python 07_build_views.py                        # ~6 min
python 08_train_shape_model.py                  # ~5 min on one MI300X (ROCm PyTorch)
python 11_build_pixel_views.py                  # difference images (downloads TPFs)
python 08_train_shape_model.py --extra-views pixel   # (b) without/with difference images
```

Run from inside this folder: the scripts import `common.py` from the current directory. Keep the
`if __name__ == "__main__":` guard in each script: TLS starts worker processes that re-import it.

Variant datasets: `--name N` (05-08) reads/writes `output/N_*` instead. 05 also takes `--max-chunks K`
(runs of up to 4 sectors per star, default 2), `--random-stars M` (M no-TOI 2-min stars whose signals
are labeled `NT`; 06/08 use them as fakes only with `--nt`) and `--star-props` (TIC magnitude, radius,
temperature, log g per star). 07 needs the same `--max-chunks` as 05. The larger set:

```bash
python 05_build_dataset.py --name dataset_v2 --max-chunks 4 --random-stars 1000 --star-props   # ~27 min
python 07_build_views.py --name dataset_v2 --max-chunks 4                                      # ~14 min
```

## Watch it live

`live.py` runs step 4 and streams what it is doing (which star, each cleaning and search stage, the
result) to the TESS visualizer in your browser:

```bash
python live.py                                 # serve the page; click Run pipeline there
python live.py --limit 5                       # or start a run right away: the first 5 stars
python live.py --tics 26826078 34077285        # pick stars by TIC ID (from data/benchmark/targets.csv)
python live.py --tics 26826078 --port 8000 --no-browser
```

It serves the page at http://127.0.0.1:8765/ (local only) and keeps serving after a run ends; Ctrl-C to
stop. Click **Run pipeline** on the page and pick the 7-star tour, the first N stars, or TIC IDs; a new
run replaces the one in progress. Extra command-line arguments go to `04_benchmark_100_stars.py`.
Endpoints: `GET /targets`, `GET /run` (running, finished, failed or stopped, and the exit code),
`POST /run` with JSON `{"tics": [...]}` or `{"limit": N}` (400 unless whole numbers and known TICs),
`POST /stop`, and `GET /events`. Opening `reports/tess-survey/index.html` directly plays the recorded sample instead.
The run's event log and CSVs go to `output/live/` (git-ignored), so the committed results stay as they are.
To record a run for replay (load the file in the page, or rebuild the page's built-in sample):

```bash
python 04_benchmark_100_stars.py --tics 26826078 34077285 --out output/live \
    --events ../reports/tess-survey/runs/sample.jsonl
```

`--out` keeps the CSVs for just those stars out of `output/`, where they would replace the full results.

## Results

**Benchmark (100 stars, one sector each).** A host counts as found if a signal has SDE >= 7 and its
period is within 1% of a cataloged transiting planet.

| | Found (of 60 known hosts) | Comparison stars flagged (of 40) |
|---|---|---|
| This search, two signals | **44** (small planets 10 of 20) | 14 |
| Repo's TLS, first signal, same scoring | 33 | - |

The repo's full pipeline reports 36/60, but it also requires BLS agreement and vetting checks, so it
isn't directly comparable. Of the 16 hosts still missed, about 9 are too faint or show too few dips in
one sector.

**Dataset** (`data/dataset/toi_labeled.csv` labels, all planet stars plus as many fake stars):
2,204 stars with 2-min data; our search finds the labeled TOI on 87% of planet stars and 64% of
fake stars. Training uses only catalog-matched signals: 1,725 dips (1,111 CP/KP planets, 614 FP/FA
fakes). Unmatched signals and half/double matches stay unlabeled. 06's `load()` drops 18 stars that have
no 2-min file of their own (older step-5 runs kept a neighbour star's file; 17 of the 18 were fakes).

**Classifiers** (5-fold cross-validation grouped by star; same folds for both), on NASA-confirmed
planets vs NASA-known fakes (1,725 dips):

| | Ranking (ROC AUC) | Planets caught at the rules' fake rate | Planets called planet | Fakes called planet | Overall correct |
|---|---|---|---|---|---|
| Shape model: snapshots + measurements (08) | **0.90 +/- 0.02** | **90% +/- 2%** | 90% | 29% | 83% |
| Gradient-boosted trees on measurements (06) | 0.86 +/- 0.01 | 83% +/- 3% | 88% | 37% | 79% |
| Fixed rules (SDE >= 7, noise SNR >= 7.1, >= 2 dips, no flags) | - | 60% | 60% | 30% | 64% |

Snapshots alone, without the measurements, score the same as the trees (0.86). Both models beat the
rules in every group of stars with the same number of months of data (planet stars get re-observed
more, so data-amount alone predicts the label; features are per-dip to limit that).

**Unsure flag** (`shape_score_spread`, the spread of the 5 copies' scores), shape model above, by spread quartile:

| Spread quartile | Q1 (lowest) | Q2 | Q3 | Q4 (highest) |
|---|---|---|---|---|
| Correct at 0.5 | 95% | 91% | 82% | 65% |
| ROC AUC within the bin | 0.96 | 0.94 | 0.87 | 0.69 |

Higher spread does mean more mistakes, but the score's closeness to 0.5 finds wrong calls slightly better
(ROC AUC 0.77 vs 0.74 for spread; together 0.77). Where it adds something: among confident scores (< 0.2 or
> 0.8), high-spread rows are wrong ~4x as often (13-14% vs 3%). `09` writes it to `candidate_ranking.csv`.

**Extra views** (`08 --extra-views pixel`, one GPU session, same folds and seeds; shape model (b)
0.90 +/- 0.02, 90% +/- 2% without):

| | ROC AUC | Planets caught at the rules' fake rate | Per-fold ROC AUC change |
|---|---|---|---|
| + difference images (11) | **0.91 +/- 0.02** | **92% +/- 1%** | +0.021 +0.006 +0.004 +0.005 +0.008 |

Difference images help a little in every fold: NASA fakes whose dimming peaks > 1 px off the target (171)
scored > 0.5 12% -> 3% of the time (centroid_snr > 3: 23% -> 18%). The gain is small, within the fold
spread, so the views stay opt-in. Each-dip views (each dip side by side) were tried too: worse in all 5 folds, not kept.

**More months per star** (`dataset_v2`: up to 16 sectors instead of 8): shape model 0.90 -> 0.91,
trees 0.86 -> 0.87, within the fold spread. **Random "nothing here" stars** (`--nt`): cost ~0.01 (shape)
to ~0.03 (trees) on the task above, within the fold spread; but without them over half of random-star
signals score > 0.5, with them ~1%.

**Undecided candidates** (`09_rank_candidates.py score --name dataset_v2 --nt`: trained with the random
"nothing here" stars, no star features; 4,338 stars with 2-min data, 22 without their own file dropped):
we find 3,377 of NASA's undecided TOIs (3,396 dips); only those are ranked. Junk dips matching no TOI
score > 0.5 just 4% of the time. On the 2,018 also scored by ExoMiner++, scores correlate at 0.55
(Spearman) and agree on planet/fake at a 0.5 cutoff 75% of the time. `score` needs
`output/dataset_v2_views.npz` and `output/candidate_views.npz` (~35 MB, not in git): rebuild them with
`07_build_views.py --name dataset_v2 --max-chunks 4` and `09_rank_candidates.py build`.

**Star features** (opt-in: `--star-features` in 06/08; writes `output/star_*_scores.csv`). Star radius,
temperature and log g from the TOI table, plus two physics checks: the size of the dimming body
(sqrt(depth) x star radius) and measured / expected dip length for that period and star. Brightness
(Tmag) is left out: alone it scores AUC 0.58 and adds nothing on top. With FP/FA-only fakes, same folds
(ROC AUC, planets caught at the rules' fake rate; measured before the 18-star drop, as is the next paragraph):

| | Without | With star features |
|---|---|---|
| Trees (06) | 0.86 +/- 0.02, 83% +/- 6% | **0.92 +/- 0.01, 92% +/- 3%** (all 5 folds better) |
| Shape model (08 b) | 0.90 +/- 0.03, 92% +/- 4% | 0.92 +/- 0.02, 92% +/- 3% (planets caught: no clear change) |
| Average of both (08 c) | 0.90 +/- 0.03, 90% +/- 3% | **0.93 +/- 0.01, 94% +/- 2%** |

Missing star values mark fakes (radius missing for 13% of fake dips, 1% of planets), but on stars with
all values known the trees still gain +0.05 AUC. The model beats the rules in every brightness quartile.

## Why it's built this way

Tested in earlier experiments (scripts in git history, commit `ada76fc`), judged mainly on injected
fake planets:
- 33 h lightkurve flattening beats the repo's 1-day median filter on small planets (41 vs 33 hosts).
- Masking the dips before re-flattening doesn't add detections, but measures depth ~22% more accurately.
- A second search with signal 1 hidden finds the weaker of two fake planets 90/120 times vs 5/120.
- Shorter smoothing, a lower SDE cutoff and a longer period range were rejected (lost planets or added
  false alarms).

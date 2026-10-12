# TESS transit search and planet-vs-fake classifier

For the confirmed-label training and discovery experiment, see [the learning pipeline guide](../docs/learning_pipeline.md).
Step 3 now trains only on signals matched to CP/KP or FP/FA labels. Unmatched signals on every host
remain unlabeled. The classifier results below describe the earlier experiment, which included
unmatched signals on false-positive hosts as negatives; rerun step 3 to evaluate the revised labels.
Legacy step-2 matches use period only; the new pipeline additionally checks transit timing.

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
| `01_benchmark.py` | Runs the search on the repo's 100-star benchmark. Writes `output/benchmark_100*.csv`. |
| `02_build_dataset.py` | Builds the labeled training set from NASA TOI dispositions. Writes `output/dataset_*.csv`. |
| `03_train_trees.py` | Trains and cross-validates gradient-boosted trees on the measurements against the fixed rules. Writes `output/classifier_scores.csv`. |
| `04_build_snapshots.py` | Folds each signal's light curve into fixed-size "snapshots" of the dip shape (whole orbit, zoom, odd, even, halfway point, star position). Writes `output/views.npz`. CPU, ~6 min. |
| `05_build_pixel_views.py` | Difference images (out-of-dip minus in-dip pixels from the SPOC target pixel files): which star dimmed. Streams TPFs through `data/raw/tpf_dataset/` (deleted per star). Writes `output/dataset_pixel_views.npz`. |
| `06_train_shape_model.py` | Shape model: a small neural network (PyTorch) on the snapshots, the pixel images and 03's measurements, same folds as 03. 5 copies per fold; their score spread is saved as `shape_score_spread` ("unsure" flag). Writes `output/shape_scores.csv`. GPU, ~5 min. `--no-pixel` leaves the pixel images out. |
| `07_rank_candidates.py` | Runs steps 2, 4 (and then 5) on every star with an undecided (PC/APC) TOI, scores each signal with 03/06 trained on all labeled rows, and compares with NASA's ExoMiner++ catalog. Writes `output/candidate*`. |
| `live.py` | Runs step 1 and shows it live in the TESS visualizer ("Mission Control"). See below. |

## Run

From the repo root with the project environment active (`pip install -r requirements.txt`):

```bash
cd tls_masked_search
python ../scripts/run_benchmark.py download    # once: raw files for the 100 benchmark stars
python 01_benchmark.py                          # ~5 min
python 02_build_dataset.py                      # ~25 min, ~15 GB in data/raw/dataset/ (--per-class N for less)
python 03_train_trees.py                        # ~1 min
python 04_build_snapshots.py                    # ~6 min
python 05_build_pixel_views.py                  # difference images (streams ~140 GB of TPFs, keeps none)
python 06_train_shape_model.py                  # ~5 min on one MI300X (ROCm PyTorch)
```

Run from inside this folder: the scripts import `common.py` from the current directory. Keep the
`if __name__ == "__main__":` guard in each script: TLS starts worker processes that re-import it.

Variant datasets: `--name N` (02-06) reads/writes `output/N_*` instead. 02 also takes `--max-chunks K`
(runs of up to 4 sectors per star, default 2), `--random-stars M` (M no-TOI 2-min stars whose signals
are labeled `NT`; 03/06 use them as fakes only with `--nt`) and `--star-props` (TIC magnitude, radius,
temperature, log g per star). 04 and 05 need the same `--max-chunks` as 02. The larger set:

```bash
python 02_build_dataset.py --name dataset_v2 --max-chunks 4 --random-stars 1000 --star-props   # ~27 min
python 04_build_snapshots.py --name dataset_v2 --max-chunks 4                                  # ~14 min
python 05_build_pixel_views.py --name dataset_v2 --max-chunks 4 --reuse dataset                # ~15 min
```

Candidates: `07_rank_candidates.py build`, `05_build_pixel_views.py --name candidates`, then
`07_rank_candidates.py score --name dataset_v2 --nt` (GPU) and `07_rank_candidates.py compare`.

## Watch it live

`live.py` runs step 1 and streams what it is doing (which star, each cleaning and search stage, the
result) to the TESS visualizer in your browser:

```bash
python live.py                                 # serve the page; click Run pipeline there
python live.py --limit 5                       # or start a run right away: the first 5 stars
python live.py --tics 26826078 34077285        # pick stars by TIC ID (from data/benchmark/targets.csv)
python live.py --tics 26826078 --port 8000 --no-browser
```

It serves the page at http://127.0.0.1:8765/ (local only) and keeps serving after a run ends; Ctrl-C to
stop. Click **Run pipeline** on the page and pick the 7-star tour, the first N stars, or TIC IDs; a new
run replaces the one in progress. Extra command-line arguments go to `01_benchmark.py`.
Endpoints: `GET /targets`, `GET /run` (running, finished, failed or stopped, and the exit code),
`POST /run` with JSON `{"tics": [...]}` or `{"limit": N}` (400 unless whole numbers and known TICs),
`POST /stop`, and `GET /events`. Opening `reports/tess-survey/index.html` directly plays the recorded sample instead.
The run's event log and CSVs go to `output/live/` (git-ignored), so the committed results stay as they are.
To record a run for replay (load the file in the page, or rebuild the page's built-in sample):

```bash
python 01_benchmark.py --tics 26826078 34077285 --out output/live \
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
fakes). Unmatched signals and half/double matches stay unlabeled. 03's `load()` drops 18 stars that have
no 2-min file of their own (older step-2 runs kept a neighbour star's file; 17 of the 18 were fakes).

**Classifiers** (5-fold cross-validation grouped by star; same folds for both), on NASA-confirmed
planets vs NASA-known fakes (1,725 dips):

| | Ranking (ROC AUC) | Planets caught at the rules' fake rate | Planets called planet | Fakes called planet | Overall correct |
|---|---|---|---|---|---|
| Shape model: snapshots + pixel images + measurements (06, default) | **0.91 +/- 0.02** | **92% +/- 1%** | 90% | 26% | 85% |
| Gradient-boosted trees on measurements (03) | 0.86 +/- 0.01 | 83% +/- 3% | 88% | 37% | 79% |
| Fixed rules (SDE >= 7, noise SNR >= 7.1, >= 2 dips, no flags) | - | 60% | 60% | 30% | 64% |

Snapshots alone, without pixel images or measurements, score 0.86 +/- 0.03 (87% caught), like the trees.
Both models beat the rules in every group of stars with the same number of months of data (planet stars
get re-observed more, so data-amount alone predicts the label; features are per-dip to limit that).

**Unsure flag** (`shape_score_spread`, the spread of the 5 copies' scores), shape model above, by spread quartile:

| Spread quartile | Q1 (lowest) | Q2 | Q3 | Q4 (highest) |
|---|---|---|---|---|
| Correct at 0.5 | 97% | 91% | 82% | 68% |
| ROC AUC within the bin | 0.98 | 0.89 | 0.89 | 0.72 |

Higher spread does mean more mistakes, but the score's closeness to 0.5 finds wrong calls slightly better
(ROC AUC 0.78 vs 0.74 for spread). Earlier, without pixel images: among confident scores (< 0.2 or
> 0.8), high-spread rows were wrong ~4x as often (13-14% vs 3%). `07` writes it to `candidate_ranking.csv`.

**Pixel images** (step 5; measured as an opt-in extra in one GPU session, same folds and seeds; now
the default, `--no-pixel` to leave them out; shape model 0.90 +/- 0.02, 90% +/- 2% without):

| | ROC AUC | Planets caught at the rules' fake rate | Per-fold ROC AUC change |
|---|---|---|---|
| + difference images (5) | **0.91 +/- 0.02** | **92% +/- 1%** | +0.021 +0.006 +0.004 +0.005 +0.008 |

Difference images help a little in every fold: NASA fakes whose dimming peaks > 1 px off the target (171)
scored > 0.5 12% -> 3% of the time (centroid_snr > 3: 23% -> 18%). The gain is small, within the fold
spread. Each-dip views (each dip side by side) and synthetic injected training rows were tried too:
no gain, not kept.

**More months per star** (`dataset_v2`: up to 16 sectors instead of 8; without pixel images): shape model 0.90 -> 0.91,
trees 0.86 -> 0.87, within the fold spread. **Random "nothing here" stars** (`--nt`): cost ~0.01 (shape)
to ~0.03 (trees) on the task above, within the fold spread; but without them over half of random-star
signals score > 0.5, with them ~1%.

**Undecided candidates** (`07_rank_candidates.py score --name dataset_v2 --nt`: trained with the random
"nothing here" stars and pixel images, no star features; 4,338 stars with 2-min data, 22 without their own
file dropped): we find 3,377 of NASA's undecided TOIs (3,396 dips); only those are ranked. Junk dips
matching no TOI score > 0.5 6% of the time. On the 2,018 also scored by ExoMiner++, scores correlate at
0.42 (Spearman) and agree on planet/fake at a 0.5 cutoff 73% of the time (without pixel images: 0.55,
75%, junk 4%; the two shape scores correlate at 0.69 and make the same call at 0.5 for 79% of dips).
ExoMiner++ is another model, not ground truth.

**Star features** (opt-in: `--star-features` in 03/06; writes `output/star_*_scores.csv`, not in git). Star radius,
temperature and log g from the TOI table, plus two physics checks: the size of the dimming body
(sqrt(depth) x star radius) and measured / expected dip length for that period and star. Brightness
(Tmag) is left out: alone it scores AUC 0.58 and adds nothing on top. With FP/FA-only fakes, same folds
(ROC AUC, planets caught at the rules' fake rate; measured before the 18-star drop and without pixel images, as is the next paragraph):

| | Without | With star features |
|---|---|---|
| Trees (03) | 0.86 +/- 0.02, 83% +/- 6% | **0.92 +/- 0.01, 92% +/- 3%** (all 5 folds better) |
| Shape model (06 b) | 0.90 +/- 0.03, 92% +/- 4% | 0.92 +/- 0.02, 92% +/- 3% (planets caught: no clear change) |
| Average of both (06 c) | 0.90 +/- 0.03, 90% +/- 3% | **0.93 +/- 0.01, 94% +/- 2%** |

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

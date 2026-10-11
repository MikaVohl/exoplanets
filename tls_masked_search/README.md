# TESS transit search and planet-vs-fake classifier

Finds repeating dips in TESS 2-min light curves with Transit Least Squares (TLS), then scores each
dip with a model trained on NASA's expert-labeled TESS candidates (TOIs).

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
| `08_train_shape_model.py` | Shape model: a small 1D neural network (PyTorch) on the snapshots plus 06's measurements, same folds as 06. Writes `output/shape_scores.csv`. GPU, ~5 min. |
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
```

Run from inside this folder: the scripts import `common.py` from the current directory. Keep the
`if __name__ == "__main__":` guard in each script: TLS starts worker processes that re-import it.

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
fake stars. 3,198 training signals: 1,112 planets, 2,086 fakes (631 FP/FA-matched dips, plus unmatched
signals on fake-only stars). Unmatched signals on planet stars and half/double matches are left out.

**Classifiers** (5-fold cross-validation grouped by star; same folds for both), on the hard task of
NASA-confirmed planets vs NASA-known fakes (1,743 dips):

| | Ranking (ROC AUC) | Planets caught at the rules' fake rate | Planets called planet | Fakes called planet | Overall correct |
|---|---|---|---|---|---|
| Shape model: snapshots + measurements (08) | **0.87 +/- 0.02** | **87% +/- 3%** | 85% | 28% | 80% |
| Gradient-boosted trees on measurements (06) | 0.83 +/- 0.02 | 81% +/- 3% | 79% | 28% | 77% |
| Fixed rules (SDE >= 7, noise SNR >= 7.1, >= 2 dips, no flags) | - | 60% | 60% | 30% | 63% |

The shape model beats the trees in all 5 folds (+0.02 to +0.06 AUC). Snapshots alone, without the
measurements, are no better than the trees. Both models beat the rules in every group of stars with
the same number of months of data (planet stars get re-observed more, so data-amount alone predicts
the label; features are per-dip to limit that).

## Why it's built this way

Tested in earlier experiments (scripts in git history, commit `ada76fc`), judged mainly on injected
fake planets:
- 33 h lightkurve flattening beats the repo's 1-day median filter on small planets (41 vs 33 hosts).
- Masking the dips before re-flattening doesn't add detections, but measures depth ~22% more accurately.
- A second search with signal 1 hidden finds the weaker of two fake planets 90/120 times vs 5/120.
- Shorter smoothing, a lower SDE cutoff and a longer period range were rejected (lost planets or added
  false alarms).

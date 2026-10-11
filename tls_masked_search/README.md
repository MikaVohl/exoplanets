# TLS transit search with transit-masked flattening

A blind transit search for TESS light curves, built on Transit Least Squares (TLS), with one
extra step: after finding a candidate, the star's slow brightness drift is re-estimated while
**ignoring the transits**, so the smoothing curve doesn't eat into the dip. It is demonstrated on the
known hot Jupiter WASP-18 b, then tested on the repo's 100-star benchmark sample.

## The method

```
raw TESS light curve
  -> clean: drop bad points, normalize, flatten (33 h window), clip upward spikes, bin to 10 min
  -> pass 1: wide TLS search, 0.5-10 d           -> rough period, t0, duration
  -> re-flatten with the transits masked         -> baseline no longer dragged into the dips
  -> pass 2: fine TLS search around the peak     -> refined period, depth, duration
  -> benchmark only: hide signal 1's dips, repeat -> second signal (finds the weaker planet of a pair)
```

Flattening removes slow drift, but the smoothing curve dips along with a transit and absorbs part of
it, making the transit look shallower. Masking the transits when re-flattening fixes that
(depth rose ~10% on WASP-18 b, ~22% median on the benchmark hits).

## Files

| File | What it does |
|---|---|
| `common.py` | Shared settings, the cleaning recipe (`clean_and_stitch`), and the two-pass search (`two_pass`). |
| `01_download_wasp18.py` | Downloads WASP-18's TESS light curves (all sectors) into `data/` (git-ignored). |
| `02_search_wasp18.py` | Runs the method on sectors 2-3. Writes `output/results.json` and `output/search.png`. |
| `03_verify_wasp18.py` | Refines the period over all sectors, runs sanity checks, prints PASS/FAIL vs published values. |
| `04_benchmark_100_stars.py` | Runs the method on the repo's 100-star sample. Writes `output/benchmark_100*.csv`. |
| `05_ablation.py` | Tests whether the method is really better, and which part helps. Writes `output/ablation_*.csv`. |
| `06_miss_analysis.py` | For the known planets both A and B miss: how strong the dip really is, how many dips have data, and whether the found period is a multiple. Writes `output/miss_table.csv`. |
| `07_fix_experiments.py` | Tests one fix at a time against B (second search, shorter smoothing, wider period range, lower score cutoff), judged mainly on fake planets. Writes `output/fix_experiments.csv`. |

## Run

From the repo root with the project environment active (`pip install -r requirements.txt`):

```bash
cd tls_masked_search
python 01_download_wasp18.py
python 02_search_wasp18.py          # a few minutes
python 03_verify_wasp18.py          # slower: uses all sectors

python ../scripts/run_benchmark.py download    # once: raw files for the 100 stars
python 04_benchmark_100_stars.py                # ~5 min
python 05_ablation.py                           # ~30 min (use --quick for a smoke test)
python 06_miss_analysis.py                      # ~1 min
python 07_fix_experiments.py                    # ~1 min on many cores (--quick, --workers N)
```

Run from inside this folder: the scripts import `common.py` from the current directory. The
`if __name__ == "__main__":` guard in each script is required; TLS starts worker processes that
re-import the script (Python 3.14 uses `forkserver`), and without it they would re-run the search.

## Results

**WASP-18 b** (sectors 2-3, then all sectors for the period). All five checks pass.

| | Found | Published |
|---|---|---|
| Period | 0.9414540 d | 0.9414522 d (0.2 s apart) |
| Depth | 1.08% | 0.88-1.04% |
| Duration | 1.88 h | 2.1-2.2 h (low; the folded data suggests ~2.1 h) |

"Close enough" is set in `common.py`: period within 1e-4 d, depth and duration within 15% of the
published range. Depth and duration legitimately vary with method, so exact matches aren't expected.

**100-star benchmark.** A host counts as recovered if the top TLS peak has SDE >= 7 and its period is
within 1% of a cataloged transiting planet. Three pipelines share the same TLS call; only the
preprocessing differs: **A** = repo's 1-day median filter, **B** = 33 h lightkurve flatten,
**C** = B + masked second pass.

| | A | B | C |
|---|---|---|---|
| Known hosts recovered (of 60) | 33 | 41 | 41 |
| Comparison stars with SDE >= 7 (of 40) | 16 | 10 | 10 |
| Injected transits recovered (of 200) | 140 | 148 | 148 |

- **B beats A on real hosts**: 8 hosts only B finds, 0 only A finds (exact paired test p = 0.008). The extra
  hosts are mostly small or medium planets and long transits; giants are found by both.
- **Injection test points the same way but is weaker**: B-only 11 vs A-only 3 (p = 0.057, not significant
  at 0.05). The gap is at shallow depths (250-500 ppm: 47 vs 39) and vanishes at 1000 ppm and above.
- **The masked second pass (C) adds no detections** (B and C are identical). Its value is accuracy of the
  measured depth, not detection.
- Limits: one sector per star, SDE-and-period criterion only (the repo's own 36/60 also requires BLS
  agreement and vetting checks), generic limb darkening, injected planets at impact parameter 0.3.

**Why the remaining hosts are missed** (`06_miss_analysis.py`, 19 hosts neither A nor B finds): about 9
are too faint or show too few dips in one sector; the rest are period mix-ups, a correct period scoring
just under 7, a stronger signal winning, or very active stars. Per-planet numbers in `output/miss_table.csv`.

**Fixes tested one at a time against B** (`07_fix_experiments.py`). Fake planets decide, since the fixes
were chosen by looking at the real misses; real hosts are supporting evidence only.

| | B | + second search | shorter smoothing | longer periods | cutoff 6 |
|---|---|---|---|---|---|
| Fake planets (of 200) | 148 | 152 | 149 | 147 | 148 |
| Weaker of two fake planets (of 120) | 5 | **90** | 2 | 5 | - |
| Slow fake planets (of 120) | 71 | 72 | 65 | **79** | - |
| Known hosts (of 60) | 41 | 44 | 36 | 41 | 42 |
| Comparison stars flagged (of 40) | 10 | 14 | 11 | 14 | 19 |

The second search is the one clear win and is now part of step 4: **44/60** hosts (small planets
7 -> 10 of 20), at the cost of 14 instead of 10 flagged comparison stars. Longer periods help slow
planets but add false alarms (on top of the second search: 2 more flags, one fewer host), so not adopted.
Shorter smoothing and a lower cutoff are rejected.

For reference, the repo's TLS first signal scored the same way finds 33/60. The repo's full pipeline
reports 36/60, but it also requires BLS agreement and vetting checks, so it isn't directly comparable.

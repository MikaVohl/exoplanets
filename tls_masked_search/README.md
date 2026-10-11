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

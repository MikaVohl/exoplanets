# TESS planet-hunting starter

For the confirmed-label learning and discovery pipeline, see [the learning pipeline guide](docs/learning_pipeline.md).
It includes a blind TLS search, a light-curve CNN, a feature baseline, separate training/validation/test
stars, inference on unmatched signals, and injection recovery through the frozen pipeline.
The [executed starter report](reports/learning-starter/README.md) includes both model comparisons and injection diagnostics.
The [scaling experiment guide](docs/scaling_experiment.md) covers nested training datasets, CNN sizes,
and CPU versus Apple Metal benchmarks.
The [controlled experiment guide](docs/controlled_experiment.md) compares current, richer-input,
and Abdullah's model on identical clean labels, then scales the selected neural family.

Real TESS observations for an initial exploratory data analysis. The starter target is **Pi Mensae (TIC 261136679)**, a known planet host used in NASA's hands-on tutorial. The downloader requests **SPOC, 120-second cadence, sectors 1, 4, and 8** from the public MAST archive. No account or API key is required.

These are extracted light curves, not detector images. Original FITS files include aperture flux (`SAP_FLUX`), pipeline-corrected flux (`PDCSAP_FLUX`), uncertainties, background, centroids, and quality flags. All original rows are retained in those files.

## 100-star BLS/TLS benchmark

For Abdullah's alternative TLS search and classifier, open [the masked-search walkthrough](notebooks/03_tls_masked_search_walkthrough.ipynb).
It starts with a dataset census, zooms in on TOI-880's cleaning and two-signal search, and returns its
result to the aggregate benchmark. A real classifier-dataset entry then connects labeling and feature
transformations to a held-out score and the aggregate classifier evaluation. Outputs are
included in the executed notebook; a local read-only export is in `reports/walkthrough/tls_masked_search.html`.

For a worked example, open [the TOI-880 notebook walkthrough](notebooks/02_toi880_pipeline_walkthrough.ipynb). It reconstructs each preprocessing transformation from the original FITS data, shows BLS/TLS intermediate results, inspects the first signal's warning flags, masks its events, finds a weaker second signal, and only then compares the periods with the catalog. All intermediate tables and figures are included after execution. A read-only export is in `reports/walkthrough/toi880.html`.

```sh
source .venv/bin/activate
jupyter lab notebooks/02_toi880_pipeline_walkthrough.ipynb
```

The project also includes a complete benchmark using **100 distinct cataloged planet hosts**: 60 with known transiting planets and 40 whose confirmed planets have no cataloged transits. The latter are comparison stars, not proven transit-free stars.

Open [the benchmark report](reports/benchmark/index.html), [method and measured results](reports/benchmark/README.md), or [the star summary](reports/benchmark/summary.csv). Each report row links to diagnostic plots and full results with detected periods, event times/depths, significance estimates, warning flags, catalog comparisons and provenance.

The transiting group includes 20 hosts each with a largest short-period planet of ≤3, 3–8, and >8 Earth radii. Targets have TESS magnitude 4–11.5 and known stellar mass/radius. Selection is deterministic from saved NASA Exoplanet Archive and MAST snapshots. It selects each star's latest available public 120-second SPOC sector. The groups differ in brightness; this is a learning benchmark, not a population-completeness measurement.

Both **Astropy BLS** and **TLS 2.0** search independently over 0.5–12 days, shortened to half the usable baseline when necessary. They receive stellar mass/radius priors but no catalog planet periods, epochs, planet sizes or labels. Up to two search iterations are performed, masking only a signal discovered by the algorithms. Catalog comparisons happen afterward.

Preprocessing retains observing gaps, filters quality flags, splits at gaps longer than 0.5 days, trims 0.1 days at segment edges, removes slow trends using a one-day median, clips only large upward excursions, and makes ten-minute weighted bins. It scales formal errors to the measured point-to-point scatter when necessary. Longer trends, short transits and gaps can still limit sensitivity.

The common signal checks include observed repeat events, duration-block correlated-noise SNR, agreement in period and phase, central-half odd/even depths, secondary eclipses, and centroid shifts with a correlated-noise correction. The fixed thresholds in `config/benchmark.json` are heuristic screening rules, not calibrated false-alarm probabilities. Conclusions are **supported transit-like signal**, **signal requiring review**, or **no significant signal in the search range**. Data/search errors are reported separately. No result automatically confirms or rules out a planet.

Run or resume from the project folder:

```sh
.venv/bin/python scripts/run_benchmark.py all
```

Stages can be run separately:

```sh
.venv/bin/python scripts/run_benchmark.py select
.venv/bin/python scripts/run_benchmark.py download
.venv/bin/python scripts/run_benchmark.py search --workers 2
.venv/bin/python scripts/run_benchmark.py report
```

The default uses four download workers and two search workers with one TLS thread each. Successful per-star searches are reused only when the input file hash, code, configuration and numerical-library versions match. `--force` reruns searches; `--limit 3` performs a small smoke run. `--refresh-catalog` on `select` or `all` fetches fresh snapshots and can change the sample. Keep the existing snapshots to reproduce this run.

Benchmark inputs live in `data/benchmark/targets.csv`, `known_planets.csv`, `selection.json`, and `download_manifest.json`. Original files are in `data/raw/benchmark/`. Generated photometry is in `data/processed/benchmark/`. Small result JSONs and summary tables can be committed; raw files, derived photometry, plot images and numerical spectra are Git-ignored and reproducible.

Run the scientific validation checks:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

These tests check an injected transit, a seeded white-noise example, transit preservation through detrending, gap handling, unobserved-event counts, alternating eclipse depths, and period harmonics. They are not a large injection-recovery sensitivity calibration.

## Start exploring

The local `.venv` is already installed. From this project folder:

```sh
source .venv/bin/activate
jupyter lab notebooks/01_tess_eda.ipynb
```

The notebook runs offline once the data have been downloaded. Use **Run → Run All Cells**. It covers schema and units, missing values, SAP versus PDCSAP, quality filtering, cadence/gaps, noise summaries, and a zoomed light curve. It also saves plots and CSVs. Outputs are already included after the initial execution.

## Files

| Location | Contents |
| --- | --- |
| `data/raw/*.fits` | Untouched light curves downloaded from MAST |
| `data/manifest_tic261136679.json` | Source URIs, target/sector metadata, sizes, and SHA-256 hashes |
| `data/search_tic261136679.ecsv` | Archive search result with product metadata |
| `data/processed/pi_mensae_all_cadences.csv` | Selected FITS columns, every original row, normalization, and filter indicators |
| `data/processed/pi_mensae_clean.csv` | Minimally filtered, per-sector normalized data |
| `notebooks/01_tess_eda.ipynb` | Editable, executed EDA notebook |
| `reports/eda_notes.md` | Measured dataset summary |
| `reports/eda_summary.csv` | Per-sector sample counts, gaps, and scatter statistics |
| `reports/*.png` | EDA figures |

CSV column names are lowercase. Flux units are electrons/second; columns ending in `_norm` are dimensionless. `time` is BTJD (BJD − 2,457,000), in TDB days. `quality` is an integer bitmask. `keep` records whether a row passes the initial filter. FITS files remain the authoritative original observations.

The initial filter uses Lightkurve's default TESS quality mask plus finite, positive PDCSAP flux and uncertainty. No detrending, outlier clipping, or gap interpolation is applied. Per-sector median normalization does not remove stellar variability or all instrumental trends.

## Reproduce the download

```sh
.venv/bin/python scripts/download_tess.py
```

Existing files are checked and reused. New files are downloaded to temporary paths, checked for target, sector, cadence, FITS structure, and embedded checksums, and then moved into place. The manifest records SHA-256 hashes for subsequent local integrity checks, plus the status of embedded FITS checksums (0 = mismatch, 1 = valid, 2 = absent). The checker verifies the actual header/data bytes using the FITS one's-complement checksum, including when the optional DATASUM keyword is absent. A checksum mismatch stops the download. All three starter files passed these checks; a separate repeat download of sector 1 was byte-identical.

If the archive search is slow, reuse the saved product list:

```sh
.venv/bin/python scripts/download_tess.py --use-saved-search
```

That still downloads from MAST, but skips the search request. It can only select products present in the saved snapshot.

To request a different sample:

```sh
.venv/bin/python scripts/download_tess.py --tic 261136679 --sectors 1 4 8 11
```

The notebook uses the Pi Mensae manifest; adjust it explicitly if you choose another TIC. More sectors increase disk usage and change the notebook output.

To recreate the environment:

```sh
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

`requirements.txt` pins the resolved versions; `requirements.in` lists the direct dependencies. Downloads and derived CSVs are excluded from Git through `.gitignore`.

## First scientific goal

Inspect sector 1 first, compare it with the other sectors, and identify which features are associated with flagged cadences or gaps. Then detrend sectors separately and run a blind Box Least Squares search. The notebook does not supply a planet period or perform the search, so that remains your next investigation.

Three sectors of one known planet host are a learning dataset, not an unbiased survey or an ML training set. Scatter estimates here are not transit detection significances or CDPP. Expand to additional stars after establishing a baseline and use star-level evaluation splits for ML.

## Sources

- [NASA: How to find an Exoplanet with TESS data](https://heasarc.gsfc.nasa.gov/docs/tess/HowToFindAnExoplanet-UserVersion.html)
- [MAST TESS archive](https://archive.stsci.edu/missions-and-data/tess)
- [Lightkurve: search_lightcurve](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.search_lightcurve.html)

If the archive is temporarily unavailable, retry the downloader later. For manual access, open [MAST Portal](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html), search `TIC 261136679`, filter to TESS/SPOC timeseries at 120 seconds, select sectors 1, 4, and 8, and download the `_lc.fits` products. Keep the archive product identifiers with the files.

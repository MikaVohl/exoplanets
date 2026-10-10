# TESS planet-hunting starter

Real TESS observations for an initial exploratory data analysis. The starter target is **Pi Mensae (TIC 261136679)**, a known planet host used in NASA's hands-on tutorial. The downloader requests **SPOC, 120-second cadence, sectors 1, 4, and 8** from the public MAST archive. No account or API key is required.

These are extracted light curves, not detector images. Original FITS files include aperture flux (`SAP_FLUX`), pipeline-corrected flux (`PDCSAP_FLUX`), uncertainties, background, centroids, and quality flags. All original rows are retained in those files.

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

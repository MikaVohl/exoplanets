"""Download WASP-18 TESS light curves (SPOC, 2-min cadence) and cache them locally."""
import lightkurve as lk

# Search MAST for WASP-18's pre-made light curve files; SPOC = NASA's own pipeline
search = lk.search_lightcurve("WASP-18", author="SPOC", exptime=120)
print(search)

# Download every sector into ./data (reused on later runs, no re-download)
lcs = search.download_all(download_dir="data")
print(lcs)

"""Step 1: download WASP-18 TESS light curves (all sectors) into ./data.

SPOC = NASA's own pipeline; 120 s = 2-minute cadence. Later runs reuse the cache.
"""
from common import download

lcs = download()
print(f"Downloaded {len(lcs)} sectors:", [lc.meta["SECTOR"] for lc in lcs])

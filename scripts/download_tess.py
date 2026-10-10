"""Download untouched, public SPOC FITS light curves from MAST."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lightkurve as lk
import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]


def validate(path, tic, sector):
    file_bytes = path.read_bytes()
    with fits.open(path) as hdus:
        hdus.verify("exception")
        header = hdus[0].header
        time_header = hdus[1].header
        checksums = []
        for hdu in hdus:
            # Verify the actual header+data bytes, including FITS padding.
            # Astropy verify_checksum assumes datasum=0 if DATASUM is absent;
            # these archive files have CHECKSUM but no DATASUM keyword.
            checksum = 2
            if "CHECKSUM" in hdu.header:
                offsets = hdu.fileinfo()
                payload = file_bytes[offsets["hdrLoc"]:offsets["datLoc"] + offsets["datSpan"]]
                total = int(np.frombuffer(payload, dtype=">u4").astype("uint64").sum())
                while total >> 32:
                    total = (total & 0xFFFFFFFF) + (total >> 32)
                checksum = int(total == 0xFFFFFFFF)
            status = {"extension": hdu.name,
                      "checksum": checksum,
                      "datasum": int(hdu.verify_datasum())}
            # Codes: 0 = mismatch, 1 = valid, 2 = keyword absent.
            if status["datasum"] == 0:
                raise ValueError(f"Invalid FITS data checksum: {path}")
            if status["checksum"] == 0:
                raise ValueError(f"Invalid FITS HDU checksum: {path}/{hdu.name}")
            checksums.append(status)
        assert int(header["TICID"]) == tic, "Unexpected target"
        assert int(header["SECTOR"]) == sector, "Unexpected sector"
        cadence = time_header["TIMEDEL"] * 86400
        assert np.isclose(cadence, 120, atol=1), "Unexpected cadence"
        assert {"TIME", "SAP_FLUX", "PDCSAP_FLUX", "QUALITY"} <= set(hdus[1].columns.names)
        return {
            "tic_id": tic, "sector": sector, "rows": len(hdus[1].data),
            "cadence_seconds": round(cadence),
            "date_obs": time_header.get("DATE-OBS"), "date_end": time_header.get("DATE-END"),
            "pipeline_version": header.get("PROCVER"),
            "camera": header.get("CAMERA"), "ccd": header.get("CCD"),
            "fits_checksum_status": checksums,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tic", type=int, default=261136679)
    parser.add_argument("--sectors", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--use-saved-search", action="store_true",
                        help="Reuse the local archive search snapshot instead of querying MAST again.")
    args = parser.parse_args()
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    search_path = ROOT / "data" / f"search_tic{args.tic}.ecsv"
    if args.use_saved_search:
        table = Table.read(search_path, format="ascii.ecsv")
        table = table[np.isin(table["sequence_number"], args.sectors)]
        print(f"Using archive search snapshot {search_path}", flush=True)
    else:
        print(f"Searching MAST for TIC {args.tic}, sectors {args.sectors} ...", flush=True)
        result = lk.search_lightcurve(
            f"TIC {args.tic}", mission="TESS", author="SPOC", exptime=120,
            sector=args.sectors,
        )
        table = result.table
        table.write(search_path, format="ascii.ecsv", overwrite=True)
        print(result, flush=True)
    if not len(table):
        raise RuntimeError("No matching public SPOC light curves found.")
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
    )))
    entries = []
    for row in table:
        sector = int(row["sequence_number"])
        filename = Path(str(row["productFilename"])).name
        uri = str(row["dataURI"])
        path = raw / filename
        if not path.exists():
            print(f"Downloading sector {sector}: {filename}", flush=True)
            temporary = path.with_suffix(".fits.part")
            try:
                with session.get("https://mast.stsci.edu/api/v0.1/Download/file",
                                 params={"uri": uri}, timeout=(20, 120), stream=True) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            output.write(chunk)
                validate(temporary, args.tic, sector)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        entry = validate(path, args.tic, sector)
        entry.update({
            "path": str(path.relative_to(ROOT)), "source_uri": uri,
            "download_url": "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + uri,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        entries.append(entry)
        print(f"Verified sector {sector}: {entry['rows']:,} rows", flush=True)
    missing = set(args.sectors) - {e["sector"] for e in entries}
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": f"TIC {args.tic}", "author": "SPOC", "archive": "MAST",
        "requested_sectors": args.sectors, "missing_sectors": sorted(missing),
        "files": entries,
    }
    manifest_path = ROOT / "data" / f"manifest_tic{args.tic}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {manifest_path}", flush=True)
    if missing:
        print(f"No matching product found for requested sectors: {sorted(missing)}", flush=True)


if __name__ == "__main__":
    main()

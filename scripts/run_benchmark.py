"""Select, download, search and report on 100 cataloged TESS planet hosts."""

import argparse
import json
import os
import sys
from pathlib import Path

# Bound CPU use before importing numerical packages or spawning workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from tess_hunt.catalog import select_targets
    from tess_hunt.benchmark import download_sample, run_searches
    from tess_hunt.report import build_report
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", "download", "search", "report", "all"])
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "benchmark.json")
    parser.add_argument("--workers", type=int, default=2, help="Search processes; each TLS process uses one thread by default")
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Run the first N selected targets for a smoke test")
    parser.add_argument("--force", action="store_true", help="Rerun cached searches")
    parser.add_argument("--refresh-catalog", action="store_true", help="Fetch fresh catalogs and recompute selection")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage in ["select", "all"]:
        targets = select_targets(ROOT, config, refresh=args.refresh_catalog)
    else:
        targets = pd.read_csv(ROOT / "data" / "benchmark" / "targets.csv")
    if args.limit:
        targets = targets.head(args.limit)
    if args.stage in ["download", "all"]:
        _, failures = download_sample(ROOT, targets, args.download_workers)
        if failures:
            print("Some downloads failed; subsequent searches record errors. Retry the download stage to resume.", flush=True)
    if args.stage in ["search", "all"]:
        run_searches(ROOT, targets, config, args.workers, args.force)
    if args.stage in ["report", "all"]:
        build_report(ROOT)


if __name__ == "__main__":
    main()

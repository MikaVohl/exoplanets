"""Run with python -m tess_hunt.learning from the repository root."""
import argparse
from pathlib import Path

from .dataset import ROOT, load_config, prepare


def main():
    parser = argparse.ArgumentParser(description="TESS blind search and supervised signal learning")
    parser.add_argument("--run", type=Path, default=ROOT / "runs/tess-learning")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare", help="Download labeled hosts and blindly search their light curves")
    build.add_argument("--config", type=Path, default=ROOT / "config/learning.json")
    build.add_argument("--per-class", type=int, default=128)
    build.add_argument("--sectors", type=int, default=1)
    build.add_argument("--download-workers", type=int, default=8)
    build.add_argument("--workers", type=int, default=2)
    build.add_argument("--refresh-catalog", action="store_true")
    train = commands.add_parser("train", help="Train CNN and baseline, select thresholds on validation, evaluate test")
    train.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    search = commands.add_parser("search", help="Score signals in local files or download a new TIC")
    source = search.add_mutually_exclusive_group(required=True)
    source.add_argument("--files", type=Path, nargs="+")
    source.add_argument("--tic", type=int)
    search.add_argument("--sectors", type=int, default=1)
    search.add_argument("--output", type=Path)
    inject = commands.add_parser("inject", help="Inject physical transits before cleaning; run search and frozen model")
    inject.add_argument("--count", type=int, default=30)
    inject.add_argument("--seed", type=int, default=2026)
    report = commands.add_parser("report", help="Export test metrics, training curves and injection diagnostics")
    report.add_argument("--output", type=Path, default=ROOT / "reports/learning-starter")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.command == "prepare":
        if min(args.per_class, args.sectors, args.workers, args.download_workers) < 1:
            parser.error("Counts and workers must be positive")
        prepare(run, load_config(args.config), args.per_class, args.sectors,
                args.download_workers, args.workers, args.refresh_catalog)
    elif args.command == "train":
        from .model import train_models
        train_models(run, args.device)
    elif args.command == "search":
        from .evaluate import discover
        discover(run, args.files, args.tic, args.sectors, args.output)
    elif args.command == "inject":
        if args.count < 1:
            parser.error("Injection count must be positive")
        from .evaluate import injection_experiment
        injection_experiment(run, args.count, args.seed)
    elif args.command == "report":
        from .report import export_report
        export_report(run, args.output.resolve())


if __name__ == "__main__":
    main()

"""Run independent experiments without changing starter data or checkpoints."""
import argparse
from pathlib import Path

from tess_hunt.learning.dataset import ROOT


def main():
    parser = argparse.ArgumentParser(description="TESS model/data scaling and CPU/Metal benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)
    expand = sub.add_parser("prepare")
    expand.add_argument("--source", type=Path, default=ROOT / "runs/tess-learning-starter")
    expand.add_argument("--run", type=Path, default=ROOT / "runs/tess-learning-scaling")
    expand.add_argument("--per-class", type=int, default=512)
    expand.add_argument("--workers", type=int, default=4)
    expand.add_argument("--download-workers", type=int, default=16)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--source", type=Path, default=ROOT / "runs/tess-learning-starter")
    bench.add_argument("--output", type=Path, default=ROOT / "runs/tess-scaling-experiment")
    bench.add_argument("--steps", type=int, default=20)
    bench.add_argument("--cpu-sweep", action="store_true")
    bench.add_argument("--epoch-compare", action="store_true")
    grid = sub.add_parser("grid")
    grid.add_argument("--source", type=Path, default=ROOT / "runs/tess-learning-scaling")
    grid.add_argument("--output", type=Path, default=ROOT / "runs/tess-scaling-experiment")
    grid.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64])
    grid.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128, 256])
    grid.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    grid.add_argument("--epochs", type=int, default=100)
    grid.add_argument("--patience", type=int, default=20)
    grid.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    report = sub.add_parser("report")
    report.add_argument("--source", type=Path, default=ROOT / "runs/tess-scaling-experiment")
    report.add_argument("--output", type=Path, default=ROOT / "reports/tess-scaling")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import expand_dataset
        expand_dataset(args.run.resolve(), args.source.resolve(), args.per_class, args.workers, args.download_workers)
    elif args.command == "benchmark":
        from .benchmark import benchmark_devices, cpu_thread_sweep, epoch_benchmark
        benchmark_devices(args.source.resolve(), args.output.resolve(), steps=args.steps)
        if args.cpu_sweep:
            cpu_thread_sweep(args.source.resolve(), args.output.resolve())
        if args.epoch_compare:
            epoch_benchmark(args.source.resolve(), args.output.resolve())
    elif args.command == "grid":
        from .experiment import scaling_grid
        scaling_grid(args.source.resolve(), args.output.resolve(), args.widths, args.sizes,
                     args.seeds, args.epochs, args.patience, args.device)
    elif args.command == "report":
        from .report import scaling_report
        scaling_report(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

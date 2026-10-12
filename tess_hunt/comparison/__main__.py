"""python -m tess_hunt.comparison prepare|compare|scale|report|audit."""
import argparse
from pathlib import Path
from tess_hunt.learning.dataset import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "compare", "scale", "report", "audit", "score"])
    parser.add_argument("--source", type=Path, default=ROOT/"runs/tess-learning-scaling")
    parser.add_argument("--prepared", type=Path, default=ROOT/"runs/tess-controlled-inputs")
    parser.add_argument("--output", type=Path, default=ROOT/"runs/tess-controlled-experiment")
    parser.add_argument("--report", type=Path, default=ROOT/"reports/tess-controlled")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fits", type=Path)
    parser.add_argument("--candidate", type=Path, help="JSON with blind period, t0, duration_days, search_sde")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        prepare(args.source.resolve(), args.prepared.resolve(), args.workers)
    elif args.command in ("compare", "scale"):
        from .experiment import compare, scale
        (compare if args.command == "compare" else scale)(args.prepared.resolve(), args.output.resolve(), args.device)
    elif args.command == "report":
        from .report import report
        report(args.prepared.resolve(), args.output.resolve(), args.report.resolve())
    elif args.command == "audit":
        from .report import audit
        audit(args.prepared.resolve(), args.output.resolve(), args.report.resolve())
    else:
        import json
        from .infer import score_candidate
        if any(v is None for v in (args.checkpoint, args.fits, args.candidate)):
            parser.error("score requires --checkpoint, --fits and --candidate")
        print(json.dumps(score_candidate(args.checkpoint, args.fits, json.loads(args.candidate.read_text()), args.device), indent=2))


if __name__ == "__main__":
    main()

"""Export compact scientific diagnostics from a completed learning run."""
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .dataset import ROOT


def export_report(run, output):
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((run / "metrics.json").read_text())
    scores = pd.read_csv(run / "scores.csv")
    test = scores[scores.split.eq("test") & scores.y.ge(0)]
    history = pd.read_csv(run / "training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {"cnn": "#2166ac", "baseline": "#b35806"}
    for name in ("cnn", "baseline"):
        precision, recall, _ = precision_recall_curve(test.y, test[f"{name}_score"])
        axes[0].step(recall, precision, where="post", color=colors[name],
                     label=f"{name.upper()} (AP {metrics[name]['test']['average_precision']:.2f})")
    axes[0].axhline(test.y.mean(), linestyle="--", color="gray", label="Test class prevalence")
    axes[0].set(xlabel="Recall on detected, labeled test signals", ylabel="Precision in selected test sample",
                xlim=(0, 1), ylim=(0, 1.03), title=f"Held-out test stars ({len(test)} signals)")
    axes[0].legend(fontsize=8)
    axes[1].plot(history.epoch, history.validation_average_precision, color=colors["cnn"])
    axes[1].axvline(metrics["best_epoch"], color="gray", linestyle="--", label="Selected checkpoint")
    axes[1].set(xlabel="Training epoch", ylabel="Validation average precision", ylim=(0, 1.03),
                title="Checkpoint selection uses validation only")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "learning_diagnostics.png", dpi=160)
    plt.close(fig)
    for name in ("config.json", "metrics.json", "training_history.csv", "dataset_provenance.json", "failures.json",
                 "catalog_source.json", "catalog_recovery.csv", "injection_metrics.json", "injections.csv",
                 "discovery_search.csv", "discovery_search.provenance.json"):
        path = run / name
        if path.exists():
            shutil.copyfile(path, output / name)
    test.to_csv(output / "test_scores.csv", index=False)
    # Keep only held-out unmatched stars in this report; full exploration scores stay in the run.
    unknown = scores[scores.split.eq("test") & scores.y.eq(-1)].sort_values("cnn_score", ascending=False)
    unknown.to_csv(output / "held_out_unlabeled.csv", index=False)
    lines = ["# TESS confirmed-label starter experiment", "",
             f"Run: `{run.relative_to(ROOT) if run.is_relative_to(ROOT) else run}`. "
             "This is a small initial experiment with frozen validation-selected thresholds.", "",
             "![Test ranking and validation training curve](learning_diagnostics.png)", "",
             "| Model | Test PR-AUC | Test ROC-AUC | Planet recall | False-positive rate |", 
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in ("cnn", "baseline"):
        m = metrics[name]["test"]
        lines.append(f"| {name.upper()} | {m['average_precision']:.3f} | {m['roc_auc']:.3f} | "
                     f"{m['recall']:.1%} | {m['false_positive_rate']:.1%} |")
    lines += ["", f"Test set: {metrics['cnn']['test']['planets']} confirmed-planet signals and "
              f"{metrics['cnn']['test']['negatives']} FP/FA signals on "
              f"{metrics['partitions']['test']['stars']} stars. "
              "These classification metrics cover detected, conservatively matched signals, not all planets on these stars.", "",
              f"The CNN has {metrics['model_parameters']:,} parameters. "
              f"{metrics['unlabeled_signals_scored']} signals remained unlabeled and were scored without entering training.", ""]
    recovery = pd.read_csv(run / "catalog_recovery.csv")
    eligible = recovery[recovery.split.eq("test") & recovery.label.isin(json.loads((run / "config.json").read_text())["positive_dispositions"])
                        & recovery.in_search_range & recovery.observed_catalog_events.ge(2)]
    if len(eligible):
        lines += [f"On test stars the blind search matched {int(eligible.recovered.sum())}/{len(eligible)} "
                  "catalog-confirmed signals with periods in the search range and at least two sampled events. "
                  "Nonmatches include search misses and conservative timing/alias exclusions.", ""]
    injection_path = run / "injection_metrics.json"
    if injection_path.exists():
        injections = json.loads(injection_path.read_text())
        m = injections["eligible"]
        lines += [f"Injection evaluation used {injections['count']} trials on {injections['held_out_stars']} held-out stars. "
                  f"Of {m['injections']} eligible trials, blind search recovered {m['search_recovered']} "
                  f"and the frozen CNN selected {m['pipeline_recovered']}. "
                  "Transits were inserted before cleaning into real PDCSAP flux; this does not test pixel extraction or SPOC correction. "
                  "The plotted Wilson intervals describe individual trials; repeated stars create dependence, "
                  "so these are not intervals for population completeness.", ""]
        table = pd.read_csv(run / "injections.csv")
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        eligible_trials = table[~table.preexisting_match & table.observed_events.ge(2)]
        bins = [300, 1000, 3000, 10001]
        centers = np.sqrt(np.asarray(bins[:-1]) * np.asarray(bins[1:]))
        for column, label, color in [("search_recovered", "Blind search", colors["baseline"]),
                                     ("pipeline_recovered", "Search + frozen CNN", colors["cnn"])]:
            fractions, errors, counts = [], [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                group = eligible_trials[eligible_trials.nominal_depth_ppm.between(lo, hi, inclusive="left")]
                n, successes = len(group), int(group[column].sum())
                fraction = successes / n if n else np.nan
                # Wilson 95% interval, including zero/all-success bins.
                z = 1.96
                if n:
                    center = (fraction + z*z / (2*n)) / (1 + z*z/n)
                    half = z / (1 + z*z/n) * np.sqrt(fraction*(1-fraction)/n + z*z/(4*n*n))
                    errors.append([fraction - (center-half), center+half - fraction])
                else:
                    errors.append([np.nan, np.nan])
                fractions.append(fraction)
                counts.append(n)
            ax.errorbar(centers, fractions, yerr=np.asarray(errors).T, marker="o", capsize=4, color=color, label=label)
        for x, n in zip(centers, counts):
            ax.text(x, 1.03, f"n={n}", ha="center", fontsize=9)
        ax.set(xscale="log", xlabel="Nominal injected depth (ppm)", ylabel="Recovery fraction",
               ylim=(-.03, 1.1), title="Held-out injection recovery (Wilson 95% intervals)")
        ax.legend()
        fig.savefig(output / "injection_recovery.png", dpi=160)
        plt.close(fig)
        lines += ["![Injection recovery by nominal depth](injection_recovery.png)", ""]
    lines += ["Scores are uncalibrated rankings. Precision describes this selected sample, not survey candidate yield. "
              "The sample is too small to establish robust discovery performance or a scaling law. "
              "Unlabeled detections require catalog updates and scientific vetting before claiming novelty.", "",
              "See [the pipeline guide](../../docs/learning_pipeline.md) for reproduction commands, label rules, "
              "data products, configuration and limitations.", ""]
    discovery_path = run / "discovery_search.csv"
    if discovery_path.exists():
        external = pd.read_csv(discovery_path)
        lines += [f"The saved pipeline also searched {external.tic.nunique()} independent stars and ranked "
                  f"{len(external)} signals in [discovery_search.csv](discovery_search.csv). "
                  "These are an inference exercise, not verified new planets. Catalog-matched KP/PC/APC "
                  "signals may be excluded from supervised training while still carrying their known catalog disposition.", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(f"Saved report to {output / 'README.md'}", flush=True)

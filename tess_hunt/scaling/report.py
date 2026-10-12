"""Scientific scaling curves and a readable experiment report."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
import numpy as np
import pandas as pd


def scaling_report(source, output):
    output.mkdir(parents=True, exist_ok=True)
    experiment = json.loads((source / "experiment.json").read_text())
    results = pd.read_csv(source / "results.csv")
    benchmark = pd.read_csv(source / "device_benchmark.csv")
    prepared = Path(experiment["source"])
    for name in ("dataset_provenance.json", "catalog_source.json", "failures.json", "config.json"):
        if (prepared / name).exists():
            shutil.copyfile(prepared / name, output / name)
    for name in ("results.csv", "experiment.json", "completion.json", "device_benchmark.json",
                 "device_benchmark.csv", "cpu_thread_sweep.csv", "epoch_benchmark.csv"):
        if (source / name).exists():
            shutil.copyfile(source / name, output / name)
    grouped = results.groupby(["model", "width", "requested_size"]).agg(
        train_signals_mean=("train_signals", "mean"), train_stars_mean=("train_stars", "mean"),
        ap_mean=("validation_average_precision", "mean"), ap_std=("validation_average_precision", "std"),
        auc_mean=("validation_roc_auc", "mean"), auc_std=("validation_roc_auc", "std"),
        recall_mean=("validation_recall", "mean"), fpr_mean=("validation_false_positive_rate", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"), fit_seconds_sum=("fit_seconds", "sum"),
        best_epoch_mean=("best_epoch", "mean"), updates_mean=("optimizer_updates", "mean"),
        cap_fraction=("hit_epoch_cap", "mean"), repeats=("seed", "count")).reset_index()
    grouped.to_csv(output / "summary.csv", index=False)
    palette = {0: "#333333", 8: "#1b9e77", 16: "#2166ac", 32: "#7570b3", 64: "#d95f02"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for width, subset in grouped.groupby("width"):
        subset = subset.sort_values("requested_size")
        label = "Feature baseline" if width == 0 else f"CNN width {width}"
        color = palette.get(width)
        for ax, metric in zip(axes, ("ap", "auc")):
            ax.errorbar(subset.train_signals_mean, subset[f"{metric}_mean"],
                        yerr=subset[f"{metric}_std"].fillna(0), marker="o", capsize=3,
                        linestyle="--" if width == 0 else "-", color=color, label=label)
            ax.set(xscale="log", xlabel="Labeled training signals (whole stars)", ylim=(0, 1.03))
            ax.grid(alpha=.15)
    axes[0].set(ylabel="Validation average precision", title="Data scaling; mean ± SD across seeds")
    axes[1].set(ylabel="Validation ROC-AUC", title="Same validation stars for every cell")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(sorted(results.requested_size.unique()))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.savefig(output / "data_scaling.png", dpi=170)
    plt.close(fig)
    sizes = sorted(results.requested_size.unique())
    widths = sorted(results[results.model.eq("cnn")].width.unique())
    fig, ax = plt.subplots(figsize=(8, 4.3), constrained_layout=True)
    image_data = np.array([[grouped[(grouped.width == width) & (grouped.requested_size == n)].ap_mean.iloc[0]
                            for n in sizes] for width in widths])
    im = ax.imshow(image_data, vmin=.4, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(sizes)), [str(n) for n in sizes])
    parameter_counts = results[results.model.eq("cnn")].groupby("width").parameters.first()
    ax.set_yticks(range(len(widths)), [f"{w}: {int(parameter_counts[w]):,} parameters" for w in widths])
    ax.set(xlabel="Requested labeled training signals", ylabel="CNN width",
           title="Model size × data size: mean validation AP")
    for i in range(len(widths)):
        for j in range(len(sizes)):
            ax.text(j, i, f"{image_data[i,j]:.3f}", ha="center", va="center", color="white" if image_data[i,j] < .8 else "black")
    fig.colorbar(im, ax=ax, label="Validation average precision")
    fig.savefig(output / "capacity_data_grid.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for batch_size, ax in zip(sorted(benchmark.batch_size.unique()), axes):
        for device, subset in benchmark[benchmark.batch_size.eq(batch_size)].groupby("device"):
            subset = subset.sort_values("parameters")
            ax.plot(subset.parameters, subset.resident_step_seconds * 1000, marker="o",
                    label=f"{device.upper()} resident inputs")
            ax.plot(subset.parameters, subset.transfer_step_seconds * 1000, marker=".", linestyle="--",
                    label=f"{device.upper()} with input transfer")
        ax.set(xscale="log", yscale="log", xlabel="Model parameters", ylabel="Training step (milliseconds)",
               title=f"Synchronized timing, batch {batch_size}")
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.savefig(output / "metal_benchmark.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.3), constrained_layout=True)
    for width, group in grouped[grouped.model.eq("cnn")].groupby("width"):
        group = group.sort_values("requested_size")
        ax.plot(group.fit_seconds_mean, group.ap_mean, marker="o", label=f"Width {width}", color=palette.get(width))
    ax.set(xlabel="Mean measured fit time per cell (seconds)", ylabel="Validation average precision",
           title="Observed compute/performance; data size varies along each line", ylim=(0, 1.03))
    ax.legend()
    ax.grid(alpha=.15)
    fig.savefig(output / "compute_performance.png", dpi=170)
    plt.close(fig)
    full = grouped[grouped.requested_size.eq(max(sizes))].sort_values("ap_mean", ascending=False)
    best_full = full[full.model.eq("cnn")].iloc[0]
    baseline_full = full[full.model.eq("baseline")].iloc[0]
    lines = ["# TESS model/data scaling and Metal experiment", "",
             f"At {int(best_full.train_signals_mean)} training signals, CNN width {int(best_full.width)} "
             f"had mean validation AP {best_full.ap_mean:.3f}, versus {baseline_full.ap_mean:.3f} for the fixed feature baseline. "
             "All four CNN sizes beat that baseline on the full pool. "
             "Data curves fluctuate at intermediate sizes; they do not show a smooth universal scaling law.", "",
             f"Completed {len(results)} fits: {len(widths)} CNN widths plus the feature baseline, "
             f"{len(sizes)} nested data levels, and {len(experiment['seeds'])} seeds. "
             f"Validation uses {experiment['validation_signals']} labeled signals on {experiment['validation_stars']} stars "
             f"({experiment['validation_planets']} CP, {experiment['validation_negatives']} FP/FA). "
             "The test partition was not evaluated.", "",
             "The feature baseline is a gradient-boosted decision-tree classifier on period, duration, depth, "
             "search strength, signal-to-noise, odd/even and secondary-eclipse measurements, and event counts. "
             "CNNs learn from full-orbit, transit-close-up and individual-event views plus four basic measurements.", "",
             "![Scaling with labeled data](data_scaling.png)", "",
             "![Capacity and data grid](capacity_data_grid.png)", "",
             "At the full training pool:", "",
             "| Model | Training signals (mean) | Validation AP, mean ± SD | ROC-AUC | Recall at validation FPR budget | Mean fit seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in full.itertuples():
        name = "Feature baseline" if row.width == 0 else f"CNN width {row.width}"
        lines.append(f"| {name} | {row.train_signals_mean:.0f} | {row.ap_mean:.3f} ± {row.ap_std:.3f} | "
                     f"{row.auc_mean:.3f} | {row.recall_mean:.1%} | {row.fit_seconds_mean:.1f} |")
    lines += ["", "![Observed compute and validation performance](compute_performance.png)", "",
              "![CPU and Metal training step timings](metal_benchmark.png)", ""]
    metadata = json.loads((source / "device_benchmark.json").read_text())
    lines += [f"Hardware: {metadata['machine']}. CPU timings use four PyTorch threads. "
              "Timings include forward, backward, gradient clipping and AdamW, exclude warmup, "
              "and explicitly synchronize Metal before and after each timed trial. "
              "MPS CPU fallback is disabled. Both resident-input and batch-transfer timings are reported.", "",
              "| Width | Batch | CPU/MPS resident-step speedup | CPU/MPS with-transfer speedup |", 
              "| --- | ---: | ---: | ---: |"]
    for (width, batch), subset in benchmark.groupby(["width", "batch_size"]):
        if set(subset.device) >= {"cpu", "mps"}:
            cpu = subset[subset.device.eq("cpu")].iloc[0]
            mps = subset[subset.device.eq("mps")].iloc[0]
            lines.append(f"| {width} | {batch} | {cpu.resident_step_seconds/mps.resident_step_seconds:.1f}× | "
                         f"{cpu.transfer_step_seconds/mps.transfer_step_seconds:.1f}× |")
    sweep_path = source / "cpu_thread_sweep.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        lines += ["", "An additional CPU sweep tested 1, 2, 4 and 8 threads at batch 32:", ""]
        for width, group in sweep.groupby("width"):
            best = group.loc[group.step_seconds.idxmin()]
            gpu = benchmark[benchmark.width.eq(width) & benchmark.batch_size.eq(32) & benchmark.device.eq("mps")].iloc[0]
            lines.append(f"- Width {width}: fastest measured CPU setting was {int(best.cpu_threads)} threads "
                         f"({best.step_seconds*1000:.1f} ms); MPS resident step {gpu.resident_step_seconds*1000:.1f} ms "
                         f"({best.step_seconds/gpu.resident_step_seconds:.1f}× faster).")
    epoch_path = source / "epoch_benchmark.csv"
    if epoch_path.exists():
        epochs = pd.read_csv(epoch_path)
        lines += ["", "Fixed-epoch full-loop timing includes on-device indexing, validation and checkpoint selection:", ""]
        for width, group in epochs.groupby("width"):
            cpu, mps = group[group.device.eq("cpu")].iloc[0], group[group.device.eq("mps")].iloc[0]
            lines.append(f"- Width {width}, {int(cpu.epochs_run)} epochs, {int(cpu.train_signals)} training signals: "
                         f"CPU {cpu.fit_seconds:.1f}s, MPS {mps.fit_seconds:.1f}s "
                         f"({cpu.fit_seconds/mps.fit_seconds:.1f}×).")
    max_difference = benchmark.eval_max_abs_logit_difference_from_cpu.max()
    max_gradient_difference = benchmark.eval_gradient_relative_l2_difference_from_cpu.max()
    lines += ["", f"Identical evaluation weights and inputs agreed across CPU/MPS to at most {max_difference:.2g} "
              f"absolute logit difference and {max_gradient_difference:.2g} relative L2 gradient difference, with dropout disabled. "
              "Training trajectories need not be identical because dropout random generators differ by backend.", "",
              "## Interpretation and limits", "",
              "These are validation development curves, not final test results. Error bars are variation across "
              "three subset/initialization seeds; they are not population confidence intervals. "
              "The available confirmed labels limit the real-data range, and this experiment does not establish an asymptotic scaling law.", "",
              "At the full pool, parameter count and mean AP increase together over the tested widths. "
              "The width-32 model has higher recall at the chosen false-positive budget than width 64, "
              "so the best model depends on the intended operating point. "
              "The fixed tree baseline's AP decreases from 256 to the full pool. "
              "Both the baseline and CNNs would need separate hyperparameter selection for a claim about their best achievable performance.", "",
              f"All cells use learning rate {experiment['learning_rate']}, batch {experiment['batch_size']}, "
              f"at most {experiment['epochs']} epochs, and patience {experiment['patience']} after at least 25 epochs. "
              "The same epoch ceiling gives larger datasets more optimizer updates; exact update counts and measured time "
              "are saved so data and optimization effects are visible. Validation AP selects the checkpoint, "
              "with validation log loss breaking ties. No model-specific learning-rate sweep was performed.", "",
              "Each seed defines a nested, approximately class-balanced ordering of whole training stars. "
              "All widths and the baseline share each subset; labels and normalization statistics come from training rows only. "
              "Adding a multi-signal host can exceed a requested row count slightly. "
              "The full training pool is identical across seeds, so its spread primarily reflects initialization and training variation.", "",
              f"CNN cells hitting the epoch cap: {int(results[results.model.eq('cnn')].hit_epoch_cap.sum())}/"
              f"{int(results.model.eq('cnn').sum())}. A cap-hit flag does not establish convergence.", "",
              "The benchmark and grid use the same photometry representation and architecture. "
              "Keeping all training/validation tensors on Metal avoids repeated input transfers. "
              "This acceleration concerns CNN training; MAST acquisition, TLS search and the tree baseline still run on CPU.", "",
              "See [the experiment guide](../../docs/scaling_experiment.md) for reproduction commands.", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(f"Report saved to {output / 'README.md'}", flush=True)

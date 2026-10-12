"""Static scientific figures, provenance audit and checkpoint reload checks."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from tess_hunt.learning.dataset import save_json
from tess_hunt.learning.model import metrics
from .experiment import load_prepared, training_hash, predict, resident
from .models import make_model, model_inputs

NAMES = {"baseline": "Current trees", "legacy_tree": "Abdullah's pipeline: trees", "current32": "Current CNN, width 32",
         "current64": "Current CNN, width 64", "masked32": "Masked cleaning, width 32",
         "rich32": "Richer inputs, width 32", "rich_masked32": "Richer inputs + masked cleaning",
         "legacy_views": "Abdullah's model: views", "legacy_hybrid": "Abdullah's model: views + measurements",
         "legacy_hybrid_aug": "Abdullah's model: views + measurements + augmentation"}


def summary(results, group):
    return results.groupby(group).agg(ap_mean=("validation_average_precision", "mean"),
                                      ap_std=("validation_average_precision", "std"),
                                      auc_mean=("validation_roc_auc", "mean"), auc_std=("validation_roc_auc", "std"),
                                      recall_mean=("validation_recall", "mean"), recall_std=("validation_recall", "std"),
                                      fpr_mean=("validation_false_positive_rate", "mean"),
                                      parameters=("parameters", "first"), fit_seconds=("fit_seconds", "mean"),
                                      train_signals=("train_signals", "mean"), best_epoch=("best_epoch", "mean"),
                                      cap_fraction=("hit_epoch_cap", "mean"), repeats=("seed", "count")).reset_index()


def paired_bootstrap(output, results, samples=1000):
    """Paired star resampling of fixed predictions, average per-seed AP differences.

    These exploratory intervals do not adjust for checkpoint/model selection.
    """
    scores = {}
    y, tics, ids = None, None, None
    for row in results.itertuples():
        p = pd.read_csv(output/"comparison/cells"/f"{row.cell_id}.validation.csv")
        if ids is None:
            ids, y, tics = p.candidate_id.tolist(), p.y.to_numpy(), p.tic.to_numpy()
        if p.candidate_id.tolist() != ids:
            raise ValueError("Mismatched paired predictions")
        scores[(row.variant, row.seed)] = p.score.to_numpy()
    contrasts = [("current64", "current32"), ("masked32", "current32"), ("rich32", "current32"),
                 ("rich_masked32", "masked32"), ("legacy_hybrid", "legacy_views"),
                 ("legacy_hybrid_aug", "legacy_hybrid"), ("legacy_hybrid", "current64")]
    rng = np.random.default_rng(20261011)
    stars = sorted(set(tics))
    groups = [np.flatnonzero(tics == tic) for tic in stars]
    draws = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(samples)]
    records = []
    for a, b in contrasts:
        seeds = sorted(s for v, s in scores if v == a)
        def delta(index):
            return np.mean([average_precision_score(y[index], scores[(a, s)][index])
                            - average_precision_score(y[index], scores[(b, s)][index]) for s in seeds])
        differences = [delta(index) for index in draws if len(np.unique(y[index])) == 2]
        low, high = np.quantile(differences, [.05, .95])
        records.append({"a": a, "b": b, "mean_ap_difference": delta(np.arange(len(y))),
                        "paired_star_bootstrap_90_low": float(low), "paired_star_bootstrap_90_high": float(high),
                        "samples": len(differences), "selection_adjusted": False})
    return pd.DataFrame(records)


def audit(prepared, output, report_dir):
    df, arrays, provenance = load_prepared(prepared)
    source = pd.read_csv(Path(provenance["source"])/"candidates.csv")
    test_ids = set(source.loc[source.split.eq("test"), "candidate_id"])
    test_stars = set(source.loc[source.split.eq("test"), "tic"])
    if set(df.candidate_id) & test_ids or set(df.tic) & test_stars:
        raise ValueError("Test data entered preparation")
    val = df[df.split.eq("validation")]
    valid_ids = val.candidate_id.tolist()
    train = df[df.split.eq("train")]
    all_cells, checkpoints = [], []
    for stage in ("comparison", "optimization", "scaling"):
        folder = output/stage
        if not (folder/"experiment.json").exists():
            continue
        protocol = json.loads((folder/"experiment.json").read_text())
        if protocol["training_code_hash"] != training_hash() or protocol["validation_candidate_ids"] != valid_ids:
            raise ValueError("Training code or validation changed")
        results = pd.read_csv(folder/"results.csv")
        if len(results) != len(protocol["specs"]):
            raise ValueError("Incomplete experiment")
        groups = {}
        for row in results.itertuples():
            cell = json.loads((folder/"cells"/f"{row.cell_id}.json").read_text())
            ids = set(cell["train_candidate_ids"])
            if cell["fingerprint"] != protocol["fingerprint"] or not ids.issubset(set(train.candidate_id)):
                raise ValueError("Training provenance/leakage failure")
            rows = df[df.candidate_id.isin(ids)]
            if set(rows.tic) & set(val.tic) or ids & test_ids:
                raise ValueError("Held-out star entered training")
            for tic in rows.tic.unique():
                if not set(train.loc[train.tic.eq(tic), "candidate_id"]).issubset(ids):
                    raise ValueError("Training subset divided a star")
            key = (row.seed, row.requested_size)
            if key in groups and groups[key] != ids:
                raise ValueError("Models did not share the same subset")
            groups[key] = ids
            p = pd.read_csv(folder/"cells"/f"{row.cell_id}.validation.csv")
            if p.candidate_id.tolist() != valid_ids or not np.array_equal(p.y, val.y):
                raise ValueError("Validation prediction alignment failed")
            calculated = metrics(p.y.to_numpy(), p.score.to_numpy(), cell["validation_threshold"])
            for name, value in calculated.items():
                stored = cell[f"validation_{name}"]
                agrees = stored is None if value is None else stored is not None and np.isclose(value, stored, rtol=0, atol=1e-10)
                if not agrees:
                    raise ValueError(f"Saved metric differs from predictions: {name}")
            all_cells.append({"stage": stage, "cell_id": row.cell_id})
        for seed in sorted(set(results.seed)):
            previous = set()
            for key in sorted(k for k in groups if k[0] == seed):
                if not previous.issubset(groups[key]):
                    raise ValueError("Subsets are not nested")
                previous = groups[key]
        neural = results[~results.family.isin(["baseline", "legacy_tree"])]
        selected = neural.sort_values("validation_average_precision", ascending=False).iloc[0]
        checkpoint_path = folder/"cells"/f"{selected.cell_id}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        row_by_id = {r.candidate_id: i for i, r in enumerate(df.itertuples())}
        train_rows = np.array([row_by_id[k] for k in checkpoint["train_candidate_ids"]])
        _, expected_norm = model_inputs(arrays, selected.family, train_rows)
        for key in ("mean", "std"):
            np.testing.assert_allclose(checkpoint["normalization"][key], expected_norm[key], rtol=1e-6, atol=1e-7)
        val_rows = np.flatnonzero(df.split.eq("validation"))
        inputs, _ = model_inputs(arrays, selected.family, val_rows, checkpoint["normalization"])
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = make_model(selected.family, int(selected.width)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        actual = predict(model, resident(inputs, device), device)
        saved = pd.read_csv(folder/"cells"/f"{selected.cell_id}.validation.csv").score.to_numpy()
        error = float(np.max(np.abs(actual-saved)))
        if error > 1e-5:
            raise ValueError("Checkpoint reload prediction mismatch")
        checkpoints.append({"stage": stage, "cell_id": selected.cell_id, "max_absolute_score_difference": error})
        del model
    report_dir.mkdir(parents=True, exist_ok=True)
    evidence = {"cells_checked": len(all_cells), "test_evaluated": False,
                "whole_star_partitions": True, "identical_subsets_within_seed": True, "nested_data_subsets": True,
                "saved_metrics_recomputed": True, "train_only_normalization": True,
                "checkpoint_reload_checks": checkpoints,
                "candidate_sha256": provenance["candidate_sha256"], "views_sha256": provenance["views_sha256"]}
    save_json(report_dir/"audit.json", evidence)
    print(json.dumps(evidence, indent=2), flush=True)
    return evidence


def report(prepared, output, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(output/"comparison/results.csv")
    c = summary(comparison, ["variant", "family", "width"]).sort_values("ap_mean", ascending=False)
    c.insert(0, "model_name", c.variant.map(NAMES))
    c.to_csv(report_dir/"comparison_summary.csv", index=False)
    paired = paired_bootstrap(output, comparison)
    paired.insert(0, "a_name", paired.a.map(NAMES))
    paired.insert(1, "b_name", paired.b.map(NAMES))
    paired.to_csv(report_dir/"paired_comparisons.csv", index=False)
    for name in ("provenance.json", "view_diagnostics.csv"):
        shutil.copyfile(prepared/name, report_dir/name)
    for stage in ("comparison", "optimization", "scaling"):
        if (output/stage).exists():
            for name in ("results.csv", "experiment.json", "completion.json"):
                if (output/stage/name).exists():
                    if stage == "comparison" and name == "results.csv":
                        exported = comparison.copy()
                        exported.insert(0, "model_name", exported.variant.map(NAMES))
                        exported.to_csv(report_dir/f"{stage}_{name}", index=False)
                    else:
                        shutil.copyfile(output/stage/name, report_dir/f"{stage}_{name}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=True, sharey=True)
    positions = np.arange(len(c))
    for ax, metric, title in zip(axes, ("ap", "auc", "recall"), ("Average precision", "ROC-AUC", "Recall at ≤3/35 false positives")):
        ax.errorbar(c[f"{metric}_mean"], positions, xerr=c[f"{metric}_std"].fillna(0), fmt="o", capsize=3)
        ax.set(xlim=(0, 1), title=title, xlabel="Validation mean ± seed SD")
        ax.grid(axis="x", alpha=.2)
    axes[0].set_yticks(positions, [NAMES[v] for v in c.variant])
    axes[0].invert_yaxis()
    fig.suptitle("Same 423 training signals; same 85 validation signals; three independent seeds")
    fig.savefig(report_dir/"controlled_comparison.png", dpi=170)
    plt.close(fig)
    diagnostics = pd.read_csv(prepared/"view_diagnostics.csv")
    lines = ["# Controlled TESS representation comparison", "",
             "All models train on the same 423 CP/FP/FA signals and validate on the same 85 signals "
             "(50 confirmed planets, 35 FP/FA), with stars kept in separate partitions. "
             "The source is the previous scaling dataset. Test signals are excluded even from view preparation.", "",
             "Abdullah's pipeline refers to [06_train_shape_model.py](../../tls_masked_search/06_train_shape_model.py) (without its pixel images) and its associated data/view stages. Its neural variants are referred to as Abdullah's model.", "",
             "![Controlled comparison](controlled_comparison.png)", "",
             "| Model | Parameters | Validation AP ± seed SD | ROC-AUC | Recall at ≤3/35 FP | Mean fit seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in c.itertuples():
        params = f"{int(r.parameters):,}" if pd.notna(r.parameters) else "—"
        lines.append(f"| {NAMES[r.variant]} | {params} | {r.ap_mean:.3f} ± {r.ap_std:.3f} | {r.auc_mean:.3f} | {r.recall_mean:.1%} | {r.fit_seconds:.1f} |")
    lines += ["", "## What changed", "",
              "Current CNN: the original full-orbit, transit-close-up and individual-event views, with flux, uncertainty and occupancy channels and four basic scalar measurements.", "",
              "Masked cleaning: exactly the current quality/gap/window recipe, but fits the flattening trend while masking the blind candidate's transit windows. It receives no catalog ephemeris.", "",
              "Richer inputs: adds separate odd/even and phase-0.5 close-ups, signed X/Y centroid residuals, and SAP/PDCSAP full-orbit views with a three-times-longer flattening window. Each observational series has uncertainty and occupancy channels. Odd/even/secondary fluxes share the same normalization to retain their relative depths. Richer models use four basic scalars, without the diagnostic feature vector from Abdullah's pipeline.", "",
              "Abdullah's model: uses the original step-4 snapshot code on this same single FITS sector, including its masked flattening, depth normalization and missing-bin interpolation. At width 16, its architecture and predictions for identical weights match Abdullah's model in step 6 (without pixel images) exactly. The hybrid adds the original twelve measurement definitions and missing flags; one-sector/chunk rates are adapted accordingly. The views-only comparison isolates the feature vector within this architecture.", "",
              "All neural fits use the same weighted binary loss, AdamW, batch size 32, learning rate 0.001, weight decay 0.01, AP checkpoint criterion, 100-epoch ceiling and patience 20. The augmentation from Abdullah's pipeline is a separate variant. This replaces Abdullah's original five-seed ensemble and cross-validation recipe with the shared protocol. Models differ in parameter count and architecture; this is not an equal-FLOP comparison.", "",
              f"All {len(diagnostics)} prepared signals have usable centroid measurements. View-validity flags from Abdullah's pipeline are true for {int(diagnostics.legacy_ok.sum())}/{len(diagnostics)} signals; no variant drops rows based on those flags, keeping cohorts identical.", "",
              "The current tree baseline and the tree baseline from Abdullah's pipeline differ in feature set and original hyperparameters; both are trained on the same clean rows. No unmatched signals are treated as negatives.", "",
              "## Paired validation contrasts", "",
              "The differences below average individual-model AP differences across the same three seeds. Intervals resample validation stars in pairs with models fixed. They are exploratory 90% intervals, do not account for repeated validation checkpoint/model selection, and are not independent test estimates.", "",
              "| Comparison | Mean AP difference | Paired star interval |", "| --- | ---: | ---: |"]
    for r in paired.itertuples():
        lines.append(f"| {NAMES[r.a]} minus {NAMES[r.b]} | {r.mean_ap_difference:+.3f} | [{r.paired_star_bootstrap_90_low:+.3f}, {r.paired_star_bootstrap_90_high:+.3f}] |")
    if (output/"selection.json").exists() and (output/"scaling/completion.json").exists():
        selection = json.loads((output/"selection.json").read_text())
        shutil.copyfile(output/"selection.json", report_dir/"selection.json")
        optimized = pd.read_csv(output/"optimization/results.csv")
        o = summary(optimized, ["family", "width", "learning_rate", "epochs"])
        o.to_csv(report_dir/"optimization_summary.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ceiling, ax in zip(sorted(o.epochs.unique()), axes):
            subset = o[o.epochs.eq(ceiling)]
            pivot = subset.pivot(index="width", columns="learning_rate", values="ap_mean")
            im = ax.imshow(pivot.to_numpy(), vmin=.65, vmax=.95, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
            ax.set(xlabel="Learning rate", ylabel="Width", title=f"Epoch ceiling {ceiling}; patience 30")
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    ax.text(j, i, f"{pivot.iloc[i,j]:.3f}", ha="center", va="center", color="white" if pivot.iloc[i,j] < .85 else "black")
        fig.colorbar(im, ax=axes, label="Mean validation AP")
        fig.savefig(report_dir/"optimization.png", dpi=170)
        plt.close(fig)
        scaled = pd.read_csv(output/"scaling/results.csv")
        s = summary(scaled, ["family", "width", "requested_size"])
        s.to_csv(report_dir/"scaling_summary.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for width, group in s.groupby("width"):
            group = group.sort_values("requested_size")
            for ax, metric in zip(axes, ("ap", "recall")):
                ax.errorbar(group.train_signals, group[f"{metric}_mean"], yerr=group[f"{metric}_std"].fillna(0), marker="o", capsize=3, label=f"Width {width}")
        for ax in axes:
            ax.set(xscale="log", xlabel="Labeled training signals; whole-star subsets", ylim=(0, 1))
            ax.set_xticks(sorted(s.requested_size.unique()))
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=.2)
            ax.legend()
        axes[0].set(ylabel="Validation average precision", title="Selected family: data × capacity")
        axes[1].set(ylabel="Recall at ≤3/35 validation false positives", title="Candidate selection operating point")
        fig.savefig(report_dir/"selected_scaling.png", dpi=170)
        plt.close(fig)
        full = s[s.requested_size.eq(s.requested_size.max())].sort_values("ap_mean", ascending=False)
        best = full.iloc[0]
        best_rows = scaled[scaled.width.eq(best.width) & scaled.requested_size.eq(scaled.requested_size.max())].sort_values("seed")
        save_json(report_dir/"selected_models.json", {
            "family": str(best.family), "width": int(best.width),
            "learning_rate": selection["learning_rate"], "epoch_ceiling": selection["epoch_ceiling"],
            "selection_rule": "Highest mean individual-model validation AP at full training data after adaptive development",
            "validation_ap_mean": float(best.ap_mean), "validation_ap_seed_sd": float(best.ap_std),
            "validation_roc_auc_mean": float(best.auc_mean), "validation_recall_mean": float(best.recall_mean),
            "checkpoints": [str((output/"scaling/cells"/f"{r.cell_id}.pt").resolve()) for r in best_rows.itertuples()],
            "ensemble": False, "test_evaluated": False})
        lines += ["", "## Follow-up scaling", "",
                  f"The selected comparison variant was **{NAMES[selection['selected_variant']]}**, by mean validation AP. "
                  "We evaluated three widths, learning rates 0.0003/0.001/0.003, and 50/150-epoch ceilings across three seeds (54 fits). "
                  f"The recipe with highest AP averaged across widths/seeds used learning rate {selection['learning_rate']:g} "
                  f"and epoch ceiling {selection['epoch_ceiling']}. This recipe then used nested whole-star subsets of 64, 128, 256 and 423 signals (36 fits).", "",
                  "![Optimization](optimization.png)", "", "![Selected-family scaling](selected_scaling.png)", "",
                  "| Width | Parameters | Full-data validation AP ± seed SD | ROC-AUC | Recall at ≤3/35 FP |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for r in full.itertuples():
            lines.append(f"| {r.width} | {int(r.parameters):,} | {r.ap_mean:.3f} ± {r.ap_std:.3f} | {r.auc_mean:.3f} | {r.recall_mean:.1%} |")
        lines += ["", "Selection is adaptive on the same validation stars. These are development results; the largest score after tuning is optimistic. "
                  "Epoch ceilings allow early stopping and do not guarantee additional compute was used. Exact epochs, updates, cap flags and timing are saved per fit. "
                  "Each seed is an individual model, not a three-model ensemble.", "",
                  "The full-data-selected training recipe is fixed across smaller subsets. At 64 training signals, some wider models produce tied scores at 1.0; "
                  "the conservative threshold then accepts fewer than the allowed three false positives, sometimes rejecting every signal. "
                  "The operating-point curves therefore show performance under an FP ceiling, not an exact matched FP count. See per-fit FPRs in the results tables."]
    lines += ["", "## Scope and reproduction", "",
              "This experiment isolates representation and model/training changes using existing single-sector detections. "
              "It does not measure multi-sector gains, increase the clean catalog-label pool, change blind TLS candidate generation, "
              "or re-evaluate end-to-end injection recovery. All scores are uncalibrated rankings; no result establishes a newly discovered planet.", "",
              "Run instructions and interpretation: [controlled experiment guide](../../docs/controlled_experiment.md). "
              "The audit checks whole-star splits, identical/nested subsets, saved metrics and checkpoint reloads; see [audit.json](audit.json)."]
    (report_dir/"README.md").write_text("\n".join(lines)+"\n")
    print(f"Report: {report_dir/'README.md'}", flush=True)

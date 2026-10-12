"""Shared training protocol, adaptive validation-only follow-up scaling."""
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss
from torch import nn

from tess_hunt.learning.dataset import code_hash, save_json, sha256, versions
from tess_hunt.learning.model import baseline_matrix, choose_threshold, metrics
from tess_hunt.scaling.experiment import nested_subsets
from tess_hunt.scaling.benchmark import synchronize
from .data import representation_hash, source_check
from .models import make_model, model_inputs


def training_hash():
    files = [Path(__file__), Path(__file__).with_name("models.py")]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def load_prepared(prepared):
    provenance = json.loads((prepared/"provenance.json").read_text())
    source_check(Path(provenance["source"]))
    if provenance["representation_code_hash"] != representation_hash() or provenance["core_code_hash"] != code_hash() or provenance["versions"] != versions():
        raise ValueError("Representation/environment changed after preparation")
    for name, key in (("candidates.csv", "candidate_sha256"), ("views.npz", "views_sha256")):
        if sha256(prepared/name) != provenance[key]:
            raise ValueError(f"Prepared {name} was modified")
    df = pd.read_csv(prepared/"candidates.csv")
    if not df.split.isin(["train", "validation"]).all() or not df.y.isin([0, 1]).all():
        raise ValueError("Only labeled train/validation rows may enter this experiment")
    if df.candidate_id.tolist() != provenance["candidate_ids"] or df.groupby("tic").split.nunique().max() != 1:
        raise ValueError("Candidate alignment or star partition changed")
    with np.load(prepared/"views.npz") as stored:
        arrays = {k: stored[k] for k in stored.files}
    if any(len(v) != len(df) for v in arrays.values()):
        raise ValueError("Misaligned observational arrays")
    return df, arrays, provenance


def resident(inputs, device):
    return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in inputs.items()}


def predict(model, inputs, device, batch_size=128):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(next(iter(inputs.values()))), batch_size):
            out.append(torch.sigmoid(model({k: v[start:start+batch_size] for k, v in inputs.items()})))
    synchronize(device)
    return torch.cat(out).cpu().numpy()


def fit_cell(df, arrays, train_rows, val_rows, spec, config, device):
    seed, family, width = spec["seed"], spec["family"], spec["width"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(device == "cpu")
    y_train = df.y.iloc[train_rows].to_numpy()
    y_val = df.y.iloc[val_rows].to_numpy()
    history, checkpoint = [], None
    start = time.perf_counter()
    if family in ("baseline", "legacy_tree"):
        x = baseline_matrix(df).to_numpy() if family == "baseline" else np.where(np.isfinite(arrays["legacy_features"]), arrays["legacy_features"], np.nan)
        settings = (dict(max_iter=150, max_leaf_nodes=15, l2_regularization=1, min_samples_leaf=10)
                    if family == "baseline" else dict(max_iter=300))
        model = HistGradientBoostingClassifier(**settings, learning_rate=.05, random_state=seed)
        model.fit(x[train_rows], y_train)
        scores = model.predict_proba(x[val_rows])[:, 1]
        extra = {"parameters": None, "epochs_run": 0, "best_epoch": 0, "optimizer_updates": 0, "hit_epoch_cap": False}
    else:
        inputs, normalization = model_inputs(arrays, family, train_rows)
        val_inputs, _ = model_inputs(arrays, family, val_rows, normalization)
        train, validation = resident(inputs, device), resident(val_inputs, device)
        y = torch.tensor(y_train, dtype=torch.float32, device=device)
        model = make_model(family, width).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=.01)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y_train == 0).sum()/(y_train == 1).sum()), device=device))
        order_rng = torch.Generator(device="cpu").manual_seed(seed+100000)
        best_ap, best_loss, best_state, best_epoch, stale, updates = -1., np.inf, None, 0, 0, 0
        synchronize(device)
        start = time.perf_counter()
        for epoch in range(spec["epochs"]):
            model.train()
            order = torch.randperm(len(train_rows), generator=order_rng).to(device)
            losses = []
            for index in order.split(config["batch_size"]):
                batch = {k: v[index] for k, v in train.items()}
                if spec.get("augmentation", False):
                    # Original 08 recipe, separately named so augmentation is not confounded.
                    flip = torch.rand(len(index), device=device) < .5
                    for key in ("global", "local"):
                        batch[key] = torch.where(flip[:, None, None], batch[key].flip(-1), batch[key]) + .05*torch.randn_like(batch[key])
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch), y[index])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(loss.detach())
                updates += 1
            scores = predict(model, validation, device)
            ap = float(average_precision_score(y_val, scores))
            val_loss = float(log_loss(y_val, np.clip(scores, 1e-7, 1-1e-7), labels=[0, 1]))
            history.append({"epoch": epoch+1, "updates": updates, "train_loss": float(torch.stack(losses).mean().cpu()),
                            "validation_average_precision": ap, "validation_log_loss": val_loss})
            if ap > best_ap+1e-7 or (abs(ap-best_ap) <= 1e-7 and val_loss < best_loss):
                best_ap, best_loss, best_epoch, stale = ap, val_loss, epoch+1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if epoch+1 >= min(25, spec["epochs"]) and stale >= spec["patience"]:
                break
        synchronize(device)
        elapsed = time.perf_counter()-start
        model.load_state_dict(best_state)
        scores = predict(model, validation, device)
        extra = {"parameters": sum(p.numel() for p in model.parameters()), "epochs_run": len(history),
                 "best_epoch": best_epoch, "optimizer_updates": updates, "hit_epoch_cap": len(history) == spec["epochs"]}
        checkpoint = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "spec": spec,
                      "normalization": normalization, "core_code_hash": code_hash(), "training_code_hash": training_hash()}
        del optimizer, train, validation, best_state
    if checkpoint is None:
        elapsed = time.perf_counter()-start
    threshold = choose_threshold(y_val, scores, config["target_validation_fpr"])
    evaluation = metrics(y_val, scores, threshold)
    evaluation["log_loss"] = float(log_loss(y_val, np.clip(scores, 1e-7, 1-1e-7), labels=[0, 1]))
    result = {**spec, **extra, "device": "cpu" if family in ("baseline", "legacy_tree") else device,
              "train_signals": len(train_rows), "train_stars": int(df.tic.iloc[train_rows].nunique()),
              "train_planets": int(y_train.sum()), "train_negatives": int((y_train == 0).sum()),
              "fit_seconds": elapsed, "train_candidate_ids": df.candidate_id.iloc[train_rows].tolist(),
              **{f"validation_{k}": v for k, v in evaluation.items()}}
    if checkpoint is not None:
        checkpoint.update({"threshold": threshold, "validation_metrics": evaluation, "train_candidate_ids": result["train_candidate_ids"]})
    del model
    if device == "mps":
        torch.mps.empty_cache()
    return result, history, scores, checkpoint


def run_cells(prepared, output, specs, stage, device="mps"):
    if device == "mps" and (not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"):
        raise ValueError("Native MPS required; fallback must be disabled")
    df, arrays, provenance = load_prepared(prepared)
    val_rows = np.flatnonzero(df.split.eq("validation"))
    if df.y.iloc[val_rows].nunique() != 2:
        raise ValueError("Validation requires both classes")
    config = provenance["config"]
    protocol = {"prepared": str(prepared), "preparation_fingerprint": provenance["fingerprint"],
                "prepared_candidate_sha256": provenance["candidate_sha256"], "prepared_views_sha256": provenance["views_sha256"],
                "training_code_hash": training_hash(), "versions": versions(), "specs": specs, "stage": stage,
                "device": device, "validation_candidate_ids": df.candidate_id.iloc[val_rows].tolist(),
                "test_evaluated": False,
                "notes": ["Every model uses the same CP vs FP/FA rows and whole-star partitions.",
                          "All neural models share weighted BCE, AdamW weight decay 0.01, batch 32 and AP checkpoint selection.",
                          "Architectures/representations differ; parameter counts and measured compute are reported.",
                          "Legacy architecture/views reproduce 07/08 at width 16; uniform training replaces its original five-seed CV recipe.",
                          "Legacy augmentation is an explicitly separate cell; other cells have no augmentation.",
                          "FP budget is 10% of validation negatives; threshold selection and model development use validation only.",
                          "Epoch budgets are ceilings with early stopping, not fixed FLOPs or update counts.",
                          "Seeds report individual model metrics, not ensemble metrics; repeated validation selection is exploratory."]}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"experiment.json").exists() and json.loads((output/"experiment.json").read_text())["fingerprint"] != fingerprint:
        raise ValueError("Training protocol changed; use a new output directory")
    save_json(output/"experiment.json", {**protocol, "fingerprint": fingerprint})
    cells = output/"cells"
    cells.mkdir(exist_ok=True)
    results = []
    start = time.perf_counter()
    subsets = {}
    for spec in specs:
        seed, target = spec["seed"], spec["requested_size"]
        if seed not in subsets:
            subsets[seed] = nested_subsets(df, [s["requested_size"] for s in specs if s["seed"] == seed], seed)
        rows = subsets[seed][min(target, int(df.split.eq("train").sum()))]
        if df.y.iloc[rows].nunique() != 2:
            raise ValueError("Training requires both classes")
        cell_id = spec["cell_id"]
        path = cells/f"{cell_id}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["fingerprint"] != fingerprint:
                raise ValueError("Stale model cell")
        else:
            result, history, scores, checkpoint = fit_cell(df, arrays, rows, val_rows, spec, config, device)
            result["fingerprint"] = fingerprint
            predictions = df.iloc[val_rows][["candidate_id", "tic", "y"]].copy()
            predictions["score"] = scores
            predictions.to_csv(cells/f"{cell_id}.validation.csv", index=False)
            if checkpoint:
                checkpoint.update({"experiment_fingerprint": fingerprint, "prepared_views_sha256": provenance["views_sha256"]})
                torch.save(checkpoint, cells/f"{cell_id}.pt")
                pd.DataFrame(history).to_csv(cells/f"{cell_id}.history.csv", index=False)
            save_json(path, result)
        results.append(result)
        pd.DataFrame([{k: v for k, v in r.items() if k != "train_candidate_ids"} for r in results]).to_csv(output/"results.csv", index=False)
        print(f"{stage} {len(results)}/{len(specs)} {cell_id}: AP={result['validation_average_precision']:.3f}, "
              f"recall={result['validation_recall']:.1%}, fit={result['fit_seconds']:.1f}s", flush=True)
    save_json(output/"completion.json", {"cells": len(results), "wall_seconds_this_invocation": time.perf_counter()-start,
                                        "fingerprint": fingerprint, "test_evaluated": False})
    return pd.DataFrame(results)


def spec(family, width, size, seed, lr=.001, epochs=100, patience=20, augmentation=False, name=None):
    return {"family": family, "width": width, "requested_size": size, "seed": seed,
            "learning_rate": lr, "epochs": epochs, "patience": patience, "augmentation": augmentation,
            "variant": name or family, "cell_id": f"{name or family}-w{width}-n{size}-lr{lr:g}-e{epochs}-s{seed}"}


def compare(prepared, output, device="mps"):
    df, _, _ = load_prepared(prepared)
    full = int(df.split.eq("train").sum())
    variants = [("baseline", 0, False, "baseline"), ("legacy_tree", 0, False, "legacy_tree"),
                ("current", 32, False, "current32"), ("current", 64, False, "current64"),
                ("masked", 32, False, "masked32"), ("rich", 32, False, "rich32"),
                ("rich_masked", 32, False, "rich_masked32"),
                ("legacy_views", 16, False, "legacy_views"), ("legacy_hybrid", 16, False, "legacy_hybrid"),
                ("legacy_hybrid", 16, True, "legacy_hybrid_aug")]
    specs = [spec(family, width, full, seed, augmentation=augmentation, name=name)
             for family, width, augmentation, name in variants for seed in (0, 1, 2)]
    return run_cells(prepared, output/"comparison", specs, "comparison", device)


def scale(prepared, output, device="mps"):
    comparison = pd.read_csv(output/"comparison/results.csv")
    neural = comparison[~comparison.family.isin(["baseline", "legacy_tree"])]
    ranking = neural.groupby("variant").validation_average_precision.mean().sort_values(ascending=False)
    chosen_variant = str(ranking.index[0])
    chosen = neural[neural.variant.eq(chosen_variant)].iloc[0]
    family, augmentation = str(chosen.family), bool(chosen.augmentation)
    widths = [8, 16, 32] if family.startswith("legacy") else [16, 32, 64]
    df, _, _ = load_prepared(prepared)
    full = int(df.split.eq("train").sum())
    selection = {"selected_variant": chosen_variant, "selected_family": family,
                 "selection_rule": "Highest mean individual-model validation AP across the three comparison seeds",
                 "comparison_results_sha256": sha256(output/"comparison/results.csv"), "test_evaluated": False}
    save_json(output/"selection.json", selection)
    recipes = [spec(family, width, full, seed, lr=lr, epochs=epochs, patience=30, augmentation=augmentation)
               for width in widths for lr in (.0003, .001, .003) for epochs in (50, 150) for seed in (0, 1, 2)]
    tuned = run_cells(prepared, output/"optimization", recipes, "optimization", device)
    recipe = tuned.groupby(["learning_rate", "epochs"]).validation_average_precision.mean().sort_values(ascending=False).index[0]
    lr, epochs = float(recipe[0]), int(recipe[1])
    data_specs = [spec(family, width, n, seed, lr=lr, epochs=epochs, patience=30, augmentation=augmentation)
                  for n in (64, 128, 256, full) for width in widths for seed in (0, 1, 2)]
    results = run_cells(prepared, output/"scaling", data_specs, "scaling", device)
    selection.update({"learning_rate": lr, "epoch_ceiling": epochs,
                      "recipe_selection_rule": "Highest validation AP averaged across all three widths and seeds",
                      "optimization_results_sha256": sha256(output/"optimization/results.csv"),
                      "scaling_results_sha256": sha256(output/"scaling/results.csv")})
    save_json(output/"selection.json", selection)
    return results

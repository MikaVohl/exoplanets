"""Nested star-grouped training subsets; validation-only capacity/data curves."""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss
from torch import nn

from tess_hunt.learning.dataset import load_config, save_json, sha256, code_hash, versions
from tess_hunt.learning.model import SignalNet, VIEW_KEYS, baseline_matrix, choose_threshold, metrics
from .benchmark import synchronize


def experiment_hash():
    return hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))
                                   if p.name != "report.py")).hexdigest()


def nested_subsets(df, targets, seed):
    """Balance star-level ordering, keep complete groups, use one nested order per seed.

    Targets count signals, but adding a multi-signal host may overshoot slightly.
    No star is divided to hit an exact row count. Subsets preserve the full pool's
    rough class proportions rather than constructing ever-changing class priors.
    """
    train = df[df.split.eq("train") & df.y.ge(0)]
    rng = np.random.default_rng(seed)
    groups = {int(tic): g.index.to_numpy() for tic, g in train.groupby("tic")}
    cohorts = {"positive": [], "negative": [], "mixed": []}
    for tic, index in groups.items():
        values = set(df.loc[index, "y"])
        cohorts["mixed" if len(values) > 1 else "positive" if 1 in values else "negative"].append(tic)
    for name in cohorts:
        cohorts[name] = list(rng.permutation(sorted(cohorts[name])))
    total = sum(len(g) for g in cohorts.values())
    original = {name: len(group) for name, group in cohorts.items()}
    consumed = {name: 0 for name in cohorts}
    order = []
    for step in range(total):
        # Weighted round-robin of randomized hosts, keeping both classes early.
        name = max((name for name in cohorts if consumed[name] < original[name]),
                   key=lambda name: (step + 1) * original[name] / total - consumed[name])
        order.append(int(cohorts[name][consumed[name]]))
        consumed[name] += 1
    requested = sorted(set([min(int(n), len(train)) for n in targets if n > 0] + [len(train)]))
    subsets = {}
    for target in requested:
        rows = []
        for tic in order:
            rows.extend(groups[tic])
            if len(rows) >= target:
                break
        subsets[target] = np.asarray(rows, dtype=int)
    return subsets


def tensor_views(arrays, index, mean, std, device):
    tensors = {k: torch.tensor(arrays[k][index], dtype=torch.float32, device=device) for k in VIEW_KEYS}
    tensors["scalar"] = (tensors["scalar"] - torch.tensor(mean, device=device)) / torch.tensor(std, device=device)
    return tensors


def resident_scores(model, views, device, batch_size=128):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(views["scalar"]), batch_size):
            batch = {k: v[start:start + batch_size] for k, v in views.items()}
            values.append(torch.sigmoid(model(batch)))
    synchronize(device)
    return torch.cat(values).cpu().numpy()


def train_cell(df, arrays, train_rows, validation_rows, width, seed, epochs, patience, device, config):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(device == "cpu")
    mean = arrays["scalar"][train_rows].mean(0)
    std = np.maximum(arrays["scalar"][train_rows].std(0), .1)
    train = tensor_views(arrays, train_rows, mean, std, device)
    validation = tensor_views(arrays, validation_rows, mean, std, device)
    labels = df.y.to_numpy()[train_rows]
    validation_y = df.y.to_numpy()[validation_rows]
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    model = SignalNet(width).to(device)
    ordering = torch.Generator(device="cpu").manual_seed(seed + 100000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=.01)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((labels == 0).sum() / (labels == 1).sum()), device=device))
    best_ap, best_loss, best_state, best_epoch, stale, updates = -1.0, np.inf, None, 0, 0, 0
    history = []
    synchronize(device)
    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        # Same CPU-generated ordering on both backends; one index transfer per epoch.
        order = torch.randperm(len(train_rows), generator=ordering).to(device)
        losses = []
        for indices in order.split(config["batch_size"]):
            batch = {k: v[indices] for k, v in train.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), y[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.detach())
            updates += 1
        scores = resident_scores(model, validation, device)
        ap = float(average_precision_score(validation_y, scores))
        validation_loss = float(log_loss(validation_y, np.clip(scores, 1e-7, 1-1e-7), labels=[0, 1]))
        train_loss = float(torch.stack(losses).mean().cpu())
        history.append({"epoch": epoch + 1, "updates": updates, "train_loss": train_loss,
                        "validation_log_loss": validation_loss, "validation_average_precision": ap})
        improved = ap > best_ap + 1e-7 or (abs(ap - best_ap) <= 1e-7 and validation_loss < best_loss)
        if improved:
            best_ap, best_loss, best_epoch, stale = ap, validation_loss, epoch + 1, 0
            # Keep checkpoint snapshots on device to avoid moving the model/optimizer.
            best_state = {k: value.detach().clone() for k, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch + 1 >= 25 and stale >= patience:
            break
    synchronize(device)
    elapsed = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    scores = resident_scores(model, validation, device)
    threshold = choose_threshold(validation_y, scores, config["target_validation_fpr"])
    evaluation = metrics(validation_y, scores, threshold)
    evaluation["log_loss"] = float(log_loss(validation_y, np.clip(scores, 1e-7, 1-1e-7), labels=[0, 1]))
    result = {"model": "cnn", "width": width, "parameters": sum(p.numel() for p in model.parameters()),
              "seed": seed, "device": device, "train_signals": len(train_rows),
              "train_stars": int(df.iloc[train_rows].tic.nunique()),
              "train_planets": int((labels == 1).sum()), "train_negatives": int((labels == 0).sum()),
              "epochs_run": len(history), "best_epoch": best_epoch, "optimizer_updates": updates,
              "hit_epoch_cap": len(history) == epochs, "fit_seconds": elapsed,
              **{f"validation_{key}": value for key, value in evaluation.items()},
              "train_candidate_ids": df.iloc[train_rows].candidate_id.tolist()}
    checkpoint = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                  "width": width, "config": config, "scalar_mean": mean.tolist(), "scalar_std": std.tolist(),
                  "threshold": threshold, "validation_metrics": evaluation, "core_code_hash": code_hash()}
    del model, optimizer, train, validation, best_state
    if device == "mps":
        torch.mps.empty_cache()
    return result, history, scores, checkpoint


def train_baseline(df, train_rows, validation_rows, seed, config):
    model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, l2_regularization=1,
                                          min_samples_leaf=10, learning_rate=.05, random_state=seed)
    start = time.perf_counter()
    model.fit(baseline_matrix(df.iloc[train_rows]), df.y.iloc[train_rows])
    elapsed = time.perf_counter() - start
    y = df.y.iloc[validation_rows].to_numpy()
    scores = model.predict_proba(baseline_matrix(df.iloc[validation_rows]))[:, 1]
    threshold = choose_threshold(y, scores, config["target_validation_fpr"])
    evaluation = metrics(y, scores, threshold)
    evaluation["log_loss"] = float(log_loss(y, np.clip(scores, 1e-7, 1-1e-7), labels=[0, 1]))
    return {"model": "baseline", "width": 0, "parameters": None, "seed": seed, "device": "cpu",
            "train_signals": len(train_rows), "train_stars": int(df.iloc[train_rows].tic.nunique()),
            "train_planets": int(df.y.iloc[train_rows].sum()), "train_negatives": int((df.y.iloc[train_rows] == 0).sum()),
            "fit_seconds": elapsed, **{f"validation_{k}": v for k, v in evaluation.items()},
            "train_candidate_ids": df.iloc[train_rows].candidate_id.tolist()}, scores


def choose_device(output, width, requested):
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        return requested
    path = output / "device_benchmark.csv"
    if not path.exists():
        raise ValueError("Run benchmark first for measured --device auto selection")
    table = pd.read_csv(path)
    measured = table[table.width.eq(width) & table.batch_size.eq(32)]
    if len(measured) == 0:
        raise ValueError(f"No backend benchmark for width {width}")
    return str(measured.loc[measured.resident_step_seconds.idxmin(), "device"])


def scaling_grid(source, output, widths, sizes, seeds, epochs=100, patience=20, device="auto"):
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(source / "config.json")
    provenance = json.loads((source / "dataset_provenance.json").read_text())
    if provenance["code_hash"] != code_hash() or provenance["versions"] != versions():
        raise ValueError("Source preprocessing code/environment differs; regenerate the source dataset")
    for name, key in [("candidates.csv", "candidate_sha256"), ("views.npz", "views_sha256")]:
        if sha256(source / name) != provenance[key]:
            raise ValueError(f"Source {name} changed after preparation")
    df = pd.read_csv(source / "candidates.csv")
    with np.load(source / "views.npz") as data:
        arrays = {k: data[k] for k in VIEW_KEYS}
    if any(len(a) != len(df) for a in arrays.values()) or df.groupby("tic").split.nunique().max() != 1:
        raise ValueError("Source view alignment or star partition mismatch")
    validation_rows = np.flatnonzero(df.y.ge(0) & df.split.eq("validation"))
    if df.y.iloc[validation_rows].nunique() != 2:
        raise ValueError("Validation requires both classes")
    experiment = {"source": str(source), "candidate_sha256": sha256(source / "candidates.csv"),
                  "views_sha256": sha256(source / "views.npz"), "core_code_hash": code_hash(),
                  "experiment_code_hash": experiment_hash(), "versions": versions(), "widths": widths,
                  "requested_sizes": sizes, "seeds": seeds, "epochs": epochs, "patience": patience,
                  "batch_size": config["batch_size"], "learning_rate": config["learning_rate"],
                  "device_by_width": {str(w): choose_device(output, w, device) for w in widths},
                  "validation_signals": len(validation_rows), "validation_stars": int(df.iloc[validation_rows].tic.nunique()),
                  "validation_planets": int(df.y.iloc[validation_rows].sum()),
                  "validation_negatives": int((df.y.iloc[validation_rows] == 0).sum()),
                  "validation_candidate_ids": df.iloc[validation_rows].candidate_id.tolist(),
                  "test_evaluated": False,
                  "notes": ["No test predictions, metrics, model selection or threshold tuning are performed.",
                            "Three seeds vary both subset ordering and initialization; widths share subsets within seed.",
                            "Each star is an indivisible training group. Subsets are nested within seed and keep approximate class balance.",
                            "Validation AP selects checkpoint; validation log loss breaks AP ties. Minimum 25 epochs, early stopping by patience.",
                            "Each cell has the same maximum epoch budget, not the same FLOPs or optimizer updates. Compute is recorded.",
                            "Best validation scores are development measurements, not independent generalization estimates.",
                            "MPS fallback is not enabled; CPU/MPS use identical architecture and FP32 inputs."]}
    fingerprint = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
    existing = output / "experiment.json"
    if existing.exists() and json.loads(existing.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Experiment changed; use a new output directory or restore its original settings")
    save_json(existing, {**experiment, "fingerprint": fingerprint})
    cells = output / "cells"
    cells.mkdir(exist_ok=True)
    results, total_start = [], time.perf_counter()
    jobs = sum(len(nested_subsets(df, sizes, seed)) * (len(widths) + 1) for seed in seeds)
    finished = 0
    for seed in seeds:
        for requested_size, train_rows in nested_subsets(df, sizes, seed).items():
            counts = df.y.iloc[train_rows].value_counts()
            if min(counts.get(0, 0), counts.get(1, 0)) < 5:
                raise ValueError("Training subsets must contain at least five signals of each class")
            for width in [0] + widths:
                cell_id = f"n{requested_size}-w{width}-seed{seed}"
                path = cells / f"{cell_id}.json"
                if path.exists():
                    row = json.loads(path.read_text())
                    if row["fingerprint"] != fingerprint:
                        raise ValueError("Stale experiment cell")
                else:
                    if width == 0:
                        row, scores = train_baseline(df, train_rows, validation_rows, seed, config)
                    else:
                        backend = experiment["device_by_width"][str(width)]
                        row, history, scores, checkpoint = train_cell(df, arrays, train_rows, validation_rows,
                                                                      width, seed, epochs, patience, backend, config)
                        pd.DataFrame(history).to_csv(cells / f"{cell_id}.history.csv", index=False)
                        checkpoint.update({"experiment_fingerprint": fingerprint, "train_candidate_ids": row["train_candidate_ids"]})
                        torch.save(checkpoint, cells / f"{cell_id}.pt")
                    row.update({"requested_size": requested_size, "cell_id": cell_id, "fingerprint": fingerprint})
                    predictions = df.iloc[validation_rows][["candidate_id", "tic", "y"]].copy()
                    predictions["score"] = scores
                    predictions.to_csv(cells / f"{cell_id}.validation.csv", index=False)
                    save_json(path, row)
                results.append(row)
                finished += 1
                pd.DataFrame([{k: v for k, v in r.items() if k != "train_candidate_ids"} for r in results]).to_csv(output / "results.csv", index=False)
                print(f"cell {finished}/{jobs} {cell_id}: n={row['train_signals']}, "
                      f"{row['device']}, validation AP={row['validation_average_precision']:.3f}, "
                      f"fit={row['fit_seconds']:.1f}s", flush=True)
    save_json(output / "completion.json", {"cells": len(results), "wall_seconds_this_invocation": time.perf_counter()-total_start,
                                          "fingerprint": fingerprint, "test_evaluated": False})
    return pd.DataFrame(results)

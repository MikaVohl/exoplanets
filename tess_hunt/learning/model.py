"""Small scalable CNN, feature baseline, and untouched test evaluation."""
import copy
import json
import platform
import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataset import load_config, save_json, sha256, code_hash, versions

BASELINE_FEATURES = ["period", "duration_days", "search_sde", "depth_ppm", "depth_snr", "pink_snr",
                     "odd_even_sigma", "secondary_snr", "observed_transits", "significant_events"]
VIEW_KEYS = ("global", "local", "events", "event_mask", "scalar")


class Encoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(3, width, 7, stride=2, padding=3), nn.GELU(),
                                    nn.Conv1d(width, 2 * width, 5, stride=2, padding=2), nn.GELU(),
                                    nn.Conv1d(2 * width, 2 * width, 5, stride=2, padding=2), nn.GELU())

    def forward(self, x):
        x = self.layers(x)
        return torch.cat([x.mean(-1), x.amax(-1)], dim=-1)


class SignalNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.global_encoder = Encoder(width)
        self.local_encoder = Encoder(width)
        self.event_encoder = Encoder(width)
        self.head = nn.Sequential(nn.Linear(16 * width + 4, 4 * width), nn.GELU(),
                                  nn.Dropout(.2), nn.Linear(4 * width, 1))

    def forward(self, batch):
        g = self.global_encoder(batch["global"])
        local = self.local_encoder(batch["local"])
        events = batch["events"]
        b, n, channels, length = events.shape
        encoded = self.event_encoder(events.reshape(b * n, channels, length)).reshape(b, n, -1)
        mask = batch["event_mask"].unsqueeze(-1)
        count = mask.sum(1).clamp(min=1)
        mean = (encoded * mask).sum(1) / count
        variance = ((encoded - mean.unsqueeze(1)).square() * mask).sum(1) / count
        spread = torch.sqrt(variance + 1e-6)
        return self.head(torch.cat([g, local, mean, spread, batch["scalar"]], dim=-1)).squeeze(-1)


class ViewDataset(Dataset):
    def __init__(self, arrays, rows, labels, scalar_mean, scalar_std):
        self.arrays = {k: torch.from_numpy(np.asarray(arrays[k][rows], dtype=np.float32)) for k in VIEW_KEYS}
        self.arrays["scalar"] = (self.arrays["scalar"] - torch.tensor(scalar_mean)) / torch.tensor(scalar_std)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {k: value[index] for k, value in self.arrays.items()}, self.y[index]


def predict(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch, _ in loader:
            scores.extend(torch.sigmoid(model({k: v.to(device) for k, v in batch.items()})).cpu().numpy())
    return np.asarray(scores)


def choose_threshold(y, scores, maximum_fpr):
    """Freeze a conservative threshold using validation negatives only, including ties."""
    negatives = np.sort(np.asarray(scores)[np.asarray(y) == 0])[::-1]
    if len(negatives) == 0:
        raise ValueError("Threshold selection requires validation negatives")
    allowed = int(np.floor(maximum_fpr * len(negatives)))
    if allowed >= len(negatives):
        return 0.0
    return float(np.nextafter(negatives[allowed], np.inf))


def metrics(y, scores, threshold):
    y, scores = np.asarray(y), np.asarray(scores)
    accepted = scores >= threshold
    positives, negatives = y == 1, y == 0
    return {"signals": len(y), "planets": int(positives.sum()), "negatives": int(negatives.sum()),
            "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, scores)) if positives.any() else None,
            "recall": float(accepted[positives].mean()) if positives.any() else None,
            "false_positive_rate": float(accepted[negatives].mean()) if negatives.any() else None,
            "sample_precision": float(y[accepted].mean()) if accepted.any() else None,
            "accepted_signals": int(accepted.sum()), "threshold": float(threshold)}


def baseline_matrix(df):
    return df.reindex(columns=BASELINE_FEATURES).replace([np.inf, -np.inf], np.nan).astype(float)


def train_models(run, device="cpu"):
    config = load_config(run / "config.json")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["torch_threads"])
    if device == "cpu":
        torch.use_deterministic_algorithms(True)
    df = pd.read_csv(run / "candidates.csv")
    provenance = json.loads((run / "dataset_provenance.json").read_text())
    if provenance["code_hash"] != code_hash():
        raise ValueError("Pipeline changed since preparing the data; rerun prepare before training")
    if sha256(run / "candidates.csv") != provenance["candidate_sha256"]:
        raise ValueError("Candidate table changed after preparing model views; rebuild the dataset")
    if sha256(run / "views.npz") != provenance["views_sha256"]:
        raise ValueError("Views changed after preparation; rebuild the dataset")
    if provenance["versions"] != versions():
        raise ValueError("Numerical dependencies changed since preparation; rerun prepare")
    arrays = dict(np.load(run / "views.npz"))
    if any(len(a) != len(df) for a in arrays.values()):
        raise ValueError("View arrays and candidate rows are misaligned")
    labeled = df.y.ge(0)
    if df.groupby("tic").split.nunique().max() != 1:
        raise ValueError("Star leakage between partitions")
    rows = {s: np.flatnonzero((labeled & df.split.eq(s)).values) for s in ("train", "validation", "test")}
    for split, index in rows.items():
        counts = df.iloc[index].y.value_counts()
        if min(counts.get(0, 0), counts.get(1, 0)) < (5 if split == "train" else 2):
            raise ValueError(f"Too few labeled examples in {split}: {counts.to_dict()}; increase --per-class")
    scalar_mean = arrays["scalar"][rows["train"]].mean(axis=0)
    scalar_std = np.maximum(arrays["scalar"][rows["train"]].std(axis=0), .1)
    loaders = {split: DataLoader(ViewDataset(arrays, index, df.y.values[index], scalar_mean, scalar_std),
                                 batch_size=config["batch_size"], shuffle=split == "train")
               for split, index in rows.items()}
    model = SignalNet(config["width"]).to(device)
    train_y = df.y.values[rows["train"]]
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((train_y == 0).sum() / (train_y == 1).sum()), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=.01)
    history, best_state, best_ap, stale = [], None, -1.0, 0
    validation_y = df.y.values[rows["validation"]]
    for epoch in range(config["epochs"]):
        model.train()
        losses = []
        for batch, y in loaders["train"]:
            optimizer.zero_grad()
            logits = model({k: v.to(device) for k, v in batch.items()})
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_scores = predict(model, loaders["validation"], device)
        ap = float(average_precision_score(validation_y, validation_scores))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_average_precision": ap})
        print(f"epoch {epoch + 1}: loss={np.mean(losses):.4f}, validation AP={ap:.3f}", flush=True)
        if ap > best_ap:
            best_ap, best_state, stale = ap, copy.deepcopy(model.cpu().state_dict()), 0
            model.to(device)
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.to(device)
    validation_scores = predict(model, loaders["validation"], device)
    threshold = choose_threshold(validation_y, validation_scores, config["target_validation_fpr"])
    checkpoint = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "config": config,
                  "scalar_mean": scalar_mean.tolist(), "scalar_std": scalar_std.tolist(),
                  "threshold": threshold, "code_hash": code_hash(), "versions": versions(),
                  "candidate_sha256": sha256(run / "candidates.csv")}
    torch.save(checkpoint, run / "model.pt")
    baseline = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, l2_regularization=1,
                                              min_samples_leaf=10, learning_rate=.05, random_state=config["seed"])
    baseline.fit(baseline_matrix(df.iloc[rows["train"]]), train_y)
    baseline_validation = baseline.predict_proba(baseline_matrix(df.iloc[rows["validation"]]))[:, 1]
    baseline_threshold = choose_threshold(validation_y, baseline_validation, config["target_validation_fpr"])
    joblib.dump({"model": baseline, "threshold": baseline_threshold,
                 "candidate_sha256": checkpoint["candidate_sha256"]}, run / "baseline.joblib")
    # Score every row, including unmatched signals. Training labels are not needed at inference.
    loader = DataLoader(ViewDataset(arrays, np.arange(len(df)), df.y.values, scalar_mean, scalar_std),
                        batch_size=config["batch_size"])
    df["cnn_score"] = predict(model, loader, device)
    df["baseline_score"] = baseline.predict_proba(baseline_matrix(df))[:, 1]
    df["cnn_selected"] = df.cnn_score >= threshold
    df["baseline_selected"] = df.baseline_score >= baseline_threshold
    df["score_context"] = np.where(df.split.eq("train"), "training_star", "held_out_star")
    df.to_csv(run / "scores.csv", index=False)
    df[df.y.eq(-1)].sort_values("cnn_score", ascending=False).to_csv(run / "discovery_candidates.csv", index=False)
    pd.DataFrame(history).to_csv(run / "training_history.csv", index=False)
    report = {"cnn": {}, "baseline": {}, "partitions": {}, "test_subgroups": {},
              "versions": {"python": platform.python_version(), **versions()}, "device": device,
              "model_parameters": sum(p.numel() for p in model.parameters()),
              "best_epoch": max(history, key=lambda h: h["validation_average_precision"])["epoch"],
              "unlabeled_signals_scored": int((df.y == -1).sum()),
              "notes": ["Scores are uncalibrated rankings, not planet probabilities.",
                        "Precision describes this selected labeled sample, not survey candidate yield.",
                        "Thresholds selected on validation only; test labels never enter fitting or selection.",
                        "A catalog false positive is a negative signal; other signals on its star remain unlabeled.",
                        "Small test samples and confirmed-planet selection limit generalization claims."]}
    for split, index in rows.items():
        subset = df.iloc[index]
        report["partitions"][split] = {"stars": subset.tic.nunique(), "signals": len(index),
                                          "planet_stars": subset[subset.y.eq(1)].tic.nunique(),
                                          "negative_stars": subset[subset.y.eq(0)].tic.nunique()}
        report["cnn"][split] = metrics(subset.y, subset.cnn_score, threshold)
        report["baseline"][split] = metrics(subset.y, subset.baseline_score, baseline_threshold)
    test = df.iloc[rows["test"]]
    for name, subset in [("depth_below_2000ppm", test[test.depth_ppm < 2000]),
                          ("depth_2000ppm_or_more", test[test.depth_ppm >= 2000]),
                          ("period_below_5days", test[test.period < 5]),
                          ("period_5days_or_more", test[test.period >= 5])]:
        report["test_subgroups"][name] = metrics(subset.y, subset.cnn_score, threshold)
    save_json(run / "metrics.json", report)
    print(json.dumps({name: report[name]["test"] for name in ("cnn", "baseline")}, indent=2), flush=True)


def load_ranker(run):
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    config = load_config(run / "config.json")
    if checkpoint["config"] != config:
        raise ValueError("Configuration changed since model training; use the original run")
    if checkpoint["code_hash"] != code_hash():
        raise ValueError("Pipeline code changed since training; prepare/train a new compatible model")
    if checkpoint["versions"] != versions():
        raise ValueError("Numerical dependencies changed since training; reproduce the model environment")
    torch.set_num_threads(config["torch_threads"])
    model = SignalNet(config["width"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def score_views(model, checkpoint, views):
    if not views:
        return np.array([])
    arrays = {k: np.stack([v[k] for v in views]) for k in VIEW_KEYS}
    loader = DataLoader(ViewDataset(arrays, np.arange(len(views)), np.zeros(len(views)),
                                    np.array(checkpoint["scalar_mean"], dtype=np.float32),
                                    np.array(checkpoint["scalar_std"], dtype=np.float32)), batch_size=32)
    return predict(model, loader, "cpu")

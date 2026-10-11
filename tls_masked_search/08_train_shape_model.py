"""Step 8: a small 1D CNN on the step-7 folded views ("shape model"), compared with the step-6 model.

Rows, labels and the 5 star-grouped folds are exactly step 6's (its `load()` and fold seed).
Model: one conv branch for the global view, one for the stacked local views (local, odd, even,
secondary, centroid), optionally concatenated with step 6's dip measurements (standardized, NaN -> 0
plus a missing flag), then a small MLP -> planet probability. Each training fold holds out 15% of
its stars for early stopping (the test fold never influences training). Augmentation: time-reversed
views and small noise. 5 seeds per fold, averaged.
Variants: (a) views only, (b) views + measurements, (c) average of (b) and the step-6 model.

Usage (from tls_masked_search/, on a GPU):  python 08_train_shape_model.py
"""
import importlib
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn

from common import OUT_DIR

m6 = importlib.import_module("06_train_classifier")
LOCAL = ["local", "odd", "even", "secondary", "centroid"]
SEEDS = 5
MAX_EPOCHS, PATIENCE, BATCH = 200, 15, 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def conv_branch(channels, blocks):
    layers, c = [], channels
    for out in blocks:
        layers += [nn.Conv1d(c, out, 5, padding=2), nn.ReLU(), nn.Conv1d(out, out, 5, padding=2), nn.ReLU(),
                   nn.MaxPool1d(3, 2)]
        c = out
    return nn.Sequential(*layers, nn.AdaptiveMaxPool1d(4), nn.Flatten())


class ShapeNet(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.glob = conv_branch(1, [16, 32, 64])
        self.loc = conv_branch(len(LOCAL), [16, 32])
        self.head = nn.Sequential(nn.Linear(64 * 4 + 32 * 4 + n_feat, 128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128, 1))

    def forward(self, g, l, x):
        return self.head(torch.cat([self.glob(g), self.loc(l), x], 1)).squeeze(1)


def features(df, train):
    """Step-6 measurements standardized on the training rows, NaN -> 0, plus missing flags."""
    x = df[m6.DIP_FEATURES].replace([np.inf, -np.inf], np.nan).values.astype(float)
    miss = np.isnan(x)
    mu, sd = np.nanmean(x[train], 0), np.nanstd(x[train], 0) + 1e-9
    z = np.nan_to_num(np.clip((x - mu) / sd, -10, 10))
    return np.hstack([z, miss]).astype(np.float32)


def train_one(g, l, x, y, train, val, seed):
    torch.manual_seed(seed)
    net = ShapeNet(x.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    tr, va = torch.as_tensor(train, device=DEV), torch.as_tensor(val, device=DEV)
    best, best_state, wait = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        net.train()
        for b in tr[torch.randperm(len(tr), device=DEV)].split(BATCH):
            gb, lb = g[b], l[b]
            flip = torch.rand(len(b), device=DEV) < 0.5
            gb = torch.where(flip[:, None, None], gb.flip(2), gb) + 0.05 * torch.randn_like(gb)
            lb = torch.where(flip[:, None, None], lb.flip(2), lb) + 0.05 * torch.randn_like(lb)
            opt.zero_grad()
            loss_fn(net(gb, lb, x[b]), y[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(g[va], l[va], x[va]), y[va]).item()
        if v < best - 1e-4:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return net


def shape_cv(df, views, use_features):
    """Out-of-fold planet scores, step 6's folds."""
    g = torch.as_tensor(views["global"][:, None], device=DEV)
    l = torch.as_tensor(np.stack([views[k] for k in LOCAL], 1), device=DEV)
    y = torch.as_tensor(df.y.values.astype(np.float32), device=DEV)
    scores, fold_of = np.zeros(len(df)), np.zeros(len(df), int)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=m6.SEED)
    for k, (train, test) in enumerate(folds.split(df, df.y, groups=df.tic)):
        x = features(df, train) if use_features else np.zeros((len(df), 0), np.float32)
        x = torch.as_tensor(x, device=DEV)
        preds = []
        for seed in range(SEEDS):
            inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
            fit, val = next(inner.split(train, groups=df.tic.values[train]))
            net = train_one(g, l, x, y, train[fit], train[val], seed)
            with torch.no_grad():
                t = torch.as_tensor(test, device=DEV)
                preds.append(torch.sigmoid(net(g[t], l[t], x[t])).cpu().numpy())
        scores[test], fold_of[test] = np.mean(preds, 0), k
        print(f"    fold {k + 1}/5 done", flush=True)
    return scores, fold_of


def evaluate(df, s, folds):
    """The step-6 numbers on the hard task, plus all-signals AUC."""
    h = df.hard.values
    mean, std = m6.fold_stats(df, s, folds, h)
    y, call = df.y.values[h], s[h] > 0.5
    r = {"ROC AUC (hard)": f"{mean.roc_auc:.2f} ± {std.roc_auc:.2f}",
         "planets caught at rules' fake rate": f"{mean.model_planets_caught:.0%} ± {std.model_planets_caught:.0%}",
         "planets called planet (>0.5)": f"{call[y == 1].mean():.0%}",
         "fakes called planet (>0.5)": f"{call[y == 0].mean():.0%}",
         "overall correct (>0.5)": f"{(call == y).mean():.0%}"}
    hd = df[h].assign(score=s[h])
    for months, grp in hd.groupby(pd.cut(hd.sectors, [0, 1, 2, 4, 8], labels=["1", "2", "3-4", "5-8"]), observed=True):
        fake_rate = grp.rules_pass[grp.y == 0].mean()
        cut = np.quantile(grp.score[grp.y == 0], 1 - fake_rate)
        r[f"{months} months ({len(grp)} dips): rules / model"] = (
            f"{grp.rules_pass[grp.y == 1].mean():.0%} / {(grp.score[grp.y == 1] > cut).mean():.0%}")
    am, asd = m6.fold_stats(df, s, folds)
    r["ROC AUC (all signals)"] = f"{am.roc_auc:.2f} ± {asd.roc_auc:.2f} (pooled {roc_auc_score(df.y, s):.2f})"
    return r


def main():
    start = time.time()
    df = m6.load().reset_index(drop=True)
    npz = np.load(OUT_DIR / "views.npz")
    pos = pd.Series(np.arange(len(npz["tic"])), index=pd.MultiIndex.from_arrays([npz["tic"], npz["signal"]]))
    idx = pos.loc[list(zip(df.tic, df.signal))].values  # KeyError if a row has no views
    views = {k: np.clip(npz[k][idx], -10, 10) for k in ["global"] + LOCAL}
    print(f"{len(df)} rows ({df.hard.sum()} hard), {npz['ok'][idx].sum()} with ok views; device {DEV} "
          f"({torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'})")

    print("step-6 model (gradient-boosted trees)...", flush=True)
    gbt, folds, _ = m6.cross_validate(df, m6.DIP_FEATURES)
    print(f"  {time.time() - start:.0f} s")
    out = {"GBT (06)": gbt}
    for name, use in [("(a) views", False), ("(b) views + features", True)]:
        t = time.time()
        print(f"{name}...", flush=True)
        out[name], f2 = shape_cv(df, views, use)
        assert (f2 == folds).all(), "folds differ from step 6"
        print(f"  {time.time() - t:.0f} s")
    out["(c) avg (b) + GBT"] = (out["(b) views + features"] + gbt) / 2

    table = pd.DataFrame({name: evaluate(df, s, folds) for name, s in out.items()})
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print("\n" + table.to_string())
    df[["tic", "signal", "label", "y", "hard"]].assign(
        gbt=gbt, views=out["(a) views"], views_features=out["(b) views + features"],
        average=out["(c) avg (b) + GBT"], fold=folds).to_csv(OUT_DIR / "shape_scores.csv", index=False)
    print(f"\ntotal {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()

"""Step 6: a small CNN on the step-4 snapshots and step-5 pixel images ("shape model"), compared with step 3's trees.

Rows, labels and the 5 star-grouped folds are exactly step 3's (its `load()` and fold seed).
Model: one 1D conv branch for the global view, one for the stacked local views (local, odd, even,
secondary, centroid), one small 2D conv branch for the difference images (plus a missing flag),
optionally concatenated with step 3's dip measurements (standardized, NaN -> 0 plus a missing flag),
then a small MLP -> planet probability. Each training fold holds out 15% of
its stars for early stopping (the test fold never influences training). Augmentation: time-reversed
views and small noise (not on the images). 5 seeds per fold, averaged; their standard deviation is saved as the
"unsure" flag (shape_score_spread).
Variants: (a) snapshots only, (b) snapshots + pixel images + measurements (the production model),
(c) average of (b) and step 3's trees.

Usage (from tls_masked_search/, on a GPU):  python 06_train_shape_model.py [--name dataset]
  --name N reads output/N_stars.csv, N_signals.csv, N_views.npz, N_pixel_views.npz and writes N_shape_scores.csv
  (default "dataset": dataset_*.csv, views.npz, dataset_pixel_views.npz -> shape_scores.csv).
  --no-pixel leaves the pixel images out of (b); output prefix nopixel_.
"""
import argparse
import importlib
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn

m3 = importlib.import_module("03_train_trees")
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


def image_branch(channels):
    return nn.Sequential(nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                         nn.AdaptiveMaxPool2d(2), nn.Flatten())


class ShapeNet(nn.Module):
    def __init__(self, n_feat, extra_shapes=()):
        """`extra_shapes`: per-row shape (C, H, W) of each extra image view (the pixel images), one 2D branch each."""
        super().__init__()
        self.glob = conv_branch(1, [16, 32, 64])
        self.loc = conv_branch(len(LOCAL), [16, 32])
        self.extra = nn.ModuleList(image_branch(s[0]) for s in extra_shapes)
        self.head = nn.Sequential(nn.Linear(64 * 4 + 32 * 4 * (1 + len(extra_shapes)) + n_feat, 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, g, l, x, *extra):
        return self.head(torch.cat([self.glob(g), self.loc(l), *(b(e) for b, e in zip(self.extra, extra)), x], 1)).squeeze(1)


def features(df, train, cols=None):
    """Step-3 measurements (`cols`, default DIP_FEATURES) standardized on the training rows, NaN -> 0, plus missing flags."""
    x = df[cols or m3.DIP_FEATURES].replace([np.inf, -np.inf], np.nan).values.astype(float)
    miss = np.isnan(x)
    mu, sd = np.nanmean(x[train], 0), np.nanstd(x[train], 0) + 1e-9
    z = np.nan_to_num(np.clip((x - mu) / sd, -10, 10))
    return np.hstack([z, miss]).astype(np.float32)


def train_one(g, l, x, y, train, val, seed, history=None, extra=()):
    """One network, early-stopped on `val`. If `history` is a list, (training loss, validation loss) is
    appended after every round (the training loss without augmentation or dropout). `extra`: extra view
    tensors (all rows), not augmented."""
    torch.manual_seed(seed)
    net = ShapeNet(x.shape[1], [e.shape[1:] for e in extra]).to(DEV)
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
            loss_fn(net(gb, lb, x[b], *(e[b] for e in extra)), y[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(g[va], l[va], x[va], *(e[va] for e in extra)), y[va]).item()
            if history is not None:
                history.append((loss_fn(net(g[tr], l[tr], x[tr], *(e[tr] for e in extra)), y[tr]).item(), v))
        if v < best - 1e-4:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(best_state)
    return net


def shape_cv(df, views, use_features, cols=None, pixel=None):
    """Out-of-fold planet scores, step 3's folds, and the spread (std) of the SEEDS copies' scores.
    `cols`: measurements to add (default DIP_FEATURES).
    `pixel`: (array, ok) of df's rows' pixel images, or None; ~ok joins the features as a flag."""
    g = torch.as_tensor(views["global"][:, None], device=DEV)
    l = torch.as_tensor(np.stack([views[k] for k in LOCAL], 1), device=DEV)
    e = [] if pixel is None else [torch.as_tensor(pixel[0], device=DEV)]
    flags = np.zeros((len(df), 0), np.float32) if pixel is None else (~pixel[1])[:, None].astype(np.float32)
    y = torch.as_tensor(df.y.values.astype(np.float32), device=DEV)
    scores, spread, fold_of = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df), int)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=m3.SEED)
    for k, (train, test) in enumerate(folds.split(df, df.y, groups=df.tic)):
        x = features(df, train, cols) if use_features else np.zeros((len(df), 0), np.float32)
        x = torch.as_tensor(np.hstack([x, flags]), device=DEV)
        preds = []
        for seed in range(SEEDS):
            inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
            fit, val = next(inner.split(train, groups=df.tic.values[train]))
            net = train_one(g, l, x, y, train[fit], train[val], seed, extra=e)
            with torch.no_grad():
                t = torch.as_tensor(test, device=DEV)
                preds.append(torch.sigmoid(net(g[t], l[t], x[t], *(a[t] for a in e))).cpu().numpy())
        scores[test], spread[test], fold_of[test] = np.mean(preds, 0), np.std(preds, 0), k
        print(f"    fold {k + 1}/5 done", flush=True)
    return scores, fold_of, spread


def evaluate(df, s, folds):
    """The step-3 numbers on the hard task, plus all-signals AUC."""
    h = df.hard.values
    mean, std = m3.fold_stats(df, s, folds, h)
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
    am, asd = m3.fold_stats(df, s, folds)
    r["ROC AUC (all signals)"] = f"{am.roc_auc:.2f} ± {asd.roc_auc:.2f} (pooled {roc_auc_score(df.y, s):.2f})"
    return r


def rows_of(npz, df):
    """Row index in `npz` of each of `df`'s (tic, signal) rows; KeyError if a row is missing."""
    pos = pd.Series(np.arange(len(npz["tic"])), index=pd.MultiIndex.from_arrays([npz["tic"], npz["signal"]]))
    return pos.loc[list(zip(df.tic, df.signal))].values


def views_of(df, name):
    """Step-4 views (clipped) of `df`'s rows from output/<name>_views.npz (views.npz for "dataset"), and ok flags."""
    npz = np.load(m3.out_path(name, "views.npz"))
    idx = rows_of(npz, df)
    return {k: np.clip(npz[k][idx], -10, 10) for k in ["global"] + LOCAL}, npz["ok"][idx]


def pixel_views_of(df, path):
    """Step-5 pixel images of `df`'s rows from `path` (an output/*_pixel_views.npz): (x clipped, NaN -> 0; ok)."""
    npz = np.load(path)
    idx = rows_of(npz, df)
    return np.clip(np.nan_to_num(npz["x"][idx].astype(np.float32)), -10, 10), npz["ok"][idx]


def spread_table(df, s, spread):
    """Hard-task rows binned by the copies' score spread: is a high spread a sign of a wrong call?"""
    h = df.hard.values
    d = pd.DataFrame({"y": df.y.values[h], "s": s[h], "spread": spread[h]})
    d["wrong"] = (d.s > 0.5) != (d.y == 1)
    rows = {}
    for q, b in d.groupby(pd.qcut(d.spread, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]), observed=True):
        rows[q] = {"dips": len(b), "spread": f"{b.spread.min():.3f}-{b.spread.max():.3f}", "fakes": f"{(b.y == 0).mean():.0%}",
                   "correct (>0.5)": f"{1 - b.wrong.mean():.0%}", "ROC AUC": f"{roc_auc_score(b.y, b.s):.2f}"}
    print(f"\nUnsure flag (spread of the {SEEDS} copies' scores), hard task, by spread quartile:")
    print(pd.DataFrame(rows).T.to_string())
    print(f"  finding the wrong calls (ROC AUC): by spread {roc_auc_score(d.wrong, d.spread):.2f}, "
          f"by closeness of the score to 0.5 {roc_auc_score(d.wrong, -abs(d.s - 0.5)):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dataset", help="dataset name, as in step 3")
    ap.add_argument("--star-features", action="store_true",
                    help="add step 3's STAR_FEATURES (trees and (b)); output prefix star_")
    ap.add_argument("--nt", action="store_true", help="NT signals as fakes, as in step 3; output prefix nt_")
    ap.add_argument("--no-pixel", action="store_true", help="leave the step-5 pixel images out of (b); output prefix nopixel_")
    args = ap.parse_args()
    cols = m3.FEATURES if args.star_features else m3.DIP_FEATURES
    start = time.time()
    df = m3.load(args.name, args.nt).reset_index(drop=True)
    views, ok = views_of(df, args.name)
    pixel = None if args.no_pixel else pixel_views_of(df, m3.OUT_DIR / f"{args.name}_pixel_views.npz")
    print(f"{len(df)} rows ({df.hard.sum()} hard), {ok.sum()} with ok views, "
          f"{'no' if pixel is None else pixel[1].sum()} with ok pixel images; device {DEV} "
          f"({torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'})")

    print("step-3 model (gradient-boosted trees)...", flush=True)
    gbt, folds, _ = m3.cross_validate(df, cols)
    print(f"  {time.time() - start:.0f} s")
    b = "(b) views + features" + ("" if pixel is None else " + pixels")
    out, spread = {"GBT (03)": gbt}, {}
    for name, use, pix in [("(a) views", False, None), (b, True, pixel)]:
        t = time.time()
        print(f"{name}...", flush=True)
        out[name], f2, spread[name] = shape_cv(df, views, use, cols, pix)
        assert (f2 == folds).all(), "folds differ from step 3"
        print(f"  {time.time() - t:.0f} s")
    out["(c) avg (b) + GBT"] = (out[b] + gbt) / 2

    table = pd.DataFrame({name: evaluate(df, s, folds) for name, s in out.items()})
    with pd.option_context("display.width", 250, "display.max_colwidth", 40):
        print("\n" + table.to_string())
    spread_table(df, out[b], spread[b])
    prefix = ("nopixel_" if pixel is None else "") + ("nt_" if args.nt else "") + ("star_" if args.star_features else "")
    df[["tic", "signal", "label", "y", "hard"]].assign(
        gbt=gbt, views=out["(a) views"], views_features=out[b], shape_score_spread=spread[b],
        average=out["(c) avg (b) + GBT"], fold=folds).to_csv(m3.out_path(args.name, prefix + "shape_scores.csv"), index=False)
    print(f"\ntotal {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()

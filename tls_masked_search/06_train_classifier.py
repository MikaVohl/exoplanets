"""Step 6: planet-vs-fake model on the step-5 dataset, compared with the repo's fixed rules.

Examples (one row per signal our search found):
  planet = signal matches a CP/KP TOI; fake = matches an FP/FA TOI (with --nt also NT: signals on
  random stars with no TOI). Left out: unmatched signals on every star (may be unlisted planets),
  half/double matches, and PC (undecided) matches.
Model: scikit-learn HistGradientBoostingClassifier on dip measurements only (--star-features: plus
STAR_FEATURES, with a star-only leak check and a model-vs-rules check per brightness quartile).
Evaluation: 5-fold cross-validation grouped by star (a star is never in both training and testing).
Checks:
  - data-amount leak: planet stars are re-observed more, so a model given ONLY how much data a star
    has already scores well. Dip features are therefore per-dip rates and strengths; compare the main
    model with the rules within groups of stars that have the same number of months.
  - the hard task: CP/KP vs FP/FA only (NASA's flagged dips), since unmatched noise is easy to reject.
  - the fixed rules (tess_hunt: SDE >= 7, noise-aware SNR >= 7.1, >= 2 dips, no warning flags):
    how many planets does the model catch when it lets through the same share of fakes?

Usage (from tls_masked_search/):  python 06_train_classifier.py [--name dataset]   (~1 min)
  --name N reads output/N_stars.csv, N_signals.csv and writes N_classifier_scores.csv
  (default "dataset": dataset_*.csv -> classifier_scores.csv).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from common import OUT_DIR, ROOT

DIP_FEATURES = ["period", "duration_h", "depth_ppm", "sde_per_transit", "depth_snr_per_transit",
                "pink_snr_per_transit", "odd_even_sigma", "secondary_snr", "centroid_snr", "transits_per_sector",
                "significant_event_fraction", "chunk_fraction"]
STAR_PROPS = ["st_rad", "st_teff", "st_logg", "st_tmag"]  # radius (Sun = 1), temperature (K), log g (cgs), TESS mag
# Brightness (st_tmag) is left out of the model: it adds ~nothing, and bright stars get more follow-up.
STAR_FEATURES = ["st_rad", "st_teff", "st_logg", "companion_radius_earth", "duration_ratio"]
FEATURES = DIP_FEATURES + STAR_FEATURES
AMOUNT_FEATURES = ["sectors", "sectors_available", "span_days", "chunks"]
SEED = 0


def star_props(stars):
    """STAR_PROPS per tic: from the stars file if step 5 wrote them (every star, same source), else the first
    non-null value per star in the TOI table (every star of the original dataset is a TOI host)."""
    if set(STAR_PROPS) <= set(stars.columns):
        return stars[["tic"] + STAR_PROPS]
    toi = pd.read_csv(ROOT / "data" / "dataset" / "toi_labeled.csv")
    return toi.groupby("tid")[STAR_PROPS].first().rename_axis("tic").reset_index()


def with_features(s, stars):
    """Signals table + its stars table -> signals with every model feature and the fixed rules' verdict
    (no labels needed)."""
    s = s.merge(stars[["tic", "sectors", "sectors_available", "span_days", "chunks"]], on="tic")
    s = s.merge(star_props(stars), on="tic", how="left")
    # Physics checks against the star (NaN if a property is missing): size of the dimming body in Earth
    # radii (above ~25 it is a star, not a planet), and measured dip length / the length expected for a
    # planet on that period (circular orbit, central crossing: a/R = (g P^2 / (4 pi^2 R))^(1/3),
    # length = P / pi * R/a).
    s["companion_radius_earth"] = np.sqrt(s.depth_ppm.clip(lower=0) * 1e-6) * s.st_rad * 109.1
    a_over_r = (10 ** s.st_logg * (s.period * 86400) ** 2 / (4 * np.pi ** 2 * s.st_rad * 6.957e10)) ** (1 / 3)
    s["duration_ratio"] = s.duration_h / (s.period * 24 / np.pi / a_over_r)
    # Counts and signal strengths grow with the amount of data (planet stars get re-observed more), so
    # they are turned into rates and per-dip strengths, so they can't stand in for "more data".
    s["transits_per_sector"] = s.observed_transits / s.sectors
    s["significant_event_fraction"] = s.significant_events / s.observed_transits.replace(0, np.nan)
    s["chunk_fraction"] = s.n_chunks_found / s.chunks
    per_dip = np.sqrt(s.observed_transits.clip(lower=1))
    s["sde_per_transit"] = s.best_chunk_sde / per_dip
    s["depth_snr_per_transit"] = s.depth_snr / per_dip
    s["pink_snr_per_transit"] = s.pink_snr / per_dip

    flags = s["flags"].fillna("")
    s["rules_pass"] = ((s.best_chunk_sde >= 7) & (s.pink_snr >= 7.1) & (s.observed_transits >= 2)
                       & flags.eq("")).astype(int)
    return s


def load(name="dataset", nt=False):
    """Labeled rows of output/<name>_*.csv; NT rows (random no-TOI stars) count as fakes only if `nt`."""
    s = with_features(pd.read_csv(OUT_DIR / f"{name}_signals.csv"), pd.read_csv(OUT_DIR / f"{name}_stars.csv"))
    planet = s.label.isin(["CP", "KP"])
    fake = s.label.isin(["FP", "FA", "NT"] if nt else ["FP", "FA"])
    s = s[planet | fake].copy()
    s["y"] = s.label.isin(["CP", "KP"]).astype(int)
    s["hard"] = s.label.isin(["CP", "KP", "FP", "FA"])  # NASA-flagged dips only
    return s


def cross_validate(df, features):
    """Out-of-fold planet scores for every row (each row scored by a model that never saw its star)."""
    scores = np.zeros(len(df))
    importances = []
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_of = np.zeros(len(df), int)
    for k, (train, test) in enumerate(folds.split(df, df.y, groups=df.tic)):
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)
        model.fit(df[features].iloc[train], df.y.iloc[train])
        scores[test] = model.predict_proba(df[features].iloc[test])[:, 1]
        fold_of[test] = k
        if len(features) > 1:
            imp = permutation_importance(model, df[features].iloc[test], df.y.iloc[test],
                                         scoring="average_precision", n_repeats=5, random_state=SEED)
            importances.append(imp.importances_mean)
    return scores, fold_of, (np.mean(importances, axis=0) if importances else None)


def fold_table(df, scores, fold_of, mask=None):
    """Per fold: ranking quality, and planets caught at the rules' fake rate."""
    rows = []
    for k in np.unique(fold_of):
        f = (fold_of == k) & (mask if mask is not None else True)
        y, s, rule = df.y.values[f], scores[f], df.rules_pass.values[f]
        rule_fake_rate = rule[y == 0].mean()
        cut = np.quantile(s[y == 0], 1 - rule_fake_rate) if rule_fake_rate > 0 else s[y == 0].max()
        rows.append({"roc_auc": roc_auc_score(y, s), "avg_precision": average_precision_score(y, s),
                     "rules_planets_caught": rule[y == 1].mean(), "rules_fakes_passed": rule_fake_rate,
                     "model_planets_caught": (s[y == 1] > cut).mean()})
    return pd.DataFrame(rows)


def fold_stats(df, scores, fold_of, mask=None):
    """Mean and spread over folds of fold_table."""
    r = fold_table(df, scores, fold_of, mask)
    return r.mean(), r.std()


def by_group(h, groups, unit):
    """Model vs rules within groups of hard-task rows (`h` has a `score` column), at each group's rule fake rate."""
    for name, g in h.groupby(groups, observed=True):
        fake_rate = g.rules_pass[g.y == 0].mean()
        cut = np.quantile(g.score[g.y == 0], 1 - fake_rate)
        print(f"  {str(name):>3} {unit}: {len(g):4d} dips, rules catch {g.rules_pass[g.y == 1].mean():.0%} of planets, "
              f"model {(g.score[g.y == 1] > cut).mean():.0%} (both letting through {fake_rate:.0%} of fakes)")


def show(name, mean, std):
    print(f"  {name}")
    print(f"    ranking quality (ROC AUC, 0.5 = guessing, 1 = perfect): {mean.roc_auc:.2f} +/- {std.roc_auc:.2f}")
    print(f"    fixed rules: catch {mean.rules_planets_caught:.0%} of planets, let through {mean.rules_fakes_passed:.0%} of fakes")
    print(f"    model at the same fake rate: catches {mean.model_planets_caught:.0%} +/- {std.model_planets_caught:.0%} of planets")


def out_path(name, suffix):
    """output/<suffix> for the default dataset, else output/<name>_<suffix>."""
    return OUT_DIR / (suffix if name == "dataset" else f"{name}_{suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dataset", help="dataset name: reads output/<name>_stars.csv, _signals.csv")
    ap.add_argument("--star-features", action="store_true", help="add STAR_FEATURES; output prefix star_")
    ap.add_argument("--nt", action="store_true", help="NT signals (random no-TOI stars) as fakes; output prefix nt_")
    args = ap.parse_args()
    features = FEATURES if args.star_features else DIP_FEATURES
    df = load(args.name, args.nt).reset_index(drop=True)
    hard = df.hard.values
    print(f"{len(df)} signals from {df.tic.nunique()} stars: {df.y.sum()} planet, {(1 - df.y).sum()} fake "
          f"({df.hard.sum()} NASA-flagged dips, of which {df[df.hard].y.sum()} planets)")

    print("\nCheck 1: data-amount leak (model sees only how much data each star has)")
    s, folds, _ = cross_validate(df, AMOUNT_FEATURES)
    print(f"  ROC AUC {roc_auc_score(df.y, s):.2f} on all signals, "
          f"{roc_auc_score(df.y[df.hard], s[df.hard]):.2f} on NASA-flagged dips")
    if args.star_features:
        print("\nCheck 3: star leak (model sees only the star; bright stars get more follow-up and confirmations)")
        for cols in [["st_tmag"], STAR_PROPS]:
            s, folds, _ = cross_validate(df, cols)
            m, sd = fold_stats(df, s, folds, hard)
            print(f"  {' + '.join(cols)}: hard-task ROC AUC {m.roc_auc:.2f} +/- {sd.roc_auc:.2f}")

    print(f"\nMain model ({'dip measurements + star' if args.star_features else 'dip measurements only'}), "
          "5-fold cross-validation by star:")
    s, folds, imp = cross_validate(df, features)
    show("catalog-labeled signals only (unmatched signals stay unlabeled)", *fold_stats(df, s, folds))
    show("hard task: NASA-flagged dips only (real planet vs known fake)", *fold_stats(df, s, folds, hard))
    if args.star_features:
        base, _, _ = cross_validate(df, DIP_FEATURES)
        d = fold_table(df, s, folds, hard) - fold_table(df, base, folds, hard)
        print("  vs dip measurements only, per fold (hard task): ROC AUC "
              + " ".join(f"{v:+.3f}" for v in d.roc_auc) + ", planets caught "
              + " ".join(f"{v:+.0%}" for v in d.model_planets_caught))

    h = df[df.hard].assign(score=s[hard])
    print("\nCheck 2: model vs rules among stars with the same amount of data (hard task):")
    by_group(h, pd.cut(h.sectors, [0, 1, 2, 4, 8], labels=["1", "2", "3-4", "5-8"]), "months")
    if args.star_features:
        print("\nCheck 4: model vs rules among stars of similar brightness (TESS magnitude quartiles, hard task):")
        by_group(h, pd.qcut(h.st_tmag, 4), "mag")

    print("\nWhich measurements the model relied on most (drop in ranking quality when shuffled):")
    for name, v in sorted(zip(features, imp), key=lambda x: -x[1]):
        print(f"  {name:<28}{v:+.3f}")

    df["planet_score"] = s
    prefix = ("nt_" if args.nt else "") + ("star_" if args.star_features else "")
    df.to_csv(out_path(args.name, prefix + "classifier_scores.csv"), index=False)


if __name__ == "__main__":
    main()

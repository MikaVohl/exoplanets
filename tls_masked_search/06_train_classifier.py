"""Step 6: planet-vs-fake model on the step-5 dataset, compared with the repo's fixed rules.

Examples (one row per signal our search found):
  planet = signal matches a CP/KP TOI; fake = matches an FP/FA TOI. Left out: unmatched signals
  on every star (may be unlisted planets), half/double
  matches, and PC (undecided) matches.
Model: scikit-learn HistGradientBoostingClassifier on dip measurements only.
Evaluation: 5-fold cross-validation grouped by star (a star is never in both training and testing).
Checks:
  - data-amount leak: planet stars are re-observed more, so a model given ONLY how much data a star
    has already scores well. Dip features are therefore per-dip rates and strengths; compare the main
    model with the rules within groups of stars that have the same number of months.
  - the hard task: CP/KP vs FP/FA only (NASA's flagged dips), since unmatched noise is easy to reject.
  - the fixed rules (tess_hunt: SDE >= 7, noise-aware SNR >= 7.1, >= 2 dips, no warning flags):
    how many planets does the model catch when it lets through the same share of fakes?

Usage (from tls_masked_search/):  python 06_train_classifier.py   (~1 min)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from common import OUT_DIR

DIP_FEATURES = ["period", "duration_h", "depth_ppm", "sde_per_transit", "depth_snr_per_transit",
                "pink_snr_per_transit", "odd_even_sigma", "secondary_snr", "centroid_snr", "transits_per_sector",
                "significant_event_fraction", "chunk_fraction"]
AMOUNT_FEATURES = ["sectors", "sectors_available", "span_days", "chunks"]
SEED = 0


def load():
    s = pd.read_csv(OUT_DIR / "dataset_signals.csv")
    stars = pd.read_csv(OUT_DIR / "dataset_stars.csv")
    s = s.merge(stars[["tic", "sectors", "sectors_available", "span_days", "chunks"]], on="tic")
    # Counts and signal strengths grow with the amount of data (planet stars get re-observed more), so
    # they are turned into rates and per-dip strengths, so they can't stand in for "more data".
    s["transits_per_sector"] = s.observed_transits / s.sectors
    s["significant_event_fraction"] = s.significant_events / s.observed_transits.replace(0, np.nan)
    s["chunk_fraction"] = s.n_chunks_found / s.chunks
    per_dip = np.sqrt(s.observed_transits.clip(lower=1))
    s["sde_per_transit"] = s.best_chunk_sde / per_dip
    s["depth_snr_per_transit"] = s.depth_snr / per_dip
    s["pink_snr_per_transit"] = s.pink_snr / per_dip
    planet = s.label.isin(["CP", "KP"])
    fake = s.label.isin(["FP", "FA"])
    s = s[planet | fake].copy()
    s["y"] = s.label.isin(["CP", "KP"]).astype(int)
    s["hard"] = s.label.ne("none")  # NASA-flagged dips only
    flags = s["flags"].fillna("")
    s["rules_pass"] = ((s.best_chunk_sde >= 7) & (s.pink_snr >= 7.1) & (s.observed_transits >= 2)
                       & flags.eq("")).astype(int)
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


def fold_stats(df, scores, fold_of, mask=None):
    """Mean and spread over folds of: ranking quality, and planets caught at the rules' fake rate."""
    rows = []
    for k in np.unique(fold_of):
        f = (fold_of == k) & (mask if mask is not None else True)
        y, s, rule = df.y.values[f], scores[f], df.rules_pass.values[f]
        rule_fake_rate = rule[y == 0].mean()
        cut = np.quantile(s[y == 0], 1 - rule_fake_rate) if rule_fake_rate > 0 else s[y == 0].max()
        rows.append({"roc_auc": roc_auc_score(y, s), "avg_precision": average_precision_score(y, s),
                     "rules_planets_caught": rule[y == 1].mean(), "rules_fakes_passed": rule_fake_rate,
                     "model_planets_caught": (s[y == 1] > cut).mean()})
    r = pd.DataFrame(rows)
    return r.mean(), r.std()


def show(name, mean, std):
    print(f"  {name}")
    print(f"    ranking quality (ROC AUC, 0.5 = guessing, 1 = perfect): {mean.roc_auc:.2f} +/- {std.roc_auc:.2f}")
    print(f"    fixed rules: catch {mean.rules_planets_caught:.0%} of planets, let through {mean.rules_fakes_passed:.0%} of fakes")
    print(f"    model at the same fake rate: catches {mean.model_planets_caught:.0%} +/- {std.model_planets_caught:.0%} of planets")


def main():
    df = load().reset_index(drop=True)
    print(f"{len(df)} signals from {df.tic.nunique()} stars: {df.y.sum()} planet, {(1 - df.y).sum()} fake "
          f"({df.hard.sum()} NASA-flagged dips, of which {df[df.hard].y.sum()} planets)")

    print("\nCheck 1: data-amount leak (model sees only how much data each star has)")
    s, folds, _ = cross_validate(df, AMOUNT_FEATURES)
    print(f"  ROC AUC {roc_auc_score(df.y, s):.2f} on all signals, "
          f"{roc_auc_score(df.y[df.hard], s[df.hard]):.2f} on NASA-flagged dips")

    print("\nMain model (dip measurements only), 5-fold cross-validation by star:")
    s, folds, imp = cross_validate(df, DIP_FEATURES)
    show("catalog-labeled signals only (unmatched signals stay unlabeled)", *fold_stats(df, s, folds))
    show("hard task: NASA-flagged dips only (real planet vs known fake)", *fold_stats(df, s, folds, df.hard.values))

    print("\nCheck 2: model vs rules among stars with the same amount of data (hard task):")
    h = df[df.hard].assign(score=s[df.hard.values])
    for months, g in h.groupby(pd.cut(h.sectors, [0, 1, 2, 4, 8], labels=["1", "2", "3-4", "5-8"]), observed=True):
        fake_rate = g.rules_pass[g.y == 0].mean()
        cut = np.quantile(g.score[g.y == 0], 1 - fake_rate)
        print(f"  {months:>3} months: {len(g):4d} dips, rules catch {g.rules_pass[g.y == 1].mean():.0%} of planets, "
              f"model {(g.score[g.y == 1] > cut).mean():.0%} (both letting through {fake_rate:.0%} of fakes)")

    print("\nWhich measurements the model relied on most (drop in ranking quality when shuffled):")
    for name, v in sorted(zip(DIP_FEATURES, imp), key=lambda x: -x[1]):
        print(f"  {name:<28}{v:+.3f}")

    df["planet_score"] = s
    df.to_csv(OUT_DIR / "classifier_scores.csv", index=False)


if __name__ == "__main__":
    main()

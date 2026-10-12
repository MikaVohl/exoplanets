"""Step 9: rank NASA's undecided TESS candidates (TOIs with disposition PC or APC) and compare with ExoMiner++.

Stages (run in order):
  build    (CPU) every star with a PC/APC TOI, minus the step-5 training stars (so nothing scored was
           trained on), goes through exactly step 5's pipeline (same sector cap, chunked search,
           vet_signal measurements, 1% period match to the star's TOIs) and step 7's views.
           -> output/candidates_stars.csv, candidates_signals.csv, candidate_views.npz
           Raw files go to tls_masked_search/data/raw_candidates/ (git-ignored, ~28 GB). ~45 min on 96 cores.
  score    (GPU) step 6's trees and step 8's shape model (views + step-6 measurements, SEEDS seeds, each
           early-stopped on a 15% star-grouped split) trained on ALL labeled step-6 rows; every
           TOI-matched candidate signal scored.  -> output/candidate_ranking.csv (sorted by shape score;
           shape_score_spread = std of the SEEDS copies' scores, the "unsure" flag).
           Unmatched signals are left out: the models never trained on them and score most of them
           as planets (86 of the top 100 when they were included).
           --star-features adds 06's STAR_FEATURES to both models. --name picks the step-5 training set
           (e.g. dataset_v2) and --nt adds its random no-TOI stars as fakes, so junk dips score low.
           Set OMP_NUM_THREADS on a busy node.
  compare  (CPU) ExoMiner++ TESS 2-min catalog (Zenodo 10.5281/zenodo.15466292): matched to our
           TOI-matched signals by TIC + period within 1% (the catalog has no TOI column); the TCE from
           the longest sector run is used.  -> output/candidate_exominer.csv

Usage (from tls_masked_search/):  python 09_rank_candidates.py build [--limit N]
                                  python 09_rank_candidates.py score [--star-features] [--name N --nt]   (on a GPU)
                                  python 09_rank_candidates.py compare
"""
import argparse
import importlib
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd
import requests

from common import HERE, OUT_DIR, ROOT, relation

m6 = importlib.import_module("06_train_classifier")
TOIS = ROOT / "data" / "dataset" / "toi_labeled.csv"  # step 5's cached TOI table
PENDING = ["PC", "APC"]
STARS, SIGNALS, VIEWS = (OUT_DIR / f for f in ["candidates_stars.csv", "candidates_signals.csv", "candidate_views.npz"])
RANKING = OUT_DIR / "candidate_ranking.csv"
EXOMINER = HERE / "data" / "exominer"
ZENODO = "https://zenodo.org/api/records/15466293/files/{}/content"
CATALOGS = ["exominerplusplus_catalog_unk_tces_s1-s67_tess-spoc-2min_complete_1-16-2025_1014.csv",
            "exominerplusplus_catalog_labeled_tces_s1-s67_tess-spoc-2min_complete_1-14-2025_1039.csv"]


def build(args):
    # Steps 5 and 7 need transitleastsquares (CPU venv only), so they are imported here.
    m5, m7 = (importlib.import_module(n) for n in ["05_build_dataset", "07_build_views"])
    m5.CHUNK_SECTORS, m5.MAX_CHUNKS = 4, 2  # the training set's sector cap (8 sectors per star)
    m5.RAW = m7.RAW = HERE / "data" / "raw_candidates"  # git-ignored; candidates' own download cache

    def safe_download(tic):
        """05's download with its retries; NaN sectors_available = gave up."""
        for attempt in range(3):
            try:
                return m5.download(tic)
            except Exception as e:
                print(f"  download TIC {tic} attempt {attempt + 1} failed: {e!r}", flush=True)
                time.sleep(10 * 2 ** attempt)
        return [], np.nan, 0.0, 0.0

    warnings.filterwarnings("ignore")
    m5.RAW.mkdir(parents=True, exist_ok=True)
    tois = m5.toi_table()
    if not tois.tfopwg_disp.isin(PENDING).any():  # an old cache without PCs: refetch
        m5.LABELS.unlink()
        tois = m5.toi_table()
    cand = tois[tois.tfopwg_disp.isin(PENDING)].tid.unique()
    trained = set(pd.read_csv(OUT_DIR / "dataset_stars.csv").tic)
    print(f"{len(cand)} stars with a PC/APC TOI; {sum(t in trained for t in cand)} are step-5 training stars, excluded")
    cand = [int(t) for t in cand if t not in trained][:args.limit]

    stars, signals, failed = {}, [], []
    start, t_down, n_search = time.time(), None, 0

    def save():
        pd.DataFrame(list(stars.values())).to_csv(STARS, index=False)
        pd.DataFrame(signals or {"tic": []}).to_csv(SIGNALS, index=False)

    # Downloads (threads) feed searches (processes) as they finish.
    with ProcessPoolExecutor(args.workers) as procs, ThreadPoolExecutor(args.download_workers) as threads:
        procs.submit(int, 0).result()  # fork every worker before any download thread exists
        pending = {threads.submit(safe_download, tic): ("download", tic) for tic in cand}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                kind, tic = pending.pop(f)
                if kind == "download":
                    paths, avail, mb, sec = f.result()
                    stars[tic] = {"tic": tic, "sectors_available": avail, "download_mb": mb,
                                  "download_seconds": sec, "sectors": len(paths)}
                    if paths:
                        job = {"tic": tic, "paths": paths, "tois": tois[tois.tid.eq(tic)]}
                        pending[procs.submit(m5.search_star, job)] = ("search", tic)
                    if len(stars) == len(cand):
                        t_down = time.time() - start
                        print(f"Downloads done: {t_down:.0f} s, {sum(s['download_mb'] for s in stars.values()):.0f} MB, "
                              f"{sum(np.isnan(s['sectors_available']) for s in stars.values())} failed", flush=True)
                    continue
                n_search += 1
                try:
                    row, sig = f.result()
                    stars[tic].update({k: v for k, v in row.items() if k != "sectors"})
                    signals.extend(sig)
                except Exception as e:
                    failed.append((tic, repr(e)))
                    stars[tic]["skipped"] = f"search failed: {e!r}"
                if n_search % 100 == 0:
                    print(f"  {n_search} searched, {len(stars)}/{len(cand)} downloaded, {time.time() - start:.0f} s", flush=True)
                    save()
    save()
    t_search = time.time() - start
    st = pd.DataFrame(list(stars.values()))
    sig = pd.DataFrame(signals)
    print(f"Download + search: {t_search:.0f} s wall (downloads finished at {t_down or 0:.0f} s)")
    print(f"{len(st)} candidate stars: {st.sectors.gt(0).sum()} with SPOC 2-min data, "
          f"{st.get('skipped', pd.Series(dtype=str)).notna().sum()} skipped/failed searches ({len(failed)} errors: {failed[:3]})")
    print(f"{len(sig)} signals; matched dispositions: {sig.label.value_counts().to_dict()}")

    start = time.time()
    jobs = [(int(t), g[["signal", "period", "t0", "duration_h"]].to_dict("records")) for t, g in sig.groupby("tic")]
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(m7.star_views, jobs, chunksize=1), 1):
            results += res
            if i % 500 == 0:
                print(f"  views: {i}/{len(jobs)} stars, {time.time() - start:.0f} s", flush=True)
    names = m7.FLUX_VIEWS + ["centroid"]
    np.savez_compressed(VIEWS, tic=np.array([r[0] for r in results], np.int64),
                        signal=np.array([r[1] for r in results], np.int64), ok=np.array([r[3] for r in results]),
                        **{k: np.stack([r[2][k] for r in results]).astype(np.float32) for k in names})
    print(f"Views: {sum(r[3] for r in results)}/{len(results)} ok, {time.time() - start:.0f} s")


def views_for(df, path, keys):
    npz = np.load(path)
    pos = pd.Series(np.arange(len(npz["tic"])), index=pd.MultiIndex.from_arrays([npz["tic"], npz["signal"]]))
    idx = pos.loc[list(zip(df.tic, df.signal))].values  # KeyError if a row has no views
    return {k: np.clip(npz[k][idx], -10, 10) for k in keys}, npz["ok"][idx]


def score(args):
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupShuffleSplit
    m8 = importlib.import_module("08_train_shape_model")
    start = time.time()
    feats = m6.FEATURES if args.star_features else m6.DIP_FEATURES
    df = m6.load(args.name, args.nt).reset_index(drop=True)
    cand = m6.with_features(pd.read_csv(SIGNALS), pd.read_csv(STARS))
    own = m6.own_data(cand.tic.values, HERE / "data" / "raw_candidates")
    print(f"candidates: dropped {(~own).sum()} signals of {cand.tic[~own].nunique()} stars with no light curve of their own")
    cand = cand[own].reset_index(drop=True)
    keys = ["global"] + m8.LOCAL
    tv, _ = views_for(df, OUT_DIR / ("views.npz" if args.name == "dataset" else f"{args.name}_views.npz"), keys)
    cv, cok = views_for(cand, VIEWS, keys)
    print(f"training: {len(df)} rows from {df.tic.nunique()} stars ({df.y.sum()} planet); "
          f"candidates: {len(cand)} signals from {cand.tic.nunique()} stars ({cok.sum()} ok views); "
          f"features: {len(feats)} ({'06 FEATURES = dip + star' if args.star_features else '06 DIP_FEATURES'}); "
          f"device {m8.DEV}", flush=True)

    trees = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=m6.SEED)  # 06's settings
    tree = trees.fit(df[feats], df.y).predict_proba(cand[feats])[:, 1]
    print(f"trees: {time.time() - start:.0f} s", flush=True)

    both = pd.concat([df, cand], ignore_index=True)
    n = len(df)
    dev = m8.DEV
    g = torch.as_tensor(np.concatenate([tv["global"], cv["global"]])[:, None], device=dev)
    l = torch.as_tensor(np.stack([np.concatenate([tv[k], cv[k]]) for k in m8.LOCAL], 1), device=dev)
    x = torch.as_tensor(m8.features(both, np.arange(n), feats), device=dev)  # standardized on training rows only
    y = torch.as_tensor(df.y.values.astype(np.float32), device=dev)
    test = torch.as_tensor(np.arange(n, len(both)), device=dev)
    preds = []
    for seed in range(m8.SEEDS):
        fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(np.arange(n), groups=df.tic))
        net = m8.train_one(g, l, x, y, fit, val, seed)
        with torch.no_grad():
            preds.append(torch.sigmoid(net(g[test], l[test], x[test])).cpu().numpy())
        print(f"  shape seed {seed + 1}/{m8.SEEDS}: {time.time() - start:.0f} s", flush=True)
    shape = np.mean(preds, 0)

    pc = cand.label.isin(PENDING)
    tois = pd.read_csv(TOIS)
    star_tois = tois[tois.tfopwg_disp.isin(PENDING)].groupby("tid").toi.agg(lambda t: ";".join(map(str, sorted(t))))
    out = pd.DataFrame({
        "tic": cand.tic, "toi": cand.tic.map(star_tois), "signal": cand.signal, "period": cand.period,
        "depth_ppm": cand.depth_ppm, "duration_h": cand.duration_h, "shape_score": shape,
        "shape_score_spread": np.std(preds, 0), "tree_score": tree,
        "average_score": (shape + tree) / 2,  # 08's variant (c), the best in StarDetails' CV with star features
        "matched_toi": np.where(pc, cand.label_toi.astype(str), "unmatched"),
        "matched_other": np.where(~pc & cand.label.ne("none"), cand.label + " " + cand.label_toi.astype(str), ""),
        "sectors": cand.sectors, "views_ok": cok,
    }).sort_values("shape_score", ascending=False)
    out[out.matched_toi.ne("unmatched")].to_csv(RANKING, index=False)
    print(f"Spearman(shape, tree) = {out[['shape_score', 'tree_score']].corr('spearman').iloc[0, 1]:.2f}; "
          f"mean shape score: PC/APC-matched {shape[pc].mean():.2f} ({pc.sum()}), "
          f"unmatched {shape[cand.label.eq('none')].mean():.2f} ({cand.label.eq('none').sum()}, "
          f"{(shape[cand.label.eq('none')] > 0.5).mean():.0%} above 0.5); "
          f"star radius or logg missing for {cand[['st_rad', 'st_logg']].isna().any(axis=1).mean():.0%} of candidate signals "
          f"(training: {df[['st_rad', 'st_logg']].isna().any(axis=1).mean():.0%})")
    print(f"score: {time.time() - start:.0f} s -> {RANKING}")


def exominer():
    """Both ExoMiner++ catalogs (unlabeled + labeled TCEs) as tic, period, score, sector run."""
    EXOMINER.mkdir(parents=True, exist_ok=True)
    parts = []
    for name in CATALOGS:
        path = EXOMINER / name
        if not path.exists():
            r = requests.get(ZENODO.format(name), timeout=600)
            r.raise_for_status()
            path.write_bytes(r.content)
        c = pd.read_csv(path, dtype=str)
        if "TIC ID" in c:
            c = pd.DataFrame({"tic": c["TIC ID"], "period": c["Orbital Period [day]"], "score": c["ExoMiner Score"],
                              "run": c["Sector Run"], "catalog_label": ""})
        else:  # labeled: uid = tic-planet-S<run>
            c = pd.DataFrame({"tic": c.target_id, "period": c.tce_period, "score": c.score,
                              "run": c.uid.str.split("-S").str[1], "catalog_label": c.label})
        parts.append(c)
    c = pd.concat(parts, ignore_index=True)
    c[["tic", "period", "score"]] = c[["tic", "period", "score"]].astype(float)
    ends = c.run.str.split("-")
    c["n_run"] = ends.str[-1].astype(int) - ends.str[0].astype(int) + 1
    c["run_end"] = ends.str[-1].astype(int)
    return c.astype({"tic": np.int64})


def compare(args):
    ours = pd.read_csv(RANKING)
    ours = ours[ours.matched_toi.ne("unmatched")].copy()
    tois = pd.read_csv(TOIS).set_index("toi")
    ours["toi_period"] = ours.matched_toi.astype(float).map(tois.pl_orbper)
    cat = exominer()
    by_tic = dict(tuple(cat.groupby("tic")))
    theirs = []
    for r in ours.itertuples():  # the TCE from the longest sector run (ties: most recent) at the TOI's period
        c = by_tic.get(r.tic)
        hit = None if c is None else c[[relation(p, r.toi_period) == "exact" for p in c.period]]
        if hit is None or hit.empty:
            theirs.append((np.nan, "", ""))
        else:
            h = hit.sort_values(["n_run", "run_end"]).iloc[-1]
            theirs.append((h.score, h.run, h.catalog_label))
    ours["exominer_score"], ours["exominer_run"], ours["exominer_label"] = zip(*theirs)
    m = ours.dropna(subset=["exominer_score"]).drop_duplicates("matched_toi")
    print(f"{ours.matched_toi.nunique()} PC/APC TOIs found by our search; {len(m)} in ExoMiner++ "
          f"(TIC + period within 1%); {m.exominer_label.ne('').sum()} of those from its labeled catalog")
    rho = m[["shape_score", "exominer_score"]].corr("spearman").iloc[0, 1]
    rho_t = m[["tree_score", "exominer_score"]].corr("spearman").iloc[0, 1]
    print(f"Spearman: our shape score vs ExoMiner++ {rho:.2f}; our tree score vs ExoMiner++ {rho_t:.2f}")
    print("Agreement at 0.5 (rows: ours, columns: ExoMiner++):")
    print(pd.crosstab(np.where(m.shape_score > 0.5, "planet", "fake"), np.where(m.exominer_score > 0.5, "planet", "fake"),
                      rownames=["ours"], colnames=["ExoMiner++"], margins=True).to_string())
    k = min(100, len(m))
    top_ours, top_theirs = set(m.nlargest(k, "shape_score").matched_toi), set(m.nlargest(k, "exominer_score").matched_toi)
    print(f"top-{k} overlap (among matched TOIs): {len(top_ours & top_theirs)}")
    m = m.assign(diff=m.shape_score - m.exominer_score)
    cols = ["matched_toi", "tic", "shape_score", "tree_score", "exominer_score", "period", "depth_ppm", "sectors"]
    with pd.option_context("display.width", 200):
        print("\nWe say planet, ExoMiner++ says fake:\n" + m.nlargest(5, "diff")[cols].round(3).to_string(index=False))
        print("\nWe say fake, ExoMiner++ says planet:\n" + m.nsmallest(5, "diff")[cols].round(3).to_string(index=False))
        top = ours.drop_duplicates("matched_toi").head(20)
        print("\nTop 20 TOI-matched candidates by our shape score:\n"
              + top[cols].round(3).to_string(index=False))
    ours.drop(columns="toi_period").to_csv(OUT_DIR / "candidate_exominer.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["build", "score", "compare"])
    parser.add_argument("--limit", type=int, default=None, help="build: only the first N candidate stars (smoke test)")
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--download-workers", type=int, default=48)
    parser.add_argument("--star-features", action="store_true", help="score: also use 06's STAR_FEATURES")
    parser.add_argument("--name", default="dataset", help="score: step-5 training set to learn from")
    parser.add_argument("--nt", action="store_true", help="score: count its random no-TOI signals as fakes")
    args = parser.parse_args()
    {"build": build, "score": score, "compare": compare}[args.stage](args)


if __name__ == "__main__":
    main()

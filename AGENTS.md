# AGENTS.md

Finding transiting planets in NASA TESS 2-min light curves and telling real planets from fakes,
trained on NASA's expert-labeled candidates (TOIs). Two pipelines share this repo.

## Owners
- `tess_hunt/`, `scripts/`, `config/`, `docs/`: Mika (benchmark, learning, scaling, comparison experiments).
- `tls_masked_search/`, `notebooks/03_*` (created by Mika, extended by Abdullah): Abdullah (masked TLS search, trees,
  shape model, candidate ranking).
- `tess_hunt/comparison/` bridges them: it imports `tls_masked_search/04_build_snapshots.py` and hashes it
  with `02_build_dataset.py` and `common.py`.

## Run
Mika (repo root; details in `docs/`):
- `python scripts/run_benchmark.py all` - 100-star BLS/TLS benchmark
- `python -m tess_hunt.learning --run runs/NAME prepare|train|inject|report` - learning pipeline
- `python -m tess_hunt.scaling prepare|benchmark|grid|report` - scaling experiment
- `python -m tess_hunt.comparison prepare|compare|scale|report|audit` - controlled comparison

Abdullah (`cd tls_masked_search`; details in its README):
- `01_benchmark.py` - masked two-pass TLS on the 100 benchmark stars
- `02_build_dataset.py` - search ~2,200 TOI stars, label signals
- `03_train_trees.py` - gradient-boosted trees, 5-fold CV
- `04_build_snapshots.py` - folded light-curve snapshots
- `05_build_pixel_views.py` - difference images from target pixel files
- `06_train_shape_model.py` - CNN on snapshots + pixel images + measurements (GPU)
- `07_rank_candidates.py build|score|compare` - rank undecided TOIs, compare with ExoMiner++

## Data and results
- `data/raw/` holds every download and is git-ignored. Most of it was deleted to stay under the home
  quota (~500 GB; pixel files are ~80 MB per star, so 05 streams and deletes them). Training (03, 06) and
  scoring (`07 score`) need only committed files. Steps 02 and `07 build` download what they need; 04 and 05
  read light curves from `data/raw/`, so re-run 02 / `07 build` first to rebuild them.
- Committed: `data/{benchmark,catalogs,dataset,starter}/`, `reports/`, and in `tls_masked_search/output/`
  the labeled datasets, search outputs, all model inputs (`*.npz`: snapshots and pixel images, slow to
  rebuild), final tables (`benchmark_100*.csv`, `candidate_ranking.csv`, `candidate_exominer.csv`) and the
  two score files the notebooks read. Other `*_scores.csv` are ignored.

## Labels
- CP/KP: planet. FP/FA: fake. PC/APC: undecided (ranked, never trained on).
- NT: signals on random stars with no TOI; used as fakes only when asked (`--nt`).
- A signal is labeled only if its period matches a TOI; unmatched signals stay unlabeled, never fakes.

## Fair evaluation
- 5-fold cross-validation grouped by star (a star is never in both train and test).
- Test on real NASA labels (CP/KP vs FP/FA) only; synthetic or NT rows may train, never grade.
- Report the spread over folds, not one number.

## GPU
`ssh absheikh@ctr-cx71-mi300x-01`, then
`docker exec -e HIP_VISIBLE_DEVICES=0 86721e6d3fa1 bash -lc 'cd /home/absheikh/exoplanets/tls_masked_search && /opt/venv/bin/python ...'`
(home is shared: `/home/AMD/absheikh` on the host is `/home/absheikh` in the container);
chown container-written files back to the host user.

## Rules for agents
- Commit as the user; no AI attribution in commits, code or docs.
- Don't change another owner's area without telling them.
- Renaming or editing `tls_masked_search/02_build_dataset.py`, `04_build_snapshots.py` or `common.py`
  changes `tess_hunt.comparison`'s representation hash and invalidates Mika's prepared comparison runs.
- Run `python -m unittest discover -s tests` before pushing.

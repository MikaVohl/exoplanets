# Controlled supervised TESS experiment

This experiment follows the request to train on confirmed TESS labels, retain the ability to rank unknown candidates, and test methods that can improve with data and compute. It compares input representations on the existing clean cohort before adapting the winning neural family across data, capacity and optimization budgets.

## Reproduce

From the repository root, using the existing learning environment and scaling dataset:

```sh
.venv/bin/python -m tess_hunt.comparison prepare --workers 6
.venv/bin/python -m tess_hunt.comparison compare --device mps
.venv/bin/python -m tess_hunt.comparison scale --device mps
.venv/bin/python -m tess_hunt.comparison report
.venv/bin/python -m tess_hunt.comparison audit
.venv/bin/python -m unittest discover -s tests -v
```

Defaults:

- Source: `runs/tess-learning-scaling`.
- Prepared observational views: `runs/tess-controlled-inputs`.
- Model checkpoints, histories and validation predictions: `runs/tess-controlled-experiment`.
- Shareable report: `reports/tess-controlled/README.md`.

All commands accept `--source`, `--prepared`, `--output`, `--report`, `--workers` and `--device`. A changed numerical protocol or representation requires a new output directory; cache/checkpoint provenance rejects stale reuse. Preparation uses only the cached FITS sector underlying each existing blind-search detection, checks its SHA256, and excludes test rows. It neither downloads new observations nor accesses test photometry. Earlier runs and their code remain intact.

An experimental checkpoint can also score a new blind candidate:

```sh
.venv/bin/python -m tess_hunt.comparison score \
  --checkpoint /absolute/path/to/cells/model.pt \
  --fits /absolute/path/to/lightcurve.fits \
  --candidate /absolute/path/to/blind_candidate.json
```

The candidate JSON needs `period`, `t0` (BTJD), `duration_days`, and `search_sde` from the blind search. Other label/catalog fields are ignored. The representation is rebuilt from the FITS observation and the saved training normalization is reused. Output is an uncalibrated ranking and a selection-for-review flag at the saved validation threshold. The checkpoint's held-out test stars are excluded during this development phase. This interface does not confirm a planet or use the catalog as an inference input.

## Controlled comparison

Abdullah's pipeline refers to [`06_train_shape_model.py`](../tls_masked_search/06_train_shape_model.py) (without its pixel images) and its associated data/view stages. The neural model is called Abdullah's model.

The cohort contains 423 training signals (264 CP, 159 FP/FA) and 85 validation signals (50 CP, 35 FP/FA), split by star. Unknown, PC, APC and KP signals are excluded from this CP-only supervised experiment. No additional signal inherits a negative label from its host.

Each variant is run for seeds 0, 1 and 2 on exactly the same complete training stars:

| Variant | What it tests |
| --- | --- |
| Current tree baseline | The previous ten diagnostic features and fixed HGB settings |
| Abdullah's pipeline: tree baseline | Abdullah's twelve measurement definitions and HGB settings, on the clean cohort |
| Current CNN, widths 32 and 64 | Previous generic light-curve representations and a larger-capacity control |
| Masked CNN, width 32 | Same model, candidate-masked flattening only |
| Rich CNN, width 32 | More observational channels, with the original cleaning |
| Rich masked CNN, width 32 | Additional observational channels plus candidate-masked cleaning |
| Abdullah's model, views only | Original step-4 representation and step-6 architecture |
| Abdullah's model, views + measurements | The same architecture plus measurements and missing flags |
| Abdullah's model + augmentation | Original time-reversal/noise augmentation, separately controlled |

Richer observations add odd/even transits, the phase-0.5 window, signed X/Y moment-centroid residuals, and longer-window SAP/PDCSAP flux. Formal uncertainties are raised to empirical noise where necessary, and missing bins carry occupancy masks. Odd/even/secondary fluxes share a normalization scale. All new input transformations use only observational data and the blind-search ephemeris. They receive no TOI ID, label, catalog period/epoch, TIC, sector number or host metadata.

The views from Abdullah's pipeline are constructed with the actual `04_build_snapshots.py` function on the same individual sector. Its flux normalization, interpolation and centroid-distance view are preserved. Width 16 is exactly the architecture of Abdullah's model in [`06_train_shape_model.py`](../tls_masked_search/06_train_shape_model.py) without pixel images; a test loads the original weights and checks identical CPU predictions. Measurements from Abdullah's pipeline adapt sector/chunk rates to the fixed one-sector context. The original feature-vector strength is therefore compared directly, while the old mixture of labels, multi-sector coverage, folds and ensemble is removed.

All neural variants use weighted BCE, AdamW at learning rate 0.001 and weight decay 0.01, gradient clipping at 5, batch size 32, validation AP checkpoint selection, a 100-epoch ceiling and patience 20 after at least 25 epochs. Features/scalars are normalized using fit rows only. This uniform recipe differs from the original training recipe for Abdullah's model; architecture/representation comparisons do not establish best achievable performance for each model. Parameter counts differ and are reported rather than treated as equal compute.

Native MPS runs use FP32 with CPU fallback disabled and resident inputs. Random minibatch order uses a dedicated CPU generator. Dropout varies by backend, so training trajectories need not be identical across CPU and MPS.

## Adaptive follow-up

The highest mean individual-model validation AP selects the neural variant. Follow-up optimization tests three widths, three learning rates (0.0003, 0.001, 0.003), and 50/150-epoch ceilings with patience 30, each with three seeds: 54 fits. Widths for Abdullah's model are 8/16/32; other neural widths are 16/32/64.

The learning-rate/epoch recipe with highest AP averaged over widths and seeds then runs data/capacity curves on 64, 128, 256 and the full 423 training signals: 36 fits. Star groups are indivisible, subsets are nested within seed, and all widths share each subset. A requested count can be slightly exceeded when adding a multi-signal star. Exact epochs, optimizer updates, measured seconds and epoch-cap flags are recorded.

This uses equal maximum epochs, not equal optimizer updates or FLOPs. Early stopping can also make a higher ceiling use no additional compute. Repeated use of the same validation set makes selected results optimistic; no selected maximum is an independent test estimate. Scores across seeds are averaged as metrics, not predictions. The report's paired star bootstrap contrasts are exploratory and do not adjust for validation/model selection.

## Evaluation boundary

Thresholds are selected on validation negatives with the existing 10% budget. With 35 negatives the maximum allowed count is three, giving an empirical FPR of 8.6%; recall differences are sensitive to a handful of examples. Scores are ranking outputs rather than calibrated planet probabilities.

The audit checks original candidate alignment, CP/FP/FA labels, no test IDs/stars, shared whole-star subsets, nested data order, validation-only predictions, recomputed saved metrics, training-only normalization, and checkpoint reload predictions.

This phase holds single-sector coverage and candidate generation fixed. It does not claim multi-sector or larger-label-pool gains, improved search completeness, new injection recovery, or a verified planet discovery. An observational representation can also encode observing/processing selection effects; a future time/sector holdout is useful once the development recipe is frozen. A final test and end-to-end search/injection evaluation should follow that freeze.

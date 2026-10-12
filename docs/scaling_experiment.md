# TESS model/data scaling experiment

The experiment measures whether more reliably labeled TESS signals and larger CNNs improve ranking, and whether PyTorch's MPS backend accelerates the actual workload on this Mac. Its modules live outside `tess_hunt/learning/`, so adding experiment/report code does not invalidate the saved starter model.

## Reproduction

From the repository root, with the learning dependencies installed:

```sh
# Same catalog and preprocessing as the starter; expand to up to 512 hosts per class.
.venv/bin/python -m tess_hunt.scaling prepare --download-workers 32

# Run these timings without other CPU/GPU-heavy work for an uncontended comparison.
.venv/bin/python -m tess_hunt.scaling benchmark \
  --source runs/tess-learning-scaling --steps 8 --cpu-sweep --epoch-compare

# Includes the full available training pool in addition to the requested levels.
.venv/bin/python -m tess_hunt.scaling grid \
  --widths 8 16 32 64 --sizes 32 64 128 256 --seeds 0 1 2 \
  --epochs 100 --patience 20 --device auto

.venv/bin/python -m tess_hunt.scaling report
```

`prepare` batches exact TIC queries through MAST, filters for SPOC two-minute light curves, and then uses the established downloader, checksum validation, blind search and signal labeling. Product inventories and FITS files are reused. The confirmed-positive catalog is finite, so requesting 512 does not guarantee 512 eligible positive hosts. Unavailable products and failed downloads are recorded. The 100-star benchmark stays outside the sampling pool.

Large original files, prepared views, checkpoints and per-cell progress are kept in ignored `runs/` and `data/raw/`. The compact results and figures are exported to [reports/tess-scaling](../reports/tess-scaling/README.md).

## Experimental controls

- Only CP signals are positive by default; period-and-phase-matched FP/FA signals are negative. Other signals do not enter supervised training.
- TIC hashing keeps each star's sectors and signals in one partition. No test predictions or test performance metrics are produced by this experiment.
- For each seed, whole training stars are shuffled within class cohorts and interleaved to preserve approximate class proportions. Training subsets are nested. A multi-signal host can make the actual signal count slightly exceed the requested target.
- Every width and the feature baseline use the same subset for a given size and seed. The full pool is identical across seeds.
- All models use the same fixed validation stars. Scalar normalization statistics come exclusively from the selected training subset.
- CNN widths 8, 16, 32 and 64 have 10,673, 41,057, 160,961 and 637,313 parameters.
- The learning rate is 0.001, batch size 32, and weight decay 0.01. Each CNN can train for 100 epochs, with at least 25 epochs before patience-based stopping. Validation average precision selects its checkpoint; log loss breaks ties.
- There is no architecture-specific learning-rate tuning. The feature baseline keeps the starter's gradient-boosted-tree settings.

The epoch ceiling is held fixed, not the number of optimizer steps or FLOPs. Larger datasets therefore receive more updates. Saved training histories include updates, train loss, validation loss and validation AP; per-cell records include fit time, best epoch and whether the ceiling was reached. These diagnostics distinguish observed data scaling from optimization limitations without claiming they are perfectly isolated.

Three seeds vary both subset composition and network initialization. Plot error bars are standard deviations across those seeds, not confidence intervals for the underlying planet population. Best validation measurements are development results and can be optimistic. A final scientific performance estimate needs a separately reserved test evaluation after model choice.

## CPU versus Metal

PyTorch reports whether MPS was built and is available. The benchmark runs the same architecture, float32 inputs and optimizer on CPU and MPS, without enabling CPU fallback for unsupported Metal operations.

For each width and batch size 32/128, it performs five warmup steps, then three timed trials. Each step contains the forward pass, binary cross-entropy, backward pass, gradient clipping and AdamW update. Metal is explicitly synchronized before and after every timed trial. Timing unsynchronized dispatch would undercount GPU work.

Two transfer policies are measured: data already resident on the device, and moving each batch from CPU to the device. The scaling trainer keeps its complete training and validation tensors on Metal, avoiding repeated transfers. This is feasible for this dataset and is part of the implementation, not an assumption that all future datasets fit in memory.

An extra CPU sweep checks 1, 2, 4 and 8 threads at widths 16 and 64. A fixed 12-epoch full-loop comparison then uses the fastest measured CPU thread setting and includes indexing, validation and checkpoint selection. This comparison uses identical data and a dedicated CPU generator for minibatch order on both backends. Dropout generators still differ across CPU/MPS; identical training trajectories are not claimed. Identical evaluation weights and inputs are checked for numerical agreement.

The GPU acceleration covers CNN training. MAST acquisition, photometry processing, TLS and the feature baseline remain CPU work. Actual speedups and machine/library versions are saved in the report; the experiment does not assume that Metal will win for every future model or batch size.

## Outputs and resuming

`runs/tess-scaling-experiment/experiment.json` stores source hashes, protocol, validation IDs, backend choices and the experiment fingerprint. `results.csv` records each completed cell; `cells/` stores histories, validation scores, training IDs and checkpoints. Completed cells are reused only when the experiment fingerprint matches. Changed source data or protocol requires a new output directory. The benchmark files should be finalized before starting a grid so automatic backend selection remains consistent.

Run the scientific controls:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

References: [PyTorch MPS backend](https://docs.pytorch.org/docs/2.14/notes/mps.html), [Metal synchronization](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html), and [MAST observation/product queries](https://astroquery.readthedocs.io/en/latest/mast/mast_obsquery.html).

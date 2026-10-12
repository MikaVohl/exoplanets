"""Apply an experimental checkpoint to a blind candidate without catalog inputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy.io import fits

from tess_hunt.learning.dataset import code_hash, sha256, versions
from tess_hunt.learning.photometry import read_raw, clean, make_views, bin_search
from tess_hunt.search import vet_signal
from .data import candidate_views, OBSERVATIONAL_FIELDS, representation_hash
from .experiment import training_hash, resident, predict
from .models import make_model, model_inputs


def score_candidate(checkpoint_path, path, candidate, device="mps"):
    """Labels/TOI fields in a candidate record are ignored, never fed to a model.

    The held-out test stars of this development experiment remain protected.
    A new candidate needs only measured period, epoch, duration and TLS SDE.
    """
    checkpoint_path, path = Path(checkpoint_path), Path(path)
    protocol = json.loads((checkpoint_path.parent.parent/"experiment.json").read_text())
    provenance = json.loads((Path(protocol["prepared"])/"provenance.json").read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint["experiment_fingerprint"] != protocol["fingerprint"]
            or checkpoint["training_code_hash"] != training_hash()
            or checkpoint["core_code_hash"] != code_hash()
            or checkpoint["prepared_views_sha256"] != provenance["views_sha256"]
            or provenance["representation_code_hash"] != representation_hash()
            or provenance["versions"] != versions()):
        raise ValueError("Checkpoint preparation/code/environment differs")
    source_path = Path(provenance["source"])/"candidates.csv"
    if sha256(source_path) != provenance["source_candidate_sha256"]:
        raise ValueError("Source star partitions changed after preparation")
    source = pd.read_csv(source_path)
    tic = int(fits.getheader(path, 0)["TICID"])
    stars = source[source.tic.eq(tic)]
    split = str(stars.split.iloc[0]) if len(stars) else "new_star"
    if split == "test":
        raise ValueError("Test stars are excluded from development inference; use a frozen final evaluation")
    raw = read_raw(path)
    config = provenance["config"]
    measured = {k: float(candidate[k]) for k in ("period", "t0", "duration_days", "search_sde")}
    original = clean(raw, config)
    diagnostics = vet_signal(bin_search(original, config["bin_minutes"]), measured["period"], measured["t0"], measured["duration_days"])
    measured.update({k: diagnostics[k] for k in OBSERVATIONAL_FIELDS if k not in measured})
    views, _ = candidate_views(path, measured, config)
    views.update(make_views(original, measured, config))
    arrays = {k: v[None] for k, v in views.items()}
    spec = checkpoint["spec"]
    inputs, _ = model_inputs(arrays, spec["family"], np.array([0]), checkpoint["normalization"])
    model = make_model(spec["family"], spec["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    score = float(predict(model, resident(inputs, device), device)[0])
    return {"score": score, "validation_threshold": checkpoint["threshold"],
            "selected_for_review": score >= checkpoint["threshold"],
            "score_kind": "uncalibrated ranking", "star_partition": split,
            "candidate_ephemeris": {k: measured[k] for k in ("period", "t0", "duration_days")},
            "fits_sha256": sha256(path), "checkpoint_sha256": sha256(checkpoint_path),
            "catalog_features_used": False}

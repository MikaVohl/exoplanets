"""Signal labels and stable partitions; catalog information never enters search."""
import hashlib

import numpy as np

from tess_hunt.search import phase_offset, period_relation


def partition(tic, config):
    """All products and signals of a TIC retain one partition as the sample grows."""
    digest = hashlib.sha256(f"{config['seed']}:{int(tic)}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < config["train_fraction"]:
        return "train"
    if value < config["train_fraction"] + config["validation_fraction"]:
        return "validation"
    return "test"


def match_signal(signal, catalog, time, config):
    """Conservative period + phase matching over the observing baseline.

    Catalog epochs are BJD_TDB; photometry/search epochs are BTJD. Missing or
    drifting ephemerides, aliases, and conflicting catalog matches stay unlabeled.
    This can lose legitimate labels with TTVs or stale ephemerides deliberately.
    """
    hits, reasons = [], []
    for row in catalog.to_dict("records"):
        period = row.get("pl_orbper", np.nan)
        if not np.isfinite(period) or period <= 0:
            continue
        relation = period_relation(signal["period"], period, config["period_tolerance"])
        if relation is None:
            continue
        if relation != "exact":
            reasons.append("catalog_period_alias")
            continue
        epoch = row.get("pl_tranmid", np.nan)
        if not np.isfinite(epoch):
            reasons.append("catalog_epoch_missing")
            continue
        epoch -= 2457000
        duration = row.get("pl_trandurh", np.nan) / 24
        duration = duration if np.isfinite(duration) and duration > 0 else signal["duration_days"]
        tolerance = min(config["epoch_tolerance_durations"] * max(duration, signal["duration_days"]),
                        0.1 * period)
        cycles = np.unique(np.rint((np.asarray(time) - signal["t0"]) / signal["period"]).astype(int))
        centers = signal["t0"] + cycles * signal["period"]
        offset = float(np.max(np.abs(phase_offset(centers, period, epoch))))
        if offset > tolerance:
            reasons.append("catalog_phase_mismatch")
            continue
        hits.append((str(row["tfopwg_disp"]), str(row["toi"]), offset))
    if len(hits) != 1:
        return {"y": -1, "label": "unlabeled", "label_toi": "",
                "label_reason": "ambiguous_catalog_matches" if len(hits) > 1 else
                ";".join(sorted(set(reasons))) or "no_catalog_match"}
    disposition, toi, offset = hits[0]
    y = (1 if disposition in config["positive_dispositions"] else
         0 if disposition in config["negative_dispositions"] else -1)
    return {"y": y, "label": disposition, "label_toi": toi,
            "label_reason": "period_and_phase_match" if y >= 0 else "unresolved_disposition",
            "catalog_phase_offset_days": offset}

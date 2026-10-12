"""Blind candidate generation, identical for labeled, discovery, and injected data."""
import numpy as np

from tess_hunt.search import bls_search, tls_search, vet_signal
from .photometry import bin_search


def find_signals(data, config, stellar=None):
    stellar = stellar or {"mass": 1.0, "radius": 1.0}
    settings = {"period_min_days": config["period_min_days"], "period_max_days": config["period_max_days"],
                "minimum_observed_transits": 2, "tls_oversampling": 3, "tls_threads_per_worker": 1,
                "tls_duration_grid_step": 1.1, "duration_min_days": .025, "duration_max_days": .3,
                "maximum_candidates_per_method": 1}
    remaining = bin_search(data, config["bin_minutes"])
    signals = []
    for index in range(config["maximum_signals"]):
        if len(remaining["time"]) < 100 or np.ptp(remaining["time"]) / 2 <= config["period_min_days"]:
            break
        method = tls_search if config["search_method"] == "tls" else bls_search
        result, _ = method(remaining, stellar, settings)
        best = result["best"]
        if not np.isfinite(best["sde"]) or not np.isfinite(best["duration_days"]):
            break
        signal = {"signal": index + 1, "period": best["period"], "t0": best["epoch_btjd"],
                  "duration_days": best["duration_days"], "search_sde": best["sde"],
                  "search_method": config["search_method"], "search_seconds": result["runtime_seconds"]}
        # Diagnostics accompany scores; they do not veto candidates before learning.
        vet = vet_signal(bin_search(data, config["bin_minutes"]), signal["period"], signal["t0"],
                         signal["duration_days"])
        # This path has no centroid series; zero would imply a measured non-shift.
        vet["centroid_snr"] = float("nan")
        signal.update({k: v for k, v in vet.items() if k != "events"})
        signal["flags"] = ";".join(vet["flags"])
        signals.append(signal)
        phase = (remaining["time"] - signal["t0"] + signal["period"] / 2) % signal["period"] - signal["period"] / 2
        keep = np.abs(phase) > max(signal["duration_days"], 1.5 / 24)
        remaining = {key: value[keep] for key, value in remaining.items()}
    return signals

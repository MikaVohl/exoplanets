"""Independent full-range BLS/TLS searches, followed by common signal vetting."""

import time

import numpy as np
from astropy.timeseries import BoxLeastSquares
from transitleastsquares import transitleastsquares, period_grid

from .photometry import robust_sigma


def phase_offset(t, period, epoch):
    return (np.asarray(t) - epoch + period / 2) % period - period / 2


def period_relation(a, b, tolerance=.01):
    for ratio, label in [(1.0, "exact"), (.5, "half_period"), (2.0, "double_period")]:
        if abs(a / (b * ratio) - 1) < tolerance:
            return label
    return None


def weighted_mean(y, error):
    weight = 1 / error ** 2
    return float(np.average(y, weights=weight)), float(1 / np.sqrt(weight.sum()))


def vet_signal(data, period, epoch, duration):
    t, y, dy = (data[k] for k in ["time", "flux", "error"])
    phase = phase_offset(t, period, epoch)
    inside = abs(phase) < duration / 2
    outside = abs(phase) > duration
    if inside.sum() < 5 or outside.sum() < 10:
        return {"observed_transits": 0, "depth_snr": 0.0, "pink_snr": 0.0, "flags": ["insufficient_transit_sampling"], "events": []}
    base, base_err = weighted_mean(y[outside], dy[outside])
    bottom, bottom_err = weighted_mean(y[inside], dy[inside])
    depth = base - bottom
    depth_err = np.hypot(base_err, bottom_err)
    events = []
    cycles = np.rint((t - epoch) / period).astype(int)
    for cycle in np.unique(cycles[inside]):
        center = epoch + cycle * period
        event = inside & (cycles == cycle)
        # Compare the central halves of events, reducing ingress/egress sampling
        # differences that can mimic alternating depths in sparsely binned data.
        core = (abs(phase) < duration / 4) & (cycles == cycle)
        # A local, symmetric reference helps identify dips from isolated ramps.
        local = (abs(t - center) > duration) & (abs(t - center) < min(period * .4, max(.4, 3 * duration)))
        if event.sum() < 3 or core.sum() < 1 or local.sum() < 5:
            continue
        reference, reference_error = weighted_mean(y[local], dy[local])
        value, value_error = weighted_mean(y[core], dy[core])
        err = np.hypot(reference_error, value_error)
        events.append({"cycle": int(cycle), "epoch_btjd": float(center), "points": int(event.sum()),
                       "depth_region": "central_half_of_transit",
                       "depth": reference - value, "depth_error": float(err),
                       "snr": float((reference - value) / err)})
    # Estimate correlated noise from out-of-transit averages on a transit-duration timescale.
    labels = np.floor((t - t.min()) / max(duration, .025)).astype(int)
    blocks = []
    for label in np.unique(labels):
        block = (labels == label) & outside
        if block.sum() >= 3:
            blocks.append(weighted_mean(y[block], dy[block])[0])
    block_noise = robust_sigma(blocks) if len(blocks) > 10 else robust_sigma(y[outside])
    pink_error = max(depth_err, block_noise / np.sqrt(max(1, len(events))))
    odd_even = float("nan")
    groups = [[e for e in events if e["cycle"] % 2 == parity] for parity in [0, 1]]
    if all(groups):
        means = [weighted_mean(np.array([e["depth"] for e in group]), np.array([e["depth_error"] for e in group])) for group in groups]
        odd_even = abs(means[0][0] - means[1][0]) / np.hypot(means[0][1], means[1][1])
    secondary = abs(phase_offset(t, period, epoch + period / 2)) < duration / 2
    secondary_snr, secondary_depth = 0.0, 0.0
    if secondary.sum() >= 5:
        value, error = weighted_mean(y[secondary], dy[secondary])
        secondary_depth = base - value
        secondary_snr = secondary_depth / max(np.hypot(base_err, error), pink_error)
    centroid_snr = 0.0
    for name in ["centroid1", "centroid2"]:
        centroid = data.get(name)
        if centroid is None:
            continue
        good = np.isfinite(centroid)
        if (good & inside).sum() >= 5 and (good & outside).sum() > 20:
            scatter = robust_sigma(centroid[good & outside])
            if scatter > 0:
                shift = np.mean(centroid[good & inside]) - np.mean(centroid[good & outside])
                uncertainty = scatter * np.sqrt(1 / (good & inside).sum() + 1 / (good & outside).sum())
                block_centroids = []
                for label in np.unique(labels):
                    block = (labels == label) & outside & good
                    if block.sum() >= 3:
                        block_centroids.append(float(np.mean(centroid[block])))
                if len(block_centroids) > 10:
                    uncertainty = max(uncertainty, robust_sigma(block_centroids) / np.sqrt(max(1, len(events))))
                centroid_snr = max(centroid_snr, float(abs(shift) / uncertainty))
    flags = []
    if depth > .05: flags.append("deep_eclipse")
    if duration / period > .2: flags.append("large_duty_cycle")
    if odd_even > 3: flags.append("odd_even_depth_mismatch")
    if secondary_snr > 3 and secondary_depth > .3 * max(depth, 0): flags.append("secondary_eclipse")
    if centroid_snr > 5: flags.append("centroid_shift_requires_pixel_review")
    if len(events) < 2: flags.append("fewer_than_two_observed_transits")
    significant_events = sum(e["snr"] >= 3 for e in events)
    if significant_events < 2: flags.append("fewer_than_two_individually_supported_events")
    return {"depth": float(depth), "depth_ppm": float(depth * 1e6), "depth_error": float(depth_err),
            "depth_snr": float(depth / depth_err), "pink_snr": float(depth / pink_error),
            "observed_transits": len(events), "significant_events": significant_events,
            "odd_even_sigma": float(odd_even), "secondary_snr": float(secondary_snr),
            "centroid_snr": centroid_snr, "flags": flags, "events": events}


def peak_indices(periods, powers, count):
    selected = []
    for index in np.argsort(powers)[::-1]:
        if not any(abs(periods[index] / periods[j] - 1) < .02 for j in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    return selected


def bls_search(data, stellar, config):
    start = time.monotonic()
    t, y, dy = (data[k] for k in ["time", "flux", "error"])
    maximum = min(config["period_max_days"], np.ptp(t) / config["minimum_observed_transits"])
    periods = period_grid(R_star=stellar["radius"], M_star=stellar["mass"], time_span=np.ptp(t),
                          period_min=config["period_min_days"], period_max=maximum,
                          oversampling_factor=config["tls_oversampling"], n_transits_min=2)
    periods = np.sort(np.asarray(periods))
    durations = np.geomspace(config["duration_min_days"], config["duration_max_days"], 14)
    result = BoxLeastSquares(t, y, dy).power(periods, durations, objective="likelihood", oversample=10)
    center = float(np.median(result.power))
    scale = max(robust_sigma(result.power), 1e-12)
    scores = (result.power - center) / scale
    peaks = []
    for index in peak_indices(result.period, result.power, config["maximum_candidates_per_method"]):
        p, t0, duration = float(result.period[index]), float(result.transit_time[index]), float(result.duration[index])
        peaks.append({"period": p, "epoch_btjd": t0, "duration_days": duration,
                      "sde": float(scores[index]), "model_depth": float(result.depth[index]),
                      "model_depth_snr": float(result.depth_snr[index]),
                      "vetting": vet_signal(data, p, t0, duration)})
    summary = {"method": "BLS", "best": peaks[0], "peaks": peaks,
               "period_min": float(periods.min()), "period_max": float(periods.max()),
               "period_trials": len(periods), "runtime_seconds": time.monotonic() - start,
               "sde_definition": "(log-likelihood power - median power) / (1.4826 MAD power); heuristic, not a calibrated TLS SDE"}
    return summary, {"periods": result.period, "power": result.power, "score": scores}


def tls_search(data, stellar, config):
    start = time.monotonic()
    t, y, dy = (data[k] for k in ["time", "flux", "error"])
    maximum = min(config["period_max_days"], np.ptp(t) / config["minimum_observed_transits"])
    radius, mass = float(stellar["radius"]), float(stellar["mass"])
    result = transitleastsquares(t, y, dy, verbose=False).power(
        period_min=config["period_min_days"], period_max=maximum, n_transits_min=2,
        R_star=radius, R_star_min=.8 * radius, R_star_max=1.2 * radius,
        M_star=mass, M_star_min=.8 * mass, M_star_max=1.2 * mass,
        oversampling_factor=config["tls_oversampling"],
        duration_grid_step=config["tls_duration_grid_step"],
        use_threads=config["tls_threads_per_worker"], show_progress_bar=False,
        backend="fused-pl", transit_depth_min=10e-6, u=[.3, .2], verbose=False,
    )
    best = {"period": float(result.period), "epoch_btjd": float(result.T0),
            "duration_days": float(result.duration), "sde": float(result.SDE),
            # TLS depth is the normalized flux at transit bottom, so decrement = 1-depth.
            "model_depth": float(1 - result.depth), "model_depth_snr": float(result.snr),
            "vetting": vet_signal(data, float(result.period), float(result.T0), float(result.duration))}
    peaks = [{"period": float(result.periods[i]), "sde": float(result.power[i])}
             for i in peak_indices(result.periods, result.power, config["maximum_candidates_per_method"])]
    summary = {"method": "TLS", "best": best, "peaks": peaks,
               "period_min": float(np.min(result.periods)), "period_max": float(np.max(result.periods)),
               "period_trials": len(result.periods), "runtime_seconds": time.monotonic() - start,
               "backend": "fused-pl", "limb_darkening": [.3, .2],
               "limb_darkening_note": "Generic fixed quadratic coefficients, not fitted stellar atmosphere values."}
    return summary, {"periods": result.periods, "power": result.power,
                     "model_phase": result.model_folded_phase, "model_flux": result.model_folded_model}


def method_supported(method, config):
    best = method["best"]
    vet = best["vetting"]
    threshold = config["bls_min_sde"] if method["method"] == "BLS" else config["tls_min_sde"]
    return (best["sde"] >= threshold and vet["pink_snr"] >= config["minimum_depth_snr"]
            and vet["observed_transits"] >= config["minimum_observed_transits"])


def conclude(bls, tls, config):
    bp, tp = bls["best"], tls["best"]
    relation = period_relation(bp["period"], tp["period"], config["period_agreement_fraction"])
    phase_difference = abs(float(phase_offset([bp["epoch_btjd"]], tp["period"], tp["epoch_btjd"])[0]))
    epoch_agreement = phase_difference <= max(bp["duration_days"], tp["duration_days"]) / 2
    supported = [method_supported(method, config) for method in [bls, tls]]
    preferred = tls if supported[1] else bls if supported[0] else tls
    flags = sorted(set(preferred["best"]["vetting"]["flags"]))
    if all(supported) and relation == "exact" and epoch_agreement and not flags:
        label = "supported_transit_like_signal"
        reason = "Independent BLS/TLS searches agree in period and phase; repeat-event and common vetting checks pass."
    elif any(supported):
        label = "signal_requires_review"
        reason = "At least one method passes the screening thresholds; agreement or false-positive checks require review."
        if relation != "exact": flags.append("method_period_disagreement_or_harmonic")
        if not epoch_agreement: flags.append("method_phase_disagreement")
    else:
        label = "no_significant_signal_in_search_range"
        reason = "Neither leading peak passes all the fixed SDE, correlated-noise SNR and observed-event thresholds. This does not establish planet absence."
    return {"conclusion": label, "reason": reason, "preferred_method": preferred["method"],
            "period_agreement": relation, "epoch_agreement": bool(epoch_agreement),
            "phase_difference_days": phase_difference, "bls_supported": bool(supported[0]),
            "tls_supported": bool(supported[1]), "flags": flags,
            "candidate_period": preferred["best"]["period"],
            "candidate_epoch_btjd": preferred["best"]["epoch_btjd"],
            "candidate_duration_days": preferred["best"]["duration_days"]}


def evaluate_catalog(signals, planets, config):
    """Called only after every blind search is complete; never influences detection."""
    rows = []
    for planet in planets:
        known_period = planet.get("pl_orbper")
        if not planet.get("tran_flag") or not known_period or not np.isfinite(known_period):
            continue
        matches = []
        for signal in signals:
            for name in ["bls", "tls"]:
                method = signal[name]
                best = method["best"]
                relation = period_relation(best["period"], known_period, config["catalog_period_tolerance_fraction"])
                if relation:
                    supported = method_supported(method, config)
                    known_epoch = planet.get("pl_tranmid")
                    offset = None
                    if known_epoch and np.isfinite(known_epoch):
                        offset = abs(float(phase_offset([best["epoch_btjd"] + 2457000], known_period, known_epoch)[0]))
                    matches.append({"method": name.upper(), "iteration": signal["iteration"],
                                    "relation": relation, "supported": bool(supported),
                                    "period": best["period"], "epoch_offset_days": offset})
        rows.append({"planet": planet["pl_name"], "catalog_period_days": float(known_period),
                     "catalog_radius_earth": planet.get("pl_rade"),
                     "in_nominal_period_range": bool(config["period_min_days"] <= known_period <= config["period_max_days"]),
                     "matches": matches,
                     "supported_exact_period_match": any(m["supported"] and m["relation"] == "exact" for m in matches),
                     "supported_harmonic_match": any(m["supported"] and m["relation"] != "exact" for m in matches),
                     "epoch_note": "Offsets are diagnostic only: catalog epochs can drift and uncertainties/time standards were not propagated."})
    return rows

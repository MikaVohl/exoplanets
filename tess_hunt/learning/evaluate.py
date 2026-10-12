"""Discovery inference and paired injection/control tests on held-out TESS stars."""
import json
from pathlib import Path

import batman
import numpy as np
import pandas as pd

from tess_hunt.search import phase_offset
from .dataset import ROOT, acquire_star, save_json, sha256
from .labels import match_signal
from .model import load_ranker, score_views, baseline_matrix
from .photometry import read_raw, clean, make_views
from .search import find_signals


def rank_data(raw, model, checkpoint):
    config = checkpoint["config"]
    data = clean(raw, config)
    signals = find_signals(data, config)
    scores = score_views(model, checkpoint, [make_views(data, signal, config) for signal in signals])
    for signal, score in zip(signals, scores):
        signal.update({"cnn_score": float(score), "cnn_selected": bool(score >= checkpoint["threshold"])})
    return signals, data


def discover(run, files=None, tic=None, sectors=1, output=None):
    import joblib
    model, checkpoint = load_ranker(run)
    if tic is not None:
        if sectors < 1:
            raise ValueError("Sector count must be positive")
        entry = acquire_star({"tic": tic, "cohort": "discovery", "split": "external"}, sectors)
        files = [ROOT / p["path"] for p in entry["files"]]
    if not files:
        raise ValueError("No matching SPOC files available")
    catalog = pd.read_csv(run / "catalog.csv")
    baseline = joblib.load(run / "baseline.joblib")
    training_table = pd.read_csv(run / "candidates.csv")
    star_splits = dict(training_table.groupby("tic").split.first())
    rows, manifest = [], []
    for path in files:
        path = Path(path).resolve()
        raw = read_raw(path)
        signals, data = rank_data(raw, model, checkpoint)
        for signal in signals:
            rows.append({**signal, "tic": raw["tic"], "sector": raw["sector"],
                         "score_context": star_splits.get(raw["tic"], "external_star"),
                         "source_path": str(path),
                         **match_signal(signal, catalog[catalog.tid.eq(raw["tic"])], data["time"], checkpoint["config"])})
        manifest.append({"path": str(path), "sha256": sha256(path), "tic": raw["tic"], "sector": raw["sector"]})
    table = pd.DataFrame(rows)
    if len(table):
        table["baseline_score"] = baseline["model"].predict_proba(baseline_matrix(table))[:, 1]
        table["baseline_selected"] = table.baseline_score >= baseline["threshold"]
        table = table.sort_values("cnn_score", ascending=False)
    output = (output or run / "discovery_search.csv").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    save_json(output.with_suffix(".provenance.json"), {"files": manifest, "model_sha256": sha256(run / "model.pt"),
                                                     "config": checkpoint["config"],
                                                     "threshold": checkpoint["threshold"],
                                                     "note": "Unlabeled means absent/uncertain in this catalog snapshot; not a verified new planet."})
    print(f"Saved {len(table)} ranked signals to {output}", flush=True)
    if len(table):
        print(table[["tic", "sector", "period", "label", "label_reason", "cnn_score", "cnn_selected"]].to_string(index=False))
    return table


def inject_transit(raw, rng, config):
    """Exposure-integrated, limb-darkened circular transits before detrending.

    Solar host, random impact parameter. Radius ratio is drawn from a nominal
    depth range; actual depth differs with limb darkening and grazing geometry.
    PDCSAP-level injection cannot test SPOC cotrending, dilution or pixel extraction.
    """
    finite_t = raw["time"][np.isfinite(raw["time"])]
    maximum = min(config["period_max_days"], np.ptp(finite_t) / 2)
    if maximum <= config["period_min_days"]:
        raise ValueError("Insufficient baseline for an injection")
    period = float(np.exp(rng.uniform(np.log(max(.8, config["period_min_days"])), np.log(maximum))))
    nominal_depth = float(np.exp(rng.uniform(np.log(300e-6), np.log(10000e-6))))
    radius_ratio = np.sqrt(nominal_depth)
    a = 4.2083 * period ** (2 / 3)  # a/R_sun for a solar-mass star
    impact = float(rng.uniform(0, .9))
    params = batman.TransitParams()
    params.t0 = float(finite_t.min() + rng.uniform(0, period))
    params.per, params.rp, params.a = period, radius_ratio, a
    params.inc = float(np.degrees(np.arccos(impact / a)))
    params.ecc, params.w, params.u, params.limb_dark = 0., 90., [.3, .2], "quadratic"
    times = raw["time"].copy()
    times[~np.isfinite(times)] = finite_t.min()
    transit = batman.TransitModel(params, times, supersample_factor=7, exp_time=120 / 86400).light_curve(params)
    modified = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in raw.items()}
    modified["flux"] *= transit
    # Preserve the observed noise/errors rather than invent new photon noise.
    duration = period / np.pi * np.arcsin(np.sqrt((1 + radius_ratio)**2 - impact**2) /
                                         (a * np.sin(np.radians(params.inc))))
    signal = {"period": period, "t0": params.t0, "duration_days": float(duration),
              "nominal_depth_ppm": nominal_depth * 1e6, "sampled_depth_ppm": float((1 - transit.min()) * 1e6),
              "impact_parameter": impact}
    return modified, signal


def injection_matches(candidate, injected, time, config):
    """Require exact period and alignment across the data; harmonics reported separately."""
    if abs(candidate["period"] / injected["period"] - 1) >= config["period_tolerance"]:
        return False
    cycles = np.unique(np.rint((np.asarray(time) - injected["t0"]) / injected["period"]))
    centers = injected["t0"] + cycles * injected["period"]
    offsets = np.abs(phase_offset(centers, candidate["period"], candidate["t0"]))
    return bool(offsets.max() < config["epoch_tolerance_durations"] *
                max(candidate["duration_days"], injected["duration_days"]))


def injection_experiment(run, count, seed):
    model, checkpoint = load_ranker(run)
    config, rng = checkpoint["config"], np.random.default_rng(seed)
    manifest = json.loads((run / "manifest.json").read_text())
    pool = [(entry, product) for entry in manifest if entry["split"] == "test" for product in entry["files"]]
    if not pool:
        raise ValueError("No held-out test photometry for injection evaluation")
    # Background signals in real curves are not assumed false. Paired controls
    # reveal accidental matches to signals already present before an injection.
    controls, rows, control_rows = {}, [], []
    for index in range(count):
        entry, product = pool[int(rng.integers(len(pool)))]
        path = ROOT / product["path"]
        if sha256(path) != product["sha256"]:
            raise ValueError(f"Photometry changed: {path}")
        raw = read_raw(path)
        if product["path"] not in controls:
            signals, data = rank_data(raw, model, checkpoint)
            controls[product["path"]] = signals
            control_rows.extend({**s, "tic": entry["tic"], "sector": raw["sector"], "source_path": product["path"]}
                                for s in signals)
        modified, injected = inject_transit(raw, rng, config)
        candidates, data = rank_data(modified, model, checkpoint)
        matches = [s for s in candidates if injection_matches(s, injected, data["time"], config)]
        preexisting = any(injection_matches(s, injected, data["time"], config) for s in controls[product["path"]])
        best = max(matches, key=lambda s: s["cnn_score"], default=None)
        inside = np.abs(phase_offset(data["time"], injected["period"], injected["t0"])) < injected["duration_days"] / 2
        observed = len(np.unique(np.rint((data["time"][inside] - injected["t0"]) / injected["period"])))
        row = {"injection": index, "tic": entry["tic"], "sector": raw["sector"], **injected,
               "observed_events": observed, "preexisting_match": preexisting,
               "search_recovered": bool(matches), "cnn_score": best["cnn_score"] if best else np.nan,
               "pipeline_recovered": bool(best and best["cnn_selected"]),
               "alias_recovered": any(abs(s["period"] / (injected["period"] * ratio) - 1) < config["period_tolerance"]
                                      for s in candidates for ratio in (.5, 2))}
        rows.append(row)
        pd.DataFrame(rows).to_csv(run / "injections.csv", index=False)
        print(f"injection {index + 1}/{count}: {injected['nominal_depth_ppm']:.0f} ppm, "
              f"P={injected['period']:.2f}, search={bool(matches)}, selected={row['pipeline_recovered']}", flush=True)
    table = pd.DataFrame(rows)
    pd.DataFrame(control_rows).to_csv(run / "injection_controls.csv", index=False)
    eligible = table[~table.preexisting_match & table.observed_events.ge(2)]
    def summarize(subset):
        return {"injections": len(subset), "search_recovered": int(subset.search_recovered.sum()),
                "pipeline_recovered": int(subset.pipeline_recovered.sum()),
                "search_recall": float(subset.search_recovered.mean()) if len(subset) else None,
                "pipeline_recall": float(subset.pipeline_recovered.mean()) if len(subset) else None}
    report = {"count": count, "seed": seed, "held_out_stars": table.tic.nunique(),
              "eligible": summarize(eligible), "all": summarize(table),
              "excluded_preexisting_matches": int(table.preexisting_match.sum()),
              "excluded_fewer_than_two_observed_events": int(table.observed_events.lt(2).sum()),
              "by_nominal_depth_ppm": {}, "threshold": checkpoint["threshold"],
              "model_sha256": sha256(run / "model.pt"),
              "notes": ["Physical batman transits injected into real PDCSAP flux before cleaning, flattening, binning, search and scoring.",
                        "Frozen model and validation threshold; held-out test stars only; no injection training or tuning.",
                        "Solar-host circular-orbit injections with fixed limb darkening; not a population model.",
                        "Tests post-PDCSAP processing, not pixel extraction, SPOC correction, blending, or all false positives.",
                        "Unmodified controls can contain real astrophysical signals; their selected scores are not a false-alarm rate."]}
    for name, subset in eligible.groupby(pd.cut(eligible.nominal_depth_ppm, [0, 1000, 3000, np.inf],
                                                labels=["300-1000", "1000-3000", "3000-10000"]), observed=True):
        report["by_nominal_depth_ppm"][str(name)] = summarize(subset)
    save_json(run / "injection_metrics.json", report)
    print(json.dumps(report["eligible"], indent=2), flush=True)
    return table

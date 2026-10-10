"""Per-star diagnostic plots, catalog comparison tables and a local HTML report."""

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .search import phase_offset


def diagnostic_plots(root, result):
    tic = result["tic"]
    folder = root / "reports" / "benchmark"
    (folder / "plots").mkdir(exist_ok=True)
    saved = np.load(root / "data" / "processed" / "benchmark" / f"tic{tic}.npz")
    time, flux, error = (saved[k] for k in ["time", "flux", "error"])
    for signal in result["signals"]:
        number = signal["iteration"]
        spectra = np.load(folder / "spectra" / f"tic{tic}_signal{number}.npz")
        decision = signal["decision"]
        p, t0, d = (decision[k] for k in ["candidate_period", "candidate_epoch_btjd", "candidate_duration_days"])
        phase = phase_offset(time, p, t0)
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        axes[0, 0].scatter(time, flux, s=2, alpha=.45)
        inside = abs(phase) <= d / 2
        axes[0, 0].scatter(time[inside], flux[inside], s=3, color="tab:red", label="Candidate transit windows")
        axes[0, 0].set(xlabel="BTJD [TDB days]", ylabel="Detrended relative flux", title="Observed 10-minute bins")
        axes[0, 0].legend(fontsize=7)
        for method, ax in [("bls", axes[0, 1]), ("tls", axes[1, 0])]:
            ax.plot(spectra[method + "_period"], spectra[method + "_power"], lw=.6)
            best = signal[method]["best"]
            ax.axvline(best["period"], color="tab:red", lw=.7)
            ax.set(xlabel="Trial period [days]", ylabel="BLS robust score" if method == "bls" else "TLS SDE",
                   title=f"{method.upper()}: P={best['period']:.5f} d, score={best['sde']:.1f}")
        ax = axes[1, 1]
        near = abs(phase) < min(p / 2, max(.25, 2.5 * d))
        ax.scatter(phase[near] * 24, flux[near], s=3, alpha=.25)
        edges = np.linspace(-min(p / 2, max(.25, 2.5 * d)), min(p / 2, max(.25, 2.5 * d)), 65)
        centers, means, errors = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (phase >= lo) & (phase < hi)
            if mask.sum() > 0:
                weights = 1 / error[mask] ** 2
                centers.append((lo + hi) * 12)
                means.append(np.average(flux[mask], weights=weights))
                errors.append(1 / np.sqrt(weights.sum()))
        ax.errorbar(centers, means, yerr=errors, fmt=".", color="tab:orange", ms=5)
        if decision["preferred_method"] == "TLS":
            ax.plot((spectra["tls_model_phase"] - .5) * p * 24, spectra["tls_model_flux"], color="tab:red", lw=1)
        ax.set(xlim=(edges[0] * 24, edges[-1] * 24), xlabel="Time from candidate transit center [hours]",
               ylabel="Relative flux", title=f"{decision['preferred_method']} candidate: phase folded")
        for ax in axes.flat:
            ax.grid(alpha=.15)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        fig.suptitle(f"{result['hostname']} · TIC {tic} · sector {result['sector']} · signal {number}\n"
                     f"{decision['conclusion'].replace('_', ' ')}", fontsize=11)
        fig.tight_layout()
        fig.savefig(folder / "plots" / f"tic{tic}_signal{number}.png", dpi=120)
        plt.close(fig)
        mask = abs(phase) > .75 * d
        time, flux, error = (x[mask] for x in [time, flux, error])


def build_report(root):
    folder = root / "reports" / "benchmark"
    targets = pd.read_csv(root / "data" / "benchmark" / "targets.csv")
    results, summaries, candidates, recoveries = [], [], [], []
    for target in targets.to_dict("records"):
        path = folder / "results" / f"tic{target['tic']}.json"
        if not path.exists():
            summaries.append({"tic": target["tic"], "hostname": target["hostname"], "sector": target["sector"],
                              "catalog_group": target["catalog_group"], "radius_bin": target["radius_bin"],
                              "status": "not_run", "conclusion": "not_run", "host_exact_period_recovery": False})
            continue
        result = json.loads(path.read_text())
        results.append(result)
        row = {"tic": result["tic"], "hostname": result["hostname"], "sector": result["sector"],
               "catalog_group": result["catalog_group"], "radius_bin": result["radius_bin"],
               "tmag": target["tmag"], "status": result["status"], "conclusion": result["conclusion"],
               "reason": result["reason"], "host_exact_period_recovery": False, "known_planets": len(result["known_planets"])}
        if result["status"] == "ok":
            # Display the signal that supports the overall star conclusion.
            first = min(result["signals"], key=lambda s: {
                "supported_transit_like_signal": 0, "signal_requires_review": 1,
                "no_significant_signal_in_search_range": 2}[s["decision"]["conclusion"]])
            row["representative_signal_iteration"] = first["iteration"]
            for method in ["bls", "tls"]:
                best = first[method]["best"]
                row.update({method + "_period_days": best["period"], method + "_sde": best["sde"],
                            method + "_pink_snr": best["vetting"]["pink_snr"],
                            method + "_observed_transits": best["vetting"]["observed_transits"]})
            row["flags"] = ";".join(first["decision"]["flags"])
            row["search_points"] = result["preprocessing"]["search_points"]
            row["span_days"] = result["preprocessing"]["span_days"]
            row["signals_searched"] = len(result["signals"])
            row["runtime_seconds"] = sum(s[m]["runtime_seconds"] for s in result["signals"] for m in ["bls", "tls"])
            plot_paths = [folder / "plots" / f"tic{result['tic']}_signal{s['iteration']}.png" for s in result["signals"]]
            if any(not p.exists() or p.stat().st_mtime < path.stat().st_mtime for p in plot_paths):
                diagnostic_plots(root, result)
            for signal in result["signals"]:
                for method in ["bls", "tls"]:
                    best = signal[method]["best"]
                    v = best["vetting"]
                    candidates.append({"tic": result["tic"], "hostname": result["hostname"],
                                       "iteration": signal["iteration"], "method": method.upper(),
                                       "period_days": best["period"], "epoch_btjd": best["epoch_btjd"],
                                       "duration_hours": best["duration_days"] * 24, "score": best["sde"],
                                       "depth_ppm": v.get("depth_ppm"), "depth_snr": v["depth_snr"],
                                       "pink_snr": v["pink_snr"], "observed_transits": v["observed_transits"],
                                       "odd_even_sigma": v.get("odd_even_sigma"), "secondary_snr": v.get("secondary_snr"),
                                       "centroid_snr": v.get("centroid_snr"), "flags": ";".join(v["flags"]),
                                       "conclusion": signal["decision"]["conclusion"]})
            for evaluation in result["catalog_evaluation"]:
                matches = evaluation["matches"]
                recoveries.append({"tic": result["tic"], "hostname": result["hostname"],
                                   "planet": evaluation["planet"], "catalog_period_days": evaluation["catalog_period_days"],
                                   "catalog_radius_earth": evaluation["catalog_radius_earth"],
                                   "in_nominal_period_range": evaluation["in_nominal_period_range"],
                                   "exact_supported_match": evaluation["supported_exact_period_match"],
                                   "harmonic_supported_match": evaluation["supported_harmonic_match"],
                                   "bls_exact_match": any(m["method"] == "BLS" and m["supported"] and m["relation"] == "exact" for m in matches),
                                   "tls_exact_match": any(m["method"] == "TLS" and m["supported"] and m["relation"] == "exact" for m in matches),
                                   "matching_methods": ";".join(f"{m['method']}:{m['relation']}:signal{m['iteration']}" for m in matches)})
            row["host_exact_period_recovery"] = any(e["supported_exact_period_match"] for e in result["catalog_evaluation"])
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    candidate_table = pd.DataFrame(candidates)
    recovery_table = pd.DataFrame(recoveries)
    summary.to_csv(folder / "summary.csv", index=False)
    candidate_table.to_csv(folder / "candidate_signals.csv", index=False)
    recovery_table.to_csv(folder / "planet_recovery.csv", index=False)
    counts = pd.crosstab(summary.catalog_group, summary.conclusion)
    known = summary[summary.catalog_group.eq("known_transiting_host")]
    comparison = summary[summary.catalog_group.eq("no_cataloged_transiting_planet")]
    recovery_count = int(known.host_exact_period_recovery.sum())
    completed = int(summary.status.eq("ok").sum())
    notes = ["# 100-star TESS transit benchmark", "",
             f"Completed searches: {completed}/{len(targets)}. Cataloged transiting hosts with a supported exact-period match: {recovery_count}/{len(known)}.", "",
             "A recovery means a leading BLS or TLS peak in up to two blind iterations passed the fixed screening thresholds and matched a catalog period within 1%. Epoch offsets are diagnostic only. This is a period-matched benchmark, not a planet confirmation.", "",
             "| Catalog group | Conclusion | Stars |", "| --- | --- | ---: |"]
    for (group, conclusion), count in summary.groupby(["catalog_group", "conclusion"]).size().items():
        notes.append(f"| {group} | {conclusion} | {count} |")
    notes += ["", "## How to read the results", "",
              "- supported_transit_like_signal: BLS/TLS agree in period and phase, with repeated observed dips and screening checks passed.",
              "- signal_requires_review: one or both methods support a signal, but agreement or false-positive diagnostics require inspection.",
              "- no_significant_signal_in_search_range: no leading peak passed all thresholds. It does not establish absence of planets or transits.",
              "- insufficient_data/error/not_run: not an astrophysical negative result.", "",
              "## Fixed scope and limitations", "",
              "Latest available public two-minute SPOC sector for each selected star; gaps retained. Search 0.5–12 days, shortened when necessary to half the usable time span. Gap-aware one-day median detrending, 0.1-day segment edge trim, upward-only 8-sigma clipping, and ten-minute weighted bins. Up to two blind iterations, masking only the discovered leading signal. BLS and TLS search independently over the full allowed range. Catalog periods, epochs and labels are introduced only after both searches.", "",
              "Screening thresholds are heuristic, not calibrated false-alarm rates. Correlated-noise SNR is a duration-block estimate. Centroid, secondary-eclipse and odd/even diagnostics are warnings, not definitive false-positive identifications. TLS uses generic quadratic limb-darkening coefficients. Detrending/bins may attenuate short or long transits. The strongest two signals can miss additional planets. Sparse transits and catalog-period drift can cause missed matches.", "",
              "The 40 comparison stars have confirmed planets but no cataloged transiting planets. A flagged comparison signal is not automatically a false positive or a new planet. The groups differ in brightness and are selected from known planet hosts with available TESS data, so performance is not representative of a blind stellar survey. No injection-recovery sensitivity calibration or pixel-level contamination vetting is included.", "",
              "## Outputs", "",
              "summary.csv: one row per star. candidate_signals.csv: leading peaks from both methods and each iteration. planet_recovery.csv: post-search known-planet comparison. results/: full JSON with parameters, events, alternative peaks, provenance and versions. plots/: diagnostic panels. spectra/: saved numerical periodograms.", "",
              "Sources: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html), [MAST TESS](https://archive.stsci.edu/missions-and-data/tess), [Astropy BLS](https://docs.astropy.org/en/stable/timeseries/bls.html), [TLS 2.0](https://transitleastsquares.readthedocs.io/en/latest/)."]
    notes += ["", "## Measured recovery by host size group", "",
              "| Largest cataloged short-period transiting planet per host | Hosts with a supported exact-period match |", "| --- | ---: |"]
    for size in ["small", "medium", "giant"]:
        part = known[known.radius_bin.eq(size)]
        notes.append(f"| {size} | {int(part.host_exact_period_recovery.sum())}/{len(part)} |")
    notes += ["", "These are host-level, period-matched recoveries for this selected sample. They do not measure detection completeness; brightness, stellar variability, observing coverage, transit duration and planet size all differ among stars."]
    (folder / "README.md").write_text("\n".join(notes) + "\n")
    if len(recovery_table):
        nominal = recovery_table[recovery_table.in_nominal_period_range]
        metrics = {"completed_stars": completed, "selected_stars": len(targets),
                   "known_transiting_hosts": len(known), "comparison_hosts": len(comparison),
                   "known_hosts_with_exact_supported_period_match": recovery_count,
                   "catalog_planets_in_nominal_range": len(nominal),
                   "catalog_planets_with_exact_supported_match": int(nominal.exact_supported_match.sum()),
                   "bls_planets_with_exact_supported_match": int(nominal.bls_exact_match.sum()),
                   "tls_planets_with_exact_supported_match": int(nominal.tls_exact_match.sum()),
                   "conclusion_counts": summary.conclusion.value_counts().to_dict()}
    else:
        metrics = {"completed_stars": completed, "selected_stars": len(targets)}
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # Compact local report: filterable table, per-star plots and full-result links.
    headers = ["Star", "TIC", "Sector", "Catalog group", "Catalog planets", "Conclusion", "Candidate BLS period", "Candidate TLS period", "Known-period match", "Details"]
    rows = []
    lookup = {r["tic"]: r for r in results}
    for row in summaries:
        tic = row["tic"]
        details = f'<a href="results/tic{tic}.json">JSON</a>'
        for signal in lookup.get(tic, {}).get("signals", []):
            n = signal["iteration"]
            details += f' · <a href="plots/tic{tic}_signal{n}.png">signal {n}</a>'
        catalog_items = []
        for planet in lookup.get(tic, {}).get("known_planets", []):
            period = planet.get("pl_orbper")
            period_text = f"{period:.4f} days" if period is not None else "period unavailable"
            label = "cataloged transit" if planet.get("tran_flag") else "no cataloged transit"
            catalog_items.append(f'<li>{html.escape(planet["pl_name"])}: {period_text}; {label}</li>')
        catalog_cell = f'<details><summary>{len(catalog_items)} planet(s)</summary><ul>{"".join(catalog_items)}</ul></details>'
        values = [html.escape(str(row["hostname"])), str(tic), str(row["sector"]),
                  html.escape(row["catalog_group"]), catalog_cell, html.escape(row["conclusion"]),
                  f"{row.get('bls_period_days', float('nan')):.5f}",
                  f"{row.get('tls_period_days', float('nan')):.5f}",
                  "yes" if row["host_exact_period_recovery"] else "no", details]
        rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>")
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>TESS benchmark</title>
<style>body{{font:15px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#172b45;background:#f8fafc}}h1{{font-size:30px}}table{{border-collapse:collapse;background:white;width:100%;font-size:12px}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #dce3ed}}th{{background:#e9eff7;position:sticky;top:0}}input{{padding:10px;width:340px;margin:18px 0}}a{{color:#1760ad}}.stat{{display:inline-block;background:white;border:1px solid #dce3ed;border-radius:8px;padding:16px;margin:5px}}p{{max-width:1050px;line-height:1.6}}</style>
<h1>TESS · 100-star transit benchmark</h1><p>Independent BLS and TLS searches of individual stars. Catalog information is used for sample selection and comparison after the searches. Conclusions are provisional signal-screening results.</p>
<div class="stat"><b>{completed}/{len(targets)}</b> stars searched</div><div class="stat"><b>{recovery_count}/{len(known)}</b> known transiting hosts with a supported exact-period match</div>
<div class="stat"><b>{int(summary.conclusion.eq('supported_transit_like_signal').sum())}</b> supported transit-like</div><div class="stat"><b>{int(summary.conclusion.eq('signal_requires_review').sum())}</b> require review</div><div class="stat"><b>{int(summary.conclusion.eq('no_significant_signal_in_search_range').sum())}</b> no significant signal</div>
<p>One sector per star · 0.5–12 day search · ten-minute bins · up to two signals.<br>Comparison stars have confirmed planets with no cataloged transits; they are not proven transit-free. A nondetection does not establish planet absence.</p>
<p><a href="README.md">Method and limitations</a> · <a href="summary.csv">Star summary CSV</a> · <a href="candidate_signals.csv">Signal CSV</a> · <a href="planet_recovery.csv">Planet comparison CSV</a></p>
<input id="filter" placeholder="Filter by star, group, sector or conclusion" aria-label="Filter stars">
<table><thead><tr>{''.join('<th>'+h+'</th>' for h in headers)}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<script>document.getElementById('filter').addEventListener('input',function(){{const q=this.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));}});</script></html>'''
    (folder / "index.html").write_text(content)
    print(f"Report: {folder / 'index.html'}; completed {completed}/{len(targets)}; known-host period matches {recovery_count}/{len(known)}", flush=True)
    return metrics

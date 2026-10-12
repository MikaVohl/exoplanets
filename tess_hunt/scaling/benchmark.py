"""Synchronized CPU/MPS benchmarks on real model inputs and optimizer steps."""
import gc
import os
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from tess_hunt.learning.dataset import save_json, sha256, versions
from tess_hunt.learning.model import SignalNet, VIEW_KEYS


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def benchmark_devices(source, output, widths=(8, 16, 32, 64), batches=(32, 128), steps=20, repeats=3):
    output.mkdir(parents=True, exist_ok=True)
    if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError("Disable PYTORCH_ENABLE_MPS_FALLBACK for a native Metal benchmark")
    torch.set_num_threads(4)
    df = pd.read_csv(source / "candidates.csv")
    with np.load(source / "views.npz") as loaded:
        arrays = {k: loaded[k] for k in VIEW_KEYS}
    rows = np.flatnonzero(df.y.ge(0) & df.split.eq("train"))
    mean = arrays["scalar"][rows].mean(0)
    std = np.maximum(arrays["scalar"][rows].std(0), .1)
    arrays["scalar"] = (arrays["scalar"] - mean) / std
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    results = []
    for width in widths:
        for batch_size in batches:
            index = rows[np.arange(batch_size) % len(rows)]
            host = {k: torch.tensor(arrays[k][index], dtype=torch.float32) for k in VIEW_KEYS}
            target = torch.tensor(df.y.values[index], dtype=torch.float32)
            torch.manual_seed(917)
            initial = SignalNet(width).state_dict()
            reference = SignalNet(width).eval()
            reference.load_state_dict(initial)
            with torch.no_grad():
                reference_logits = reference(host)
            reference_loss = nn.functional.binary_cross_entropy_with_logits(reference(host), target)
            reference_loss.backward()
            reference_gradients = {name: p.grad.detach().clone() for name, p in reference.named_parameters()}
            for device in devices:
                torch.use_deterministic_algorithms(device == "cpu")
                model = SignalNet(width).to(device)
                model.load_state_dict(initial)
                batch = {k: v.to(device) for k, v in host.items()}
                y = target.to(device)
                model.eval()
                with torch.no_grad():
                    logits = model(batch).cpu()
                max_error = float((logits - reference_logits).abs().max())
                # Disable dropout for numerical gradient parity, then time training mode.
                loss_check = nn.functional.binary_cross_entropy_with_logits(model(batch), y)
                loss_check.backward()
                squared_difference, squared_reference, max_gradient_error = 0., 0., 0.
                for name, parameter in model.named_parameters():
                    difference = parameter.grad.detach().cpu() - reference_gradients[name]
                    squared_difference += float(difference.square().sum())
                    squared_reference += float(reference_gradients[name].square().sum())
                    max_gradient_error = max(max_gradient_error, float(difference.abs().max()))
                gradient_relative_error = float(np.sqrt(squared_difference / max(squared_reference, 1e-30)))
                model.train()
                torch.manual_seed(33)
                loss_fn = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
                def step(transfer=False):
                    inputs = {k: v.to(device) for k, v in host.items()} if transfer else batch
                    labels = target.to(device) if transfer else y
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(inputs), labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    return loss
                for _ in range(5):
                    step()
                resident, transfer = [], []
                for copy_inputs, timings in ((False, resident), (True, transfer)):
                    for _ in range(repeats):
                        synchronize(device)
                        start = time.perf_counter()
                        for _ in range(steps):
                            step(copy_inputs)
                        synchronize(device)
                        timings.append((time.perf_counter() - start) / steps)
                memory = int(torch.mps.current_allocated_memory()) if device == "mps" else None
                row = {"width": width, "parameters": sum(p.numel() for p in model.parameters()),
                       "batch_size": batch_size, "device": device, "cpu_threads": 4,
                       "resident_step_seconds": float(np.median(resident)),
                       "transfer_step_seconds": float(np.median(transfer)),
                       "resident_trials": resident, "transfer_trials": transfer,
                       "eval_max_abs_logit_difference_from_cpu": max_error,
                       "eval_max_abs_gradient_difference_from_cpu": max_gradient_error,
                       "eval_gradient_relative_l2_difference_from_cpu": gradient_relative_error,
                       "eval_abs_loss_difference_from_cpu": abs(float(loss_check.detach().cpu()) - float(reference_loss.detach())),
                       "mps_allocated_bytes": memory}
                results.append(row)
                print(f"benchmark width={width}, batch={batch_size}, {device}: "
                      f"resident={np.median(resident)*1000:.1f} ms, transfer={np.median(transfer)*1000:.1f} ms, "
                      f"logit difference={max_error:.2g}", flush=True)
                save_json(output / "device_benchmark.json", {"results": results})
                del model, optimizer, batch, y
                gc.collect()
                if device == "mps":
                    torch.mps.empty_cache()
    machine = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    report = {"machine": machine, "platform": platform.platform(), "versions": versions(),
              "mps_available": torch.backends.mps.is_available(), "mps_built": torch.backends.mps.is_built(),
              "steps_per_trial": steps, "trials": repeats, "warmup_steps": 5,
              "candidate_sha256": sha256(source / "candidates.csv"), "results": results,
              "backend_environment": {k: v for k, v in os.environ.items()
                                      if k.startswith("PYTORCH_MPS_") or k == "PYTORCH_ENABLE_MPS_FALLBACK"},
              "notes": ["Median synchronized forward/backward/clipped-gradient/AdamW step; five warmup steps excluded.",
                        "Resident means inputs already on device; transfer includes moving each batch to device.",
                        "Real training photometry inputs; repeating a batch here benchmarks compute, not learning.",
                        "CPU uses four threads; this is the configured CPU workload, not an exhaustive CPU tuning sweep.",
                        "MPS CPU fallback is not enabled. Logit, loss and gradient agreement checked with identical weights/input and dropout disabled."]}
    save_json(output / "device_benchmark.json", report)
    pd.DataFrame([{k: v for k, v in row.items() if not k.endswith("_trials")} for row in results]).to_csv(output / "device_benchmark.csv", index=False)
    return report


def cpu_thread_sweep(source, output, widths=(16, 64), steps=12):
    """Check that a poor CPU thread setting does not explain an apparent GPU win."""
    df = pd.read_csv(source / "candidates.csv")
    torch.use_deterministic_algorithms(True)
    rows = np.flatnonzero(df.y.ge(0) & df.split.eq("train"))[:32]
    with np.load(source / "views.npz") as data:
        batch = {k: torch.tensor(data[k][rows], dtype=torch.float32) for k in VIEW_KEYS}
    mean, std = batch["scalar"].mean(0), batch["scalar"].std(0).clamp(min=.1)
    batch["scalar"] = (batch["scalar"] - mean) / std
    y = torch.tensor(df.y.values[rows], dtype=torch.float32)
    results = []
    for width in widths:
        for threads in (1, 2, 4, 8):
            torch.set_num_threads(threads)
            torch.manual_seed(917)
            model = SignalNet(width)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            def step():
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(batch), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            for _ in range(5):
                step()
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                for _ in range(steps):
                    step()
                timings.append((time.perf_counter() - start) / steps)
            row = {"width": width, "batch_size": 32, "cpu_threads": threads,
                   "step_seconds": float(np.median(timings))}
            results.append(row)
            print(f"CPU sweep width={width}, threads={threads}: {row['step_seconds']*1000:.1f}ms", flush=True)
            del model, optimizer
    torch.set_num_threads(4)
    pd.DataFrame(results).to_csv(output / "cpu_thread_sweep.csv", index=False)
    return results


def epoch_benchmark(source, output, widths=(16, 64), epochs=12):
    """Fixed epochs on identical data, including validation/checkpoint overhead."""
    from .experiment import train_cell
    from tess_hunt.learning.dataset import load_config
    df = pd.read_csv(source / "candidates.csv")
    with np.load(source / "views.npz") as data:
        arrays = {k: data[k] for k in VIEW_KEYS}
    train_rows = np.flatnonzero(df.y.ge(0) & df.split.eq("train"))
    validation_rows = np.flatnonzero(df.y.ge(0) & df.split.eq("validation"))
    config = load_config(source / "config.json")
    sweep_path = output / "cpu_thread_sweep.csv"
    sweep = pd.read_csv(sweep_path) if sweep_path.exists() else None
    results = []
    for width in widths:
        best_threads = 4
        if sweep is not None:
            candidates = sweep[sweep.width.eq(width)]
            best_threads = int(candidates.loc[candidates.step_seconds.idxmin(), "cpu_threads"])
        for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
            settings = {**config, "torch_threads": best_threads if device == "cpu" else 4}
            result, _, _, _ = train_cell(df, arrays, train_rows, validation_rows, width, 917,
                                         epochs, epochs + 1, device, settings)
            result["cpu_threads"] = settings["torch_threads"]
            results.append({key: value for key, value in result.items() if key != "train_candidate_ids"})
            print(f"full-loop width={width}, {device}, threads={settings['torch_threads']}: "
                  f"{epochs} epochs in {result['fit_seconds']:.2f}s", flush=True)
    pd.DataFrame(results).to_csv(output / "epoch_benchmark.csv", index=False)
    torch.set_num_threads(4)
    return results

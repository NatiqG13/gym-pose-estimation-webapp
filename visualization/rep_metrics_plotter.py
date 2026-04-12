"""
rep_metrics_plotter.py
By: Natiq Ghafoor

Per-rep charts and summary metrics.

Outputs (PNG):
- rep_rom.png        ROM per rep (pass=green, fail=red)
- rep_duration.png   duration per rep (pass=green, fail=red)
- rep_outcomes.png   pass vs fail counts

Outputs (JSON):
- rep_metrics.json   summary statistics used for README bullets

Expected per-rep fields:
- rep_index (int)
- rom (float)
- duration (float)
- label ("pass" / "fail")
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _safe_float(v: Any, default: float = 0.0) -> float:
    """
    Best-effort float conversion used by plotting utilities.
    """
    try:
        return float(v)
    except Exception:
        return default


def plot_rep_metrics(
    rep_feature_list: List[Dict[str, Any]],
    output_dir: str,
    rom_plot_name: str = "rep_rom.png",
    duration_plot_name: str = "rep_duration.png",
    outcomes_plot_name: str = "rep_outcomes.png",
    metrics_json_name: str = "rep_metrics.json",
) -> Dict[str, Any]:
    """
    Saves rep-level charts and a compact JSON summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not rep_feature_list:
        raise ValueError("rep_feature_list is empty")

    reps = sorted(rep_feature_list, key=lambda r: int(r.get("rep_index", 0)))

    x = [int(r.get("rep_index", 0)) for r in reps]
    rom = [_safe_float(r.get("rom", 0.0)) for r in reps]
    dur = [_safe_float(r.get("duration", 0.0)) for r in reps]
    labels = [str(r.get("label", "fail")).lower() for r in reps]
    colors = ["green" if lab == "pass" else "red" for lab in labels]

    pass_count = sum(1 for lab in labels if lab == "pass")
    fail_count = len(labels) - pass_count

    # ROM chart
    plt.figure(figsize=(10, 4))
    plt.bar(x, rom, color=colors)
    plt.title("ROM per Rep")
    plt.xlabel("Rep #")
    plt.ylabel("ROM (deg)")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    rom_path = os.path.join(output_dir, rom_plot_name)
    plt.savefig(rom_path, dpi=150)
    plt.close()

    # Duration chart
    plt.figure(figsize=(10, 4))
    plt.bar(x, dur, color=colors)
    plt.title("Duration per Rep")
    plt.xlabel("Rep #")
    plt.ylabel("Duration (s)")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    dur_path = os.path.join(output_dir, duration_plot_name)
    plt.savefig(dur_path, dpi=150)
    plt.close()

    # Outcomes chart
    plt.figure(figsize=(6, 4))
    plt.bar(["pass", "fail"], [pass_count, fail_count])
    plt.title("Rep Outcomes")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    out_path = os.path.join(output_dir, outcomes_plot_name)
    plt.savefig(out_path, dpi=150)
    plt.close()

    metrics = {
        "rep_count": len(reps),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "avg_rom": sum(rom) / len(rom) if rom else 0.0,
        "avg_duration": sum(dur) / len(dur) if dur else 0.0,
        "max_rom": max(rom) if rom else 0.0,
        "min_rom": min(rom) if rom else 0.0,
        "max_duration": max(dur) if dur else 0.0,
        "min_duration": min(dur) if dur else 0.0,
    }

    metrics_path = os.path.join(output_dir, metrics_json_name)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "rep_rom_plot": rom_path,
        "rep_duration_plot": dur_path,
        "rep_outcomes_plot": out_path,
        "rep_metrics_json": metrics_path,
        "metrics": metrics,
    }
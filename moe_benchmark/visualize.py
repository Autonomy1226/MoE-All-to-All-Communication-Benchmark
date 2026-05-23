"""Visualization utilities for MoE communication benchmarks.

Generates comparison charts and summary reports from benchmark results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def plot_benchmark_results(
    results: list[dict[str, Any]],
    output_dir: str = "results",
    save: bool = True,
) -> None:
    """Generate comparison charts from benchmark result dicts.

    Args:
        results: list of dicts, each with keys:
            - strategy
            - avg_total_ms
            - avg_sent_bytes
            - tokens_per_second
            - comm_ratio (optional)
            - config dict
        output_dir: where to save figures.
        save: if True, write PNG files; always writes a summary JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping chart generation.")
        _save_summary_json(results, output_dir)
        return

    strategies = [r["strategy"] for r in results]
    avg_times = [r.get("avg_total_ms", 0) for r in results]
    tokens_p_s = [r.get("tokens_per_second", 0) for r in results]
    sent_bytes = [r.get("avg_sent_bytes", 0) for r in results]

    colors = ["#5B9BD5", "#ED7D31", "#70AD47"]
    bar_colors = [colors[i % len(colors)] for i in range(len(strategies))]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Latency per strategy
    ax = axes[0]
    bars = ax.bar(strategies, avg_times, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Average Latency per MoE Forward Pass", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Dispatch Strategy")
    for bar, val in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(avg_times) * 0.02,
                f"{val:.2f}", ha="center", fontsize=9)

    # 2. Throughput
    ax = axes[1]
    bars = ax.bar(strategies, tokens_p_s, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Token Throughput", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tokens / second")
    ax.set_xlabel("Dispatch Strategy")
    for bar, val in zip(bars, tokens_p_s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(tokens_p_s) * 0.02,
                f"{val:.1f}", ha="center", fontsize=9)

    # 3. Communication volume
    ax = axes[2]
    useful = [r.get("avg_useful_bytes", s) for r, s in zip(results, sent_bytes)]
    padding = [s - u for s, u in zip(sent_bytes, useful)]

    x = np.arange(len(strategies))
    width = 0.35
    ax.bar(x, useful, width, label="Useful Data", color="#70AD47", edgecolor="white")
    ax.bar(x, padding, width, bottom=useful, label="Padding Overhead", color="#FF6B6B", edgecolor="white")
    ax.set_title("Communication Volume Breakdown", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bytes per Iteration")
    ax.set_xlabel("Dispatch Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend(fontsize=8)

    fig.suptitle("MoE All-to-All Communication Benchmark", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(output_dir, "benchmark_comparison.png"), dpi=150, bbox_inches="tight")
        print(f"[visualize] saved chart → {output_dir}/benchmark_comparison.png")

    plt.close(fig)

    # --- Latency histogram overlay ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, r in enumerate(results):
        times = r.get("iteration_times_ms", [])
        if times:
            ax2.hist(times, bins=30, alpha=0.5, label=r["strategy"], color=colors[i % len(colors)])
    ax2.set_title("Iteration Latency Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Count")
    ax2.legend()

    if save:
        fig2.savefig(os.path.join(output_dir, "latency_distribution.png"), dpi=150, bbox_inches="tight")
        print(f"[visualize] saved chart → {output_dir}/latency_distribution.png")

    plt.close(fig2)

    _save_summary_json(results, output_dir)


def _save_summary_json(results: list[dict], output_dir: str) -> None:
    path = os.path.join(output_dir, "benchmark_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"[visualize] saved summary → {path}")

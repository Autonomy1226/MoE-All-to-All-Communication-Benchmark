#!/usr/bin/env python3
"""Main entry point for MoE All-to-All communication benchmarks.

Usage:
    # Single benchmark
    python scripts/run_benchmark.py --strategy bucketed --tokens 1024 --hidden 1024

    # Compare all three strategies
    python scripts/run_benchmark.py --compare-all --tokens 1024 --hidden 1024

    # Sweep over token counts
    python scripts/run_benchmark.py --sweep-tokens --hidden 1024

    # Full experiment suite
    python scripts/run_benchmark.py --run-all
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from moe_benchmark.benchmark import BenchmarkConfig, BenchmarkResult, Strategy
from moe_benchmark.moe_layer import MoELayer
from moe_benchmark.dispatch_strategies import get_dispatch_fn
from moe_benchmark.visualize import plot_benchmark_results


# ---------------------------------------------------------------------------
# Single-process worker (spawned per rank)
# ---------------------------------------------------------------------------

def _worker(rank: int, world_size: int, config: BenchmarkConfig, result_queue: mp.Queue):
    """Worker process that runs the benchmark on one rank."""
    backend = config.backend
    if backend == "nccl" and not torch.cuda.is_available():
        print(f"[rank {rank}] NCCL not available, falling back to gloo")
        backend = "gloo"

    # Use file-based rendezvous to avoid Windows hostname resolution
    # issues (e.g. DNS suffix appending wustat.windows.com).
    init_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", f".rendezvous_{config.master_port}")
    )
    init_uri = f"file:///{init_file.replace(os.sep, '/')}"

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=init_uri,
    )

    device = torch.device("cuda" if backend == "nccl" else "cpu")

    moe = MoELayer(
        hidden_dim=config.hidden_dim,
        num_experts=config.num_experts,
        top_k=config.top_k,
        world_size=world_size,
    ).to(device)

    dispatch_fn = get_dispatch_fn(config.strategy.value)

    # --- Warmup ---
    for _ in range(config.num_warmup):
        x = torch.randn(config.num_tokens_per_rank, config.hidden_dim, device=device)
        _ = moe(x, dispatch_fn, rank=rank)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    # --- Benchmark iterations ---
    iteration_times_ms = []
    for _ in range(config.num_iterations):
        x = torch.randn(config.num_tokens_per_rank, config.hidden_dim, device=device)

        t0 = time.perf_counter()
        _ = moe(x, dispatch_fn, rank=rank)

        if device.type == "cuda":
            torch.cuda.synchronize()
        dist.barrier()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        iteration_times_ms.append(elapsed_ms)

    # --- Gather timings from all ranks ---
    local_t = torch.tensor(iteration_times_ms, dtype=torch.float64)
    all_t_list = [torch.zeros_like(local_t) for _ in range(world_size)]
    dist.all_gather(all_t_list, local_t)

    if rank == 0:
        all_times = torch.cat(all_t_list).tolist()
        t_tensor = torch.tensor(all_times)
        avg_ms = t_tensor.mean().item()
        std_ms = t_tensor.std().item()
        p50 = t_tensor.quantile(0.5).item()
        p99 = t_tensor.quantile(0.99).item()
        throughput = (config.num_tokens_per_rank * world_size * config.num_iterations) / (sum(all_times) / 1000)

        result = {
            "strategy": config.strategy.value,
            "hidden_dim": config.hidden_dim,
            "num_experts": config.num_experts,
            "num_tokens_per_rank": config.num_tokens_per_rank,
            "world_size": world_size,
            "backend": backend,
            "avg_total_ms": avg_ms,
            "std_total_ms": std_ms,
            "p50_total_ms": p50,
            "p99_total_ms": p99,
            "tokens_per_second": throughput,
            "iteration_times_ms": all_times,
        }
        result_queue.put(result)

    dist.barrier()
    dist.destroy_process_group()

    # Clean up the rendezvous file (only rank 0)
    if rank == 0:
        try:
            os.remove(init_file)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# High-level runner (orchestrates multiprocessing)
# ---------------------------------------------------------------------------

def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run a single benchmark config across multiple processes."""
    world_size = config.world_size

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(target=_worker, args=(rank, world_size, config, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if not result_queue.empty():
        return result_queue.get()
    else:
        raise RuntimeError("Benchmark failed: no result from rank 0")


_port_counter = 29500


def _next_port() -> int:
    global _port_counter
    p = _port_counter
    _port_counter += 1
    return p


def _make_config(args) -> BenchmarkConfig:
    return BenchmarkConfig(
        hidden_dim=args.hidden,
        num_experts=args.experts,
        top_k=args.top_k,
        num_tokens_per_rank=args.tokens,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        strategy=Strategy(args.strategy) if isinstance(args.strategy, str) else args.strategy,
        pipelined_chunks=args.chunks,
        backend=args.backend,
        world_size=args.world_size,
        master_port=_next_port(),
    )


def run_compare_all(args) -> list[dict]:
    """Run all three strategies with the same parameters."""
    results = []
    for strategy in [Strategy.NAIVE, Strategy.BUCKETED, Strategy.PIPELINED]:
        print(f"\n{'='*60}")
        print(f"  Running strategy: {strategy.value}")
        print(f"{'='*60}")
        args.strategy = strategy
        config = _make_config(args)
        result = run_benchmark(config)
        results.append(result)
        print(f"  avg={result['avg_total_ms']:.2f}ms  p50={result['p50_total_ms']:.2f}ms  "
              f"p99={result['p99_total_ms']:.2f}ms  throughput={result['tokens_per_second']:.0f} tok/s")

    return results


def run_sweep_tokens(args) -> list[dict]:
    """Run benchmark sweeping over different token counts."""
    all_results = []
    token_counts = [256, 512, 1024, 2048, 4096]

    for nt in token_counts:
        args.tokens = nt
        print(f"\n--- Token count: {nt} ---")
        results = run_compare_all(args)
        all_results.extend(results)

    return all_results


def run_sweep_hidden(args) -> list[dict]:
    """Run benchmark sweeping over different hidden dimensions."""
    all_results = []
    hidden_dims = [512, 1024, 2048, 4096]

    for hd in hidden_dims:
        args.hidden = hd
        args.tokens = max(256, 4096 * 1024 // hd)  # scale tokens to keep memory roughly constant
        print(f"\n--- Hidden dim: {hd}, tokens: {args.tokens} ---")
        results = run_compare_all(args)
        all_results.extend(results)

    return all_results


def run_full_suite(args) -> list[dict]:
    """Run the complete experiment suite."""
    all_results = []

    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Varying token count (hidden_dim=1024)")
    print("=" * 70)
    args.hidden = 1024
    all_results.extend(run_sweep_tokens(args))

    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Varying hidden dimension (tokens auto-scaled)")
    print("=" * 70)
    all_results.extend(run_sweep_hidden(args))

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MoE All-to-All Communication Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model parameters
    parser.add_argument("--hidden", type=int, default=1024, help="Hidden dimension (default: 1024)")
    parser.add_argument("--experts", type=int, default=8, help="Total number of experts (default: 8)")
    parser.add_argument("--top-k", type=int, default=1, help="Top-K routing (default: 1)")

    # Workload
    parser.add_argument("--tokens", type=int, default=1024, help="Tokens per rank (default: 1024)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations (default: 50)")

    # Strategy
    parser.add_argument("--strategy", type=str, default="bucketed",
                        choices=["naive", "bucketed", "pipelined"],
                        help="Dispatch strategy (default: bucketed)")
    parser.add_argument("--chunks", type=int, default=4, help="Pipeline chunks (default: 4)")

    # Backend
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"],
                        help="torch.distributed backend (default: gloo)")
    parser.add_argument("--world-size", type=int, default=4, help="Number of processes (default: 4)")

    # Output
    parser.add_argument("--output", type=str, default="results", help="Output directory (default: results)")

    # Modes
    parser.add_argument("--compare-all", action="store_true",
                        help="Run all three strategies and compare")
    parser.add_argument("--sweep-tokens", action="store_true",
                        help="Sweep over token counts")
    parser.add_argument("--sweep-hidden", action="store_true",
                        help="Sweep over hidden dimensions")
    parser.add_argument("--run-all", action="store_true",
                        help="Run full experiment suite")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.run_all:
        results = run_full_suite(args)
    elif args.sweep_tokens:
        results = run_sweep_tokens(args)
    elif args.sweep_hidden:
        results = run_sweep_hidden(args)
    elif args.compare_all:
        results = run_compare_all(args)
    else:
        # Single benchmark
        config = _make_config(args)
        results = [run_benchmark(config)]

    # Save and visualize
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Save raw results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[✓] Results saved → {result_path}")

    # Also save as latest
    latest_path = os.path.join(output_dir, "results_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Visualize
    if results:
        plot_benchmark_results(results, output_dir=output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  [{r['strategy']:12s}]  avg={r['avg_total_ms']:8.2f}ms  "
              f"p50={r['p50_total_ms']:8.2f}ms  "
              f"throughput={r['tokens_per_second']:10.0f} tok/s")


if __name__ == "__main__":
    mp.freeze_support()
    main()

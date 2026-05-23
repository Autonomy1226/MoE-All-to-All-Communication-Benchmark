"""Benchmark runner for MoE communication strategies.

Orchestrates distributed benchmark runs across multiple processes,
collects timing and communication metrics, and saves results.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable

import torch
import torch.distributed as dist

from .moe_layer import MoELayer
from .dispatch_strategies import get_dispatch_fn


class Strategy(str, Enum):
    NAIVE = "naive"
    BUCKETED = "bucketed"
    PIPELINED = "pipelined"


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    # Model parameters
    hidden_dim: int = 1024
    num_experts: int = 8
    top_k: int = 1

    # Workload parameters
    num_tokens_per_rank: int = 1024
    num_warmup: int = 5
    num_iterations: int = 50

    # Dispatch parameters
    strategy: Strategy = Strategy.BUCKETED
    pipelined_chunks: int = 4

    # Backend
    backend: str = "gloo"  # "gloo" or "nccl"
    world_size: int = 4
    master_port: int = 29500


@dataclass
class BenchmarkResult:
    """Aggregated results from a benchmark run."""

    config: BenchmarkConfig
    strategy: str

    # Timing (seconds)
    avg_total_ms: float = 0.0
    std_total_ms: float = 0.0
    p50_total_ms: float = 0.0
    p99_total_ms: float = 0.0

    # Communication breakdown (from dispatch info)
    avg_sent_bytes: float = 0.0
    avg_useful_bytes: float = 0.0
    avg_padding_overhead: float = 0.0

    # Throughput
    tokens_per_second: float = 0.0
    effective_bandwidth_gbps: float = 0.0

    # Per-iteration timings (for trace export)
    iteration_times_ms: list[float] = field(default_factory=list)

    # Extra
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Runs distributed MoE communication benchmarks."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.rank = 0
        self.world_size = config.world_size
        self._initialized = False

    def init_distributed(self):
        """Initialize the distributed process group."""
        if self._initialized:
            return

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")

        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.backend,
                rank=self.rank,
                world_size=self.world_size,
            )
        else:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self._initialized = True

    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return aggregated results."""
        self.init_distributed()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        config = self.config
        device = torch.device("cuda" if torch.cuda.is_available() and config.backend == "nccl" else "cpu")

        # Build MoE layer
        moe = MoELayer(
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            world_size=self.world_size,
        ).to(device)

        dispatch_fn = get_dispatch_fn(config.strategy.value)

        # Warmup
        for _ in range(config.num_warmup):
            x = torch.randn(config.num_tokens_per_rank, config.hidden_dim, device=device)
            _ = moe(x, dispatch_fn, rank=self.rank)

        if self.rank == 0 and device.type == "cuda":
            torch.cuda.synchronize()
        dist.barrier()

        # Benchmark iterations
        iteration_times_ms = []
        all_dispatch_info = []

        for i in range(config.num_iterations):
            x = torch.randn(config.num_tokens_per_rank, config.hidden_dim, device=device)

            t0 = time.perf_counter()
            # The dispatch info dict is returned inside the MoE forward;
            # we wrap it to capture comm stats.
            # Note: our current MoE.forward returns output; we need to
            # collect dispatch info from intermediate calls.

            _ = moe(x, dispatch_fn, rank=self.rank)

            if device.type == "cuda":
                torch.cuda.synchronize()
            dist.barrier()

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            iteration_times_ms.append(elapsed_ms)

        # Collect communication stats from all ranks
        # (simplified: rank 0 gathers timings; comm stats are rank-local estimates)
        local_t = torch.tensor(iteration_times_ms, dtype=torch.float64)
        all_t_list = [torch.zeros_like(local_t) for _ in range(self.world_size)]
        dist.all_gather(all_t_list, local_t)

        # For the result, use rank 0's timings + cross-rank stats
        all_times = torch.cat(all_t_list).tolist() if self.rank == 0 else []
        if self.rank == 0:
            t_tensor = torch.tensor(all_times)
            avg_ms = t_tensor.mean().item()
            std_ms = t_tensor.std().item()
            p50 = t_tensor.quantile(0.5).item()
            p99 = t_tensor.quantile(0.99).item()
        else:
            avg_ms = std_ms = p50 = p99 = 0.0

        # Broadcast result metadata from rank 0
        if self.rank == 0:
            result = BenchmarkResult(
                config=config,
                strategy=config.strategy.value,
                avg_total_ms=avg_ms,
                std_total_ms=std_ms,
                p50_total_ms=p50,
                p99_total_ms=p99,
                avg_sent_bytes=config.num_tokens_per_rank * config.hidden_dim * 4 * self.world_size,
                avg_useful_bytes=config.num_tokens_per_rank * config.hidden_dim * 4,
                avg_padding_overhead=(self.world_size - 1) / self.world_size if config.strategy == Strategy.NAIVE else 0.0,
                tokens_per_second=(config.num_tokens_per_rank * self.world_size) / (avg_ms / 1000) if avg_ms > 0 else 0,
                effective_bandwidth_gbps=0.0,  # computed below
                iteration_times_ms=all_times,
            )
            result.effective_bandwidth_gbps = (
                (result.avg_sent_bytes / (avg_ms / 1000)) / 1e9 if avg_ms > 0 else 0
            )
        else:
            result = BenchmarkResult(config=config, strategy="")

        return result

    def cleanup(self):
        if self._initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False


def run_single_rank(
    rank: int,
    world_size: int,
    config: BenchmarkConfig,
    return_dict: dict,
):
    """Entry point for a single spawned process."""
    runner = BenchmarkRunner(config)
    runner.rank = rank
    runner.world_size = world_size

    try:
        result = runner.run()
        if rank == 0:
            return_dict["result"] = result
    finally:
        runner.cleanup()

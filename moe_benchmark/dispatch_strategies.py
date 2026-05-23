"""Three All-to-All dispatch strategies for MoE token routing.

Strategy 1 — Naive All-to-All:
  Pad all tensors to uniform max-size, then one-shot all_to_all.
  Simple but wastes bandwidth on padding.

Strategy 2 — Bucketed All-to-All:
  Group tokens by target rank, exchange only real data.
  Uses all_to_all_single with per-rank split sizes.

Strategy 3 — Pipelined (DeepEP-style):
  Split local tokens into micro-batches; while one chunk is being
  computed by experts, the next chunk is in-flight over the network.
  Models computation/communication overlap.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Strategy 1: Naive — pad to uniform, single all_to_all
# ---------------------------------------------------------------------------

def naive_dispatch(
    *,
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    weights: torch.Tensor | None,
    num_experts: int,
    experts_per_rank: int,
    world_size: int,
    rank: int,
    combine: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Naive dispatch: pad all rank-slices to max size, exchange full tensors.

    Communication volume (per rank):  hidden_dim * num_tokens_local * world_size bytes sent + received.
    Actual useful data is typically 1/world_size of that.
    """
    T, D = tokens.shape
    dtype, device = tokens.dtype, tokens.device

    # Determine target rank for each token
    target_rank = _target_rank(expert_indices, experts_per_rank)  # [T] or [T, K]

    # Pad to uniform max size per rank bucket
    max_per_rank = T
    padded_send = torch.zeros(world_size, max_per_rank, D, dtype=dtype, device=device)
    send_masks = []

    for r in range(world_size):
        mask = (target_rank == r)
        if expert_indices.dim() == 2:  # top_k > 1
            mask = mask.any(dim=-1)
        send_masks.append(mask)
        count = mask.sum().item()
        if count > 0:
            padded_send[r, :count] = tokens[mask]

    # Build independent tensors for all_to_all (unbind returns views,
    # which cannot be modified inplace by all_to_all on newer PyTorch).
    padded_send_list = [padded_send[i].clone() for i in range(world_size)]
    padded_recv_list = [torch.zeros_like(padded_send[i]) for i in range(world_size)]
    dist.all_to_all(padded_recv_list, padded_send_list)
    padded_recv = torch.stack(padded_recv_list, dim=0)

    # Concatenate received tokens
    recv_concat = padded_recv.reshape(-1, D)
    # Map each received token back to a local expert index
    recv_expert_idx = torch.zeros(recv_concat.shape[0], dtype=torch.int64, device=device)
    offset = 0
    for src_rank in range(world_size):
        count = (target_rank == src_rank).sum().item() if not expert_indices.dim() == 2 else (target_rank.any(dim=-1) == (src_rank)).sum().item()
        # Source rank owns experts [src_rank * epr, (src_rank+1) * epr)
        base_expert = src_rank * experts_per_rank
        for i in range(count):
            if i < max_per_rank:
                recv_expert_idx[offset + i] = base_expert + (expert_indices if expert_indices.dim() == 1 else expert_indices).min() % experts_per_rank

    return recv_concat, recv_expert_idx, {
        "strategy": "naive",
        "sent_bytes": T * D * dtype.itemsize * world_size,
        "useful_bytes": T * D * dtype.itemsize,
        "padding_overhead": (world_size - 1) / world_size,
    }


# ---------------------------------------------------------------------------
# Strategy 2: Bucketed — variable-size all_to_all with split sizes
# ---------------------------------------------------------------------------

def bucketed_dispatch(
    *,
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    weights: torch.Tensor | None,
    num_experts: int,
    experts_per_rank: int,
    world_size: int,
    rank: int,
    combine: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Bucketed dispatch: group tokens by target rank, exchange with exact sizes.

    Uses all_to_all_single with per-rank split sizes to avoid padding overhead.
    This is the standard implementation used in most MoE frameworks (DeepSpeed-MoE, Tutel, etc.)
    """
    T, D = tokens.shape
    dtype, device = tokens.dtype, tokens.device

    target_rank = _target_rank(expert_indices, experts_per_rank)

    # Count tokens per target rank
    send_counts = torch.zeros(world_size, dtype=torch.int64, device=device)
    for r in range(world_size):
        mask = (target_rank == r)
        if expert_indices.dim() == 2:
            mask = mask.any(dim=-1)
        send_counts[r] = mask.sum().item()

    # Exchange counts — use all_gather (reliable on gloo) instead of
    # all_to_all_single to avoid gloo compatibility issues on Windows.
    all_send_counts = [torch.zeros(world_size, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(all_send_counts, send_counts)
    recv_counts = torch.stack([c[rank] for c in all_send_counts])

    # Group tokens by target rank — build a flat send buffer ordered by target rank
    send_order = target_rank.argsort()
    if expert_indices.dim() == 2:
        # For top_k>1 we need a stable ordering; flatten
        flat_expert = expert_indices.reshape(T, -1)
        target_rank_flat = flat_expert // experts_per_rank
        target_rank_single = target_rank_flat[:, 0]  # use first expert for ordering
        send_order = target_rank_single.argsort()

    sorted_tokens = tokens[send_order].contiguous()

    # --- Use point-to-point isend/irecv instead of all_to_all_single ---
    # gloo backend has poor support for all_to_all_single with variable
    # split sizes; P2P is more portable and models the real data flow.

    # 1. Build send buffers: one tensor per target rank
    send_bufs: list[torch.Tensor] = []
    send_offsets = (torch.cat([torch.zeros(1, device=device, dtype=torch.int64), send_counts.cumsum(0)]) * D).tolist()
    for r in range(world_size):
        start = send_offsets[r]
        end = send_offsets[r + 1]
        send_bufs.append(sorted_tokens.reshape(-1)[start:end].clone().contiguous())

    # 2. Allocate recv buffers based on recv_counts
    recv_bufs: list[torch.Tensor] = []
    for r in range(world_size):
        n = int(recv_counts[r].item())
        recv_bufs.append(torch.zeros(n * D, dtype=dtype, device=device))

    # 3. Post non-blocking sends and receives
    reqs = []
    for r in range(world_size):
        if r == rank:
            continue  # self-copy handled below
        if send_bufs[r].numel() > 0:
            reqs.append(dist.isend(send_bufs[r], dst=r))
        if recv_bufs[r].numel() > 0:
            reqs.append(dist.irecv(recv_bufs[r], src=r))

    # 4. Self-copy (rank → rank, no network)
    recv_bufs[rank] = send_bufs[rank].clone()

    # 5. Wait for all transfers
    for req in reqs:
        req.wait()

    # 6. Concatenate received buffers
    total_recv = int(recv_counts.sum().item())
    if total_recv > 0:
        recv_tokens = torch.cat(recv_bufs, dim=0).reshape(total_recv, D)
    else:
        recv_tokens = torch.zeros(0, D, dtype=dtype, device=device)

    recv_expert_idx = _build_recv_expert_idx(
        recv_counts, experts_per_rank, world_size, device
    )

    total_sent = int(send_counts.sum().item())
    total_useful = total_sent  # no padding

    return recv_tokens, recv_expert_idx, {
        "strategy": "bucketed",
        "sent_bytes": total_sent * D * dtype.itemsize,
        "useful_bytes": total_useful * D * dtype.itemsize,
        "padding_overhead": 0.0,
        "send_counts": send_counts,
        "recv_counts": recv_counts,
    }


# ---------------------------------------------------------------------------
# Strategy 3: Pipelined — chunked dispatch with compute/comm overlap
# ---------------------------------------------------------------------------

def pipelined_dispatch(
    *,
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    weights: torch.Tensor | None,
    num_experts: int,
    experts_per_rank: int,
    world_size: int,
    rank: int,
    combine: bool = False,
    num_chunks: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Pipelined dispatch: split tokens into micro-batches for overlap.

    Modeled after DeepEP's approach:
      Chunk 0: send ─┬─ recv ─ compute ─ send_back ─┬─ recv_back → output
      Chunk 1:       send ─┬─ recv ─ compute ─ send_back ─┬─ recv_back → output
      ...

    On a single GPU this won't achieve real overlap (serial execution),
    but it models the scheduling pattern and measures the theoretical
    overlap potential. With CUDA streams this could achieve real overlap.
    """
    T, D = tokens.shape
    dtype, device = tokens.dtype, tokens.device
    chunk_size = max(1, T // num_chunks)

    all_recv_tokens = []
    all_recv_expert_idx = []
    total_sent_bytes = 0
    total_useful_bytes = 0

    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        end = start + chunk_size if chunk_id < num_chunks - 1 else T
        if start >= T:
            break

        chunk_tokens = tokens[start:end]
        chunk_expert_idx = expert_indices[start:end]

        # Use bucketed dispatch for each chunk (could be on a separate stream)
        # In a real implementation, comm for chunk N+1 runs while compute for chunk N
        recv, recv_idx, info = bucketed_dispatch(
            tokens=chunk_tokens,
            expert_indices=chunk_expert_idx,
            weights=weights[start:end] if weights is not None else None,
            num_experts=num_experts,
            experts_per_rank=experts_per_rank,
            world_size=world_size,
            rank=rank,
        )

        # Point: after dispatch returns, we would launch expert compute on a
        # CUDA stream while the next chunk's dispatch begins on another stream.
        # torch.cuda.synchronize() here simulates the compute barrier.

        all_recv_tokens.append(recv)
        all_recv_expert_idx.append(recv_idx)
        total_sent_bytes += info["sent_bytes"]
        total_useful_bytes += info["useful_bytes"]

    if len(all_recv_tokens) > 0:
        combined_tokens = torch.cat(all_recv_tokens, dim=0)
        combined_expert_idx = torch.cat(all_recv_expert_idx, dim=0)
    else:
        combined_tokens = torch.zeros(0, D, dtype=dtype, device=device)
        combined_expert_idx = torch.zeros(0, dtype=torch.int64, device=device)

    return combined_tokens, combined_expert_idx, {
        "strategy": "pipelined",
        "sent_bytes": total_sent_bytes,
        "useful_bytes": total_useful_bytes,
        "padding_overhead": 0.0,
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target_rank(expert_indices: torch.Tensor, experts_per_rank: int) -> torch.Tensor:
    """Map expert indices to the rank that owns them."""
    return expert_indices // experts_per_rank


def _build_recv_expert_idx(
    recv_counts: torch.Tensor,
    experts_per_rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Assign received tokens to local experts round-robin.

    The receiving rank only knows how many tokens arrived from each source
    rank, not the original expert indices (which were computed by the
    sender's router).  For communication benchmarking the exact expert
    assignment is irrelevant — we distribute tokens evenly to keep the
    compute path exercised.
    """
    total = int(recv_counts.sum().item())
    if total == 0:
        return torch.zeros(0, dtype=torch.int64, device=device)

    recv_expert_idx = torch.arange(total, device=device) % experts_per_rank
    return recv_expert_idx


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

DISPATCH_REGISTRY = {
    "naive": naive_dispatch,
    "bucketed": bucketed_dispatch,
    "pipelined": pipelined_dispatch,
}


def get_dispatch_fn(name: str):
    """Look up dispatch strategy by name."""
    if name not in DISPATCH_REGISTRY:
        raise ValueError(f"Unknown dispatch strategy: {name}. Options: {list(DISPATCH_REGISTRY)}")
    return DISPATCH_REGISTRY[name]

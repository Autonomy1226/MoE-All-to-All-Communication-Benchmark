"""Simplified MoE Layer for communication benchmarking.

Models a standard MoE (Mixture of Experts) FFN block:
  Input -> Router -> Top-K Gate -> Token Dispatch -> Expert FFNs -> Token Combine -> Output

Supports Expert Parallelism (EP): each process holds a subset of experts,
and tokens are routed across processes via All-to-All.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """A single expert FFN: two linear layers with GELU activation."""

    def __init__(self, hidden_dim: int, intermediate_dim: int | None = None):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 4
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class MoELayer(nn.Module):
    """MoE layer that replaces a standard FFN in a Transformer block.

    Args:
        hidden_dim:     dimensionality of input/output tokens.
        num_experts:    total number of experts across *all* ranks.
        top_k:          how many experts each token is routed to.
        world_size:     number of processes (ranks).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        world_size: int = 4,
    ):
        super().__init__()
        assert num_experts % world_size == 0, (
            f"num_experts ({num_experts}) must be divisible by world_size ({world_size})"
        )
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.world_size = world_size
        self.experts_per_rank = num_experts // world_size

        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        # Each rank only instantiates the experts it owns
        self.local_experts = nn.ModuleList([
            ExpertFFN(hidden_dim) for _ in range(self.experts_per_rank)
        ])

    def forward(
        self,
        x: torch.Tensor,
        dispatch_fn,
        rank: int = 0,
    ) -> torch.Tensor:
        """Forward pass with expert-parallel dispatch.

        Args:
            x:            [num_tokens_local, hidden_dim] — local tokens on this rank.
            dispatch_fn:  function that handles token routing across ranks.
            rank:         current process rank.

        Returns:
            output:  [num_tokens_local, hidden_dim] — same shape as input.
        """
        num_tokens_local = x.shape[0]

        # 1. Route each token to top-k experts
        router_logits = self.router(x)                    # [T, E]
        routing_probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_probs, self.top_k, dim=-1)

        # Normalize top-k weights so they sum to 1.0 per token
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # For top_k=1 we can squeeze; for top_k>1 we flatten
        if self.top_k == 1:
            expert_idx = topk_indices.squeeze(-1)         # [T]
            weight = topk_weights.squeeze(-1)             # [T]
        else:
            expert_idx = topk_indices                     # [T, K]
            weight = topk_weights                         # [T, K]

        # 2. Dispatch tokens to their target experts across ranks
        received_tokens, received_expert_idx, recv_info = dispatch_fn(
            tokens=x,
            expert_indices=expert_idx,
            weights=weight if self.top_k > 1 else None,
            num_experts=self.num_experts,
            experts_per_rank=self.experts_per_rank,
            world_size=self.world_size,
            rank=rank,
        )

        # 3. Each expert computes on the tokens it receives
        expert_outputs = _compute_experts(
            received_tokens, received_expert_idx,
            self.local_experts, self.experts_per_rank,
        )

        # 4. Combine: route expert outputs back to original ranks
        output = dispatch_fn(
            tokens=expert_outputs,
            expert_indices=received_expert_idx,
            weights=None,
            num_experts=self.num_experts,
            experts_per_rank=self.experts_per_rank,
            world_size=self.world_size,
            rank=rank,
        )

        # 5. The dispatch/combine pair returns a tuple; extract the return token tensor
        if isinstance(output, tuple):
            output = output[0]

        # Slice to original local token count (padding may have been added)
        output = output[:num_tokens_local]
        return output


def _compute_experts(
    tokens: torch.Tensor,
    expert_idx: torch.Tensor,
    experts: nn.ModuleList,
    experts_per_rank: int,
) -> torch.Tensor:
    """Compute expert FFN for each token and combine weighted results."""
    outputs = torch.zeros_like(tokens)
    for e in range(experts_per_rank):
        mask = (expert_idx == e)
        if mask.any():
            expert_in = tokens[mask]
            expert_out = experts[e](expert_in)
            outputs[mask] = expert_out
    return outputs

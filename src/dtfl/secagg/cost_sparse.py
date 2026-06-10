"""Sparse-graph (SecAgg+, Bell et al.) secure-aggregation cost model.

Replaces the complete communication graph with a random k-regular graph of
logarithmic degree, so each client masks with only O(log n) neighbors instead of
all n-1. This yields polylogarithmic per-client overhead while retaining
semi-honest security and bounded-dropout tolerance -- the design needed when
tiers are large.

Same SecAggCost output as the complete-graph model, so the engine/experiment can
swap topologies by changing one call. The neighbor degree defaults to
ceil(log2(n)) scaled by a constant, matching the SecAgg+ family's guidance.
"""

from __future__ import annotations

import math

from dtfl.secagg.cost_complete import CostConstants, SecAggCost

__all__ = ["sparse_graph_cost", "default_degree"]


def default_degree(tier_size: int, degree_scale: float = 1.0) -> int:
    """Neighbor degree for a sparse SecAgg graph: ~ scale * ceil(log2 n).

    A small constant times log n gives the polylog overhead while keeping the
    graph connected and dropout-robust with high probability (SecAgg+ analysis).
    Floored at a minimum so tiny tiers still mask with a few peers.
    """
    n = max(2, tier_size)
    deg = math.ceil(degree_scale * math.log2(n))
    return max(2, min(deg, n - 1))


def sparse_graph_cost(
    tier_size: int,
    model_dim: int,
    c: CostConstants | None = None,
    degree_scale: float = 1.0,
) -> SecAggCost:
    """Per-client cost in a sparse (log-degree) SecAgg tier."""
    c = c or CostConstants()
    n = max(1, tier_size)
    degree = default_degree(n, degree_scale)

    bytes_shares = degree * c.bytes_per_share
    bytes_vector = model_dim * c.bytes_per_param
    per_client_bytes = bytes_shares + bytes_vector

    dh_ops = degree
    prg_params = (degree + 1) * model_dim

    setup = (
        c.base_setup_sec
        + degree * c.sec_per_dh
        + degree * c.sec_per_share_op
        + prg_params * c.sec_per_prg_param
    )
    upload = model_dim * c.sec_per_param_upload

    return SecAggCost(
        topology="sparse",
        tier_size=n,
        degree=degree,
        per_client_bytes=per_client_bytes,
        per_client_dh_ops=dh_ops,
        per_client_prg_params=prg_params,
        setup_latency_sec=setup,
        upload_latency_sec=upload,
    )

"""dtfl.secagg: secure-aggregation COST model (not real crypto).

Produces communication/computation overhead and per-phase latency that feed the
latent t_setup/t_upload terms, plus the analytic tier-success probability used
for the predicted-vs-empirical release calibration.
"""
from dtfl.secagg.cost_complete import CostConstants, SecAggCost, complete_graph_cost
from dtfl.secagg.cost_sparse import default_degree, sparse_graph_cost
from dtfl.secagg.success_prob import (
    SuccessPrediction,
    hoeffding_success_lower_bound,
    tier_success_probability,
)

__all__ = [
    "CostConstants", "SecAggCost", "complete_graph_cost",
    "sparse_graph_cost", "default_degree",
    "SuccessPrediction", "tier_success_probability", "hoeffding_success_lower_bound",
]

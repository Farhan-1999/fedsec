"""Complete-graph (Bonawitz-style) secure-aggregation cost model.

Classic Practical SecAgg pairs every client with every other in the tier
(complete communication graph). Per-client cost therefore scales with tier size
n: O(n) pairwise key agreements / mask seeds and O(n) secret shares, plus the
O(d) masked vector upload (d = model dimension).

This is a COST MODEL, not real crypto. It produces:
  - per-client bytes (shares + masked vector),
  - per-client crypto operations (DH + PRG expansions),
  - a wall-clock setup latency and upload latency, via calibration constants that
    a small real-protocol run (Flower SecAgg+) would fit.

The latencies feed back into the latent t_setup / t_upload terms so that the
response time being TIERED ON includes secure-aggregation overhead -- the
experimental design flags ignoring this as invalidating the whole argument.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SecAggCost", "CostConstants", "complete_graph_cost"]


@dataclass(frozen=True)
class CostConstants:
    """Calibration constants (fit from a small real-protocol run).

    Defaults are plausible order-of-magnitude placeholders; replace with values
    measured via experiments/calibrate_secagg.py on real hardware.
    """

    bytes_per_share: float = 64.0  # encrypted Shamir share (AEAD ciphertext)
    bytes_per_param: float = 4.0  # quantized model parameter (32-bit)
    sec_per_dh: float = 5e-4  # one X25519 key agreement
    sec_per_prg_param: float = 1e-7  # PRG expansion per output element
    sec_per_share_op: float = 2e-4  # create/encrypt or reconstruct one share
    sec_per_param_upload: float = 1e-7  # network time per uploaded parameter
    base_setup_sec: float = 0.05  # fixed per-session handshake overhead


@dataclass(frozen=True)
class SecAggCost:
    """Per-client cost of one tier session under a given topology."""

    topology: str
    tier_size: int
    degree: int  # number of neighbors each client masks with
    per_client_bytes: float
    per_client_dh_ops: int
    per_client_prg_params: int
    setup_latency_sec: float
    upload_latency_sec: float

    @property
    def total_latency_sec(self) -> float:
        return self.setup_latency_sec + self.upload_latency_sec

    def summary(self) -> str:
        return (
            f"{self.topology}: n={self.tier_size} deg={self.degree} "
            f"bytes={self.per_client_bytes/1e3:.1f}KB "
            f"setup={self.setup_latency_sec*1e3:.1f}ms "
            f"upload={self.upload_latency_sec*1e3:.1f}ms"
        )


def complete_graph_cost(
    tier_size: int,
    model_dim: int,
    c: CostConstants | None = None,
) -> SecAggCost:
    """Per-client cost in an all-to-all SecAgg tier of size ``tier_size``."""
    c = c or CostConstants()
    n = max(1, tier_size)
    degree = n - 1  # every other client

    # bytes: one encrypted share per neighbor + the masked model vector
    bytes_shares = degree * c.bytes_per_share
    bytes_vector = model_dim * c.bytes_per_param
    per_client_bytes = bytes_shares + bytes_vector

    # crypto ops: one DH per neighbor; PRG expands a d-length mask per neighbor
    # (pairwise masks) plus one self-mask
    dh_ops = degree
    prg_params = (degree + 1) * model_dim

    # latency: setup dominated by DH + share ops scaling with degree; upload by d
    setup = (
        c.base_setup_sec
        + degree * c.sec_per_dh
        + degree * c.sec_per_share_op
        + prg_params * c.sec_per_prg_param
    )
    upload = model_dim * c.sec_per_param_upload

    return SecAggCost(
        topology="complete",
        tier_size=n,
        degree=degree,
        per_client_bytes=per_client_bytes,
        per_client_dh_ops=dh_ops,
        per_client_prg_params=prg_params,
        setup_latency_sec=setup,
        upload_latency_sec=upload,
    )

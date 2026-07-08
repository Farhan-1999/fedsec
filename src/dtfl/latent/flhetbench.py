"""FLHetBench-grounded population construction (Path 3).

Replaces the *parametric* latent model (evenly spaced class means + chosen
``s_within``/``eta``) with device profiles derived from the FLHetBench
real-world device database (CVPR 2024). This module lives in the LATENT layer on
purpose: everything it produces is ground-truth device state on the hidden side
of the observation boundary. The server/attacker never see any of it -- they see
only the transcript, exactly as before. The import-separation tests keep this
module unreachable from ``attack`` / ``controller``.

Provenance (Path 3 -- each FLHetBench asset used where it is DENSE):
  * compute latency  <- training_latency.json (978 real phone models, ResNet ms),
    joined to the sampled device set by AI-Benchmark *rank* (real2benchmark ->
    benchmark2score), because the two sources use different model-naming schemes
    and exact string joins cover <20% of devices. Devices with no benchmark
    mapping draw from the empirical (log) distribution of the mapped devices, so
    the population stays within real FLHetBench latencies.
  * network throughput <- the paired ``tcp_speed_results`` in each device record
    (MobiPerf); present for 100% of records, so this keeps the real
    device<->network pairing.
  * heterogeneity level <- controlled spread subsampling over the compute
    latencies (DPGMM-style ``sigma``), NOT the case1..4 partition, whose compute
    join is too sparse to define clean class structure (case1 maps only 2/100).

The output is a list of ``LatentDeviceState`` -- identical type and semantics to
``latent.population.build_population`` -- so the engine, protocol, transcript,
attack, defense, and metrics layers are all untouched.

Capability CLASS remains a latent label: compute-latency quantile bands over the
constructed population, used only to score the attack. It is never an input to
tiering; devices still self-assign to tiers by response time against blind,
transcript-calibrated deadlines.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from dtfl.types import LatentDeviceState

__all__ = [
    "FLHetBenchConfig",
    "load_flhetbench_assets",
    "build_flhetbench_population",
]


# Default location of the staged FLHetBench data inside the repo. Resolved
# relative to the repo root (three levels up from this file:
# src/dtfl/latent/flhetbench.py -> repo root).
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "data", "flhetbench")


@dataclass(frozen=True)
class FLHetBenchConfig:
    """Knobs for FLHetBench-grounded population construction.

    Parameters
    ----------
    num_classes:
        C -- number of capability classes (compute-latency quantile bands).
    case:
        Which paired device/network sample set to draw network profiles from
        (``case1``..``case4``). Compute latency comes from the full
        training-latency table regardless of case (Path 3).
    hetero_spread:
        DPGMM-style ``sigma``. Multiplicative widening of the compute-latency
        spread around its geometric mean, in log space. 1.0 = use the real
        latencies as-is; >1 exaggerates heterogeneity, <1 compresses it. This is
        the controlled heterogeneity-sweep axis.
    proxy_noise_eta:
        The ONE modeled quantity: round-to-round transient jitter stddev (log),
        added in ``draw_completion_times``. Not in a static device profile, so it
        stays modeled -- but anchored to measured per-device spread by default.
    data_dir:
        Directory holding the staged FLHetBench JSON files.
    """

    num_classes: int = 5
    case: str = "case2"
    hetero_spread: float = 1.0
    proxy_noise_eta: float = 0.20
    data_dir: str = _DEFAULT_DATA_DIR


@dataclass(frozen=True)
class _Assets:
    """Parsed FLHetBench source tables (loaded once, reused across builds)."""

    tl_latency_sorted: np.ndarray  # real ResNet latencies (ms), ascending
    score_rank: dict  # benchmark device name -> rank (0 = fastest)
    n_scores: int
    real2benchmark: dict  # real model string -> benchmark name (or 'unknown')


def load_flhetbench_assets(data_dir: str = _DEFAULT_DATA_DIR) -> _Assets:
    """Load and index the FLHetBench compute-join tables."""
    with open(os.path.join(data_dir, "training_latency.json")) as f:
        tl = json.load(f)
    with open(os.path.join(data_dir, "real2benchmark.json")) as f:
        r2b = json.load(f)
    with open(os.path.join(data_dir, "benchmark2score.json")) as f:
        b2s = json.load(f)

    tl_lat = np.array(
        sorted(x["ResNet Latency"] for x in tl if x.get("ResNet Latency", 0) > 0),
        dtype=np.float64,
    )
    # Higher AI-Benchmark score = faster device => rank 0 is the fastest, which
    # should map to the LOWEST latency. Sort scores descending for that alignment.
    b2s_desc = sorted(b2s.items(), key=lambda kv: -kv[1])
    score_rank = {name: i for i, (name, _) in enumerate(b2s_desc)}

    return _Assets(
        tl_latency_sorted=tl_lat,
        score_rank=score_rank,
        n_scores=len(b2s_desc),
        real2benchmark=r2b,
    )


def _rank_aligned_latency(real_model: str, assets: _Assets) -> float | None:
    """Map a real device model to a real ResNet latency by benchmark rank.

    Returns None if the model has no (non-'unknown') benchmark mapping; the caller
    fills those from the empirical distribution of the mapped devices.
    """
    bench = assets.real2benchmark.get(real_model)
    if bench is None or "unknown" in str(bench) or bench not in assets.score_rank:
        return None
    frac = assets.score_rank[bench] / max(1, assets.n_scores - 1)  # 0=fast..1=slow
    idx = int(round(frac * (assets.tl_latency_sorted.size - 1)))
    return float(assets.tl_latency_sorted[idx])


def build_flhetbench_population(
    n: int,
    config: FLHetBenchConfig,
    rng: np.random.Generator,
    assets: _Assets | None = None,
) -> list[LatentDeviceState]:
    """Construct ``n`` device states grounded in FLHetBench real data (Path 3).

    Steps:
      1. Load the case device records (real model + paired network vector).
      2. Map each record's model to a real ResNet latency by benchmark rank;
         fill unmapped models from the empirical log-latency distribution of the
         mapped ones (stays within real FLHetBench latencies).
      3. Apply the ``hetero_spread`` (sigma) widening in log space to set the
         heterogeneity level.
      4. Sample-with-replacement up to ``n`` clients, each cloning a real device
         template (compute base + its paired availability proxy).
      5. Assign capability CLASS by compute-latency quantile band (latent label,
         scoring-only).

    ``mu`` is set to the natural log of the (spread-adjusted) compute latency in
    ms, so it is a base log-latency directly comparable to the parametric model's
    ``m_c + u_i``. ``draw_completion_times`` then adds the modeled transient jitter
    and heavy tail on top, exactly as in the synthetic path.
    """
    if assets is None:
        assets = load_flhetbench_assets(config.data_dir)

    case_path = os.path.join(config.data_dir, "device", f"{config.case}.json")
    with open(case_path) as f:
        records = json.load(f)

    # --- per-template compute latency + availability proxy ---
    comp_ms: list[float | None] = []
    avail: list[float] = []
    for rec in records:
        model = rec["device_properties"]["device_info"]["model"]
        comp_ms.append(_rank_aligned_latency(model, assets))
        # Availability proxy from the paired network vector: faster/steadier
        # networks proxy higher participation. Normalized to (0, 1] later.
        net = rec.get("tcp_speed_results") or [0.0]
        avail.append(float(np.mean(net)))

    # Fill unmapped compute latencies from the empirical log-distribution of the
    # mapped devices, so the population stays within real FLHetBench latencies.
    mapped = np.array([c for c in comp_ms if c is not None], dtype=np.float64)
    if mapped.size == 0:
        raise ValueError(
            f"No devices in {config.case} mapped to a benchmark latency; "
            "cannot build a FLHetBench-grounded population from this case."
        )
    log_mapped = np.log(mapped)
    mu_log, sd_log = float(log_mapped.mean()), float(log_mapped.std())
    comp_filled = np.array(
        [c if c is not None else float(np.exp(rng.normal(mu_log, sd_log))) for c in comp_ms],
        dtype=np.float64,
    )

    # --- heterogeneity control (DPGMM-style sigma), in log space ---
    log_comp = np.log(comp_filled)
    center = log_comp.mean()
    log_comp = center + (log_comp - center) * float(config.hetero_spread)
    comp_filled = np.exp(log_comp)

    # --- availability proxy -> (0, 1] participation rate ---
    avail_arr = np.array(avail, dtype=np.float64)
    if np.ptp(avail_arr) > 0:
        avail_norm = 0.10 + 0.30 * (avail_arr - avail_arr.min()) / np.ptp(avail_arr)
    else:
        avail_norm = np.full_like(avail_arr, 0.20)

    # --- sample n clients with replacement from the real device templates ---
    idx = rng.integers(0, len(records), size=n)
    client_comp = comp_filled[idx]
    client_avail = avail_norm[idx]

    # --- capability class = compute-latency quantile band (latent scoring label) ---
    C = config.num_classes
    edges = np.quantile(client_comp, np.linspace(0.0, 1.0, C + 1))
    edges[-1] = np.inf  # ensure the slowest device lands in the last class
    # class 1 = fastest = lowest latency
    class_of = np.searchsorted(edges[1:], client_comp, side="right") + 1
    class_of = np.clip(class_of, 1, C)

    return [
        LatentDeviceState(
            device_id=i,
            capability_class=int(class_of[i]),
            mu=float(np.log(client_comp[i])),
            availability_rate=float(client_avail[i]),
        )
        for i in range(n)
    ]
"""dtfl.sim: orchestration. Wires latent->protocol->transcript over R rounds."""
from dtfl.sim.engine import (
    DeadlinePolicy,
    Engine,
    EngineConfig,
    SimulationOutput,
)
from dtfl.sim.gates import GateReport, evaluate_gates, snr_diagnostic
from dtfl.sim.round import RoundConfig, RoundLatentLog, RoundResult, run_round

__all__ = [
    "Engine", "EngineConfig", "SimulationOutput", "DeadlinePolicy",
    "RoundConfig", "RoundResult", "RoundLatentLog", "run_round",
    "GateReport", "evaluate_gates", "snr_diagnostic",
]

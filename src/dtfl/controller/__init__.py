"""dtfl.controller: deadline-setting policies (the system being measured).

NOTE: OracleQuantileController is intentionally NOT re-exported here. It reads
latent ground truth and lives in dtfl.controller.oracle, which the attack package
is forbidden to import. Import it explicitly (from dtfl.controller.oracle import
OracleQuantileController) in experiment code only.
"""
from dtfl.controller.base import Controller, DeadlinePolicy, project_monotone
from dtfl.controller.ewma import EWMAQuantileController
from dtfl.controller.fixed import FixedEqualWidth, FixedQuantile
from dtfl.controller.quantile import QuantileTrackingController

__all__ = [
    "Controller", "DeadlinePolicy", "project_monotone",
    "FixedEqualWidth", "FixedQuantile",
    "QuantileTrackingController", "EWMAQuantileController",
]

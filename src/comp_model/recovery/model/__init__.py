"""Model-level recovery analysis utilities."""

from comp_model.recovery.model.analysis import confusion_matrix, recovery_rates
from comp_model.recovery.model.config import (
    CandidateModelSpec,
    GeneratingModelSpec,
    ModelRecoveryConfig,
)
from comp_model.recovery.model.display import confusion_matrix_table, recovery_rate_table
from comp_model.recovery.model.runner import (
    ModelRecoveryResult,
    ReplicationResult,
    run_model_recovery,
)

__all__ = [
    "CandidateModelSpec",
    "GeneratingModelSpec",
    "ModelRecoveryConfig",
    "ModelRecoveryResult",
    "ReplicationResult",
    "confusion_matrix",
    "confusion_matrix_table",
    "recovery_rate_table",
    "recovery_rates",
    "run_model_recovery",
]

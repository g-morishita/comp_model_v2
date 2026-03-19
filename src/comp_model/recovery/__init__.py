"""Parameter and model recovery analysis utilities."""

from comp_model.recovery.config import ParamDist, RecoveryStudyConfig, sample_true_params
from comp_model.recovery.display import recovery_summary, recovery_table
from comp_model.recovery.extraction import ReplicationEstimates, SubjectEstimates
from comp_model.recovery.metrics import (
    ParameterRecoveryMetrics,
    RecoveryMetrics,
    compute_recovery_metrics,
)
from comp_model.recovery.model import (
    CandidateModelSpec,
    GeneratingModelSpec,
    ModelRecoveryConfig,
    ModelRecoveryResult,
    ReplicationResult,
    confusion_matrix,
    confusion_matrix_table,
    recovery_rate_table,
    recovery_rates,
    run_model_recovery,
)
from comp_model.recovery.runner import RecoveryResult, run_recovery

__all__ = [
    "CandidateModelSpec",
    "GeneratingModelSpec",
    "ModelRecoveryConfig",
    "ModelRecoveryResult",
    "ParamDist",
    "ParameterRecoveryMetrics",
    "RecoveryMetrics",
    "RecoveryResult",
    "RecoveryStudyConfig",
    "ReplicationEstimates",
    "ReplicationResult",
    "SubjectEstimates",
    "compute_recovery_metrics",
    "confusion_matrix",
    "confusion_matrix_table",
    "recovery_rate_table",
    "recovery_rates",
    "recovery_summary",
    "recovery_table",
    "run_model_recovery",
    "run_recovery",
    "sample_true_params",
]

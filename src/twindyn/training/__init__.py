"""Training: optimiser, schedule, precision, checkpoints, seeding and the engine.

Ref: Algorithms 1-2, Sec. 3.5, Sec. 3.10, Sec. 4.4.
"""

from twindyn.training.checkpoint import (
    CHECKPOINT_FORMAT,
    CheckpointContents,
    load_checkpoint,
    payload_digest,
    restore,
    save_checkpoint,
    verify_round_trip,
)
from twindyn.training.engine import (
    StageReport,
    TrainingReport,
    score,
    train_readout,
    train_representation,
    train_twindyn,
)
from twindyn.training.optim import (
    SCHEDULE_NAMES,
    CosineSchedule,
    LambdaScheduleAdapter,
    LinearWarmupConstantSchedule,
    OptimiserBundle,
    WarmupCosineSchedule,
    build_optimiser,
    build_schedule,
    effective_batch_size,
    parameter_groups,
    trainable_parameter_count,
)
from twindyn.training.precision import (
    ExponentialMovingAverage,
    build_grad_scaler,
    precision_context,
)
from twindyn.training.seeding import (
    InitialisationPlan,
    apply_initialisation,
    plan_initialisations,
)

__all__ = [
    "CHECKPOINT_FORMAT",
    "SCHEDULE_NAMES",
    "CheckpointContents",
    "CosineSchedule",
    "ExponentialMovingAverage",
    "InitialisationPlan",
    "LambdaScheduleAdapter",
    "LinearWarmupConstantSchedule",
    "OptimiserBundle",
    "StageReport",
    "TrainingReport",
    "WarmupCosineSchedule",
    "apply_initialisation",
    "build_grad_scaler",
    "build_optimiser",
    "build_schedule",
    "effective_batch_size",
    "load_checkpoint",
    "parameter_groups",
    "payload_digest",
    "plan_initialisations",
    "precision_context",
    "restore",
    "save_checkpoint",
    "score",
    "train_readout",
    "train_representation",
    "train_twindyn",
    "trainable_parameter_count",
    "verify_round_trip",
]

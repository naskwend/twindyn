"""Study-wide constants taken from the manuscript.

Every value here is either quoted from a stated location in the paper or marked
as an engineering default because the manuscript is silent. The silence markers
are exported so that the integrity manifest can list them.

Ref: Sec. 3.10 (selected quantities), Sec. 4.2 (criteria), Sec. 4.4 (protocol),
Table 4 (selected axis length and state dimension).
"""

from __future__ import annotations

from typing import Final

PAPER_TITLE: Final[str] = (
    "Self-Supervised Digital Twin Modeling of Tumor Microenvironment Dynamics "
    "for Survival Prognosis in Kidney Cancer"
)
FRAMEWORK_NAME: Final[str] = "TwinDyn"

SELECTED_AXIS_LENGTH: Final[int] = 16
SELECTED_STATE_DIM: Final[int] = 64
AXIS_LENGTH_GRID: Final[tuple[int, ...]] = (4, 8, 16, 32)
STATE_DIM_GRID: Final[tuple[int, ...]] = (16, 32, 64, 128)

INITIALISATION_COUNT: Final[int] = 10
BOOTSTRAP_RESAMPLES: Final[int] = 1000
CONFIDENCE_LEVEL: Final[float] = 0.95
EXTERNAL_THESIS_THRESHOLD: Final[float] = 0.03
DECISION_MARGIN: Final[float] = 0.005
LABEL_EFFICIENCY_FRACTIONS: Final[tuple[float, ...]] = (0.10, 0.25, 0.50, 1.00)
RESOURCE_CHANCE_LEVEL: Final[float] = 0.25
PROBE_REPEATS: Final[int] = 10
PARAMETER_COUNT_MILLIONS: Final[float] = 3.16
MATCHED_CONTROL_PARAMETERS_MILLIONS: Final[float] = 3.14
READ_OUT_DEPTH: Final[int] = 1

REPORTED_COHORT_SIZES: Final[dict[str, int]] = {
    "cptac_ccrcc": 103,
    "gse29609": 39,
    "emtab1980": 101,
}

ENGINEERING_DEFAULT_MASK_RATIO: Final[float] = 0.30
ENGINEERING_DEFAULT_LAMBDA_MASK: Final[float] = 1.0
ENGINEERING_DEFAULT_LAMBDA_ALIGN: Final[float] = 1.0
ENGINEERING_DEFAULT_LAMBDA_TRAJECTORY: Final[float] = 0.5
ENGINEERING_DEFAULT_BATCH_SIZE: Final[int] = 64
ENGINEERING_DEFAULT_REPRESENTATION_EPOCHS: Final[int] = 60
ENGINEERING_DEFAULT_READOUT_EPOCHS: Final[int] = 40
ENGINEERING_DEFAULT_LEARNING_RATE: Final[float] = 1e-3
ENGINEERING_DEFAULT_READOUT_LEARNING_RATE: Final[float] = 1e-3
ENGINEERING_DEFAULT_WEIGHT_DECAY: Final[float] = 1e-4
ENGINEERING_DEFAULT_WARMUP_STEPS: Final[int] = 100
ENGINEERING_DEFAULT_GRAD_CLIP: Final[float] = 1.0
ENGINEERING_DEFAULT_SEED: Final[int] = 0
ENGINEERING_DEFAULT_SHARED_FEATURES: Final[int] = 512
ENGINEERING_DEFAULT_MORPHOLOGY_FEATURES: Final[int] = 256

UNREPORTED_QUANTITIES: Final[tuple[str, ...]] = (
    "mask_ratio",
    "lambda_mask",
    "lambda_align",
    "lambda_trajectory",
    "batch_size",
    "optimizer",
    "learning_rate",
    "readout_learning_rate",
    "weight_decay",
    "scheduler",
    "warmup_steps",
    "grad_clip",
    "precision",
    "representation_epochs",
    "readout_epochs",
    "tau_to_compartment_assignment_rule",
    "deconvolution_configuration",
    "shared_feature_space_size",
)

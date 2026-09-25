"""The numbers the manuscript reports, transcribed with their table anchors.

Every entry here is a value printed in the manuscript, kept in one module so the
verification pass can compare an executed quantity against the published one and
so the release can state which of them this environment can and cannot reproduce.
Nothing in this module is computed; it is quoted.

Ref: Table 1 (main comparison on the held-out split of the training resource),
Table 2 (ablation on the mean external concordance), Table 3 (generalisation and
robustness), Table 4 (properties of the learned state), Table 5 (mechanism
evidence triad), Table 6 (resource-disjoint external validation), Sec. 4.10
(label efficiency).
"""

from __future__ import annotations

from typing import Final

TWINDYN_MAIN: Final[dict[str, object]] = {
    "identifier": "TwinDyn",
    "label": "TwinDyn (label-free representation)",
    "family": "this work",
    "c_index": 0.734,
    "interval": (0.716, 0.752),
    "sd": 0.011,
}

MAIN_COMPARISON_MARGIN_OVER_B16: Final[float] = 0.003
MAIN_COMPARISON_VERDICT: Final[str] = "parity"

ABLATION: Final[dict[str, dict[str, object]]] = {
    "full_model": {
        "mean_external": 0.655,
        "delta": 0.000,
        "params_millions": 3.16,
        "role": "reference",
    },
    "minus_sst_static_matched": {
        "mean_external": 0.614,
        "delta": -0.041,
        "params_millions": 3.14,
        "role": "level 0 innovation",
    },
    "minus_msi_alignment": {
        "mean_external": 0.631,
        "delta": -0.024,
        "params_millions": 3.16,
        "role": "level 1a",
    },
    "minus_masked_reconstruction": {
        "mean_external": 0.645,
        "delta": -0.010,
        "params_millions": 3.16,
        "role": "training objective",
    },
    "minus_trajectory_pooling": {
        "mean_external": 0.648,
        "delta": -0.007,
        "params_millions": 3.15,
        "role": "read-out input",
    },
    "readout_depth_1_to_2": {
        "mean_external": 0.653,
        "delta": -0.002,
        "params_millions": 3.16,
        "role": "head capacity",
    },
    "alignment_clinical_only": {
        "mean_external": 0.652,
        "delta": -0.003,
        "params_millions": 3.16,
        "role": "designed null below the margin",
    },
    "equal_budget_default_hyperparameters": {
        "mean_external": 0.652,
        "delta": -0.003,
        "params_millions": 3.16,
        "role": "margin retention 92.7%",
    },
}
ABLATION_MAX_DROP: Final[float] = 0.041
ABLATION_MIN_DROP: Final[float] = 0.003
ABLATION_DROP_RATIO: Final[float] = 13.7
ABLATION_MARGIN_RETENTION: Final[float] = 0.927

GENERALISATION: Final[dict[str, dict[str, object]]] = {
    "full": {"definition": "all external samples", "c_index": 0.655, "delta": 0.000},
    "stage_1": {"definition": "localised", "c_index": 0.681, "delta": 0.026},
    "stage_2": {"definition": "intermediate", "c_index": 0.634, "delta": -0.021},
    "stage_3": {"definition": "locally advanced", "c_index": 0.662, "delta": 0.007},
    "stage_4": {"definition": "advanced", "c_index": 0.648, "delta": -0.007},
    "grade_1": {"definition": "lowest", "c_index": 0.702, "delta": 0.047},
    "grade_2": {"definition": "low intermediate", "c_index": 0.667, "delta": 0.012},
    "grade_3": {"definition": "high intermediate", "c_index": 0.641, "delta": -0.014},
    "grade_4": {"definition": "highest", "c_index": 0.612, "delta": -0.043},
    "compartments_aggregated": {
        "definition": "coarser resolution",
        "c_index": 0.636,
        "delta": -0.019,
    },
    "minus_morphology": {"definition": "dropout", "c_index": 0.651, "delta": -0.004},
    "minus_compartments": {"definition": "dropout", "c_index": 0.622, "delta": -0.033},
    "b10_minus_morphology": {"definition": "dropout", "c_index": 0.610, "delta": -0.039},
}
B10_FULL_MODALITY_EXTERNAL: Final[float] = 0.649

STATE_PROPERTIES: Final[dict[str, object]] = {
    "axis_length_sweep": {
        "values": (0.631, 0.646, 0.655, 0.657),
        "grid": (4, 8, 16, 32),
        "selected": 16,
        "note": "plateau after 16",
    },
    "state_dimension_sweep": {
        "values": (0.638, 0.652, 0.655, 0.654),
        "grid": (16, 32, 64, 128),
        "selected": 64,
        "note": "saturation after 64",
    },
    "shuffled_ordering": {"c_index": 0.618, "delta": -0.037},
    "hyperparameter_spread": 0.006,
    "alignment_magnitude_relative": {
        "training|cptac_ccrcc": 0.21,
        "training|gse29609": 0.34,
        "training|emtab1980": 0.39,
    },
    "alignment_pair_gains": {
        "training|cptac_ccrcc": 0.046,
        "training|gse29609": 0.039,
        "training|emtab1980": 0.040,
    },
}

MECHANISM: Final[dict[str, float]] = {
    "necessity_delta_sst_matched": -0.041,
    "invariance_probe_raw_composition": 0.612,
    "invariance_probe_learned_state": 0.271,
    "probe_chance_level": 0.25,
    "perturbation_early_small": -0.012,
    "perturbation_early_large": -0.034,
    "perturbation_late_large": -0.009,
    "unperturbed_external": 0.655,
}
PERTURBATION_SCALES: Final[dict[str, float]] = {"small": 0.25, "large": 1.0}
PERTURBATION_POSITIONS: Final[dict[str, int]] = {"early": 0, "late": 15}

EXTERNAL_VALIDATION: Final[dict[str, dict[str, object]]] = {
    "cptac_ccrcc": {
        "cohort": "CPTAC-ccRCC (GDC/TCIA)",
        "samples": 103,
        "twindyn": 0.658,
        "twindyn_interval": (0.613, 0.703),
        "best_static": 0.612,
        "best_static_interval": (0.565, 0.659),
        "difference": 0.046,
        "difference_interval": (0.018, 0.074),
        "p_corrected": 0.009,
        "loss_vs_train": 0.076,
    },
    "gse29609": {
        "cohort": "GSE29609 (GEO)",
        "samples": 39,
        "twindyn": 0.667,
        "twindyn_interval": (0.626, 0.708),
        "best_static": 0.628,
        "best_static_interval": (0.585, 0.671),
        "difference": 0.039,
        "difference_interval": (0.011, 0.067),
        "p_corrected": 0.012,
        "loss_vs_train": 0.067,
    },
    "emtab1980": {
        "cohort": "E-MTAB-1980 (AE)",
        "samples": 101,
        "twindyn": 0.641,
        "twindyn_interval": (0.592, 0.690),
        "best_static": 0.601,
        "best_static_interval": (0.551, 0.651),
        "difference": 0.040,
        "difference_interval": (0.009, 0.071),
        "p_corrected": 0.014,
        "loss_vs_train": 0.093,
    },
}
EXTERNAL_MEAN: Final[dict[str, float]] = {
    "twindyn": 0.655,
    "best_static": 0.614,
    "difference": 0.042,
    "loss_vs_train": 0.079,
}
STATIC_BASELINE_TRANSFER_LOSSES: Final[tuple[float, ...]] = (0.119, 0.103, 0.130)

LABEL_EFFICIENCY: Final[dict[str, dict[str, object]]] = {
    "0.10": {"twindyn": 0.716, "best_static": 0.640, "difference": 0.076},
    "0.25": {"difference": 0.038},
    "0.50": {"difference": 0.012},
    "1.00": {"difference": 0.003, "verdict": "parity"},
}
ZERO_LABEL_EXTERNAL: Final[float] = 0.658

WITHIN_STRATUM_GAINS: Final[dict[str, float]] = {
    "stage_1": 0.021,
    "stage_2": 0.036,
    "stage_3": 0.051,
    "stage_4": 0.007,
}

MEASURED_TRANSFER_TAX_POINTS: Final[tuple[float, ...]] = (0.076, 0.067, 0.093)


def reported_main_table() -> list[dict[str, object]]:
    """Table 1 as a list of rows, with the TwinDyn row appended."""
    from twindyn.models.baselines.inventory import BASELINES

    rows: list[dict[str, object]] = []
    for identifier, spec in BASELINES.items():
        rows.append(
            {
                "id": identifier,
                "method": spec.method,
                "family": spec.family,
                "c_index": spec.reported_c_index,
                "interval": spec.reported_interval,
                "sd": spec.reported_sd,
                "comparator": spec.comparator,
            }
        )
    rows.append(
        {
            "id": "TwinDyn",
            "method": TWINDYN_MAIN["label"],
            "family": TWINDYN_MAIN["family"],
            "c_index": TWINDYN_MAIN["c_index"],
            "interval": TWINDYN_MAIN["interval"],
            "sd": TWINDYN_MAIN["sd"],
            "comparator": False,
        }
    )
    return rows


def strongest_static_baseline() -> dict[str, object]:
    from twindyn.models.baselines.inventory import BASELINES, COMPARATOR_ID

    spec = BASELINES[COMPARATOR_ID]
    return {
        "id": COMPARATOR_ID,
        "method": spec.method,
        "reported_c_index": spec.reported_c_index,
        "reported_interval": spec.reported_interval,
    }

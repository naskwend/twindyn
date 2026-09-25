"""Pass one: the paper claim to code mapping.

Every entry names the claim, the location in the manuscript it comes from, the
files and symbols that carry it, and the execution checks of pass two that test
it. Entries are written once and read by the artefact writer, so a claim that has
no code behind it cannot quietly disappear.

Ref: the equation, algorithm, table and protocol numbering of the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Claim:
    identifier: str
    claim: str
    paper_location: str
    code: tuple[str, ...]
    checks: tuple[str, ...] = ()
    notes: str = ""


CLAIMS: tuple[Claim, ...] = (
    Claim(
        "C01",
        "The microenvironment observation for a tumour is a compartment block followed by a morphology block.",
        "Sec. 3.1, Eq. (1)",
        (
            "src/twindyn/data/schema.py:SampleRecord",
            "src/twindyn/data/axis.py:MicroenvironmentAxis",
        ),
        ("eq1-01", "eq1-02", "data-01"),
    ),
    Claim(
        "C02",
        "The state evolves along an ordered axis with z(tau) = A(tau) z(tau-1) + B(tau) u(tau) and x(tau) = C z(tau) + noise.",
        "Sec. 3.1, Eq. (2)",
        (
            "src/twindyn/models/scan.py:selective_scan",
            "src/twindyn/models/sst.py:SelectiveStateSpaceTransition",
            "src/twindyn/data/synthetic.py:LatentLaw",
        ),
        ("sst-02", "sst-03", "sst-04", "data-05"),
    ),
    Claim(
        "C03",
        "The ordered axis is a predefined organisation of the sample's compartments, fixed before training and never adjusted to an outcome.",
        "Sec. 3.2, Sec. 3.10, Sec. 4.4",
        (
            "src/twindyn/data/compartments.py:RAW_COMPARTMENTS",
            "src/twindyn/data/axis.py:MicroenvironmentAxis",
        ),
        ("eq1-02", "axis-01", "axis-02", "axis-03", "axis-04"),
    ),
    Claim(
        "C04",
        "Using the common map C, MSI links the observation to a hidden state that carries no information about the resource that produced it.",
        "Sec. 3.3",
        (
            "src/twindyn/models/msi.py:MultiSourceInvariantEncoder",
            "src/twindyn/losses/alignment.py",
        ),
        ("msi-01", "msi-02", "msi-04", "pair-02", "mech-01"),
    ),
    Claim(
        "C05",
        "The alignment term stays unspecified for a resource pair that shares no matched samples and contributes only to the other objectives.",
        "Sec. 3.3",
        ("src/twindyn/data/pairs.py:batch_alignment_pairs", "src/twindyn/losses/alignment.py"),
        ("pair-02", "pair-04", "pair-05"),
    ),
    Claim(
        "C06",
        "SST is selective: it chooses how much new observation to write and how much of the existing state to keep.",
        "Sec. 3.4",
        ("src/twindyn/models/sst.py:SelectiveStateSpaceTransition", "src/twindyn/models/scan.py"),
        ("sst-05", "sst-06"),
    ),
    Claim(
        "C07",
        "The operator is instantiated with an exponential-trapezoidal discretisation, a complex-valued state and a multi-input multi-output form.",
        "Sec. 3.4; reference 54",
        (
            "src/twindyn/models/scan.py:exponential_trapezoidal_step",
            "src/twindyn/models/scan.py:selective_scan",
        ),
        ("sst-02", "sst-03", "sst-04", "sst-08", "sst-09"),
    ),
    Claim(
        "C08",
        "The scan cost grows linearly as O(tau_max d^2), which is what makes the resolution of the microenvironment affordable.",
        "Sec. 3.8",
        ("src/twindyn/models/scan.py:scan_cost", "src/twindyn/models/sst.py:analytic_cost"),
        ("sst-01",),
    ),
    Claim(
        "C09",
        "TOR combines trajectory characteristics along the axis and passes them through a light survival head with the depth kept to a minimum.",
        "Sec. 3.5",
        (
            "src/twindyn/models/tor.py:TrajectoryPool",
            "src/twindyn/models/tor.py:TrajectoryOutcomeReadout",
        ),
        ("model-01", "model-02", "train-05"),
    ),
    Claim(
        "C10",
        "The read-out is trained with the conventional survival objective for a continuous risk under right censoring, and it is the only place a survival label enters.",
        "Sec. 3.5; reference 47",
        (
            "src/twindyn/losses/survival.py:cox_partial_likelihood",
            "src/twindyn/training/engine.py:train_readout",
        ),
        ("loss-06", "loss-07", "loss-08"),
    ),
    Claim(
        "C11",
        "The representation-learning objective is the weighted sum of the masked reconstruction, the cross-resource alignment and the trajectory consistency terms, with no survival label.",
        "Sec. 3.6, Eq. (3)",
        ("src/twindyn/losses/composite.py:composite_objective",),
        ("loss-01", "loss-02", "loss-08"),
    ),
    Claim(
        "C12",
        "L_mask is the squared error of the reconstruction of a masked fraction of the observed locations, taken from the resulting state.",
        "Sec. 3.6, Eq. (4)",
        (
            "src/twindyn/losses/reconstruction.py:masked_reconstruction_loss",
            "src/twindyn/data/masking.py:sample_mask",
        ),
        ("mask-01", "mask-02", "mask-03", "mask-04", "train-06"),
    ),
    Claim(
        "C13",
        "L_align is the squared state distance over matched pairs drawn from different resources.",
        "Sec. 3.6, Eq. (5)",
        ("src/twindyn/losses/alignment.py:cross_resource_alignment_loss",),
        ("pair-03", "loss-03"),
    ),
    Claim(
        "C14",
        "L_traj requires the operator to predict the future positions of a trajectory from its past, with g the pooled summary and cos the similarity.",
        "Sec. 3.6, Eq. (6)",
        ("src/twindyn/losses/trajectory.py:trajectory_consistency_loss",),
        ("loss-04", "loss-05"),
    ),
    Claim(
        "C15",
        "Masking is applied at different densities because the same biological system appears at different observation densities across resources.",
        "Sec. 3.6",
        (
            "src/twindyn/data/masking.py:position_availability",
            "src/twindyn/data/schema.py:SampleRecord",
        ),
        ("mask-01", "mask-02", "data-01", "data-08"),
    ),
    Claim(
        "C16",
        "Identifiability: under a shared full-rank observation map, resource overlap and a separated spectrum, the state and the transition operator are identifiable up to an invertible linear transformation.",
        "Sec. 3.7, Assumptions 1-3, Theorem 1, Appendix 1",
        (
            "src/twindyn/models/msi.py:state_map_rank",
            "src/twindyn/models/scan.py:spectrum_separation",
            "src/twindyn/models/probe.py:invariance_gap",
        ),
        ("msi-01", "sst-07", "mech-01"),
    ),
    Claim(
        "C17",
        "The invariance of discrimination follows from the theorem: a ranking-based read-out is unchanged by the transformation the theorem leaves free.",
        "Sec. 3.7, Corollary 1",
        (
            "src/twindyn/models/tor.py:TrajectoryOutcomeReadout",
            "src/twindyn/metrics/concordance.py",
        ),
        ("metric-02", "mech-01"),
    ),
    Claim(
        "C18",
        "Algorithm 1: the representation phase samples (observation, resource) pairs, runs MSI then SST, evaluates the three terms and updates MSI and SST by gradient descent.",
        "Algorithm 1",
        ("src/twindyn/training/engine.py:train_representation",),
        ("train-01", "train-06", "train-07"),
    ),
    Claim(
        "C19",
        "Algorithm 2: inference runs MSI, then SST along the axis, pools the trajectory and returns a risk score.",
        "Algorithm 2",
        ("src/twindyn/training/engine.py:score", "src/twindyn/models/twindyn.py:TwinDyn.forward"),
        ("model-01", "train-01"),
    ),
    Claim(
        "C20",
        "The deconvolution configuration that produces the compartment ratios is fixed before any findings and is never tuned on an outcome, and its sensitivity is measured.",
        "Sec. 3.10, Sec. 4.4",
        ("src/twindyn/data/deconvolution.py",),
        ("deconv-01", "deconv-02", "deconv-03"),
    ),
    Claim(
        "C21",
        "The only quantities chosen on internal validation are the latent dimension, the masking ratio and the objective weights, with the read-out depth stated next to them.",
        "Sec. 3.10, Sec. 4.4",
        ("src/twindyn/evaluation/selection.py", "src/twindyn/cli/sweep.py"),
        ("reported-06", "reported-07", "loss-02"),
    ),
    Claim(
        "C22",
        "Random initialisation methods are predetermined and indexed, so a representation is learned once per initialisation and every read-out variant reuses it.",
        "Sec. 3.10, Sec. 4.4",
        (
            "src/twindyn/training/seeding.py:plan_initialisations",
            "src/twindyn/pipeline.py:run_experiment",
        ),
        ("ckpt-01", "ckpt-03", "ckpt-04"),
    ),
    Claim(
        "C23",
        "Four public resources are used: TCGA-KIRC for training and CPTAC-ccRCC, GSE29609 and E-MTAB-1980 as external cohorts from different repositories.",
        "Sec. 4.1, Data Availability Statement",
        ("src/twindyn/data/resources.py:RESOURCES",),
        ("data-03", "data-04"),
    ),
    Claim(
        "C24",
        "The shared feature space is the intersection of observable features rather than a union with imputation, and missing features cannot be filled in.",
        "Sec. 4.1",
        ("src/twindyn/data/feature_space.py:intersect_inventories",),
        ("fs-01", "fs-02"),
    ),
    Claim(
        "C25",
        "The splitting is at the level of tumour samples and zero overlap is verified rather than presumed.",
        "Sec. 4.1",
        ("src/twindyn/data/splits.py:assert_zero_overlap",),
        ("split-01", "split-02"),
    ),
    Claim(
        "C26",
        "The primary criterion is a paired external concordance difference of at least 0.03 against the strongest static composition baseline, assessed per resource.",
        "Sec. 4.2",
        ("src/twindyn/metrics/decision.py:primary_criterion",),
        ("metric-09", "metric-11", "reported-08"),
    ),
    Claim(
        "C27",
        "The external family holds three paired comparisons and the label-efficiency family four, each corrected with Holm-Bonferroni.",
        "Sec. 4.2",
        (
            "src/twindyn/metrics/correction.py:holm_bonferroni",
            "src/twindyn/metrics/decision.py:hypothesis_families",
        ),
        ("metric-06", "metric-07"),
    ),
    Claim(
        "C28",
        "Confidence intervals are 95% bootstrap intervals over 1,000 resamples and pairwise comparisons report an uncorrected and a corrected p-value.",
        "Sec. 4.4",
        ("src/twindyn/metrics/bootstrap.py",),
        ("metric-04", "metric-05"),
    ),
    Claim(
        "C29",
        "The predetermined decision margin is 0.005 concordance units, and a result below it counts as a lack of improvement.",
        "Sec. 4.4",
        ("src/twindyn/metrics/decision.py:decide",),
        ("metric-09", "metric-10"),
    ),
    Claim(
        "C30",
        "The main comparison spans 0.612 for a clinical Cox model to 0.731 for the static composition representation, with TwinDyn at 0.734 and a margin of 0.003 reported as parity.",
        "Table 1, Sec. 4.5",
        ("src/twindyn/models/baselines/inventory.py", "src/twindyn/reported.py"),
        ("reported-03", "table1-01"),
    ),
    Claim(
        "C31",
        "All rows of the main comparison are evaluated on the identical held-out sample set with identical preprocessing.",
        "Table 1 caption",
        (
            "src/twindyn/models/baselines/protocol.py:BaselineProtocol",
            "src/twindyn/pipeline.py:run_comparators",
        ),
        ("compare-01", "train-04"),
    ),
    Claim(
        "C32",
        "The parameter-matched control replaces the selective operator with a static encoder of matched count, so the drop is attributable to the transition law rather than to capacity.",
        "Table 2, Sec. 4.6",
        (
            "src/twindyn/models/static_transition.py",
            "src/twindyn/evaluation/sweeps.py:AblationVariant",
        ),
        ("sst-10", "mech-03"),
    ),
    Claim(
        "C33",
        "The full model reports 0.655 against 0.614 for the static control, and max drop over min drop is 0.041/0.003 = 13.7x with 92.7% margin retention at equal budget.",
        "Table 2, caption of the ablation",
        (
            "src/twindyn/evaluation/sweeps.py:assemble_ablation",
            "src/twindyn/metrics/decision.py:drop_ratio",
        ),
        ("metric-10", "reported-02"),
    ),
    Claim(
        "C34",
        "The clinical-covariate-only alignment variant is a designed null that falls below the prespecified margin and is reported as an ablation with no effect.",
        "Table 2, Sec. 4.6",
        ("src/twindyn/evaluation/sweeps.py", "src/twindyn/data/pairs.py"),
        ("pair-05",),
    ),
    Claim(
        "C35",
        "The stage profile across the external cohorts is non-monotone with a trough at stage II, and the grade profile is monotone from 0.702 to 0.612.",
        "Table 3, Sec. 4.7, Sec. 4.11",
        (
            "src/twindyn/metrics/strata.py:monotonicity",
            "src/twindyn/metrics/strata.py:stratum_rows",
        ),
        ("metric-12",),
    ),
    Claim(
        "C36",
        "Withholding the morphology features costs 0.004 and withholding the compartment fractions costs 0.033, against 0.039 for the strongest supervised multimodal baseline.",
        "Table 3, Sec. 4.7",
        (
            "src/twindyn/data/splits.py:modality_dropout",
            "src/twindyn/metrics/strata.py:dropout_rows",
        ),
        ("split-05", "data-08"),
    ),
    Claim(
        "C37",
        "Trajectory length is swept over 4/8/16/32 with a plateau after 16 and the state dimension over 16/32/64/128 with saturation after 64.",
        "Table 4, Sec. 4.8",
        (
            "src/twindyn/evaluation/sweeps.py:axis_length_sweep",
            "src/twindyn/evaluation/sweeps.py:plateau_point",
        ),
        ("axis-02", "reported-05", "reported-07"),
    ),
    Claim(
        "C38",
        "Shuffling the axis before training costs 0.037 concordance, more than any hyperparameter effect, so the ordering is load-bearing.",
        "Table 4, Sec. 4.8, Sec. 5.1",
        (
            "src/twindyn/data/axis.py:MicroenvironmentAxis.shuffled",
            "src/twindyn/evaluation/sweeps.py:shuffled_axis",
        ),
        ("axis-03",),
    ),
    Claim(
        "C39",
        "The alignment magnitude is reported per resource pair relative to its value at initialisation and decreases during training.",
        "Table 4, Sec. 4.8",
        (
            "src/twindyn/losses/alignment.py:relative_alignment",
            "src/twindyn/pipeline.py:relative_alignment",
        ),
        ("pair-03",),
        notes=(
            "Equation (5) is used unnormalised, so its raw relative magnitude follows the state's "
            "growth during training; the release reports it and marks the reading as a deviation."
        ),
    ),
    Claim(
        "C40",
        "Necessity: replacing the selective operator with the matched static control costs 0.041, the largest single drop.",
        "Table 5, Table 2, Sec. 4.9",
        ("src/twindyn/models/static_transition.py", "src/twindyn/evaluation/sweeps.py"),
        ("sst-10", "mech-03"),
    ),
    Claim(
        "C41",
        "Invariance: a linear probe recovers resource identity at 0.612 from the raw composition and only 0.271 from the learned state, against chance 0.25.",
        "Table 5, Sec. 4.9, Sec. 4.6",
        (
            "src/twindyn/models/probe.py:resource_probe",
            "src/twindyn/evaluation/mechanism.py:invariance_evidence",
        ),
        ("mech-01", "data-07"),
    ),
    Claim(
        "C42",
        "Perturbation response: noise injected early costs more than noise injected late, and the loss grows with the noise scale.",
        "Table 5, Sec. 4.9",
        (
            "src/twindyn/evaluation/mechanism.py:perturbation_evidence",
            "src/twindyn/models/twindyn.py:perturbed_risk",
        ),
        ("mech-02", "model-03"),
    ),
    Claim(
        "C43",
        "Resource-disjoint external validation reports 0.658 over 0.612, 0.667 over 0.628 and 0.641 over 0.601 on cohorts of 103, 39 and 101 samples.",
        "Table 6, Sec. 4.10",
        ("src/twindyn/evaluation/external.py:evaluate_external",),
        ("external-01", "table6-01"),
    ),
    Claim(
        "C44",
        "TwinDyn loses 0.076, 0.067 and 0.093 from the training resource while the static baselines lose 0.119, 0.103 and 0.130, a reduction of about one third.",
        "Sec. 4.10, Sec. 5.2, Conclusion",
        (
            "src/twindyn/metrics/transfer.py:TransferTax",
            "src/twindyn/metrics/transfer.py:tax_reduction",
        ),
        ("metric-14", "reported-01"),
    ),
    Claim(
        "C45",
        "At ten percent of the labels the margin is 0.076, at twenty-five percent 0.038, at fifty percent 0.012, and the no-label frozen representation still reaches 0.658 on an unseen resource.",
        "Sec. 4.10, Fig. 4 of the manuscript",
        (
            "src/twindyn/metrics/efficiency.py:efficiency_curve",
            "src/twindyn/pipeline.py:run_label_efficiency",
        ),
        ("metric-13", "split-03"),
    ),
    Claim(
        "C46",
        "The learned state carries information beyond stage and grade, with non-monotone within-stratum gains of 0.021, 0.036, 0.051 and 0.007.",
        "Sec. 4.11, Table 3",
        (
            "src/twindyn/metrics/strata.py:within_stratum_gain",
            "src/twindyn/pipeline.py:run_generalisation",
        ),
        ("metric-12",),
    ),
    Claim(
        "C47",
        "The ablation, the probe and the perturbation response together form the mechanism case, and the ablation family is descriptive rather than a corrected hypothesis family.",
        "Sec. 4.9, Sec. 4.2, Table 5",
        (
            "src/twindyn/metrics/decision.py:hypothesis_families",
            "src/twindyn/evaluation/reporting.py",
        ),
        ("metric-06", "reported-02"),
    ),
    Claim(
        "C48",
        "Everything the manuscript reports rests on public resources whose access conditions are declared, and the code release itself carries the third-party provenance.",
        "Data Availability Statement, Conflict of Interest Statement",
        ("dataset_urls.txt", "THIRD_PARTY_NOTICES.txt", "src/twindyn/data/resources.py"),
        ("data-03", "data-04"),
    ),
)


@dataclass
class ClaimRecord:
    claim: Claim
    check_status: dict[str, str] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if not self.claim.checks:
            return "UNVERIFIED"
        statuses = [
            self.check_status.get(identifier, "NOT_RUN") for identifier in self.claim.checks
        ]
        if all(status == "PASS" for status in statuses):
            return "PASS"
        if any(status == "PASS" for status in statuses) and any(
            status == "NOT_RUN" for status in statuses
        ):
            return "PARTIALLY_VERIFIED"
        if any(status == "PASS" for status in statuses):
            return "PARTIALLY_VERIFIED"
        if all(status == "NOT_RUN" for status in statuses):
            return "NOT_RUN"
        if any(status == "BLOCKED" for status in statuses):
            return "BLOCKED"
        return "FAIL"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.claim.identifier,
            "claim": self.claim.claim,
            "paper_location": self.claim.paper_location,
            "code": list(self.claim.code),
            "checks": list(self.claim.checks),
            "check_status": dict(self.check_status),
            "status": self.status,
            "notes": self.claim.notes,
        }


def claim_records() -> list[ClaimRecord]:
    return [ClaimRecord(claim=claim) for claim in CLAIMS]


def code_reference_exists(root: object, reference: str) -> bool:
    """Whether a ``path:symbol`` reference names a file that exists under ``root``."""
    from pathlib import Path

    base = Path(str(root))
    path_part = reference.split(":", 1)[0]
    return (base / path_part).exists()

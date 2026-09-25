"""The mechanism evidence triad.

Ref: Sec. 4.9 and Table 5 (three kinds of evidence: the necessity ablation, the
resource-classification probe applied to the raw composition and to the learned
state, and the perturbation response; a true dynamic object shows a monotone
degradation of the risk ordering with the size of the disturbance and with how
early it is applied, which a static encoder cannot produce), Sec. 4.6 (the
invariance objective is accomplished in a tangible way rather than just being
present).

This module produces the probe and perturbation arms. The necessity arm is the
matched-capacity ablation, which lives with the ablation table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from twindyn.data.dataset import AxisDataset, collate
from twindyn.metrics.concordance import concordance_index, pairwise_ranking_agreement
from twindyn.models.probe import ProbeConfig, ProbeResult, invariance_gap, resource_probe
from twindyn.models.twindyn import TwinDyn
from twindyn.utils.config import EvaluationConfig


@dataclass
class PerturbationResult:
    position: int
    scale: float
    c_index: float
    delta_vs_unperturbed: float
    ranking_agreement: float

    def as_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "scale": self.scale,
            "c_index": self.c_index,
            "delta": self.delta_vs_unperturbed,
            "ranking_agreement": self.ranking_agreement,
        }


@dataclass
class MechanismReport:
    raw_probe: dict[str, object]
    learned_probe: dict[str, object]
    invariance: dict[str, float]
    perturbation: list[PerturbationResult]
    unperturbed_c_index: float

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_probe": dict(self.raw_probe),
            "learned_probe": dict(self.learned_probe),
            "invariance": dict(self.invariance),
            "perturbation": [entry.as_dict() for entry in self.perturbation],
            "unperturbed_c_index": self.unperturbed_c_index,
        }


def pooled_state_features(model: TwinDyn, dataset: AxisDataset, device: torch.device) -> Tensor:
    loader = DataLoader(
        dataset, batch_size=min(128, max(1, len(dataset))), shuffle=False, collate_fn=collate
    )
    rows: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model.encode_representation(
                batch["positions"].to(device), batch["availability"].to(device)
            )
            states = output.states.mean(dim=0)
            rows.append(torch.cat([torch.real(states), torch.imag(states)], dim=-1).cpu())
    model.train()
    return torch.cat(rows) if rows else torch.zeros((0, 2 * model.config.state_dim))


def raw_composition_features(dataset: AxisDataset) -> Tensor:
    return dataset.anchors.clone()


def invariance_evidence(
    model: TwinDyn,
    dataset: AxisDataset,
    config: EvaluationConfig,
    device: torch.device | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, float]]:
    """The two probe rows of Table 5 and the gap between them."""
    device = device or torch.device("cpu")
    resources = dataset.view.resources
    labels = torch.where(resources < 0, torch.zeros_like(resources), resources)
    classes = int(torch.unique(labels).numel())
    probe_config = ProbeConfig(repeats=config.probe_repeats, seed=config.random_state)
    raw = resource_probe(raw_composition_features(dataset).float(), labels, classes, probe_config)
    learned = resource_probe(
        pooled_state_features(model, dataset, device), labels, classes, probe_config
    )
    return _probe_dict(raw), _probe_dict(learned), invariance_gap(raw, learned)


def _probe_dict(result: ProbeResult) -> dict[str, object]:
    return {
        "accuracy": result.accuracy,
        "interval": list(result.interval),
        "standard_error": result.standard_error,
        "repeats": result.repeats,
        "chance_level": result.chance_level,
        "classes": result.classes,
        "per_repeat": list(result.per_repeat),
    }


def perturbation_evidence(
    model: TwinDyn,
    dataset: AxisDataset,
    positions: dict[str, int],
    scales: dict[str, float],
    config: EvaluationConfig,
    device: torch.device | None = None,
) -> tuple[float, list[PerturbationResult]]:
    """Inject state noise at the stated positions and scales, then re-score."""
    device = device or torch.device("cpu")
    observations = dataset.positions.to(device)
    availability = dataset.availability.to(device)
    time = dataset.view.times.numpy().astype(np.float64)
    event = dataset.view.events.numpy().astype(np.float64)
    labelled = dataset.view.has_label.numpy().astype(bool)
    model.eval()
    results: list[PerturbationResult] = []
    with torch.no_grad():
        baseline = model(observations, availability).risk.cpu().numpy().astype(np.float64)
    unperturbed = concordance_index(baseline[labelled], time[labelled], event[labelled]).c_index
    for scale in scales.values():
        for position in positions.values():
            generator = torch.Generator().manual_seed(config.random_state)
            with torch.no_grad():
                risk = (
                    model.perturbed_risk(
                        observations, availability, int(position), float(scale), generator
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
            c_index = concordance_index(risk[labelled], time[labelled], event[labelled]).c_index
            results.append(
                PerturbationResult(
                    position=int(position),
                    scale=float(scale),
                    c_index=c_index,
                    delta_vs_unperturbed=c_index - unperturbed,
                    ranking_agreement=pairwise_ranking_agreement(
                        torch.tensor(baseline[labelled]), torch.tensor(risk[labelled])
                    ),
                )
            )
    model.train()
    return unperturbed, results


def monotonicity_of_response(results: list[PerturbationResult]) -> dict[str, object]:
    """Whether the degradation grows with the scale and shrinks with the position."""
    by_position: dict[int, list[PerturbationResult]] = {}
    for entry in results:
        by_position.setdefault(entry.position, []).append(entry)
    scale_monotone: dict[int, bool] = {}
    for position, entries in by_position.items():
        ordered = sorted(entries, key=lambda entry: entry.scale)
        values = [entry.delta_vs_unperturbed for entry in ordered]
        scale_monotone[position] = all(
            values[index + 1] <= values[index] for index in range(len(values) - 1)
        )
    positions = sorted(by_position)
    era_monotone = True
    if len(positions) > 1:
        largest = {
            position: max(entry.scale for entry in by_position[position]) for position in positions
        }
        comparable = [
            position for position in positions if largest[position] == max(largest.values())
        ]
        if len(comparable) > 1:
            values = [
                min(entry.delta_vs_unperturbed for entry in by_position[position])
                for position in comparable
            ]
            era_monotone = values[-1] >= values[0]
    return {
        "monotone_in_scale": scale_monotone,
        "later_positions_degrade_less": era_monotone,
        "positions": positions,
    }


def collect_mechanism(
    model: TwinDyn,
    dataset: AxisDataset,
    config: EvaluationConfig,
    positions: dict[str, int],
    scales: dict[str, float],
    device: torch.device | None = None,
) -> MechanismReport:
    raw, learned, gap = invariance_evidence(model, dataset, config, device)
    unperturbed, perturbation = perturbation_evidence(
        model, dataset, positions, scales, config, device
    )
    return MechanismReport(
        raw_probe=raw,
        learned_probe=learned,
        invariance=gap,
        perturbation=perturbation,
        unperturbed_c_index=unperturbed,
    )


def summary_arrays(report: MechanismReport) -> dict[str, float]:
    return {
        "raw_probe_accuracy": float(report.raw_probe["accuracy"]),
        "learned_probe_accuracy": float(report.learned_probe["accuracy"]),
        "chance_level": float(report.raw_probe["chance_level"]),
        "readability_removed": float(report.invariance["readability_removed"]),
        "unperturbed_c_index": float(report.unperturbed_c_index),
        "mean_perturbation_delta": float(
            np.mean([entry.delta_vs_unperturbed for entry in report.perturbation])
        )
        if report.perturbation
        else float("nan"),
    }

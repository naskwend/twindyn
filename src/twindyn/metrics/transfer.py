"""Transfer tax accounting.

Ref: Sec. 4.10 (the static baselines weaken from the training resource to the
external cohorts by 0.119, 0.103 and 0.130, reproducing the size of the transfer
tax noted for the strongest published kidney-cancer model, while TwinDyn loses
0.076, 0.067 and 0.093, a reduction of about one third rather than an elimination),
Sec. 5.2 (the transfer loss varies across cohorts rather than shifting uniformly
and is reported as a range rather than a single figure).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransferTax:
    method: str
    training_value: float
    external_values: dict[str, float]

    @property
    def losses(self) -> dict[str, float]:
        return {key: self.training_value - value for key, value in self.external_values.items()}

    @property
    def mean_loss(self) -> float:
        values = self.losses
        return sum(values.values()) / len(values) if values else float("nan")

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "training_value": self.training_value,
            "external_values": dict(self.external_values),
            "losses": self.losses,
            "mean_loss": self.mean_loss,
        }


def tax_reduction(baseline: TransferTax, candidate: TransferTax) -> dict[str, object]:
    """Per-cohort reduction in the transfer tax, and its share of the baseline tax."""
    if baseline.external_values.keys() != candidate.external_values.keys():
        raise ValueError("both methods must be evaluated on the same cohorts")
    per_cohort: dict[str, dict[str, float]] = {}
    for key in baseline.external_values:
        base_loss = baseline.losses[key]
        candidate_loss = candidate.losses[key]
        per_cohort[key] = {
            "baseline_loss": base_loss,
            "candidate_loss": candidate_loss,
            "reduction": base_loss - candidate_loss,
            "reduction_share": (base_loss - candidate_loss) / base_loss
            if base_loss
            else float("nan"),
        }
    shares = [
        entry["reduction_share"]
        for entry in per_cohort.values()
        if entry["reduction_share"] == entry["reduction_share"]
    ]
    return {
        "per_cohort": per_cohort,
        "mean_reduction_share": sum(shares) / len(shares) if shares else float("nan"),
        "baseline_range": (
            min(baseline.losses.values()) if baseline.losses else float("nan"),
            max(baseline.losses.values()) if baseline.losses else float("nan"),
        ),
        "candidate_range": (
            min(candidate.losses.values()) if candidate.losses else float("nan"),
            max(candidate.losses.values()) if candidate.losses else float("nan"),
        ),
    }


def retained_share(baseline: TransferTax, candidate: TransferTax) -> dict[str, float]:
    """The share of the external value the candidate keeps where the baseline loses it."""
    out: dict[str, float] = {}
    for key in baseline.external_values:
        base_loss = baseline.losses[key]
        candidate_loss = candidate.losses[key]
        out[key] = candidate_loss / base_loss if base_loss else float("nan")
    return out

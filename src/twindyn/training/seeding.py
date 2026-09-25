"""Initialisation planning.

Ref: Sec. 3.10 and Sec. 4.4 (random initialisation methods are predetermined and
indexed, so representation learning runs once for each initialisation and every
read-out variant reuses it in a paired fashion).
"""

from __future__ import annotations

from dataclasses import dataclass

from twindyn.registry import INITIALISATION_COUNT
from twindyn.utils.runtime import initialisation_seeds, set_seed


@dataclass(frozen=True)
class InitialisationPlan:
    """The indexed seeds every paired comparison is built on."""

    seeds: tuple[int, ...]
    base_seed: int

    @property
    def count(self) -> int:
        return len(self.seeds)

    def seed(self, index: int) -> int:
        if index < 0 or index >= self.count:
            raise IndexError(f"initialisation index {index} outside the plan")
        return self.seeds[index]

    def as_dict(self) -> dict[str, object]:
        return {"base_seed": self.base_seed, "count": self.count, "seeds": list(self.seeds)}


def plan_initialisations(base_seed: int, count: int | None = None) -> InitialisationPlan:
    resolved = INITIALISATION_COUNT if count is None else int(count)
    if resolved <= 0:
        raise ValueError("the initialisation count must be positive")
    return InitialisationPlan(seeds=initialisation_seeds(base_seed, resolved), base_seed=base_seed)


def apply_initialisation(plan: InitialisationPlan, index: int, deterministic: bool = True) -> int:
    seed = plan.seed(index)
    set_seed(seed, deterministic=deterministic)
    return seed


__all__ = ["InitialisationPlan", "apply_initialisation", "plan_initialisations", "set_seed"]

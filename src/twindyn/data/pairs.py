"""Cross-resource sample matching for the alignment term.

Ref: Sec. 3.3 (associations are determined by the goal rather than by the data,
using the concept of matching samples across resources; when a resource pair
shares no common samples the alignment term stays unspecified for that pair and
contributes only to the other objectives), Sec. 3.6 (equation 5 is taken over
pairs drawn from different resources), Assumption 2 (at least two resources
provide matched observations occupying overlapping regions of the latent state
space), Sec. 4.6 (the clinical-covariate-only alignment variant is a designed
null that falls below the prespecified margin).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor


@dataclass
class PairSet:
    """Matched index pairs between two distinct resources."""

    left_resource: str
    right_resource: str
    left_index: Tensor
    right_index: Tensor
    similarity: Tensor = field(default_factory=lambda: torch.zeros(0))

    @property
    def available(self) -> bool:
        return int(self.left_index.numel()) > 0

    def __len__(self) -> int:
        return int(self.left_index.numel())


def mutual_nearest_pairs(
    left: Tensor,
    right: Tensor,
    max_pairs: int,
    min_similarity: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Mutually nearest neighbours in cosine similarity, symmetrised by construction."""
    if left.dim() != 2 or right.dim() != 2:
        raise ValueError("anchor matrices must be two-dimensional")
    if left.shape[1] != right.shape[1]:
        raise ValueError("anchor spaces must share a dimensionality")
    left_norm = torch.nn.functional.normalize(left, dim=1)
    right_norm = torch.nn.functional.normalize(right, dim=1)
    similarity = left_norm @ right_norm.t()
    forward = similarity.argmax(dim=1)
    backward = similarity.argmax(dim=0)
    left_ids: list[int] = []
    right_ids: list[int] = []
    scores: list[float] = []
    for left_id in range(left.shape[0]):
        right_id = int(forward[left_id])
        if int(backward[right_id]) != left_id:
            continue
        score = float(similarity[left_id, right_id])
        if score < min_similarity:
            continue
        left_ids.append(left_id)
        right_ids.append(right_id)
        scores.append(score)
    if len(left_ids) > max_pairs:
        order = np.argsort(np.asarray(scores))[::-1][:max_pairs]
        left_ids = [left_ids[int(i)] for i in order]
        right_ids = [right_ids[int(i)] for i in order]
        scores = [scores[int(i)] for i in order]
    return (
        torch.tensor(left_ids, dtype=torch.long),
        torch.tensor(right_ids, dtype=torch.long),
        torch.tensor(scores, dtype=torch.float32),
    )


def batch_alignment_pairs(
    anchors: Tensor,
    resources: Tensor,
    max_pairs_per_resource: int,
    min_similarity: float = 0.0,
    clinical_only: bool = False,
) -> list[PairSet]:
    """Matched pairs inside one batch, with indices local to that batch.

    The matching runs per step over the samples actually present in the batch, so
    the alignment term sees pairs of samples that the loss can differentiate
    through. Section 3.3 makes the same point about the term remaining unspecified
    for a pair of resources with no matched samples in scope.
    """
    if anchors.dim() != 2:
        raise ValueError("anchors must be (batch, anchor_dim)")
    if resources.shape[0] != anchors.shape[0]:
        raise ValueError("resources must label every anchor row")
    grouped: dict[str, list[int]] = {}
    for position, value in enumerate(resources.tolist()):
        grouped.setdefault(str(int(value)), []).append(position)
    keys = sorted(grouped)
    pairs: list[PairSet] = []
    for position, left_key in enumerate(keys):
        for right_key in keys[position + 1 :]:
            left_rows = torch.tensor(grouped[left_key], dtype=torch.long)
            right_rows = torch.tensor(grouped[right_key], dtype=torch.long)
            left_anchor = anchors[left_rows]
            right_anchor = anchors[right_rows]
            if min(left_rows.numel(), right_rows.numel()) < 1:
                pairs.append(
                    PairSet(
                        left_key,
                        right_key,
                        torch.zeros(0, dtype=torch.long),
                        torch.zeros(0, dtype=torch.long),
                    )
                )
                continue
            if clinical_only:
                left_anchor = _clinical_view(left_anchor)
                right_anchor = _clinical_view(right_anchor)
            left_local, right_local, scores = mutual_nearest_pairs(
                left_anchor, right_anchor, max_pairs_per_resource, min_similarity
            )
            pairs.append(
                PairSet(
                    left_resource=left_key,
                    right_resource=right_key,
                    left_index=left_rows[left_local],
                    right_index=right_rows[right_local],
                    similarity=scores,
                )
            )
    return pairs


def build_alignment_pairs(
    anchors: dict[str, Tensor],
    indices: dict[str, list[int]],
    max_pairs_per_resource: int,
    min_similarity: float = 0.0,
    clinical_only: bool = False,
) -> list[PairSet]:
    """One :class:`PairSet` per unordered resource pair.

    A pair set with no matched samples is returned as an available-false entry so
    the caller can report which resource pairs contributed to the objective.
    """
    keys = sorted(anchors)
    pairs: list[PairSet] = []
    for position, left_key in enumerate(keys):
        for right_key in keys[position + 1 :]:
            left_anchor = anchors[left_key]
            right_anchor = anchors[right_key]
            if clinical_only:
                left_anchor = _clinical_view(left_anchor)
                right_anchor = _clinical_view(right_anchor)
            left_ids, right_ids, scores = mutual_nearest_pairs(
                left_anchor, right_anchor, max_pairs_per_resource, min_similarity
            )
            pairs.append(
                PairSet(
                    left_resource=left_key,
                    right_resource=right_key,
                    left_index=torch.tensor(
                        [indices[left_key][int(i)] for i in left_ids], dtype=torch.long
                    ),
                    right_index=torch.tensor(
                        [indices[right_key][int(i)] for i in right_ids], dtype=torch.long
                    ),
                    similarity=scores,
                )
            )
    return pairs


def _clinical_view(anchor: Tensor) -> Tensor:
    """The designed null: keep only the leading clinical covariates."""
    width = min(2, anchor.shape[1])
    view = torch.zeros_like(anchor)
    view[:, :width] = anchor[:, :width]
    return view


def alignment_magnitude(
    states: Tensor,
    pairs: list[PairSet],
    reference: Tensor | None = None,
) -> dict[str, float]:
    """Mean squared state distance per resource pair, optionally relative to a reference."""
    out: dict[str, float] = {}
    for pair in pairs:
        key = f"{pair.left_resource}|{pair.right_resource}"
        if not pair.available:
            out[key] = float("nan")
            continue
        left = states[pair.left_index]
        right = states[pair.right_index]
        difference = left - right
        value = float((difference.conj() * difference).real.sum(dim=1).mean().detach())
        if reference is not None:
            baseline = float(reference.get(key, 0.0)) if isinstance(reference, dict) else 0.0
            value = value / baseline if baseline else value
        out[key] = value
    return out


def pairs_to_index(
    pairs: list[PairSet], device: torch.device | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Flatten the available pair sets into left/right index tensors plus a pair mask."""
    lefts: list[Tensor] = []
    rights: list[Tensor] = []
    tags: list[int] = []
    for tag, pair in enumerate(pairs):
        if not pair.available:
            continue
        lefts.append(pair.left_index)
        rights.append(pair.right_index)
        tags.append(tag)
    if not lefts:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty, empty
    left = torch.cat(lefts).to(device)
    right = torch.cat(rights).to(device)
    mask = torch.ones(left.shape[0], dtype=torch.bool, device=device)
    return left, right, mask


def pair_resource_tags(pairs: list[PairSet]) -> list[str]:
    return [f"{pair.left_resource}|{pair.right_resource}" for pair in pairs]

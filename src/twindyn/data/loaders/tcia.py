"""Diagnostic whole-slide images and the collection access conditions.

Ref: Data Availability Statement (diagnostic whole-slide images from The Cancer
Imaging Archive; the CPTAC-ccRCC imaging collection is version 14, released
2025-07-07, DOI 10.7937/k9/tcia.2018.oblamn27, under CC BY 4.0), Sec. 3.10 and
Sec. 4.4 (the ordered axis is fixed before training), Table 3 (withholding the
morphology features at test time).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from twindyn.data.loaders.tabular import (
    ResourcePresence,
    TableBlock,
    locate,
    read_delimited,
)

TCIA_PATTERNS: dict[str, tuple[str, ...]] = {
    "series_manifest": ("*Series*.csv", "*series*.csv", "**/*series*.csv"),
    "study_manifest": ("*Study*.csv", "**/*study*.csv"),
    "patient_manifest": ("*Patient*.csv", "**/*patient*.csv"),
    "slides": ("**/*.svs", "**/*.tiff", "**/*.ndpi"),
    "annotations": ("**/*.xml", "**/*geojson"),
}


@dataclass(frozen=True)
class CollectionAccess:
    """The declared access conditions of an imaging collection."""

    collection: str
    version: str | None
    release_date: str | None
    doi: str | None
    licence: str | None
    requires_application: bool
    notes: str


TCIA_COLLECTIONS: dict[str, CollectionAccess] = {
    "cptac_ccrcc": CollectionAccess(
        collection="CPTAC-ccRCC",
        version="14",
        release_date="2025-07-07",
        doi="10.7937/k9/tcia.2018.oblamn27",
        licence="CC BY 4.0",
        requires_application=False,
        notes="Imaging collection served openly; the PDC proteomic arm is matched by case identifier.",
    ),
    "tcga_kirc": CollectionAccess(
        collection="TCGA-KIRC diagnostic slides",
        version=None,
        release_date=None,
        doi=None,
        licence=None,
        requires_application=False,
        notes="Diagnostic slides are distributed through the GDC open tier alongside the molecular data.",
    ),
}


@dataclass
class CancerImagingArchive:
    """A reader for a locally staged TCIA manifest and slide tree."""

    root: Path
    collection: str = "cptac_ccrcc"

    def presence(self) -> ResourcePresence:
        found, missing = locate(self.root, TCIA_PATTERNS)
        return ResourcePresence(self.collection, self.root, found, missing)

    def access(self) -> CollectionAccess:
        return TCIA_COLLECTIONS[self.collection]

    def series_table(self) -> TableBlock:
        presence = self.presence()
        if "series_manifest" not in presence.found:
            raise FileNotFoundError("no TCIA series manifest under the resource root")
        return read_delimited(Path(presence.found["series_manifest"]))

    def case_uids(self) -> list[str]:
        block = self.series_table()
        column = block.find_column("PatientID", "Patient ID", "PatientId")
        if column is None:
            raise ValueError("series manifest lacks a patient identifier")
        return [row[block.header.index(column)] for row in block.rows]

    def slide_index(self) -> dict[str, list[str]]:
        """Map case identifiers to the slide files staged on disk."""
        out: dict[str, list[str]] = {}
        for path in self.root.glob("**/*"):
            if path.suffix.lower() not in {".svs", ".tiff", ".ndpi"}:
                continue
            case = _case_from_filename(path.name)
            if case is None:
                continue
            out.setdefault(case, []).append(str(path))
        return out

    def modality_availability(self) -> dict[str, bool]:
        presence = self.presence()
        return {
            "imaging": "slides" in presence.found,
            "series_manifest": "series_manifest" in presence.found,
            "annotations": "annotations" in presence.found,
        }


def _case_from_filename(name: str) -> str | None:
    stem = name.split(".")[0]
    parts = stem.split("-")
    if len(parts) >= 3 and parts[0].upper().startswith("C3"):
        return "-".join(parts[:3])
    if len(parts) >= 2 and parts[0].startswith("TCGA"):
        return "-".join(parts[:3]) if len(parts) >= 3 else stem
    return None


def write_manifest_template(path: Path, columns: list[str]) -> Path:
    """Emit an empty manifest header so a staged tree can be described uniformly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
    return path

"""CPTAC-ccRCC through the Proteomic Data Commons and Cancer Imaging Archive.

Ref: Data Availability Statement (the first external cohort is the Clinical
Proteomic Tumor Analysis Consortium clear cell renal cell carcinoma collection,
available from The Cancer Imaging Archive under CC BY 4.0 with matched proteomic
and clinical data in the Proteomic Data Commons; the imaging collection is
version 14, released 2025-07-07, DOI 10.7937/k9/tcia.2018.oblamn27).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twindyn.data.loaders.tabular import (
    ResourcePresence,
    locate,
    parse_event,
    parse_grade,
    parse_stage,
    parse_survival_value,
    read_delimited,
)

PDC_PATTERNS: dict[str, tuple[str, ...]] = {
    "proteome": ("*_proteome*.tsv", "**/*proteome*.tsv", "*_proteome*.csv"),
    "phosphoproteome": ("*_phosphoproteome*.tsv", "**/*phosphoproteome*.tsv"),
    "biospecimen": (
        "*biospecimen*.tsv",
        "**/*biospecimen*.tsv",
        "*cases*.tsv",
        "**/*clinical*.tsv",
    ),
    "manifest": ("*manifest*.csv", "*manifest*.tsv", "**/*manifest*.csv"),
}

ALIQUOT_COLUMN_HINTS = (
    "CPTAC_CCRCC",
    "CPTAC-CCRCC",
    "aliquot",
    "Aliquot",
    "Sample",
)


@dataclass
class CptacCcrcc:
    """A reader for a locally staged PDC proteomic archive of CPTAC-ccRCC."""

    root: Path

    def presence(self) -> ResourcePresence:
        found, missing = locate(self.root, PDC_PATTERNS)
        return ResourcePresence("cptac_ccrcc", self.root, found, missing)

    def proteome(self) -> tuple[list[str], np.ndarray, list[str]]:
        """Protein identifiers, a sample-major abundance matrix and aliquot identifiers."""
        presence = self.presence()
        if "proteome" not in presence.found:
            raise FileNotFoundError("no PDC proteome table under the resource root")
        block = read_delimited(Path(presence.found["proteome"]))
        gene_column = block.find_column("Gene", "gene", "Gene_Name", "gene_name")
        identifiers = (
            [row[block.header.index(gene_column)] for row in block.rows]
            if gene_column is not None
            else [row[0] for row in block.rows]
        )
        first_aliquot = _first_aliquot_column(block.header)
        if first_aliquot is None:
            raise ValueError("proteome table has no aliquot intensity columns")
        samples = block.header[first_aliquot:]
        rows = []
        for row in block.rows:
            values = []
            for raw in row[first_aliquot:]:
                try:
                    values.append(float(raw))
                except ValueError:
                    values.append(float("nan"))
            rows.append(values)
        matrix = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(samples)))
        return identifiers, matrix.T, samples

    def biospecimen(self) -> tuple[list[str], dict[str, dict[str, str]]]:
        presence = self.presence()
        if "biospecimen" not in presence.found:
            raise FileNotFoundError("no PDC clinical or biospecimen table under the resource root")
        block = read_delimited(Path(presence.found["biospecimen"]))
        case_column = block.find_column("Case ID", "case_id", "case", "Participant ID")
        if case_column is None:
            raise ValueError("clinical table lacks a case identifier")
        cases: list[str] = []
        fields: dict[str, dict[str, str]] = {}
        for row in block.rows:
            case = row[block.header.index(case_column)]
            cases.append(case)
            fields[case] = {column: row[block.header.index(column)] for column in block.header}
        return cases, fields

    def survival(self) -> dict[str, tuple[float, bool]]:
        cases, fields = self.biospecimen()
        out: dict[str, tuple[float, bool]] = {}
        for case in cases:
            entry = fields[case]
            time_raw = _pick(
                entry, ("survival_time", "days_to_death", "days_to_last_followup", "os_time")
            )
            event_raw = _pick(entry, ("vital_status", "event", "os_event"))
            if time_raw is None:
                continue
            parsed = parse_survival_value(time_raw)
            if parsed is None or parsed <= 0.0:
                continue
            flag = parse_event(event_raw) if event_raw is not None else None
            out[case] = (float(parsed), bool(flag))
        return out

    def strata(self) -> dict[str, tuple[int | None, int | None]]:
        cases, fields = self.biospecimen()
        out: dict[str, tuple[int | None, int | None]] = {}
        for case in cases:
            entry = fields[case]
            stage_raw = _pick(entry, ("tumor_stage", "ajcc_pathologic_stage", "stage"))
            grade_raw = _pick(entry, ("tumor_grade", "grade", "isup_grade"))
            out[case] = (
                parse_stage(stage_raw) if stage_raw is not None else None,
                parse_grade(grade_raw) if grade_raw is not None else None,
            )
        return out


def _first_aliquot_column(header: list[str]) -> int | None:
    reserved = {
        "Proteome",
        "Gene",
        "gene",
        "Gene_Name",
        "NCBIGeneID",
        "Uniprot",
        "UniprotID",
        "Protein",
        "Protein_ID",
        "Organism",
        "Chromosome",
        "Number_of_Peptides",
        "Number_of_Unique_Peptides",
        "Number_of_PSMs",
    }
    for position, column in enumerate(header):
        if column in reserved:
            continue
        if any(hint in column for hint in ALIQUOT_COLUMN_HINTS):
            return position
    for position, column in enumerate(header):
        if column not in reserved and not column.lower().startswith(("x", "id")):
            return position
    return None


def _pick(fields: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in fields:
            return fields[key]
    for candidate, value in fields.items():
        lowered = candidate.lower()
        for key in keys:
            if key.lower() in lowered:
                return value
    return None

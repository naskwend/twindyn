"""TCGA-KIRC through the Genomic Data Commons open tier.

Ref: Data Availability Statement (the training resource is the kidney renal clear
cell carcinoma cohort of The Cancer Genome Atlas, available from the National
Cancer Institute Genomic Data Commons under project accession TCGA-KIRC, with
diagnostic whole-slide images from The Cancer Imaging Archive).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twindyn.data.loaders.tabular import (
    ResourcePresence,
    TableBlock,
    locate,
    parse_event,
    parse_float_matrix,
    parse_grade,
    parse_stage,
    parse_survival_value,
    read_delimited,
)

MANIFEST_PATTERNS: dict[str, tuple[str, ...]] = {
    "manifest": (
        "gdc_sample_sheet*.tsv",
        "gdc_sample_sheet*.csv",
        "sample_sheet*.tsv",
        "**/gdc_sample_sheet*.tsv",
    ),
    "clinical": ("clinical*.tsv", "clinical*.csv", "**/clinical*.tsv", "**/clinical*.csv"),
    "clinical_json": ("clinical*.json", "**/clinical*.json"),
    "star_counts": ("**/*star_gene_counts.tsv", "**/*augmented_star_gene_counts.tsv"),
    "htseq_counts": ("**/*htseq.counts", "**/*htseq.counts.gz"),
    "wsi": ("**/*.svs", "**/*.tiff"),
}

CLINICAL_COLUMNS = {
    "case_id": ("case_id", "Case ID", "submitter_id"),
    "vital": ("vital_status", "Vital Status"),
    "days_death": ("days_to_death", "Days to Death"),
    "days_last": ("days_to_last_follow_up", "Days to Last Follow up"),
    "stage": ("ajcc_pathologic_stage", "tumor_stage", "Stage"),
    "grade": ("ajcc_pathologic_grade", "neoplasm_histologic_grade", "Grade"),
}

MANIFEST_COLUMNS = {
    "case_id": ("Case ID", "Case_ID"),
    "sample_id": ("Sample ID", "Sample_ID"),
    "sample_type": ("Sample Type", "Sample_Type"),
    "file_name": ("File Name", "File_Name"),
    "data_category": ("Data Category", "Data_Category"),
    "project": ("Project ID", "Project_ID"),
}


@dataclass
class GdcKirc:
    """A reader for a locally staged GDC download of TCGA-KIRC."""

    root: Path

    def presence(self) -> ResourcePresence:
        found, missing = locate(self.root, MANIFEST_PATTERNS)
        return ResourcePresence("tcga_kirc", self.root, found, missing)

    def manifest(self) -> TableBlock:
        presence = self.presence()
        if "manifest" not in presence.found:
            raise FileNotFoundError("no GDC sample sheet under the resource root")
        return read_delimited(Path(presence.found["manifest"]))

    def clinical(self) -> TableBlock:
        presence = self.presence()
        if "clinical" not in presence.found:
            raise FileNotFoundError("no GDC clinical table under the resource root")
        return read_delimited(Path(presence.found["clinical"]))

    def expression(self) -> tuple[list[str], np.ndarray, list[str]]:
        """Gene-length-normalised counts, one row per sample, plus gene identifiers."""
        presence = self.presence()
        files: list[Path] = []
        if "star_counts" in presence.found:
            files = sorted(self.root.glob("**/*star_gene_counts.tsv"))
        elif "htseq_counts" in presence.found:
            files = sorted(self.root.glob("**/*htseq.counts"))
            if not files:
                files = sorted(self.root.glob("**/*htseq.counts.gz"))
        if not files:
            raise FileNotFoundError("no expression files under the resource root")
        genes: list[str] = []
        rows: list[np.ndarray] = []
        samples: list[str] = []
        for path in files:
            block = read_delimited(path)
            gene_column = block.find_column("gene_name", "gene_id", "Ensembl_ID")
            if gene_column is None:
                continue
            index = block.header.index(gene_column)
            count_column = block.find_column("unstranded", "count", "tpm_unstranded", "FPKM")
            if count_column is None:
                continue
            count_index = block.header.index(count_column)
            identifiers = [row[index] for row in block.rows]
            values = np.asarray(
                [
                    _safe_float(row[count_index]) if count_index < len(row) else np.nan
                    for row in block.rows
                ],
                dtype=np.float64,
            )
            if not genes:
                genes = identifiers
            rows.append(values)
            samples.append(path.name.split(".")[0])
        if not rows:
            raise ValueError("expression files carried no usable count column")
        matrix = np.vstack(rows)
        return genes, matrix, samples

    def survival(self) -> dict[str, tuple[float, bool]]:
        block = self.clinical()
        case_column = block.find_column(*CLINICAL_COLUMNS["case_id"])
        vital_column = block.find_column(*CLINICAL_COLUMNS["vital"])
        death_column = block.find_column(*CLINICAL_COLUMNS["days_death"])
        last_column = block.find_column(*CLINICAL_COLUMNS["days_last"])
        if case_column is None:
            raise ValueError("clinical table lacks a case identifier")
        out: dict[str, tuple[float, bool]] = {}
        for row in block.rows:
            case = row[block.header.index(case_column)]
            vital = row[block.header.index(vital_column)] if vital_column else ""
            event = parse_event(vital)
            death = (
                parse_survival_value(row[block.header.index(death_column)])
                if death_column
                else None
            )
            last = (
                parse_survival_value(row[block.header.index(last_column)]) if last_column else None
            )
            if event is None:
                event = death is not None
            time = death if (event and death is not None) else last
            if time is None or time <= 0.0:
                continue
            out[case] = (float(time), bool(event))
        return out

    def strata(self) -> dict[str, tuple[int | None, int | None]]:
        block = self.clinical()
        case_column = block.find_column(*CLINICAL_COLUMNS["case_id"])
        stage_column = block.find_column(*CLINICAL_COLUMNS["stage"])
        grade_column = block.find_column(*CLINICAL_COLUMNS["grade"])
        if case_column is None:
            raise ValueError("clinical table lacks a case identifier")
        out: dict[str, tuple[int | None, int | None]] = {}
        for row in block.rows:
            case = row[block.header.index(case_column)]
            stage = parse_stage(row[block.header.index(stage_column)]) if stage_column else None
            grade = parse_grade(row[block.header.index(grade_column)]) if grade_column else None
            out[case] = (stage, grade)
        return out

    def tumour_samples(self) -> list[str]:
        block = self.manifest()
        sample_column = block.find_column(*MANIFEST_COLUMNS["sample_id"])
        type_column = block.find_column(*MANIFEST_COLUMNS["sample_type"])
        if sample_column is None:
            raise ValueError("sample sheet lacks a sample identifier")
        selected: list[str] = []
        for row in block.rows:
            sample_type = row[block.header.index(type_column)] if type_column else "Primary Tumor"
            if "tumor" not in sample_type.lower():
                continue
            selected.append(row[block.header.index(sample_column)])
        return selected

    def gene_matrix_block(self, block: TableBlock) -> tuple[list[str], np.ndarray]:
        return parse_float_matrix(block, 1)


def _safe_float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        return float("nan")

"""E-MTAB-1980 through the ArrayExpress/BioStudies file layout.

Ref: Data Availability Statement (the third external cohort is ArrayExpress study
E-MTAB-1980, openly released since 2013-10-16).
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
    parse_grade,
    parse_stage,
    parse_survival_value,
    read_delimited,
)

SDRF_PATTERNS: dict[str, tuple[str, ...]] = {
    "sdrf": ("E-MTAB-1980.sdrf.txt", "*.sdrf.txt", "**/*.sdrf.txt"),
    "processed": (
        "E-MTAB-1980-processed-data-*.txt",
        "**/E-MTAB-1980-processed*.txt",
        "**/*processed-data*.txt",
    ),
    "idf": ("E-MTAB-1980.idf.txt", "*.idf.txt", "**/*.idf.txt"),
    "mage": ("*.MAGE-TAB.zip", "**/*.MAGE-TAB.zip"),
}

SDRF_COLUMNS = {
    "source": ("Source Name", "source name"),
    "stage": (
        "Characteristics[stage]",
        "Characteristics[clinical information]",
        "Characteristics[tumour stage]",
    ),
    "grade": ("Characteristics[grade]", "Characteristics[tumour grade]"),
    "survival": (
        "Characteristics[survival time]",
        "Characteristics[overall survival]",
        "Characteristics[event free survival time]",
    ),
    "event": ("Characteristics[event]", "Characteristics[status]"),
    "age": ("Characteristics[age]",),
    "sex": ("Characteristics[sex]",),
}


@dataclass
class ArrayExpressStudy:
    """A reader for the MAGE-TAB layout an ArrayExpress study publishes."""

    root: Path
    accession: str = "E-MTAB-1980"

    def presence(self) -> ResourcePresence:
        found, missing = locate(self.root, SDRF_PATTERNS)
        return ResourcePresence("emtab1980", self.root, found, missing)

    def sdrf(self) -> TableBlock:
        presence = self.presence()
        if "sdrf" not in presence.found:
            raise FileNotFoundError("no SDRF file under the resource root")
        return read_delimited(Path(presence.found["sdrf"]))

    def processed_matrix(self) -> tuple[list[str], np.ndarray, list[str]]:
        """Feature identifiers, a sample-major matrix and sample identifiers."""
        presence = self.presence()
        if "processed" not in presence.found:
            raise FileNotFoundError("no processed data matrix under the resource root")
        block = read_delimited(Path(presence.found["processed"]))
        identifier_column = block.find_column("Reporter Ref", "Reporter Identifier", "ID", "Gene")
        start = 0 if identifier_column is None else block.header.index(identifier_column) + 1
        identifiers = (
            [row[0] for row in block.rows]
            if identifier_column is None
            else [row[block.header.index(identifier_column)] for row in block.rows]
        )
        matrix_rows = []
        for row in block.rows:
            values = []
            for raw in row[start:]:
                try:
                    values.append(float(raw))
                except ValueError:
                    values.append(float("nan"))
            matrix_rows.append(values)
        matrix = np.asarray(matrix_rows, dtype=np.float64) if matrix_rows else np.zeros((0, 0))
        samples = block.header[start:]
        return identifiers, matrix.T, samples

    def sample_fields(self) -> dict[str, dict[str, str]]:
        block = self.sdrf()
        source_column = block.find_column(*SDRF_COLUMNS["source"])
        if source_column is None:
            raise ValueError("SDRF lacks a source name column")
        out: dict[str, dict[str, str]] = {}
        for row in block.rows:
            source = row[block.header.index(source_column)]
            entry = out.setdefault(source, {})
            for column in block.header:
                if column.startswith("Characteristics[") or column.startswith("Factor Value["):
                    entry[column] = row[block.header.index(column)]
        return out

    def survival(self) -> dict[str, tuple[float, bool]]:
        fields = self.sample_fields()
        out: dict[str, tuple[float, bool]] = {}
        for source, entry in fields.items():
            time_raw = _pick(entry, SDRF_COLUMNS["survival"])
            event_raw = _pick(entry, SDRF_COLUMNS["event"])
            if time_raw is None:
                continue
            parsed = parse_survival_value(time_raw)
            if parsed is None or parsed <= 0.0:
                continue
            flag = parse_event(event_raw) if event_raw is not None else None
            out[source] = (float(parsed), bool(flag))
        return out

    def strata(self) -> dict[str, tuple[int | None, int | None]]:
        fields = self.sample_fields()
        out: dict[str, tuple[int | None, int | None]] = {}
        for source, entry in fields.items():
            stage_raw = _pick(entry, SDRF_COLUMNS["stage"])
            grade_raw = _pick(entry, SDRF_COLUMNS["grade"])
            out[source] = (
                parse_stage(stage_raw) if stage_raw is not None else None,
                parse_grade(grade_raw) if grade_raw is not None else None,
            )
        return out


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

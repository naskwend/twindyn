"""GSE29609 through the Gene Expression Omnibus series-matrix layout.

Ref: Data Availability Statement (the second external cohort is Gene Expression
Omnibus series GSE29609, freely downloadable with no access restriction).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from twindyn.data.loaders.tabular import (
    ResourcePresence,
    locate,
    open_text,
    parse_event,
    parse_grade,
    parse_stage,
    parse_survival_value,
)

SOFT_PATTERNS: dict[str, tuple[str, ...]] = {
    "series_matrix": ("*series_matrix.txt", "*series_matrix.txt.gz", "**/*series_matrix.txt*"),
    "soft": ("*_family.soft", "*_family.soft.gz", "**/*_family.soft*"),
    "supplementary": ("*_RAW.tar", "GSE29609*.tar"),
    "platform": ("*GPL*.annot*", "**/*GPL*.annot*"),
}

CHARACTERISTIC_KEYS = {
    "stage": ("stage", "tnm stage", "ajcc stage", "pathologic stage"),
    "grade": ("grade", "fuhrman grade", "isup grade", "nuclear grade"),
    "survival": ("survival time", "follow up time", "time to event", "months to death", "os.time"),
    "event": ("event", "status", "vital status", "death", "os.event"),
    "age": ("age",),
    "sex": ("sex", "gender"),
}


@dataclass
class GeoSeries:
    """A reader for the GEO series-matrix text format."""

    root: Path
    accession: str = "GSE29609"

    def presence(self) -> ResourcePresence:
        found, missing = locate(self.root, SOFT_PATTERNS)
        return ResourcePresence("gse29609", self.root, found, missing)

    def series_matrix(self) -> tuple[dict[str, list[list[str]]], list[str], np.ndarray]:
        """Metadata, sample identifiers and a sample-major expression matrix."""
        presence = self.presence()
        if "series_matrix" not in presence.found:
            raise FileNotFoundError("no series matrix file under the resource root")
        path = Path(presence.found["series_matrix"])
        metadata: dict[str, list[list[str]]] = {}
        samples: list[str] = []
        identifiers: list[str] = []
        values: list[np.ndarray] = []
        in_table = False
        with open_text(path) as handle:
            reader = csv.reader(handle, delimiter="\t", quotechar='"')
            for row in reader:
                if not row:
                    continue
                if row[0].startswith("!series_matrix_table_begin"):
                    in_table = True
                    continue
                if row[0].startswith("!series_matrix_table_end"):
                    in_table = False
                    continue
                first = row[0].strip('"').strip()
                if not in_table:
                    if first.upper().startswith("ID_REF"):
                        in_table = True
                        samples = [entry.strip('"') for entry in row[1:]]
                        continue
                    if first.startswith("!"):
                        field, _, value = first[1:].partition(" = ")
                        if field:
                            entries = [value.strip('"').strip()] if value else []
                            entries.extend(entry.strip('"').strip() for entry in row[1:])
                            metadata.setdefault(field, []).append(entries)
                    continue
                if first.upper().startswith("ID_REF"):
                    samples = [entry.strip('"') for entry in row[1:]]
                    continue
                identifiers.append(row[0].strip('"'))
                row_values = []
                for entry in row[1:]:
                    try:
                        row_values.append(float(entry))
                    except ValueError:
                        row_values.append(float("nan"))
                values.append(np.asarray(row_values, dtype=np.float64))
        matrix = np.vstack(values) if values else np.zeros((len(samples), 0))
        metadata["_identifiers"] = [identifiers]
        return metadata, samples, matrix.T

    def sample_characteristics(self) -> dict[str, dict[str, str]]:
        """One field map per sample.

        Each characteristic line of the series matrix carries the value of every
        sample, so a repeated field contributes one row per line rather than one
        entry per sample.
        """
        metadata, samples, _ = self.series_matrix()
        out: dict[str, dict[str, str]] = {sample: {} for sample in samples}
        for key, rows in metadata.items():
            if not key.lower().startswith("sample_characteristics"):
                continue
            for values in rows:
                for position, entry in enumerate(values):
                    if position >= len(samples) or ":" not in entry:
                        continue
                    label, value = entry.split(":", 1)
                    out[samples[position]][label.strip().lower()] = value.strip()
        for key in ("sample_title", "sample_source_name_ch1"):
            for values in metadata.get(key, []):
                for position, entry in enumerate(values):
                    if position < len(samples):
                        out[samples[position]][key] = entry
        return out

    def survival(self) -> dict[str, tuple[float, bool]]:
        characteristics = self.sample_characteristics()
        out: dict[str, tuple[float, bool]] = {}
        for sample, fields in characteristics.items():
            time = _pick(fields, CHARACTERISTIC_KEYS["survival"])
            event = _pick(fields, CHARACTERISTIC_KEYS["event"])
            if time is None:
                continue
            parsed = parse_survival_value(time)
            if parsed is None or parsed <= 0.0:
                continue
            flag = parse_event(event) if event is not None else None
            out[sample] = (float(parsed), bool(flag))
        return out

    def strata(self) -> dict[str, tuple[int | None, int | None]]:
        characteristics = self.sample_characteristics()
        out: dict[str, tuple[int | None, int | None]] = {}
        for sample, fields in characteristics.items():
            stage = _pick(fields, CHARACTERISTIC_KEYS["stage"])
            grade = _pick(fields, CHARACTERISTIC_KEYS["grade"])
            out[sample] = (
                parse_stage(stage) if stage is not None else None,
                parse_grade(grade) if grade is not None else None,
            )
        return out

    def platform_genes(self) -> list[str]:
        metadata, _, _ = self.series_matrix()
        rows = metadata.get("_identifiers", [])
        return list(rows[0]) if rows else []


def _pick(fields: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in fields:
            return fields[key]
    for candidate, value in fields.items():
        for key in keys:
            if key in candidate:
                return value
    return None


def read_soft_characteristics(path: Path) -> dict[str, dict[str, str]]:
    """Fallback reader for the SOFT family layout when a matrix file is absent."""
    out: dict[str, dict[str, str]] = {}
    current: str | None = None
    with open_text(path) as handle:
        buffer = io.StringIO(handle.read())
    for line in buffer:
        if line.startswith("^SAMPLE"):
            current = line.split("=", 1)[1].strip()
            out[current] = {}
        elif current is not None and "=" in line:
            key, value = line.split("=", 1)
            out[current][key.strip().lstrip("!").lower()] = value.strip().strip('"')
    return out

"""Readers for the public resources the manuscript draws its cohorts from.

Each reader parses the file layout its repository actually publishes and reports
which pieces are present on disk. Nothing is downloaded here: the readers operate
on a local resource root, and a reader whose files are absent reports itself
unavailable so that the pipeline can record the cohort as not run rather than
substituting a number.

Ref: Sec. 4.1 (public resources and validation design), Data Availability
Statement (the four accessions).
"""

from __future__ import annotations

import csv
import gzip
import io
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class TableBlock:
    """A parsed delimited table with its provenance."""

    path: Path
    header: list[str]
    rows: list[list[str]]

    def column(self, name: str) -> list[str]:
        try:
            position = self.header.index(name)
        except ValueError as exc:
            raise KeyError(f"{self.path.name} has no column {name!r}") from exc
        return [row[position] if position < len(row) else "" for row in self.rows]

    def find_column(self, *candidates: str) -> str | None:
        lowered = {column.lower(): column for column in self.header}
        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]
        return None

    def as_float(self, name: str) -> np.ndarray:
        values = []
        for raw in self.column(name):
            try:
                values.append(float(raw))
            except ValueError:
                values.append(float("nan"))
        return np.asarray(values, dtype=np.float64)


@dataclass
class ResourcePresence:
    resource: str
    root: Path
    found: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.found)

    def describe(self) -> dict[str, object]:
        return {
            "resource": self.resource,
            "root": str(self.root),
            "available": self.available,
            "found": dict(self.found),
            "missing": list(self.missing),
        }


def open_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def read_delimited(
    path: Path,
    delimiter: str | None = None,
    comment: str | None = None,
    skip_header_rows: int = 0,
) -> TableBlock:
    """Read a delimited table, honouring comments and a header offset."""
    if not path.exists():
        raise FileNotFoundError(path)
    if delimiter is None:
        delimiter = (
            "\t" if path.suffix in {".tsv", ".txt", ".gz"} and ".csv" not in path.name else ","
        )
        if path.name.endswith(".csv") or path.name.endswith(".csv.gz"):
            delimiter = ","
    header: list[str] = []
    rows: list[list[str]] = []
    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter=delimiter, quotechar='"')
        for raw in reader:
            if not raw:
                continue
            if comment and raw[0].startswith(comment):
                continue
            if skip_header_rows > 0:
                skip_header_rows -= 1
                continue
            if not header:
                header = [value.strip() for value in raw]
                continue
            rows.append([value.strip() for value in raw])
    return TableBlock(path=path, header=header, rows=rows)


def iter_matching(root: Path, patterns: Iterable[str]) -> Iterator[Path]:
    for pattern in patterns:
        yield from sorted(root.glob(pattern))


def locate(root: Path, patterns: dict[str, tuple[str, ...]]) -> tuple[dict[str, str], list[str]]:
    """Resolve each logical artefact to the first matching path under ``root``."""
    found: dict[str, str] = {}
    missing: list[str] = []
    for key, candidates in patterns.items():
        hits = list(iter_matching(root, candidates))
        if hits:
            found[key] = str(hits[0])
        else:
            missing.append(key)
    return found, missing


def parse_survival_value(raw: str) -> float | None:
    """Follow-up time in days, converting the units each repository declares."""
    cleaned = raw.strip().lower()
    if not cleaned or cleaned in {"na", "nan", "not reported", "unknown", "--", "[not available]"}:
        return None
    if cleaned.endswith("months"):
        value = _leading_float(cleaned)
        return None if value is None else value * 30.4375
    if cleaned.endswith("years"):
        value = _leading_float(cleaned)
        return None if value is None else value * 365.25
    return _leading_float(cleaned)


def _leading_float(text: str) -> float | None:
    token = ""
    for char in text:
        if char.isdigit() or char in ".-+":
            token += char
        elif token:
            break
    try:
        return float(token)
    except ValueError:
        return None


def parse_event(raw: str) -> bool | None:
    cleaned = raw.strip().lower()
    if cleaned in {"dead", "deceased", "1", "true", "yes", "event", "died of disease"}:
        return True
    if cleaned in {"alive", "living", "0", "false", "no", "censored", "not reported"}:
        return False
    if cleaned in {"", "na", "nan", "unknown", "--", "[not available]"}:
        return None
    return None


def parse_stage(raw: str) -> int | None:
    cleaned = raw.strip().lower()
    for token in ("iv", "iii", "ii", "i"):
        if cleaned.endswith(token) or f" {token}" in f" {cleaned}":
            mapping = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
            return mapping[token]
    return None


def parse_grade(raw: str) -> int | None:
    cleaned = raw.strip().lower()
    for digit in "4321":
        if cleaned.endswith(digit) or f" {digit}" in f" {cleaned}" or f"g{digit}" in cleaned:
            return int(digit)
    return None


def parse_float_matrix(block: TableBlock, start_column: int) -> tuple[list[str], np.ndarray]:
    """Extract a numeric block from a parsed table starting at ``start_column``."""
    matrix = []
    for row in block.rows:
        values = []
        for raw in row[start_column:]:
            try:
                values.append(float(raw))
            except ValueError:
                values.append(float("nan"))
        matrix.append(values)
    array = np.asarray(matrix, dtype=np.float64) if matrix else np.zeros((0, 0))
    return block.header[start_column:], array

"""The four public resources and their declared access conditions.

Ref: Sec. 4.1 (public resources and validation design); Data Availability
Statement (project accessions, imaging collection version and licence).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ResourceRole(str, Enum):
    TRAINING = "training"
    EXTERNAL = "external"


class AssayClass(str, Enum):
    TRANSCRIPTOMIC = "transcriptomic"
    PROTEOMIC = "proteomic"
    MIXED = "mixed"


@dataclass(frozen=True)
class ResourceSpec:
    key: str
    display_name: str
    cohort_label: str
    repository: str
    accession: str
    role: ResourceRole
    assay: AssayClass
    has_histology: bool
    reported_samples: int | None
    access_url: str
    licence: str | None
    version: str | None


RESOURCES: dict[str, ResourceSpec] = {
    "tcga_kirc": ResourceSpec(
        key="tcga_kirc",
        display_name="TCGA-KIRC",
        cohort_label="TCGA-KIRC",
        repository="NCI Genomic Data Commons",
        accession="TCGA-KIRC",
        role=ResourceRole.TRAINING,
        assay=AssayClass.TRANSCRIPTOMIC,
        has_histology=True,
        reported_samples=None,
        access_url="https://portal.gdc.cancer.gov/projects/TCGA-KIRC",
        licence=None,
        version=None,
    ),
    "cptac_ccrcc": ResourceSpec(
        key="cptac_ccrcc",
        display_name="CPTAC-ccRCC",
        cohort_label="CPTAC-ccRCC (GDC/TCIA)",
        repository="Clinical Proteomic Tumor Analysis Consortium",
        accession="PDC000127",
        role=ResourceRole.EXTERNAL,
        assay=AssayClass.MIXED,
        has_histology=True,
        reported_samples=103,
        access_url="https://pdc.cancer.gov/pdc/study/PDC000127",
        licence="CC BY 4.0",
        version="14",
    ),
    "gse29609": ResourceSpec(
        key="gse29609",
        display_name="GSE29609",
        cohort_label="GSE29609 (GEO)",
        repository="Gene Expression Omnibus",
        accession="GSE29609",
        role=ResourceRole.EXTERNAL,
        assay=AssayClass.TRANSCRIPTOMIC,
        has_histology=False,
        reported_samples=39,
        access_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE29609",
        licence=None,
        version=None,
    ),
    "emtab1980": ResourceSpec(
        key="emtab1980",
        display_name="E-MTAB-1980",
        cohort_label="E-MTAB-1980 (AE)",
        repository="ArrayExpress",
        accession="E-MTAB-1980",
        role=ResourceRole.EXTERNAL,
        assay=AssayClass.TRANSCRIPTOMIC,
        has_histology=False,
        reported_samples=101,
        access_url="https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-1980",
        licence=None,
        version=None,
    ),
}

TRAINING_RESOURCE = "tcga_kirc"
EXTERNAL_RESOURCES: tuple[str, ...] = ("cptac_ccrcc", "gse29609", "emtab1980")


def resource_spec(key: str) -> ResourceSpec:
    try:
        return RESOURCES[key]
    except KeyError as exc:
        raise KeyError(f"unknown resource {key!r}") from exc


def external_specs() -> tuple[ResourceSpec, ...]:
    return tuple(resource_spec(key) for key in EXTERNAL_RESOURCES)


def training_spec() -> ResourceSpec:
    return resource_spec(TRAINING_RESOURCE)

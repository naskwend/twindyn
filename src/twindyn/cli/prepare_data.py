"""Data preparation entry point.

Ref: Sec. 4.1 (the four public resources and their accessions), Data Availability
Statement (the declared access conditions, versions and licences), Sec. 3.10 (the
deconvolution configuration is fixed in advance).

The command inspects a local resource root, reports which artefacts each reader
can see, and writes the manifest the training pipeline consumes. It never fetches
anything.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from twindyn.cli.common import base_parser, compose, report_failure
from twindyn.data.compartments import COMPARTMENT_ORDER
from twindyn.data.loaders.arrayexpress import ArrayExpressStudy
from twindyn.data.loaders.cptac import CptacCcrcc
from twindyn.data.loaders.gdc import GdcKirc
from twindyn.data.loaders.geo import GeoSeries
from twindyn.data.loaders.tcia import TCIA_COLLECTIONS, CancerImagingArchive
from twindyn.data.resources import EXTERNAL_RESOURCES, RESOURCES, TRAINING_RESOURCE
from twindyn.utils.io import atomic_write_json, atomic_write_text
from twindyn.utils.runtime import get_logger

LOGGER = get_logger("cli.prepare_data")


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("Inspect the staged resources and write the preparation manifest")
    parser.add_argument(
        "--write-ordered-axis",
        action="store_true",
        help="write the predefined compartment ordering used to build the axis",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = compose(parser, args)
        root = Path(config.data.root)
        root.mkdir(parents=True, exist_ok=True)
        readers = {
            TRAINING_RESOURCE: GdcKirc(root / TRAINING_RESOURCE),
            "gse29609": GeoSeries(root / "gse29609"),
            "emtab1980": ArrayExpressStudy(root / "emtab1980"),
            "cptac_ccrcc": CptacCcrcc(root / "cptac_ccrcc"),
        }
        presence = {key: reader.presence().describe() for key, reader in readers.items()}
        imaging = {
            collection: {
                "version": access.version,
                "release_date": access.release_date,
                "doi": access.doi,
                "licence": access.licence,
                "requires_application": access.requires_application,
                "notes": access.notes,
                "staged": CancerImagingArchive(root / collection).presence().available,
            }
            for collection, access in TCIA_COLLECTIONS.items()
        }
        available = [key for key, entry in presence.items() if entry["available"]]
        manifest = {
            "root": str(root),
            "resources": presence,
            "imaging": imaging,
            "available": available,
            "missing": [key for key in RESOURCES if key not in available],
            "reported_cohort_sizes": {
                key: RESOURCES[key].reported_samples
                for key in (TRAINING_RESOURCE, *EXTERNAL_RESOURCES)
            },
            "deconvolution_configuration": {
                "method": config.deconvolution.method,
                "reference": config.deconvolution.reference,
                "marker_genes_per_compartment": config.deconvolution.marker_genes_per_compartment,
                "gene_panel_size": config.deconvolution.gene_panel_size,
                "normalise_fractions": config.deconvolution.normalise_fractions,
            },
            "ordered_axis_rule": config.axis.ordering_rule,
            "tau_max": config.axis.tau_max,
            "note": (
                "The readers never download; a resource whose files are absent is reported here so "
                "the pipeline can record its cohort-level values as not run."
            ),
        }
        destination = Path(args.output_dir or config.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        atomic_write_json(destination / "prepared_data.json", manifest)
        if args.write_ordered_axis:
            atomic_write_text(
                destination / "ordered_axis.txt",
                "\n".join(f"{index:02d}\t{name}" for index, name in enumerate(COMPARTMENT_ORDER))
                + "\n",
            )
    except Exception as error:
        return report_failure("prepare_data", error)
    for key, entry in presence.items():
        LOGGER.info("%s: available=%s samples=%s", key, entry["available"], entry)
    if not available:
        LOGGER.warning("no resource is staged under %s", root)
    LOGGER.info("wrote %s", destination / "prepared_data.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The verification layer and the three root artefacts.

The writers are parameterised by ``root`` so this module can exercise them against
a temporary tree. A test that ran them against the repository root would replace
the shipped report with a partial one, which is why the last test here asserts the
shipped report is the complete run rather than the one these fixtures produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twindyn.verification import (
    BLOCKED,
    CHECKS,
    CLAIMS,
    DEVIATIONS,
    FAIL,
    MANIFEST_EXCLUDED_DIRECTORIES,
    MANIFEST_EXCLUDED_NAMES,
    NOT_RUN,
    PASS,
    build_integrity_manifest,
    claim_records_with_statuses,
    iter_manifest_files,
    overall_verification_status,
    summarise_checks,
    verify_manifest,
    write_artefacts,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def written(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("release")
    (root / "src").mkdir()
    (root / "src" / "module_notes.txt").write_text("content", encoding="utf-8")
    (root / "README.md").write_text("release", encoding="utf-8")
    outcome = write_artefacts(root, root / ".scratch")
    return {"root": root, "outcome": outcome}


def test_artefacts_are_written(written) -> None:
    root = written["root"]
    for name in (
        "claim_to_code.json",
        "verification_report.json",
        "verification_summary.txt",
        "dataset_urls.txt",
        "integrity_manifest.json",
    ):
        assert (root / name).exists(), name


def test_report_records_a_status_from_the_allowed_set(written) -> None:
    report = json.loads((written["root"] / "verification_report.json").read_text())
    assert report["verification_status"] in {"VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"}
    assert report["summary"]["total"] == len(CHECKS)
    assert report["summary"]["counts"]["PASS"] >= 1


def test_every_check_carries_a_paper_location_and_a_detail(written) -> None:
    report = json.loads((written["root"] / "verification_report.json").read_text())
    for check in report["checks"]:
        assert check["paper_location"]
        assert check["detail"]
        assert check["status"] in {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}


def test_claim_map_binds_every_claim_to_code_and_checks(written) -> None:
    payload = json.loads((written["root"] / "claim_to_code.json").read_text())
    assert payload["claim_count"] == len(CLAIMS)
    assert isinstance(payload["code_references_missing"], list)
    for claim in payload["claims"]:
        assert claim["code"]
        assert claim["checks"]
        assert claim["status"] in {
            "PASS",
            "FAIL",
            "NOT_RUN",
            "BLOCKED",
            "PARTIALLY_VERIFIED",
            "UNVERIFIED",
        }


def test_claim_map_references_resolve_in_the_repository() -> None:
    """Every file a claim points at must exist under the release root."""
    missing = [
        reference
        for claim in CLAIMS
        for reference in claim.code
        if reference.endswith(".py") and not (REPOSITORY_ROOT / reference).exists()
    ]
    assert missing == []


def test_claim_map_carries_the_deviations_and_the_open_quantities(written) -> None:
    payload = json.loads((written["root"] / "claim_to_code.json").read_text())
    assert payload["deviations"]
    assert all(entry["justification"] for entry in payload["deviations"])
    assert payload["unreported_quantities"]
    assert payload["absent_identifiers"] == ["B14"]
    assert payload["selected_values"]["tau_max"] == 16
    assert payload["selected_values"]["state_dim"] == 64


def test_deviations_name_a_paper_location() -> None:
    for entry in DEVIATIONS:
        assert entry["paper_location"]
        assert entry["departs"]
        assert entry["justification"]


def test_dataset_urls_lists_only_content_checked_links(written) -> None:
    text = (written["root"] / "dataset_urls.txt").read_text()
    for accession in ("TCGA-KIRC", "GSE29609", "E-MTAB-1980", "PDC000127"):
        assert accession in text
    assert "content check:" in text
    assert "not reachable" in text


def test_manifest_covers_the_tree_and_excludes_itself(written) -> None:
    root = written["root"]
    manifest = json.loads((root / "integrity_manifest.json").read_text())
    paths = {entry["path"] for entry in manifest["files"]}
    assert "integrity_manifest.json" not in paths
    assert "README.md" in paths
    assert "claim_to_code.json" in paths
    assert manifest["manifest_digest"]
    assert manifest["file_count"] == len(manifest["files"])


def test_manifest_is_consistent_with_the_written_tree(written) -> None:
    root = written["root"]
    manifest = json.loads((root / "integrity_manifest.json").read_text())
    report = verify_manifest(root, manifest)
    assert report["consistent"] is True
    assert report["missing"] == []
    assert report["added"] == []
    assert report["changed"] == []


def test_manifest_skips_caches_and_binary_payloads(written) -> None:
    root = written["root"]
    cache = root / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "module.pyc").write_text("compiled", encoding="utf-8")
    paths = {entry.name for entry in iter_manifest_files(root)}
    assert "module.pyc" not in paths
    assert "integrity_manifest.json" not in paths
    for path in iter_manifest_files(root):
        assert not set(path.parts) & set(MANIFEST_EXCLUDED_DIRECTORIES)
    assert not paths & set(MANIFEST_EXCLUDED_NAMES)


def test_manifest_digest_changes_when_a_file_changes(written) -> None:
    root = written["root"]
    manifest = json.loads((root / "integrity_manifest.json").read_text())
    (root / "README.md").write_text("changed", encoding="utf-8")
    report = verify_manifest(root, manifest)
    assert report["consistent"] is False
    assert "README.md" in report["changed"]
    (root / "README.md").write_text("release", encoding="utf-8")


def test_summary_counts_partition_the_checks(written) -> None:
    summary = written["outcome"]["summary"]
    counts = summary["counts"]
    assert sum(counts.values()) == summary["total"]
    assert isinstance(written["outcome"]["summary"]["failed_checks"], list)
    assert isinstance(written["outcome"]["summary"]["areas"], dict)


def test_overall_status_reflects_the_failure_mix() -> None:
    assert (
        overall_verification_status({"counts": {"PASS": 3, "FAIL": 0, "NOT_RUN": 0, "BLOCKED": 0}})
        == "VERIFIED"
    )
    assert (
        overall_verification_status({"counts": {"PASS": 3, "FAIL": 0, "NOT_RUN": 1, "BLOCKED": 0}})
        == "PARTIALLY_VERIFIED"
    )
    assert (
        overall_verification_status({"counts": {"PASS": 2, "FAIL": 1, "NOT_RUN": 0, "BLOCKED": 0}})
        == "PARTIALLY_VERIFIED"
    )
    assert (
        overall_verification_status({"counts": {"PASS": 0, "FAIL": 0, "NOT_RUN": 0, "BLOCKED": 0}})
        == "UNVERIFIED"
    )


def test_summary_text_is_plain_text(written) -> None:
    text = (written["root"] / "verification_summary.txt").read_text()
    assert text.endswith("\n")
    assert "verification summary" in text
    assert "overall:" in text
    assert not text.startswith("#")


from twindyn.verification import CheckResult  # noqa: E402


def test_claim_statuses_come_from_the_executed_checks() -> None:

    results = [CheckResult("x", "area", "claim", "loc", PASS, "detail")]
    records = claim_records_with_statuses(results)
    assert len(records) == len(CLAIMS)
    for record in records:
        for identifier in record.claim.checks:
            assert record.check_status[identifier] in {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}


def test_summarise_checks_groups_by_area() -> None:
    results = [
        CheckResult("a", "one", "c", "l", PASS, "d"),
        CheckResult("b", "two", "c", "l", FAIL, "d"),
        CheckResult("c", "two", "c", "l", NOT_RUN, "d"),
        CheckResult("d", "two", "c", "l", BLOCKED, "d"),
    ]
    summary = summarise_checks(results)
    assert summary["total"] == 4
    assert summary["areas"]["two"][FAIL] == 1
    assert summary["failed_checks"] == ["b"]
    assert summary["pass_rate"] == pytest.approx(0.25)


def test_shipped_report_is_the_complete_run() -> None:
    """The shipped artefacts must be the full run, not a fixture's partial report."""
    report_path = REPOSITORY_ROOT / "verification_report.json"
    if not report_path.exists():
        pytest.skip("the shipped artefacts are produced by the verification entry point")
    report = json.loads(report_path.read_text())
    assert report["summary"]["total"] == len(CHECKS)
    manifest = json.loads((REPOSITORY_ROOT / "integrity_manifest.json").read_text())
    assert manifest["file_count"] >= len(CHECKS)
    assert (REPOSITORY_ROOT / "claim_to_code.json").exists()


def test_shipped_manifest_is_consistent_with_the_repository() -> None:
    path = REPOSITORY_ROOT / "integrity_manifest.json"
    if not path.exists():
        pytest.skip("the shipped artefacts are produced by the verification entry point")
    manifest = json.loads(path.read_text())
    live = build_integrity_manifest(REPOSITORY_ROOT)
    assert live["manifest_digest"] == manifest["manifest_digest"], (
        "the tree changed after the manifest was written; rerun the verification entry point"
    )

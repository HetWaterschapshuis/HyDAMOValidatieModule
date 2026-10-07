import csv
import json
import sys
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import overzicht_validatieregels as overview


@pytest.mark.parametrize("branch", [None, "fix/lagen&kolommen"])
def test_download_branch(tmp_path, monkeypatch, branch):
    urls = []
    rules = {"schema": "1.5", "objects": []}
    download_url = "https://example.org/ValidationRules_1.5.json"

    def download_json(url):
        urls.append(url)
        if url == download_url:
            return rules
        return [
            {
                "type": "file",
                "name": "ValidationRules_1.5.json",
                "download_url": download_url,
            }
        ]

    monkeypatch.setattr(overview, "_download_json", download_json)
    if branch is None:
        overview.download_validation_rules(tmp_path)
    else:
        overview.download_validation_rules(tmp_path, branch=branch)

    assert parse_qs(urlsplit(urls[0]).query) == {"ref": [branch or "main"]}
    assert urls[1:] == [download_url]
    assert json.loads((tmp_path / "ValidationRules_1.5.json").read_text()) == rules


@pytest.mark.parametrize("branch", [None, "fix/validatieregels"])
def test_branch_argument(monkeypatch, branch):
    arguments = ["overzicht_validatieregels.py"]
    if branch is not None:
        arguments.extend(["--branch", branch])
    monkeypatch.setattr(sys, "argv", arguments)

    assert overview._parse_arguments().branch == (branch or "main")


@pytest.mark.parametrize("ignored_log_columns", [None, ["brug.ontbrekende_kolom"]])
@pytest.mark.parametrize(
    "layer, column, missing_field, expected",
    [
        ("brug", "ontbrekende_kolom", "ontbrekende_kolommen", "brug.ontbrekende_kolom"),
        (
            "ontbrekende_laag",
            "code",
            "ontbrekende_lagen",
            "ontbrekende_laag (ontbreekt in HyDAMO-schema)",
        ),
        ("brug", "code", None, None),
    ],
)
def test_report_missing_dependencies(
    tmp_path, capsys, layer, column, missing_field, expected, ignored_log_columns
):
    rules_file = tmp_path / "ValidationRules_1.5.json"
    rules_file.write_text(
        json.dumps(
            {
                "schema": "1.5",
                "hydamo_version": "2.4",
                "objects": [
                    {
                        "object": layer,
                        "validation_rules": [
                            {
                                "id": 7,
                                "name": "Testregel",
                                "function": {"NOTNA": {"parameter": column}},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    reports, summary, versions = overview.create_reports(
        rules_file,
        tmp_path / "csv",
        hydamo_versions=["2.4"],
        ignored_log_columns=ignored_log_columns,
    )

    output = capsys.readouterr().out
    with reports[0].open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter=";"))
    assert len(rows) == 1
    assert summary.is_file()
    assert versions == ["2.4"]
    if missing_field is None:
        assert output == ""
        assert rows[0]["uitvoerbaar"] == "True"
    else:
        assert rows[0][missing_field] == expected
        assert rows[0]["uitvoerbaar"] == "False"
        with summary.open(encoding="utf-8-sig", newline="") as file:
            assert next(csv.DictReader(file, delimiter=";"))["niet_uitvoerbare_regels"] == "1"
        if expected in (ignored_log_columns or []):
            assert output == ""
            return
        assert (
            f"ValidationRules_1.5.json | HyDAMO 2.4 | {layer} | validatieregel 7 | Testregel"
        ) in output
        label = (
            "Ontbrekende lagen"
            if missing_field == "ontbrekende_lagen"
            else "Ontbrekende kolommen"
        )
        assert f"  {label}: {expected}" in output


def test_ignore_log_column_arguments(monkeypatch):
    monkeypatch.setattr(overview, "IGNORED_LOG_COLUMNS", ["brug.onbekend"])
    monkeypatch.setattr(sys, "argv", ["overzicht_validatieregels.py"])
    assert overview._parse_arguments().ignored_log_columns == ["brug.onbekend"]

    monkeypatch.setattr(
        sys, "argv", [
            "overzicht_validatieregels.py",
            "--ignore-log-column", "regelmiddel.maximalehoogteopening",
            "--ignore-log-column", "stuw.onbekend",
        ]
    )
    assert overview._parse_arguments().ignored_log_columns == [
        "brug.onbekend", "regelmiddel.maximalehoogteopening", "stuw.onbekend",
    ]
    assert overview.IGNORED_LOG_COLUMNS == ["brug.onbekend"]

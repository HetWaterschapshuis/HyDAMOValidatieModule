"""Tests voor argumenten, processtatus en uitvoer van de CLI-ingangen."""

import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from hydamo_validation import __main__ as cli
from hydamo_validation.summaries import ResultSummary


@pytest.fixture(params=["module", "console", "directory"])
def command(request):
    if request.param == "module":
        return [sys.executable, "-m", "hydamo_validation"]
    if request.param == "directory":
        return [sys.executable, str(Path(cli.__file__).parent)]
    executable = Path(sys.executable).with_name(
        "hydamo-validate.exe" if sys.platform == "win32" else "hydamo-validate"
    )
    assert executable.is_file(), "Installeer de module voordat de CLI-tests worden uitgevoerd"
    return [str(executable)]


@pytest.fixture
def project(tmp_path):
    directory = tmp_path / "project met spaties"
    datasets = directory / "datasets"
    datasets.mkdir(parents=True)
    gdf = gpd.GeoDataFrame(
        {
            "globalid": ["brug-1"],
            "nen3610id": ["brug-1"],
            "objectid": [1],
            "hoogteonderzijde": [1.0],
            "intreeverlies": [0.0],
            "uittreeverlies": [0.0],
            "lengte": [2.0],
            "ruwheid": [0.03],
            "typemethoderuwheid": ["Manning"],
        },
        geometry=[Point(100000, 400000)],
        crs="EPSG:28992",
    )
    gdf.to_file(datasets / "hydamo.gpkg", layer="brug", driver="GPKG")
    rules = {
        "schema": "1.5",
        "hydamo_version": "2.5",
        "objects": [{
            "object": "brug",
            "validation_rules": [{
                "id": 1,
                "name": "Controleer lengte",
                "type": "logic",
                "error_type": "critical",
                "result_variable": "lengte_ok",
                "error_message": "Brug te kort",
                "active": True,
                "function": {"GE": {"left": "lengte", "right": 3.0}},
            }],
        }],
    }
    (directory / "validationrules.json").write_text(json.dumps(rules), encoding="utf-8")
    return directory


def invoke(command, *arguments):
    return subprocess.run(
        [*command, *map(str, arguments)], capture_output=True, text=True, timeout=60
    )


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252"])
def test_help(command, monkeypatch, encoding):
    monkeypatch.setenv("PYTHONIOENCODING", encoding)
    result = invoke(command, "--help")
    assert result.returncode == 0
    for option in ["directory", "--log-level", "--output-types", "--coverages", "--raise-error"]:
        assert option in result.stdout
    assert "KEY=VALUE" in result.stdout
    assert (
        "directory/\n"
        "|-- datasets/\n"
        "|   `-- hydamo.gpkg\n"
        "`-- validationrules.json\n"
    ) in result.stdout
    assert "directory/results/" in result.stdout
    assert not result.stderr


@pytest.mark.parametrize("raise_error", [False, True])
def test_invalid_data_is_success(command, project, raise_error):
    arguments = [project, "--output-types", "csv", "geojson", "geopackage"]
    if raise_error:
        arguments.append("--raise-error")
    result = invoke(command, *arguments)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Projectmap: {project}" in result.stdout
    assert "Status: technisch succesvol afgerond" in result.stdout
    assert "Doorlooptijd:" in result.stdout
    assert f"Resultaten: {project / 'results'}" in result.stdout
    assert "ResultSummary object" not in result.stdout
    assert not result.stderr
    results = project / "results"
    summary = json.loads((results / "validation_result.json").read_text())
    assert summary["success"] is True
    assert not summary.get("errors")
    assert (results / "validator.log").is_file()
    frame = pd.read_csv(results / "csv" / "brug.csv")
    assert frame.loc[0, "invalid_critical"] == 1
    assert not frame.loc[0, "validate_001_lengte_ok"]
    assert list(results.glob("*.gpkg"))
    assert list((results / "geojson").glob("*.geojson"))


def test_rule_exception_is_technical_failure(command, project):
    path = project / "validationrules.json"
    rules = json.loads(path.read_text())
    rules["objects"][0]["validation_rules"][0]["function"]["GE"]["left"] = "onbekend"
    path.write_text(json.dumps(rules), encoding="utf-8")
    result = invoke(command, project, "--output-types", "csv")
    assert result.returncode == 1
    assert "Status: technisch mislukt" in result.stdout
    assert "Fout:" in result.stderr
    assert "onbekend" in result.stderr
    assert "Traceback" not in result.stderr
    summary = json.loads((project / "results" / "validation_result.json").read_text())
    # De bestaande Python-API rondt af, maar registreert de gecrashte regel.
    assert summary["success"] is True
    assert summary["errors"]


def test_valid_data_with_default_output(command, project):
    path = project / "validationrules.json"
    rules = json.loads(path.read_text())
    rules["objects"][0]["validation_rules"][0]["function"]["GE"]["right"] = 1.0
    path.write_text(json.dumps(rules), encoding="utf-8")
    result = invoke(command, project)
    assert result.returncode == 0, result.stdout + result.stderr
    frame = gpd.read_file(project / "results" / "results.gpkg", layer="brug")
    assert frame.loc[0, "validate_001_lengte_ok"]
    assert not (project / "results" / "csv").exists()
    assert not (project / "results" / "geojson").exists()


def test_missing_project(command, tmp_path):
    result = invoke(command, tmp_path / "ontbreekt")
    assert result.returncode == 1
    assert "Status: technisch mislukt" in result.stdout
    assert "Fout: Validatie afgebroken" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("raise_error", [False, True])
def test_aborted_validation(command, tmp_path, raise_error):
    arguments = [tmp_path]
    if raise_error:
        arguments.append("--raise-error")
    result = invoke(command, *arguments)
    assert result.returncode == 1
    assert "missing_paths" in result.stderr
    assert ("Traceback" in result.stderr) == raise_error
    assert (tmp_path / "results" / "validation_result.json").is_file()
    assert "Status: technisch mislukt" in result.stdout


@pytest.mark.parametrize("arguments", [
    [], ["project", "--output-types", "xlsx"], ["project", "--log-level", "WARNING"],
    ["project", "--coverages", "AHN"], ["project", "--coverages", "=pad"],
    ["project", "--coverages", "AHN="], ["project", "--coverages", " =pad"],
    ["project", "--coverages", "AHN= "],
])
def test_invalid_arguments(command, arguments):
    result = invoke(command, *arguments)
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "Traceback" not in result.stderr
    if "--coverages" in arguments:
        assert "verwacht KEY=VALUE" in result.stderr


@pytest.mark.parametrize("options, expected", [
    ([], {"output_types": ["geopackage"], "log_level": "INFO", "coverages": {}}),
    (["--log-level", "DEBUG", "--output-types", "csv", "geojson",
      "--coverages", "AHN=oud", "--coverages", "AHN=C:/ahn data/a=b", "--coverages", "DTM=/dtm"],
     {"output_types": ["csv", "geojson"], "log_level": "DEBUG",
      "coverages": {"AHN": "C:/ahn data/a=b", "DTM": "/dtm"}}),
])
def test_arguments_passed_to_validator(monkeypatch, options, expected):
    configured = {}
    called = {}
    summary = ResultSummary()
    summary.success = True

    def validator(**kwargs):
        configured.update(kwargs)

        def run(**kwargs):
            called.update(kwargs)
            return None, None, summary

        return run

    monkeypatch.setattr(cli, "validator", validator)
    monkeypatch.setattr(sys, "argv", ["hydamo-validate", "project", *options])
    assert cli.main() == 0
    assert configured == expected
    assert called == {"directory": "project", "raise_error": False}


def test_unhandled_exception(monkeypatch, capsys):
    def validator(**kwargs):
        raise OSError("Geen toegang")

    monkeypatch.setattr(cli, "validator", validator)
    monkeypatch.setattr(sys, "argv", ["hydamo-validate", "project"])
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert "Status: technisch mislukt" in captured.out
    assert "OSError: Geen toegang" in captured.err
    assert "Traceback" not in captured.err

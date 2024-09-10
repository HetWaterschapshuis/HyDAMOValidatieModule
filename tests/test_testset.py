from pathlib import Path
from hydamo_validation import validator
import pandas as pd
import pytest
import shutil

DATA_DIR = Path(__file__).parent.joinpath("data")
COVERAGE = {"AHN": DATA_DIR.joinpath(r"dtm")}
DIRECTORY = DATA_DIR.joinpath("tasks", "test_testset")

hydamo_validator = validator(
    output_types=["geopackage"], coverages=COVERAGE, log_level="INFO"
)


@pytest.fixture
def result():
    results_dir = DIRECTORY / "results"
    if results_dir.exists():
        shutil.rmtree(DIRECTORY / "results")
    return hydamo_validator(directory=DIRECTORY, raise_error=True)


def test_results_available(result):
    # assert if result is generated
    assert result is not None


def test_validate_result(result):
    # unpack result
    datamodel, layer_summary, result_summary = result

    # check if profielpunt is available
    assert hasattr(layer_summary, "profielpunt")

    # invalid geometries contain (7) in syntax_geometry column
    result = layer_summary.profielpunt.syntax_geometry.apply(
        lambda x: "(7)" not in x
    ).reset_index(drop=True)

    # we expecte only the first geometry to be valid, containing PointZ, Point, LineString, LineStringZ
    expected_result = pd.Series([True, False, False, False], name="syntax_geometry")
    assert result.equals(expected_result)

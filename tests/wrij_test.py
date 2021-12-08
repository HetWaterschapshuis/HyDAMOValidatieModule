from hydamo_validation import validator
from pathlib import Path

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR

coverage = {"AHN": DATA_DIR.joinpath(r"dtm")}
directory = DATA_DIR.joinpath(r"tasks/test_wrij")
exports_dir = Path(__file__).parent / "exports"
exports_dir.mkdir(exist_ok=True)


hydamo_validator = validator(output_types=["geopackage", "csv", "geojson"],
                             coverages=coverage,
                             log_level="INFO"
                             )


datamodel, layer_summary, result_summary = hydamo_validator(
    directory=directory,
    raise_error=True
    )


def test_finished():
    assert result_summary.status == "finished"


def test_non_existing_dir():
    try:
        hydamo_validator(directory="i_do_not_exist", raise_error=True)
        assert False
    except FileNotFoundError:
        assert True


def test_missing_data():
    rules_json = exports_dir.joinpath("validationrules.json")
    if rules_json.exists():
        rules_json.unlink()
    try:
        hydamo_validator(directory=exports_dir, raise_error=True)
        assert False
    except FileNotFoundError:
        assert True


def test_output_type_data():
    _hydamo_validator = validator(output_types=["not_supported"],
                                  coverages=coverage,
                                  log_level="INFO"
                                  )
    try:
        _hydamo_validator(directory=directory, raise_error=True)
        assert False
    except TypeError:
        assert True

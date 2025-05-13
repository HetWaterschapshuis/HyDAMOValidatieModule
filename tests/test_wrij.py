# %%
from hydamo_validation import validator
from hydamo_validation import __version__
from pathlib import Path
from datetime import datetime
import shutil

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR

coverage = {"AHN": DATA_DIR.joinpath(r"dtm")}
directory = DATA_DIR.joinpath(r"tasks/test_wrij")
exports_dir = Path(__file__).parent / "exports"
if exports_dir.exists():
    shutil.rmtree(exports_dir)
exports_dir.mkdir(exist_ok=True)


hydamo_validator = validator(
    output_types=["geopackage", "csv", "geojson"], coverages=coverage, log_level="INFO"
)


datamodel, layer_summary, result_summary = hydamo_validator(
    directory=directory, raise_error=True
)


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
    _hydamo_validator = validator(
        output_types=["not_supported"], coverages=coverage, log_level="INFO"
    )
    try:
        _hydamo_validator(directory=directory, raise_error=True)
        assert False
    except TypeError:
        assert True


def test_result_success():
    assert result_summary.success


def test_result_version():
    assert result_summary.module_version == __version__


def test_result_check_date():
    try:
        datetime.fromisoformat(result_summary.date_check)
        assert True
    except ValueError:
        assert False


def test_result_duration():
    assert isinstance(result_summary.duration, float)


def test_result_finished():
    assert result_summary.status == "finished"


def test_result_validation_results():
    expected_result = [
        "duikersifonhevel",
        "regelmiddel",
        "kunstwerkopening",
        "stuw",
        "brug",
        "pomp",
        "gemaal",
        "hydroobject",
    ]
    assert all([i in expected_result for i in result_summary.validation_result])


def test_result_dataset_layers():
    expected_result = [
        "Afvoergebiedaanvoergebied",
        "Duikersifonhevel",
        "Hydroobject",
        "Regelmiddel",
        "Stuw",
        "Brug",
        "Bodemval",
        "Gemaal",
        "Kunstwerkopening",
        "Pomp",
    ]
    assert all([i in expected_result for i in result_summary.dataset_layers])


def test_result_result_layers():
    expected_result = [
        "afvoergebiedaanvoergebied",
        "duikersifonhevel",
        "hydroobject",
        "regelmiddel",
        "stuw",
        "brug",
        "bodemval",
        "gemaal",
        "kunstwerkopening",
        "pomp",
    ]
    assert all([i in expected_result for i in result_summary.result_layers])


def test_result_missing_layers():
    expected_result = [
        "admingrenswaterschap",
        "afsluitmiddel",
        "aquaduct",
        "beheergrenswaterschap",
        "bijzonderhydraulischobject",
        "doorstroomopening",
        "grondwaterinfolijn",
        "grondwaterinfopunt",
        "grondwaterkoppellijn",
        "grondwaterkoppelpunt",
        "hydrologischerandvoorwaarde",
        "hydroobject_normgp",
        "lateraleknoop",
        "meetlocatie",
        "meetwaardeactiewaarde",
        "normgeparamprofiel",
        "normgeparamprofielwaarde",
        "peilafwijkinggebied",
        "peilbesluitgebied",
        "peilgebiedpraktijk",
        "peilgebiedvigerend",
        "profielgroep",
        "profiellijn",
        "profielpunt",
        "reglementgrenswaterschap",
        "ruwheidprofiel",
        "streefpeil",
        "sturing",
        "vispassage",
        "vispassagevlak",
        "vuilvang",
        "zandvang",
    ]
    assert all([i in expected_result for i in result_summary.missing_layers])


def test_result_error_layers():
    assert result_summary.error_layers == []


def test_result_syntax_layers():
    expected_result = [
        "afvoergebiedaanvoergebied",
        "duikersifonhevel",
        "hydroobject",
        "regelmiddel",
        "stuw",
        "brug",
        "bodemval",
        "gemaal",
        "kunstwerkopening",
        "pomp",
    ]
    assert all([i in expected_result for i in result_summary.syntax_result])

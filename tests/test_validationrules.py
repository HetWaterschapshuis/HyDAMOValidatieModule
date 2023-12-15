from hydamo_validation.validator import read_validation_rules
from jsonschema import ValidationError
from json import JSONDecodeError

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR


def test_schema_1_1():
    validation_rules_json = DATA_DIR.joinpath(
        "validation_rules", "ValidationRules_1.1.json"
    )

    validation_rules = read_validation_rules(validation_rules_json)
    assert validation_rules["schema"] == "1.1"


def test_schema_1_2():
    validation_rules_json = DATA_DIR.joinpath(
        "validation_rules", "ValidationRules_1.2.json"
    )

    validation_rules = read_validation_rules(validation_rules_json)
    assert validation_rules["schema"] == "1.2"


def test_invalid_json():
    validation_rules_json = DATA_DIR.joinpath("validation_rules", "NotAjson.json")

    try:
        read_validation_rules(validation_rules_json)
        assert False
    except JSONDecodeError:
        assert True


def test_missing_schema():
    validation_rules_json = DATA_DIR.joinpath(
        "validation_rules", "MissingSchemaValidationRules_1.1.json"
    )

    try:
        read_validation_rules(validation_rules_json)
        assert False
    except FileNotFoundError:
        assert True


def test_validation_error():
    validation_rules_json = DATA_DIR.joinpath(
        "validation_rules", "MissingOrderValidationRules_1.1.json"
    )

    try:
        read_validation_rules(validation_rules_json)
        assert False
    except ValidationError:
        assert True

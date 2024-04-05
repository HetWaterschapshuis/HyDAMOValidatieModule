from pathlib import Path
from hydamo_validation import validator

DATA_DIR = Path(r"data").absolute().resolve()
coverage = {"AHN": DATA_DIR.joinpath(r"dtm")}
directory = Path(r"d:\projecten\D2401.ValidatieModule\01.Issues\HYV-200")

hydamo_validator = validator(
    output_types=["geopackage", "csv", "geojson"], coverages=coverage, log_level="INFO"
)


datamodel, layer_summary, result_summary = hydamo_validator(
    directory=directory, raise_error=True
)

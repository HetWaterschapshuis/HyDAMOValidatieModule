from pathlib import Path

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR

coverage = {"AHN": DATA_DIR.joinpath(r"dtm")}
directory = DATA_DIR.joinpath(r"tasks/test_profielen")
exports_dir = Path(__file__).parent / "exports"
exports_dir.mkdir(exist_ok=True)

# hydamo_validator = validator(output_types=["geopackage", "csv", "geojson"],
#                              coverages=coverage,
#                              log_level="INFO"
#                              )


# datamodel, layer_summary, result_summary = hydamo_validator(
#     directory=directory,
#     raise_error=True
#     )

# datamodel.to_geopackage(exports_dir / "datamodel.gpkg",
#                         use_schema=False)

# %%
from hydamo_validation import topologic_functions
from hydamo_validation import validator
from pathlib import Path
from .config import COVERAGE

hydamo_validator = validator(
    output_types=["geopackage", "csv", "geojson"], coverages=COVERAGE, log_level="INFO"
)

directory = Path(r"d:\projecten\D2401.ValidatieModule\01.Issues\GIT-017")

datamodel, layer_summary, result_summary = hydamo_validator(
    directory=directory, raise_error=True
)

gdf = datamodel.hydroobject

# %%
# make a geodataframe from structures list
struc_series = topologic_functions._layers_from_datamodel(
    ["duikersifonhevel"], datamodel
)
# create spatial indices
sindex = gdf.sindex
struc_sindex = struc_series.sindex

for row in gdf.itertuples():
    geometry = row.geometry

    # get a selection of hydroobjects that intersect geometry
    indices = sindex.intersection(geometry.bounds)
    indices = [i for i in indices if i != gdf.index.get_loc(row.Index)]

# %%

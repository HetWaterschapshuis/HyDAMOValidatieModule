import fiona
import time
import logging
import geopandas as gpd
import pandas as pd
from inspect import getmembers, isfunction


def normalize_fiona_schema(schema):
    schema["properties"] = {
        k: fiona.schema.normalize_field_type(v) for k, v in schema["properties"].items()
    }
    return schema


def schema_properties_to_dtypes(properties):
    properties = {
        k: fiona.schema.normalize_field_type(v) for k, v in properties.items()
    }
    return properties


def dataset_layers(dataset_properties):
    dataset_layers_dict = {k: list(v.keys()) for k, v in dataset_properties.items()}

    layers = [
        item for sublist in list(dataset_layers_dict.values()) for item in sublist
    ]

    return layers


def get_functions(module):
    return [i[0] for i in getmembers(module, isfunction) if i[0][0] != "_"]


def read_geopackage(file_path, layer):
    """Read file as GeoDataFrame."""
    gdf = gpd.read_file(file_path, layer=layer, engine="pyogrio", use_fid_as_index=True)

    if type(gdf) == pd.DataFrame:
        gdf = gpd.GeoDataFrame(gdf, geometry=gpd.GeoSeries())

    # fix integers to nullable type
    with fiona.open(file_path, "r", layer=layer) as src:
        dtypes = normalize_fiona_schema(src.schema)["properties"]

    for k, v in dtypes.items():
        if v == "int64":
            gdf[k] = gdf[k].astype(pd.Int64Dtype())

    return gdf


class Timer(object):
    """Record function efficiency."""

    def __init__(self, logger=logging):
        self.start = time.time()
        self.milestone = self.start
        self.logger = logger

    def report(self, message=""):
        """Set milestone and report."""
        delta_time = time.time() - self.milestone
        self.logger.debug(f"{message} in {delta_time:.3f} sec")
        self.milestone = time.time()
        return delta_time

    def reset(self, message=None):
        """Report task-efficiency and reset."""
        if message:
            self.logger.debug(f"{message} in {(time.time() - self.start):.3f} sec")
        self.start = time.time()
        self.milestone = self.start

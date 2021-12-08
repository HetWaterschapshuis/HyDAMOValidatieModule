import fiona
from shapely.geometry import LineString, Point, Polygon, shape
import time
import logging
import sys
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
    src = fiona.open(file_path, "r", layer=layer)
    rows = []
    columns = list(src.schema["properties"].keys()) + ["geometry"]
    dtypes = normalize_fiona_schema(src.schema)["properties"]
    crs = src.crs
    for feature in src:
        # load geometry
        if hasattr(feature, "__geo_interface__"):
            feature = feature.__geo_interface__
        row = {"geometry": shape(feature["geometry"]) if feature["geometry"] else None}
        # load properties
        row.update(feature["properties"])
        rows.append(row)
    src.close()
    gdf = gpd.GeoDataFrame(rows, columns=columns, crs=crs)
    # fix integers with None
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

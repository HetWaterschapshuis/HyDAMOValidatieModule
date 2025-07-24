"""functions to be executed on gdf."""

import geopandas as gpd
from typing import Literal
import numpy as np
from pathlib import Path

from geopandas import GeoDataFrame
from pandas import Series
from rasterstats import zonal_stats
import logging
import pandas as pd

from hydamo_validation.datamodel import HyDAMO
from hydamo_validation.functions import custom as custom_functions

try:
    import rasterio
except ImportError:
    import gdal  # noqa to avoid rasterio.version error: https://github.com/conda-forge/rasterio-feedstock/issues/240
    import rasterio

COVERAGES = {}
# DATA_MODEL = None
# OBJECT_LAYER = None

# We get a false-positive settingwithcopywarning in buffer-function that we supress
pd.options.mode.chained_assignment = None


def _set_coverage(coverage: str, directory: str):
    """Add a coverage for functions."""
    global COVERAGES
    coverage_path = Path(directory)
    if not coverage_path.exists():
        logging.error(
            (
                f"Path to coverage {coverage} does not exist: ",
                f"{coverage_path.absolute().resolve()}",
                ". Functions using this coverage fail without data.",
            )
        )
        raise FileNotFoundError(f"{coverage_path.absolute().resolve()}")
    COVERAGES[coverage] = coverage_path


def _buffer_row(row, column):
    radius = max(row[column], 0.5)
    return row.geometry.buffer(radius)


def _get_geometric_attribute(gdf, geom_parameter):
    geometry, method = geom_parameter.split(".")
    return getattr(gdf[geometry], method)


def sum(gdf, array: list):
    """Return a sum expression."""
    expression = " + ".join(map(str, array))
    return gdf.eval(expression)


def difference(gdf, left, right, absolute=False):
    """
    Difference between 'left' and 'right'

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    left : str, numeric
        Left column or value in expression
    right : TYPE
        Right column or value in expression
    absolute : bool, optional
        Absolute (True) or relative difference (False) to left.
        The default is False.

    Returns
    -------
    result : Series
        Float series

    """

    if left in gdf.columns:
        left = gdf[left]
    if right in gdf.columns:
        right = gdf[right]
    if absolute:
        result = (left - right).abs()
    else:
        result = left - right

    return result


def divide(gdf, left, right):
    """
    Division of 'left' by 'right'

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    left : str, numeric
        Left column or value in expression
    right : TYPE
        Right column or value in expression

    Returns
    -------
    result : Series
        Float series

    """
    expression = " / ".join(map(str, [left, right]))
    return gdf.eval(expression)


def multiply(gdf, left, right):
    """
    Multiply 'left' with 'right'

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    left : str, numeric
        Left column or value in expression
    right : str, numeric
        Right column or value in expression

    Returns
    -------
    result : Series
        Float series

    """
    expression = " * ".join(map(str, [left, right]))
    return gdf.eval(expression)


def buffer(gdf, radius, percentile, coverage="ahn", fill_value: float = None):
    """
    Percentile of coverage-value of an area defined by a radius around the
    object

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    radius: str, numeric
        Radius around object used to define a cirular area
    percentile : int
        The percentile of the coverage within area around object
    coverage : str, optional
        The coverage to use. The default value is 'ahn'
    fill_value : float, optional
        The fill_value to use when the area is not intersecting the coverage.
        The default is None

    Returns
    -------
    result : Series
        Float series

    """
    gdf_out = gdf.copy()
    gdf_out["result"] = np.nan
    xmin, ymin, xmax, ymax = gdf_out.total_bounds
    coverage_path = COVERAGES[coverage]

    index_gdf = gpd.read_file(coverage_path.joinpath("index.shp"))

    for idx, row in index_gdf.cx[xmin:xmax, ymin:ymax].iterrows():
        try:
            bathymetrie_raster = coverage_path.joinpath(
                f'{row["bladnr"].upper()}_CM.tif'
            )

            gdf_select = gdf_out.loc[
                gdf_out["geometry"].centroid.within(row["geometry"])
            ]
            if not gdf_select.empty:
                if isinstance(radius, str):
                    gdf_select.loc[:, ("geometry")] = gdf_select.apply(
                        _buffer_row, args=(radius,), axis=1
                    )
                else:
                    radius = max(radius, 0.5)
                    gdf_select.loc[:, ("geometry")] = gdf_select["geometry"].buffer(
                        radius
                    )

                with rasterio.open(bathymetrie_raster, "r") as src:
                    profile = src.profile
                    raster_data = src.read(1)
                    affine = src.transform
                    scale = src.scales[0]

                raster_stats = zonal_stats(
                    gdf_select,
                    raster_data,
                    affine=affine,
                    stats=f"percentile_{percentile}",
                    nodata=profile["nodata"],
                    raster_out=True,
                )

                gdf_out.loc[gdf_select.index.to_list(), "result"] = [
                    np.nan if item is None else round(item * scale, 2)
                    for item in [
                        item[f"percentile_{percentile}"] for item in raster_stats
                    ]
                ]
        except Exception as e:
            print(
                (
                    f"bathymetrie: {bathymetrie_raster}\n"
                    f"indices: {gdf_select.index}\n"
                    f"geometrien: {gdf_select['geometry']}"
                )
            )
            raise e

    # fill series if if provided
    if fill_value is not None:
        gdf_out.loc[gdf_out["result"].isna(), "result"] = fill_value

    return gdf_out["result"]


def join_parameter(
    gdf,
    join_object: str,
    join_gdf: gpd.GeoDataFrame,
    join_parameter: str,
    fill_value=None,
):
    """Joins a parameteer of other object to geodataframe."""
    _gdf = gdf.copy()

    _join_gdf = join_gdf.copy()
    _join_gdf.set_index("globalid", inplace=True)
    series = _join_gdf[join_parameter]
    series.name = "result"
    _gdf = _gdf.merge(series, how="left", left_on=f"{join_object}id", right_index=True)

    # fill series if if provided
    if fill_value is not None:
        _gdf.loc[_gdf["result"].isna(), "result"] = fill_value

    return _gdf["result"]


def object_relation(
    gdf,
    related_gdf: gpd.GeoDataFrame,
    code_relation: str,
    statistic: Literal["min", "max", "sum", "count"],
    related_parameter: str = None,
    fill_value=None,
):
    """
    Statistic of related object to geodataframe

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    related_gdf : GeoDataFrame
        GeoDataFrame with related attributes
    code_relation : str
        Column in related_gdf used to relate to gdf. Example 'stuwid'
    statistic : str, options: 'min', 'max', 'sum', 'count'
        Statistic to compute over related values
    related_parameter: str
        Column in related_gdf over which the statistic is to be computed
    fill_value : float, optional
        The fill_value to use when the area is not intersecting the coverage.
        The default is None

    Returns
    -------
    result : Series
        Float series

    """

    gdf_out = gdf.copy()

    # remove NaN values in from related_gdf[related_parameter]
    if related_parameter:
        if "geometry" in related_parameter:
            related_gdf[related_parameter] = _get_geometric_attribute(
                related_gdf, related_parameter
            )
        related_gdf = related_gdf.loc[related_gdf[related_parameter].notna()]

    # compute statistic
    if statistic == "count":
        series = related_gdf.groupby(by=[code_relation])[code_relation].count()
    elif statistic == "sum":
        series = related_gdf.groupby(by=[code_relation])[related_parameter].sum()
    elif statistic == "min":
        series = related_gdf.groupby(by=[code_relation])[related_parameter].min()
    elif statistic == "max":
        series = related_gdf.groupby(by=[code_relation])[related_parameter].max()
    elif statistic == "majority":
        series = related_gdf.groupby(by=[code_relation])[related_parameter].agg(
            pd.Series.mode
        )

    # join series with gdf
    series.name = "result"
    series = pd.DataFrame(series.loc[series.index.isin(gdf["globalid"])]).reset_index()
    gdf_out = gdf_out.merge(
        series, how="left", left_on="globalid", right_on=code_relation
    )

    # fill series if if provided
    if fill_value is not None:
        gdf_out.loc[gdf_out["result"].isna(), "result"] = fill_value
    return gdf_out["result"]


def custom_hydamo(
        gdf: GeoDataFrame,
        hydamo: HyDAMO,
        custom_function_name: str
) -> Series:
    """
    Call ``custom_function_name``, which needs to be defined in custom.py

    Parameters
    ----------
    gdf: GeoDataFrame
        GeoDataFrame to which the output series should conform
    hydamo : HyDAMO
        Input HyDAMO object
    custom_function_name: name of the function to be called
    """
    return getattr(custom_functions, custom_function_name)(gdf, hydamo)

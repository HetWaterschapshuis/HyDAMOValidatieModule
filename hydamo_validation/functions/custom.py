import pandas as pd
from geopandas import GeoDataFrame
from pandas import Series

from hydamo_validation.datamodel import HyDAMO


def example_custom_func(
        gdf: GeoDataFrame,  # required in signature of all custom funcs
        hydamo: HyDAMO,  # required in signature of all custom funcs
        **kwargs
) -> Series:
    """
    Proof of concept of a general hydamo rule

    Parameters
    ----------
    gdf: GeoDataFrame
        GeoDataFrame to which the output series should conform
    hydamo: HyDAMO
        Input HyDAMO object
    """
    # Do something with layers, e.g. hydamo.hydroobject is an ExtendedGeoDataFrame of hydroobject features
    nr_hydro_objects = len(hydamo.hydroobject)

    # Return a series that has the same length as the input gdf
    the_result = pd.Series(nr_hydro_objects, index=gdf.index)
    return the_result

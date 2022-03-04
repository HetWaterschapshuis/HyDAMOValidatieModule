"""Logic functions to be used in eval-method."""

import pandas as pd

def _overlapping_period(row, df, start_date, end_date):
    _df = df[df.index != row.name]
    return ~((row[start_date] <= _df[end_date]) & (row[end_date] >= _df[end_date])).any()

def _check_attributes(gdf, attributes):
    for i in attributes:
        if type(i) == str:
            if not i in gdf.columns:
                raise KeyError(fr"'{i}' not in columns: {gdf.columns.to_list()}. Rule cannot be executed")

def LE(gdf, left, right, dtype=bool):
    """
    Evaluate if left is less or equal to/than right

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    left : str, numeric
        Left column or value in expression
    right : TYPE
        Right column or value in expression
    dtype : dtype, optional
        dtype assigned to result Series
        The default is bool.

    Returns
    -------
    result : Series
        Pandas Series (default dtype = bool)

    """
    _check_attributes(gdf, [left, right])
    expression = f"{left} <= {right}".lower()
    return gdf.eval(expression).astype(dtype)


def LT(gdf, left, right, dtype=bool):
    """
    Evaluate if left is less than right

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    left : str, numeric
        Left column or value in expression
    right : TYPE
        Right column or value in expression
    dtype : dtype, optional
        dtype assigned to result Series
        The default is bool.

    Returns
    -------
    result : Series
        Pandas Series (default dtype = bool)

    """
    _check_attributes(gdf, [left, right])
    expression = f"{left} < {right}".lower()
    return gdf.eval(expression).astype(dtype)


def GT(gdf, left, right, dtype=bool):
    """
    Evaluate if left is greater than right

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    left : str, numeric
        Left column or value in expression
    right : TYPE
        Right column or value in expression
    dtype : dtype, optional
        dtype assigned to result Series
        The default is bool.

    Returns
    -------
    result : Series
        Pandas Series (default dtype = bool)

    """
    _check_attributes(gdf, [left, right])
    expression = f"{left} > {right}".lower()
    return gdf.eval(expression).astype(dtype)


def GE(gdf, left, right, dtype=bool):
    """Evaluate if left is greater or equal to/than right

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    left : str, numeric
        Left column or value in expression
    right : TYPE
        Right column or value in expression
    dtype : dtype, optional
        dtype assigned to result Series
        The default is bool.

    Returns
    -------
    result : Series
        Pandas Series (default dtype = bool)

    """
    _check_attributes(gdf, [left, right])
    expression = f"{left} >= {right}".lower()
    return gdf.eval(expression).astype(dtype)


def EQ(gdf, left, right, dtype=bool):
    """Evalate if left an right expression are equal

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    left : str, numeric
        Left column or value in expression
    right : TYPE
        Right column or value in expression
    dtype : dtype, optional
        dtype assigned to result Series
        The default is bool.

    Returns
    -------
    result : Series
        Pandas Series (default dtype = bool)

    """
    _check_attributes(gdf, [left, right])
    expression = f"{left} == {right}".lower()
    return gdf.eval(expression).astype(dtype)


def BE(gdf, parameter, min, max, inclusive=False):
    """Evaluate if parameter-value is between min/max inclusive (true/false)

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    parameter: str
        Input column with numeric values
    min : numeric
        Lower limit of function
    max : numeric
        Upper limit of function
    inclusive : bool, optional
        To include min and max
        The default is False.

    Returns
    -------
    result : Series
        Pandas Series (default dtype = bool)

    """
    _check_attributes(gdf, [parameter, min, max])
    if inclusive:
        series = GE(gdf, parameter, min, dtype=bool) & LE(
            gdf, parameter, max, dtype=bool
        )
    else:
        series = GT(gdf, parameter, min, dtype=bool) & LT(
            gdf, parameter, max, dtype=bool
        )
    return series


def ISIN(gdf, parameter, array):
    """Evaluate if values in parameter are in array

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    parameter: str
        Input column with numeric values
    array : list
        list of possible values that return True

    Returns
    -------
    result : Series
        Pandas Series (default dtype = bool)

    """
    _check_attributes(gdf, [parameter])
    return gdf[parameter].isin(array)


def NOTIN(gdf, parameter, array):
    """Evaluate if values in parameter are not in array

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    parameter: str
        Input column with numeric values
    array : list
        list of possible values that return False

    Returns
    -------
    result : Series
        Pandas Series (default dtype = bool)

    """
    _check_attributes(gdf, [parameter])
    return ~ISIN(gdf, parameter, array)


def NOTNA(gdf, parameter):
    """Evaluate if values in parameter ar not NaN or None

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    parameter: str
        Input column with numeric values

    Returns
    -------
    result : Series
        Pandas Series (default dtype = bool)

    """
    _check_attributes(gdf, [parameter])
    return gdf[parameter].notna()


def join_object_exists(gdf, join_gdf, join_object):
    """Evaluate if defined related_object id exists in globalid parameter of
    related object-table.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    related_gdf : GeoDataFrame
        Input GeoDataFrame with related objects
    object: str
        HyDAMO object name of related object-layer

    Returns
    -------
    result : Series
        Pandas Series (default dtype = bool)

    """
    _check_attributes(join_gdf, ["globalid"])
    _check_attributes(gdf, [f"{join_object}id"])
    return gdf[f"{join_object}id"].isin(join_gdf["globalid"])

def consistent_period(gdf,
                      max_gap=1,
                      groupers=["pompid", "regelmiddelid"],
                      priority="prioriteit",
                      start_date="beginperiode",
                      date_format = "%d%m",
                      end_date="eindperiode"):
    """Check if a periodic-based table is time-consistent

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame
    max_gap: int
        max gap in days between too adjacent periods

    Returns
    -------
    result : Series
        Pandas Series (default dtype = bool)

    """

    # create an empty result
    _gdf = gdf.copy()
    result = pd.Series(index = _gdf.index)

    # convert start_parameter and end_parameter to datetime
    _gdf[start_date] = pd.to_datetime(_gdf[start_date], format=date_format)
    _gdf[end_date] = pd.to_datetime(_gdf[end_date], format=date_format)

    index_select = _gdf[start_date] > _gdf[end_date]
    _gdf.loc[index_select, end_date] = _gdf[index_select][end_date] + pd.offsets.DateOffset(years=1)

    for group in groupers:
        grouper = _gdf.groupby(by=[group, "prioriteit"])
        
        for _, df in _gdf.groupby(by=["pompid","prioriteit"]):
            df.sort_values(by=start_date, inplace=True)

            #check for overlap
            bool_series = df.apply((lambda x:_overlapping_period(x, df, start_date, end_date)), axis=1)

            #check for gaps
            gaps_series = df[start_date] - df.shift(1)[end_date]
            gaps_series.iloc[0] = pd.Timedelta(days=0) # due to shift we have NaT here

            # add to result
            bool_series = (gaps_series <= pd.Timedelta(days=int(max_gap))) & bool_series
            bool_series = bool_series[bool_series.index.isin(result[result.isna() | (result == True)].index)]
            result.loc[result.index.isin(bool_series.index)] = bool_series

    return result

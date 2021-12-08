"""Logic functions to be used in eval-method."""


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
    return ~ISIN(gdf, parameter, array)

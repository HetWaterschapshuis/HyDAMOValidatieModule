"""Syntax validation."""

import numpy as np
import pandas as pd
from fiona.schema import FIELD_TYPES_MAP


# %% validation functions applied on datasets
def datamodel_layers(dm_layers, ds_layers):
    """Find non-datamodel layers in datasets."""
    layers = [i for i in ds_layers if dm_layers.count(i.lower()) == 1]
    return layers


def missing_layers(dm_layers, ds_layers):
    """Find missing datamodel-layers in datasets."""
    layers = [i.lower() for i in ds_layers]
    layers = [i for i in dm_layers if i not in layers]
    return layers


def duplicate_layers(layers):
    """Find duplicate layers in datasets."""
    error_layers = list(set([item for item in layers if layers.count(item) > 1]))
    error = {}
    critical = False

    # clean layers and report error
    if error_layers:
        error_layers_str = ",".join(error_layers)
        layers = [i for i in layers if i not in error_layers]
        error_layers_str = ",".join(error_layers)
        error = {
            "type": "critical",
            "message": f"datasets bevatten lagen vaker dan 1x die worden verwijderd: {error_layers_str}",
        }

    return layers, error, critical


# %% validation functions applied on dataframes
def non_unique_values(series):
    """Return non-unique values in series."""
    return series.duplicated(keep=False)


def missing_values(series, fill_values=[]):
    """Return non-unique values in series."""
    return series.isnull() | series.isin(fill_values)


def fill_values(series, dtype, fill_values=[-999]):
    """Return filled values in series."""
    return series.isin(np.array(fill_values).astype(dtype))


def non_domain_values(series, dtype, domain_values):
    """Return non-domain values in series."""
    bool_series = ~series.isin(np.array(domain_values).astype(dtype))
    bool_series.loc[series.isna()] = False  # for not na values
    return bool_series


def _convertable_dtype(x, dtype):
    try:
        if dtype == "datetime":
            return False
        elif dtype in ["int", "int64", "float"]:
            if x is None:
                return False
            result = pd.to_numeric(x)
            if dtype in ["int", "int64"]:
                return not "." in str(result)
        return True
    except ValueError:
        return False


def convertable_dtypes(series, dtype):
    """Find convertable dtypes."""
    return series.apply(_convertable_dtype, args=(dtype,))


def _get_constant_series(gdf, value=True, dtype=bool):
    return pd.Series([value] * len(gdf), index=gdf.index, dtype=dtype)


#%% to execute it all

# all fields syntax validation
def fields_syntax(gdf, schema, validation_schema, index, keep_columns=[]):
    """Validate fields in a gdf to a validation-schema."""
    # initialize results
    columns_tuple = tuple([index] + keep_columns + ["geometry"])
    validation_gdf = gdf.loc[:, columns_tuple].copy()
    result_gdf = gdf.copy()
    valid_summary = _get_constant_series(result_gdf)

    # check iteratively if column fails a valiation rule
    for col in [i for i in validation_schema if i["id"] != "geometry"]:
        print(col)
        valid_series = _get_constant_series(result_gdf)

        replace_series = _get_constant_series(result_gdf, value="", dtype=str)

        col_exists = col["id"] in result_gdf.columns
        result_col = f"syntax_{col['id']}"
        validation_gdf[result_col] = [[] for _ in range(len(validation_gdf))]

        if (not col_exists) and ("required" in col.keys()):
            # required columns should not be missing!
            if col["required"]:
                validation_gdf.loc[:, result_col].apply(lambda x: x.append(0))
                if col["dtype"] == "float":
                    result_gdf.loc[:, col["id"]] = np.NaN
                else:
                    result_gdf.loc[:, col["id"]] = pd.NA

        # first check if a column as the correct dtype
        else:
            dtype_fixed = True
            dtype = schema["properties"][col["id"]]
            if dtype != col["dtype"]:
                # try to convert it into the correct data-type
                convertable_rows = convertable_dtypes(
                    result_gdf.loc[:, col["id"]], col["dtype"]
                )

                # replace un-convertable rows to pd.NA so they can be converted
                replace_series[~convertable_rows] = result_gdf.loc[
                    ~convertable_rows, col["id"]
                ].apply(lambda x: f"{x} -> NULL ")
                if col["dtype"] == "float":
                    result_gdf.loc[~convertable_rows, col["id"]] = np.NaN
                else:
                    result_gdf.loc[~convertable_rows, col["id"]] = pd.NA

                if col["dtype"] in ["int", "int64"]:
                    result_gdf.loc[:, col["id"]] = pd.to_numeric(
                        result_gdf[col["id"]]
                    ).astype(pd.Int64Dtype())
                elif col["dtype"] == "datetime":
                    result_gdf.loc[:, col["id"]] = pd.to_numeric(
                        result_gdf[col["id"]]
                    ).astype("datetime64[ns]")
                else:
                    result_gdf.loc[:, col["id"]] = result_gdf[col["id"]].astype(
                        col["dtype"]
                    )
                validation_gdf.loc[convertable_rows, result_col].apply(
                    lambda x: x.append(2)
                )
                # mark all un-convertable rows as un-convertable
                validation_gdf.loc[~convertable_rows, result_col].apply(
                    lambda x: x.append(1)
                )
                valid_series.loc[~convertable_rows] = False
            else:
                convertable_rows = _get_constant_series(result_gdf)

            if dtype_fixed:
                # only unique values if specified
                if "unique" in col.keys():
                    if col["unique"] == True:
                        bool_series = non_unique_values(result_gdf[col["id"]])
                        validation_gdf.loc[bool_series, result_col].apply(
                            lambda x: x.append(3)
                        )
                        valid_series.loc[bool_series] = False

                # non missing if required
                if "required" in col.keys():
                    if col["required"] == True:
                        if col_exists:
                            nan_values = []
                            if "domain" in col.keys():
                                nan_values = [
                                    i["value"]
                                    for i in col["domain"]
                                    if i["value"] in [98, 99]
                                ]
                            bool_series = missing_values(
                                result_gdf[col["id"]], fill_values=nan_values
                            )
                            validation_gdf.loc[bool_series, result_col].apply(
                                lambda x: x.append(4)
                            )
                            valid_series.loc[bool_series] = False
                        else:
                            validation_gdf.loc[:, result_col].apply(
                                lambda x: x.append(4)
                            )
                            valid_series.loc[:] = False

                # return filled values
                if col["dtype"] != "datetime":
                    bool_series = fill_values(result_gdf[col["id"]], col["dtype"])
                    validation_gdf.loc[bool_series, result_col].apply(
                        lambda x: x.append(5)
                    )
                    valid_series.loc[bool_series] = False

                # return non-domain values
                if "domain" in col.keys():
                    domain_values = [i["value"] for i in col["domain"]]
                    bool_series = non_domain_values(
                        result_gdf[col["id"]], col["dtype"], domain_values
                    )
                    validation_gdf.loc[bool_series, result_col].apply(
                        lambda x: x.append(6)
                    )
                    valid_series.loc[bool_series] = False
                    valid_summary.loc[~valid_series] = False
        #%%
        # set all invalid data from original gdf to Null
        if col_exists:
            valid_series[~convertable_rows] = True
            replace_series.loc[~valid_series] = result_gdf.loc[
                ~valid_series, col["id"]
            ].apply(lambda x: f"{x} -> NULL ")
            if col["dtype"] == "float":
                result_gdf.loc[~valid_series, col["id"]] = np.NaN
            else:
                result_gdf.loc[~valid_series, col["id"]] = pd.NA

        # map list to string
        validation_gdf.loc[:, result_col] = validation_gdf[result_col].apply(
            lambda x: f'{",".join(map(str, x))}'
        )
        validation_gdf.loc[:, result_col] = validation_gdf[result_col].apply(
            lambda x: f"(fouten: {x})" if len(x) > 0 else x
        )
        validation_gdf.loc[:, result_col] = replace_series.str.cat(
            validation_gdf[result_col]
        )

    # check for invalid geometries and delete these geometries
    geotype = next(
        (i["dtype"] for i in validation_schema if i["id"] == "geometry"), None
    )
    if geotype:
        result_col = "syntax_geometry"
        validation_gdf[result_col] = ""

        bool_series = ~(
            result_gdf["geometry"].is_valid
            | (result_gdf["geometry"].type.isin(geotype))
        )
        validation_gdf.loc[bool_series, result_col] = f"{geotype} -> NULL (7)"
        result_gdf.loc[bool_series, "geometry"] = None
        valid_summary.loc[bool_series] = False

    # report if any validation error occured
    validation_gdf.loc[:, "syntax_oordeel"] = valid_summary.astype(bool)

    return result_gdf, validation_gdf

import pandas as pd
from geopandas import GeoDataFrame

from hydamo_validation.datamodel import HyDAMO
import numpy as np


def on_profiellijn_compute_wet_profile_distance(
    gdf: GeoDataFrame,
    hydamo: HyDAMO,
) -> pd.Series:
    """
    Computes the wet profile distance for each profiellijn in gdf.

    Returns a Series of distances (afstandNatProfiel) matching the gdf index.
    """
    profielpunt = hydamo.profielpunt

    if "afstand" not in profielpunt.columns:
        raise ValueError("afstand column is not present in profielpunt.")
    if "typeprofielpunt" not in profielpunt.columns:
        raise ValueError("typeprofielpunt column is not present in profielpunt.")

    wet_profile_distances = (
        profielpunt[profielpunt["typeprofielpunt"] == 22]
        .groupby("profiellijnid")["afstand"]
        .agg(lambda x: round(abs(x.iloc[-1] - x.iloc[0]), 2) if len(x) > 1 else np.nan)
        .reset_index()
    )
    wet_profile_distances = wet_profile_distances.rename(
        columns={"afstand": "afstandNatProfiel"}
    )

    # Map distances to gdf by globalid
    merged = gdf.merge(
        wet_profile_distances, left_on="globalid", right_on="profiellijnid", how="left"
    )
    result = merged["afstandNatProfiel"]
    result.index = gdf.index  # Ensure index matches input gdf

    num_computed = result.notna().sum()
    print(f"Succesfully computed wet profile distances for {num_computed} profiellijns")
    if len(result) - num_computed > 0:
        print(
            f"Failed to compute wet profile distances for {len(result) - num_computed} profiellijns"
        )

    return result


def on_profiellijn_compute_wet_profile_depth(
    gdf: GeoDataFrame,
    hydamo: HyDAMO,
) -> pd.Series:
    """
    Computes the wet profile depth for each profiellijn in gdf.

    Returns a Series of depths (dieptenatprofiel) matching the gdf index.
    """
    profielpunt = hydamo.profielpunt

    if "afstand" not in profielpunt.columns:
        raise ValueError("afstand column is not present in profielpunt.")
    if "typeprofielpunt" not in profielpunt.columns:
        raise ValueError("typeprofielpunt column is not present in profielpunt.")
    if "hoogte" not in profielpunt.columns:
        raise ValueError("hoogte column is not present in profielpunt.")

    wet_profile_depths = []
    for pl_id, group in profielpunt.groupby("profiellijnid"):
        wet_points = group[group["typeprofielpunt"] == 22]
        if len(wet_points) > 1:
            min_afstand = wet_points["afstand"].min()
            max_afstand = wet_points["afstand"].max()
            in_range = group[
                (group["afstand"] >= min_afstand) & (group["afstand"] <= max_afstand)
            ]
            if not in_range.empty:
                min_hoogte = in_range["hoogte"].min()
                max_hoogte = in_range["hoogte"].max()
                wet_profile_depths.append(
                    {
                        "profiellijnid": pl_id,
                        "dieptenatprofiel": round(max_hoogte - min_hoogte, 2),
                    }
                )
            else:
                wet_profile_depths.append(
                    {"profiellijnid": pl_id, "dieptenatprofiel": np.nan}
                )
        else:
            wet_profile_depths.append(
                {"profiellijnid": pl_id, "dieptenatprofiel": np.nan}
            )

    wet_profile_depths_df = pd.DataFrame(wet_profile_depths)
    merged = gdf.merge(
        wet_profile_depths_df, left_on="globalid", right_on="profiellijnid", how="left"
    )
    result = merged["dieptenatprofiel"]
    result.index = gdf.index  # Ensure index matches input gdf

    num_computed = result.notna().sum()
    print(f"Succesfully computed wet profile depths for {num_computed} profiellijns")
    if len(result) - num_computed > 0:
        print(
            f"Failed to compute wet profile depths for {len(result) - num_computed} profiellijns"
        )

    return result


def on_profiellijn_compute_max_cross_product(
    gdf: GeoDataFrame,
    hydamo: HyDAMO,
) -> pd.Series:
    """
    Add the maximum cross product of the segments of the LineString to the profiellijn.
    This is used to check if the LineString is straight (enough).
    """
    if not all(
        gdf.geometry.apply(
            lambda geom: geom.geom_type in ["LineString", "MultiLineString"]
        )
    ):
        raise ValueError(
            "All geometries in profiellijn must be LineString or MultiLineString."
        )

    max_cross_product = gdf["geometry"].apply(_is_linestring_straight)
    return max_cross_product


def on_profiellijn_compute_jaarinwinning(
    gdf: GeoDataFrame,
    hydamo: HyDAMO,
) -> pd.Series:
    """
    Compute the year of inwinning of the profiellijn based on the datuminwinning
    and add as a new column to self.profiellijn.
    """
    if "datuminwinning" not in gdf.columns:
        raise ValueError("datuminwinning column is not present in profiellijn.")

    # Convert datuminwinning to datetime
    gdf["datuminwinning"] = pd.to_datetime(gdf["datuminwinning"], errors="coerce")

    # Extract the year from datuminwinning
    jaarinwinning = gdf["datuminwinning"].dt.year

    return jaarinwinning


def on_profiellijn_add_breedte_value_from_hydroobject(
    gdf: GeoDataFrame,
    hydamo: HyDAMO,
) -> pd.Series:
    """
    Add the 'BREEDTE' value from hydroobject to profiellijn.
    The profielgroep is connected to the hydroobject by hydroobjectid.
    The profiellijn is connected to the profielgroep by profielgroepid.
    """
    hydroobject = hydamo.hydroobject
    profielgroep = hydamo.profielgroep

    # Ensure 'breedte' column exists in hydroobject
    if "breedte" not in hydroobject.columns:
        raise ValueError("Column 'breedte' is not present in hydroobject.")

    # Merge profiellijn with profielgroep to get hydroobjectid
    profiellijn_with_pg = gdf.merge(
        profielgroep[["globalid", "hydroobjectid"]],
        left_on="profielgroepid",
        right_on="globalid",
        how="left",
        suffixes=("", "_profielgroep"),
    )
    # Merge with hydroobject to get breedte
    profiellijn_with_breedte = profiellijn_with_pg.merge(
        hydroobject[["globalid", "breedte"]],
        left_on="hydroobjectid",
        right_on="globalid",
        how="left",
        suffixes=("", "_hydroobject"),
    )

    return profiellijn_with_breedte["breedte"]


def on_profiellijn_compute_if_ascending(
    gdf: GeoDataFrame,
    hydamo: HyDAMO,
) -> pd.Series:
    """
    Compute if the profielpunt features are in ascending order based on the 'hoogte' column.
    We determine the point with the lowest height
    And check if ascending both to the left and right of this point, based on the 'afstand' column.
    This is used to check if the profile is ascending.
    """
    profielpunt = hydamo.profielpunt

    if "hoogte" not in profielpunt.columns:
        raise ValueError("hoogte column is not present in profielpunt.")
    if "afstand" not in profielpunt.columns:
        raise ValueError("afstand column is not present in profielpunt.")

    # Ensure the hoogte column is numeric
    profielpunt["hoogte"] = pd.to_numeric(profielpunt["hoogte"], errors="coerce")

    ascending_per_lijn = profielpunt.groupby("profiellijnid").apply(_is_ascending)
    ascending_per_lijn.name = "isascending"

    # Convert to DataFrame for merge
    ascending_df = ascending_per_lijn.reset_index()

    # Merge on GlobalID (in profiellijn) = profielLijnID (in profielpunt)
    profiellijn = gdf.merge(
        ascending_df, how="left", left_on="globalid", right_on="profiellijnid"
    )

    return profiellijn["isascending"]


def _is_ascending(group: pd.DataFrame) -> int:
    """
    Check if the 'hoogte' values in a group form a V-shape based on 'afstand' order.

    The function determines whether the 'hoogte' (height) values first decrease (to a minimum)
    and then increase, forming a V-shape when sorted by 'afstand' (distance).

    Parameters
    ----------
    group : pd.DataFrame
        DataFrame with at least two columns: 'afstand' and 'hoogte'.

    Returns
    -------
    int
        1: if the group is V-shaped (descending to minimum then ascending),
        0: otherwise
    Returns False if the group is empty.
    """
    if group.empty:
        return False
    group = group.sort_values(by="afstand").reset_index(drop=True)

    min_index = group["hoogte"].idxmin()
    left = group.loc[:min_index, "hoogte"]
    right = group.loc[min_index:, "hoogte"]

    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)

    left_descending = all(
        left.iloc[i] >= left.iloc[i + 1] for i in range(len(left) - 1)
    )
    right_ascending = all(
        right.iloc[i] <= right.iloc[i + 1] for i in range(len(right) - 1)
    )

    if left_descending and right_ascending:
        return 1
    else:
        return 0


def _is_linestring_straight(line) -> float:
    """
    Check if a LineString is straight by checking the collinearity of each segment.
    A LineString is considered straight if all segments are collinear, but we allow a small tolerance.
    This tolerance is determined in the validation module (validationrules.json).
    In this function, the maximum cross product of the segments is computed.
    """

    coords = list(line.coords)
    if len(coords) < 3:
        return True  # Two points always form a straight line

    # Use the first segment as the reference vector
    x0, y0 = coords[0]
    x1, y1 = coords[1]
    dx_ref = x1 - x0
    dy_ref = y1 - y0

    max_cross = 0
    for i in range(1, len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        # Use cross product to check colinearity (should be zero if vectors are colinear)
        cross = dx_ref * dy - dy_ref * dx

        if abs(cross) > max_cross:
            max_cross = abs(cross)

    return max_cross

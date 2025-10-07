# %%
import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from hydamo_validation.datamodel import HyDAMO
import numpy as np
from shapely.geometry import LineString
# %%


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


def intersected_peilgebieden(gdf: GeoDataFrame, hydamo: HyDAMO):
    """ "
    Check if the peilgebieden intersect with each other.
    """
    # Check for intersections
    combinatiepeilgebied_gdf = hydamo.combinatiepeilgebied
    polder_polygon_gdf = hydamo.polder

    # hydamo = r"E:\09.modellen_speeltuin\test_jk1\01_source_data\HyDAMO.gpkg"
    # hydro_object_gdf = gpd.read_file(hydamo, layer="hydroobject", driver="GPKG")
    # combinatiepeilgebied_gdf = gpd.read_file(
    #     hydamo, layer="combinatiepeilgebied", driver="GPKG"
    # )
    # polder_polygon_gdf = gpd.read_file(hydamo, layer="polder", driver="GPKG")

    # print(type(polder_polygon_gdf))
    # make a copy of the gdf
    combinatiepeilgebied_copy = combinatiepeilgebied_gdf.copy()

    # clip peilgebieden with polder polygon
    combinatiepeilgebied_copy = combinatiepeilgebied_copy.clip(polder_polygon_gdf)

    # intersect combinatiepeilgebied with itself to find overlapping areas
    intersection = combinatiepeilgebied_copy.overlay(
        combinatiepeilgebied_copy, how="intersection"
    )
    # if code 1 and code 2 are the same, drop the row
    intersection = intersection[intersection["code_1"] != intersection["code_2"]]

    # add column to combinatiepeilgebied_gdf 'intersectie_codes' with code_1 and code_2 save it as list separated by ', '
    intersection["intersectie_codes"] = (
        intersection["code_1"].astype(str) + ", " + intersection["code_2"].astype(str)
    )

    codes = intersection["code_1"].to_list()
    for code in codes:
        # select row
        code_selection = intersection[intersection["code_1"] == code]
        code_string = " - ".join(code_selection["intersectie_codes"].values.tolist())

        # calculate the number of codes in the intersectie_codes column
        number_of_codes = len(
            (
                " - ".join(
                    str(x) for x in code_selection["intersectie_codes"].values.tolist()
                )
            ).split(" - ")
        )

        # add number of codes to combinatiepeilgebied_gdf in column 'aantal_intersecties'
        combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "aantal_intersecties"
        ] = number_of_codes
        # add the column 'intersectie_codes' to the combinatiepeilgebied_gdf
        combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "intersectie_codes"
        ] = code_string

    # fill empty values in aantal_intersecties with 0
    combinatiepeilgebied_gdf["aantal_intersecties"] = combinatiepeilgebied_gdf[
        "aantal_intersecties"
    ].fillna(0)

    # save combinatiepeilgebied_gdf into hydamo

    return combinatiepeilgebied_gdf


def intersected_pump_peilgebieden(gdf: GeoDataFrame, hydamo: HyDAMO):
    """
    Calculate the distance from each gemaal to the nearest peilgebied in combinatiepeilgebied.
    Add this distance as a new column 'distance_to_peilgebied' in the gemaal layer.
    The distance is calculated in centimeters.
    """
    combinatiepeilgebied_gdf = hydamo.combinatiepeilgebied
    polder_polygon_gdf = hydamo.polder
    gemaal_gdf = hydamo.gemaal

    # make a copy of the combinatiepeilgebied.
    gdf_peilgebiedpraktijk_linestring = combinatiepeilgebied_gdf.copy()

    # transform multiploygon to lines geom.boundary
    gdf_peilgebiedpraktijk_linestring["geometry"] = gdf_peilgebiedpraktijk_linestring[
        "geometry"
    ].apply(lambda geom: geom.boundary if geom.geom_type == "MultiPolygon" else geom)

    # explode lines into part and reset index
    gdf_peilgebiedpraktijk_linestring = gdf_peilgebiedpraktijk_linestring.explode(
        index_parts=True
    )
    gdf_peilgebiedpraktijk_linestring = gdf_peilgebiedpraktijk_linestring.reset_index(
        drop=True
    )

    # clip linestrings to the polder_polygon
    gdf_peilgebiedcombinatie = gpd.clip(
        gdf_peilgebiedpraktijk_linestring, polder_polygon_gdf
    )

    # make spatial join between gdf_gemaal and gdf_peilgebiedpraktijk_linestring distance 1000 cm
    gemaal_spatial_join = gpd.sjoin_nearest(
        gemaal_gdf,
        gdf_peilgebiedcombinatie,
        how="left",
        max_distance=1000,
        distance_col="distance_to_peilgebied",
    )
    # rename column code_left to code
    if "code_left" in gemaal_spatial_join.columns:
        gemaal_spatial_join = gemaal_spatial_join.rename(columns={"code_left": "code"})

    # Join the column 'distance_to_peilgebied' from gemaal_spatial_join into gdf_gemaal based on the 'code' column
    gdf_gemaal_merge = gemaal_gdf.merge(
        gemaal_spatial_join[["code", "distance_to_peilgebied"]],
        on="code",
        how="left",
    )

    gdf_gemaal = gdf_gemaal_merge
    return gdf_gemaal


def gemaal_streefpeil_value(gdf: GeoDataFrame, hydamo: HyDAMO):
    """
    Add the columns 'pgd_codes', 'streefpeil_peilgebide_zomer', 'streefpeil_peilgebide_winter',
    'soort_streefpeilom_comb', 'peilgebied_soort_comb', 'gemaal_functie_value' to the gemaal layer.
    The values are derived from the intersection of the gemaal point with the combinatiepeilgebied polygons.
    If a gemaal intersects multiple peilgebieden, the values are concatenated into a comma-separated string.
    The column 'gemaal_functie_value' is determined based on the uniqueness of the summer target levels.

    """
    # load data
    combinatiepeilgebied_gdf = hydamo.combinatiepeilgebied
    gemaal_hydamo = hydamo.gemaal

    # get the border of the poligons
    combinatiepeilgebied_lines = combinatiepeilgebied_gdf.copy()
    combinatiepeilgebied_lines["geometry"] = (
        combinatiepeilgebied_lines.geometry.boundary
    )

    # get the border of the poligons
    peilgrens = combinatiepeilgebied_lines.copy()
    peilgrens["geometry"] = peilgrens.buffer(0.10)

    # Make a buffer using the gemaal point shapefile to be intersected with the combinatiepeilgebied
    buffer_gemaal = gemaal_hydamo.buffer(distance=1)

    # Intersect buffer_gemaal with combinatiepeilgebied
    buffer_gemaal_gdf = gpd.GeoDataFrame(
        gemaal_hydamo.copy(), geometry=buffer_gemaal, crs=gemaal_hydamo.crs
    )
    # intersect gemaaal buffer with the combined peilgebied
    gemaal_intersect_peilgebied = gpd.overlay(
        buffer_gemaal_gdf, combinatiepeilgebied_gdf, how="intersection"
    )

    # Rename all the columns, removing the "_1" if they have it
    gemaal_intersect_peilgebied.columns = [
        col[:-2] if col.endswith("_1") else col
        for col in gemaal_intersect_peilgebied.columns
    ]

    # select columns 'streefpeil_zomer', 'streefpeil_winter', 'soort_streefpeilom' to join from gemaal_intersect_peilgebied
    columns_to_join = [
        "streefpeil_zomer",
        "streefpeil_winter",
        "soort_streefpeilom",
        "peilgebied_soort",
        "code_2",
    ]

    # gather all the columns to be used
    columns_to_keep = gemaal_hydamo.columns.to_list() + columns_to_join

    # create a subset using the desired columns, and drop duplicates from the intersection between gemaal and the combined peilgebied
    gemaal = gemaal_intersect_peilgebied[columns_to_keep].drop_duplicates()

    # Select codes in a list to be used in a for loop
    codes = gemaal["code"].to_list()
    for code in codes:
        # Select features with the same code value
        code_selection = gemaal[gemaal["code"] == code]

        # Save the codes from the peilgebiede to be store in a new column that columns come from code_2 (code of the peilgebied)
        pgd_codes = ", ".join(str(x) for x in code_selection["code_2"].values.tolist())
        values_zomer = ", ".join(
            str(x) for x in code_selection["streefpeil_zomer"].values.tolist()
        )
        values_winter = ", ".join(
            str(x) for x in code_selection["streefpeil_winter"].values.tolist()
        )
        peil_gebied_soort = ", ".join(
            str(x) for x in code_selection["peilgebied_soort"].values.tolist()
        )
        soort_streefpeilom = ", ".join(
            str(x) for x in code_selection["soort_streefpeilom"].values.tolist()
        )

        # Save the code from de peilgebied in the column pgd_codes
        gemaal.loc[gemaal["code"] == code, "pgd_codes"] = pgd_codes
        gemaal.loc[gemaal["code"] == code, "streefpeil_peilgebide_zomer"] = values_zomer
        gemaal.loc[gemaal["code"] == code, "streefpeil_peilgebide_winter"] = (
            values_winter
        )
        gemaal.loc[gemaal["code"] == code, "soort_streefpeilom_comb"] = (
            soort_streefpeilom
        )
        gemaal.loc[gemaal["code"] == code, "peilgebied_soort_comb"] = peil_gebied_soort

    # dissolve the values using the column code and drop the columns:"streefpeil_zomer", "streefpeil_winter", "soort_streefpeilom", "code_2"
    gemaal_disolve = gemaal.dissolve(by="code").drop(
        columns=[
            "streefpeil_zomer",
            "streefpeil_winter",
            "soort_streefpeilom",
            "code_2",
        ]
    )

    # fucntie gemaal: 1 Sumply, 2, dranage: To avoid the level raises to much, 3.
    # Select columns to join
    columns_to_merge = [
        "code",
        "pgd_codes",
        "streefpeil_peilgebide_zomer",
        "streefpeil_peilgebide_winter",
        "soort_streefpeilom_comb",
        "peilgebied_soort_comb",
    ]
    # reset the index column and select only the columns set it in the previous step
    gemaal_buffer_subset = gemaal_disolve.reset_index(drop=False)[columns_to_merge]

    # Merge the dataframes on the 'code' column using the gemaal buffer subet
    gemaal_point = gemaal_hydamo.drop_duplicates().merge(
        gemaal_buffer_subset, on="code", how="left"
    )

    # Add the column 'gemaal_functie_value' based on the 'soort_streefpeilom_comb' column, use the list gemaal_functie_test to store the values
    gemaal_functie_test = []
    for zomer_values in gemaal_point["streefpeil_peilgebide_zomer"]:
        diff = np.diff([float(zomer_value) for zomer_value in zomer_values.split(",")])
        if len(diff) <= 1 and (not (diff.tolist()) or diff.tolist()[0] == 0):
            gemaal_functie_test.append(4)
        else:
            gemaal_functie_test.append(-999)
    # add the gemaal_functie_test to the gemaal_point dataframe
    gemaal_point["gemaal_functie_test"] = gemaal_functie_test
    # copy column functiegemaalcode in new column functiegemaalcode_DAMO
    gemaal_point["functiegemaalcode_damo"] = gemaal_point["functiegemaal"]

    # Add the column 'aantal_peilgebieden' based on the 'pgd_codes' column
    for count_peilgebieden in gemaal_point["pgd_codes"]:
        if count_peilgebieden is not None and count_peilgebieden != "":
            # count the number of peilgebieden
            count = len(count_peilgebieden.split(","))
            gemaal_point.loc[
                gemaal_point["pgd_codes"] == count_peilgebieden, "aantal_peilgebieden"
            ] = count
        else:
            gemaal_point.loc[
                gemaal_point["pgd_codes"] == count_peilgebieden, "aantal_peilgebieden"
            ] = 999
    # save the gemaal_point into hydamo
    hydamo.gemaal = gemaal_point
    return hydamo.gemaal


# TODO: ask: IS DISTANCE FINE? EVERY TIME I USE IT DO I GET A NEW SECTIONS, OR THEY KEPP THE SAME
# todo: TEST THE FUNCTION: MAYBE A PYTEST?
def split_segments_atvertex_and_distance(peilgrens, distance=100):
    """
    helper function that split the boundary of the peilgebiede, also known as peilgrens is smaller sections.
    It also split the lines in in the vertex to keep the original shape.
    """

    # get lines from peilegebied-> Peilgrens
    segments = []
    peilgrens_explode = peilgrens.explode()

    # initialize the loop
    for idx, row in peilgrens_explode.iterrows():
        # get coordinate of each geometry
        coords = list(row.geometry.coords)

        # split at vertices
        for i in range(len(coords) - 1):
            segment = LineString([coords[i], coords[i + 1]])
            length = segment.length

            # If the segment is shorter than 100 m, keep it as is and do next
            if length <= distance:
                new_row = row.copy()
                new_row.geometry = segment
                segments.append(new_row)
            else:
                # split into the choossen distance
                start = 0
                while start < length:
                    end = min(start + distance, length)
                    # create the new segment according to the coordinate
                    new_seg = LineString(
                        [segment.interpolate(start), segment.interpolate(end)]
                    )

                    new_row = row.copy()
                    new_row.geometry = new_seg
                    segments.append(new_row)

                    # sum the distance
                    start += distance

    return segments


def peil_basic_properties(gdf: GeoDataFrame, hydamo: HyDAMO):
    """
    Calculate basic properties per combinatiepeilgebied:
    - Area of each peilgebied
    - Number of hydroobjects inside each peilgebied
    - 'code_holes' indicating if the peilgebied intersects any hole inside the polder

    """
    # Load layers
    hydro_object_gdf = hydamo.hydroobject
    combinatiepeilgebied_gdf = hydamo.combinatiepeilgebied
    polder_gdf = hydamo.polder

    # hydamo = r"E:\09.modellen_speeltuin\test_jk1\01_source_data\HyDAMO.gpkg"
    # hydro_object_gdf = gpd.read_file(hydamo, layer="hydroobject", driver="GPKG")
    # combinatiepeilgebied_gdf = gpd.read_file(
    #     hydamo, layer="combinatiepeilgebied", driver="GPKG"
    # )
    # polder_gdf = gpd.read_file(hydamo, layer="polder", driver="GPKG")

    # clip combinatiepeilgebied_gdf using the polder polygon
    polder_geom = polder_gdf.geometry.union_all()
    peil_clipped = gpd.clip(combinatiepeilgebied_gdf, polder_geom)

    # Do spatial join between hydrooboject and peil_clipped to get the
    # hydrooboject that are "within" each peilgebiede
    joined = gpd.sjoin(hydro_object_gdf, peil_clipped, predicate="within")

    # Loop through unique peilgebied codes
    for code in combinatiepeilgebied_gdf["code"]:
        # sum the numbers peilgebieden codes.
        count = (joined["code_right"] == code).sum()
        # new column and add the numbers of codes
        combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "hydroobject_count"
        ] = count

        # select geometry from de combinatiepeilgebied_gdf
        geometry = combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code
        ].geometry.iloc[0]
        holes = []
        # Check if it is polygon or multipolygon
        if geometry.type == "MultiPolygon":
            for polygon in geometry.geoms:
                # append the number of holes per polygon.interirors (cool one)
                holes.append(len(polygon.interiors))
                # print(code, len(polygon.interiors))

        else:
            # append the number of holes per polygon
            holes.append(len(polygon.interiors))

        # sum how many holes does each feature polygon has
        count_holes = np.sum(holes)
        combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "holes_in_polygon"
        ] = count_holes

    # Area
    combinatiepeilgebied_gdf["area"] = combinatiepeilgebied_gdf.geometry.area

    # Add peil_meer_30cm to be use for validation rules
    combinatiepeilgebied_gdf["peil_meer_30cm"] = (
        combinatiepeilgebied_gdf["streefpeil_zomer_bovengrens"] + 0.30
    )

    # Add limit_winter_ondergrens_minder_10cm to be use for validation rules
    combinatiepeilgebied_gdf["limit_winter_ondergrens_minder_10cm"] = (
        combinatiepeilgebied_gdf["streefpeil_winter_ondergrens"] - 0.10
    )

    # Add limit_zomer_bovengrens_plus_10cm to be use for validation rules
    combinatiepeilgebied_gdf["limit_zomer_bovengrens_plus_10cm"] = (
        combinatiepeilgebied_gdf["streefpeil_zomer_bovengrens"] + 0.10
    )

    return combinatiepeilgebied_gdf


def peil_verbonde(gdf: GeoDataFrame, hydamo: HyDAMO):
    """check if the polygons of the combinatiepeilgebiede are connected. It return two columns:
    'connected', 'num_vertices'
    """

    # Load layers
    combinatiepeilgebied_gdf = hydamo.combinatiepeilgebied
    polder_gdf = hydamo.polder

    # hydamo = r"E:\09.modellen_speeltuin\test_jk1\01_source_data\HyDAMO.gpkg"
    # combinatiepeilgebied_gdf = gpd.read_file(hydamo, layer='combinatiepeilgebied', driver = 'GPKG')
    # polder_gdf = gpd.read_file(hydamo, layer='polder', driver = 'GPKG')

    # clip with polder
    peil_clipped = gpd.clip(combinatiepeilgebied_gdf, polder_gdf)

    # initialized columns
    peil_clipped["connected"] = True
    peil_clipped["num_vertices"] = 0

    # loop over each polyogn and count vertex. Check if they are connected
    for idx, row in peil_clipped.iterrows():
        geom = row["geometry"]

        # count vertex if the strucutre is polygon othewise do as multiploygon
        if geom.type == "Polygon":
            peil_clipped.at[idx, "num_vertices"] = len(geom.exterior.coords)
            peil_clipped.at[idx, "connected"] = True

        elif geom.type == "MultiPolygon":
            peil_clipped.at[idx, "num_vertices"] = sum(
                len(p.exterior.coords) for p in geom.geoms
            )
            # check if a part touches anohter
            polygons = list(geom.geoms)
            connected = False
            for i in range(len(polygons)):
                for j in range(i + 1, len(polygons)):
                    if polygons[i].touches(polygons[j]):
                        connected = True
                        break
                if connected:
                    break
            peil_clipped.at[idx, "connected"] = connected

    # columns to copy
    cols_to_copy = ["code", "connected", "num_vertices"]
    peil_copy = peil_clipped[cols_to_copy]

    # merge by code
    combinatiepeilgebied_gdf = combinatiepeilgebied_gdf.merge(
        peil_copy, on="code", how="left"
    )
    return combinatiepeilgebied_gdf


def peilgebieded_waterstand_dm(gdf: GeoDataFrame, hydamo: HyDAMO):
    """
    Calculates the median water level (waterstand) for each peilgebied and compares it to the
    streefpeil (target water level) boundaries defined in HyDAMO. Adds buffered limits (+/- 10 cm)
    to the streefpeil values and stores the results in the GeoDataFrame.
    """
    # Load layers
    combinatiepeilgebied_gdf = hydamo.combinatiepeilgebied
    polder_gdf = hydamo.polder
    # TODO ASK: is it correct to put this path here? Or should it be passed as a parameter?
    water_stand_punten_path = r"G:\01_Componenten\01_Beheer\03_Objectdata_HHNK\01_Watersysteem\01_Kwantiteit\01_Waterlopen\Datamining_waterpeilhoogtes_AHN.gdb"

    # hydamo = r"E:\09.modellen_speeltuin\test_jk1\01_source_data\HyDAMO.gpkg"
    # combinatiepeilgebied_gdf = gpd.read_file(
    #     hydamo, layer="combinatiepeilgebied", driver="GPKG", engine="pyogrio"
    # )
    # polder_gdf = gpd.read_file(hydamo, layer="polder", driver="GPKG", engine="pyogrio")
    water_stand_gdf = gpd.read_file(
        water_stand_punten_path, layer="Waterpeilhoogtes_AHN4", engine="pyogrio"
    )

    # clip polder
    peil_clipped = gpd.clip(combinatiepeilgebied_gdf, polder_gdf)
    # jon the water s
    water_stand_clip = gpd.sjoin_nearest(
        peil_clipped,
        water_stand_gdf,
        max_distance=0.01,
        distance_col="distance_to_peil",
        how="left",
    )
    codes = water_stand_clip["code"].to_list()

    for code in codes:
        code_group = water_stand_clip.loc[water_stand_clip["code"] == code]

        # May be needed in the futre?
        # waterstand_min = min(code_group['height'].values)
        # waterstand_max = max(code_group['height'].values)
        # waterstand_mean = np.mean(code_group['height'].values)

        # take waterstand median.
        waterstand_median = np.median(code_group["height"].values)

        if pd.isna(waterstand_median):
            waterstand_median = -999
        # Get upper and lower limit from the peilgebieden
        streefpeil_winter_ondergrens = combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "streefpeil_winter_ondergrens"
        ]
        streefpeil_winter_bovengrens = combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "streefpeil_winter_bovengrens"
        ]
        streefpeil_zomer_ondergrens = combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "streefpeil_zomer_ondergrens"
        ]
        streefpeil_zomer_bovengrens = combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "streefpeil_zomer_bovengrens"
        ]

        # Set the upper limit according to AHN5 (+-10cm)
        limit_winter_ondergrens = streefpeil_winter_ondergrens.values[0] - 0.10
        limit_winter_bovengrens = streefpeil_winter_bovengrens.values[0] + 0.10
        limit_zomer_ondergrens = streefpeil_zomer_ondergrens.values[0] - 0.10
        limit_zome_bovengrens = streefpeil_zomer_bovengrens.values[0] + 0.10

        # WRITE RESUTLS
        combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "limit_winter_ondergrens[10cm]"
        ] = limit_winter_ondergrens
        combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "limit_winter_bovengrens[+ 10cm]"
        ] = limit_winter_bovengrens
        combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "limit_zomer_ondergrens[- 10cm]"
        ] = limit_zomer_ondergrens
        combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "limit_zomer_bovengrens[+ 10cm]"
        ] = limit_zome_bovengrens
        combinatiepeilgebied_gdf.loc[
            combinatiepeilgebied_gdf["code"] == code, "waterstand_median"
        ] = waterstand_median

    # path  = r"E:\09.modellen_speeltuin\test_jk1\01_source_data\waterstand.gpkg"
    return combinatiepeilgebied_gdf


# %%

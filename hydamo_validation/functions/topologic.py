"""Topologic functions executed on extended geodataframe."""

from typing import Literal
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from hydamo_validation import geometry

""" 
In this block we define supporting functions. Supporting functions are ignored
by the logical-validation module. They are used by and shared between topological
validation rules.
"""


def _layers_from_datamodel(layers, datamodel):
    if len(layers) > 1:
        series = pd.concat([getattr(datamodel, i)["geometry"] for i in layers])
    else:
        series = getattr(datamodel, layers[0])["geometry"]
    return series.loc[series.index.notna()]


def _lines_snap_at_boundaries(line, other_line, tolerance):
    snaps_start = any(
        line.boundary.geoms[0].distance(i) < tolerance
        for i in other_line.boundary.geoms
    )
    snaps_end = any(
        line.boundary.geoms[-1].distance(i) < tolerance
        for i in other_line.boundary.geoms
    )
    return snaps_start, snaps_end


def _point_not_overlapping_line(point, line, tolerance):
    if line.boundary:
        snaps_start = line.boundary.geoms[0].distance(point) < tolerance
        snaps_end = line.boundary.geoms[-1].distance(point) < tolerance
    else:
        snaps_start = snaps_end = (
            Point(list(line.coords)[0]).distance(point) < tolerance
        )
    if not any([snaps_start, snaps_end]):
        not_overlapping = line.distance(point) > tolerance
    else:
        not_overlapping = True
    return not_overlapping


def _line_not_overlapping_line(line, other_line, tolerance):
    # check if lines snap at boundaries
    snaps_start, snaps_end = _lines_snap_at_boundaries(line, other_line, tolerance)

    # in the case we have two lines with only and-points and all end-points overlap, these are overlapping lines
    if (
        (snaps_start and snaps_end)
        and (len(line.coords) == 2)
        and (len(other_line.coords) == 2)
    ):
        return False

    # compare the shortest line with the longest line
    # if ends snap, we check all coordinates in between, else we check all.
    if line.length < other_line.length:
        not_overlapping = all(
            (
                _point_not_overlapping_line(Point(i), other_line, tolerance)
                for i in line.coords
            )
        )
    else:
        not_overlapping = all(
            (
                _point_not_overlapping_line(Point(i), line, tolerance)
                for i in other_line.coords
            )
        )

    return not_overlapping


def _not_overlapping_line(row, gdf, sindex, tolerance, exclude_row=True):
    geometry = row["geometry"]
    indices = list(sindex.intersection(geometry.buffer(tolerance).bounds))
    if exclude_row:
        indices = [i for i in indices if i != gdf.index.get_loc(row.name)]
    if indices:
        gdf_select = gdf.iloc[indices]
        if geometry.geom_type == "LineString":
            not_overlapping = all(
                _line_not_overlapping_line(geometry, i, tolerance)
                for i in gdf_select["geometry"]
            )
        elif geometry.geom_type == "Point":
            not_overlapping = all(
                _point_not_overlapping_line(geometry, i, tolerance)
                for i in gdf_select["geometry"]
            )
    else:
        not_overlapping = True
    return not_overlapping


def _not_overlapping_point(row, gdf, sindex, tolerance, exclude_row=True):
    geometry = row["geometry"]
    indices = list(sindex.intersection(geometry.buffer(tolerance).bounds))
    if exclude_row:
        indices = [i for i in indices if i != gdf.index.get_loc(row.name)]
    if indices:
        gdf_select = gdf.iloc[indices]
        not_overlapping = all(
            geometry.distance(i) > tolerance for i in gdf_select["geometry"]
        )
    else:
        not_overlapping = True
    return not_overlapping


def _snap_nodes(row, series, tolerance):
    series_selec = series.loc[~(series.index == row.name)]
    indices = series_selec.loc[
        (series_selec.distance(row["geometry"]) < tolerance)
    ].index.to_list()
    geom = None
    if indices:
        indices.sort()
        if indices[0] < row.name:
            geom = series.loc[indices[0]]
    if geom is None:
        geom = row["geometry"]

    return geom


def _get_nodes(gdf, tolerance):
    # start and end-nodes to GeoSeries
    nodes_series = gdf["geometry"].apply(lambda x: Point(x.coords[0]))
    nodes_series = pd.concat(
        [nodes_series, gdf["geometry"].apply(lambda x: Point(x.coords[-1]))]
    ).reset_index(drop=True)

    # snap nodes within tolerance: nodes within tolerance get the coordinate
    # of the first node.
    nodes_series = gpd.GeoSeries(
        gpd.GeoDataFrame(nodes_series, columns=["geometry"]).apply(
            lambda x: _snap_nodes(x, nodes_series, tolerance), axis=1
        )
    )

    # as all is snapped we can filter unique points
    nodes_series = gpd.GeoSeries(nodes_series.unique())

    return nodes_series


def _only_end_nodes(row, series, sindex, tolerance):
    geometry = row["geometry"]
    indices = list(sindex.intersection(geometry.bounds))
    if indices:
        series_select = series.loc[indices]
        only_end_nodes = all(
            _point_not_overlapping_line(i, geometry, tolerance) for i in series_select
        )
    else:
        only_end_nodes = True

    return only_end_nodes


def _structure_at_intersection(
    geometry, other_geometry, struc_gdf, struc_sindex, tolerance
):
    # compute the intersection and buffer it with tolerance
    intersection = geometry.intersection(other_geometry).buffer(tolerance)

    # get the structures that intersect the intersection bounds
    indices = struc_sindex.intersection(intersection.bounds)

    # check if any of these structures are within intersection
    structure_present = any(i.intersects(intersection) for i in struc_gdf.iloc[indices])

    return structure_present


def _structures_at_intersections(
    row, gdf, sindex, struc_series, struc_sindex, tolerance
):
    geometry = row["geometry"]

    # get a selection of hydroobjects that intersect geometry
    indices = sindex.intersection(geometry.bounds)
    indices = [i for i in indices if i != gdf.index.get_loc(row.name)]
    gdf_select = gdf.iloc[indices]
    gdf_select = gdf_select.loc[gdf_select["geometry"].crosses(geometry)]

    # if not emtpy, check if there is a structure at every intersection
    if not gdf_select.empty:
        structure_at_intersections = (
            gdf_select["geometry"]
            .apply(
                lambda x: _structure_at_intersection(
                    x, geometry, struc_series, struc_sindex, tolerance
                )
            )
            .all()
        )
    else:
        structure_at_intersections = True

    return structure_at_intersections


def _intersects_end_node(geometry, series, sindex, tolerance):
    # get inidices in bounds around geometry
    geometry = geometry.buffer(tolerance)
    indices = list(sindex.intersection(geometry.bounds))
    # see if there are start-nodes within within tolerance
    series_select = series.iloc[indices]
    series_select = series_select.loc[series_select.within(geometry)]

    # if dataframe is emtpy, there are no start nodes intersecting end-node
    if series_select.empty:
        intersects_start_node = False
    else:
        intersects_start_node = True
    return intersects_start_node


def _structure_at_boundary(geometry, series, distance):
    return (series.distance(geometry) < distance).any()


def _structures_at_boundaries(
    row, areas_gdf, areas_sindex, struc_series, struc_sindex, tolerance, distance
):
    geometry = row["geometry"]

    # check if hydroobject crosses boundary
    indices = list(areas_sindex.intersection(geometry.bounds))
    areas_gdf_select = areas_gdf.iloc[indices]
    areas_gdf_select = areas_gdf_select.loc[
        areas_gdf_select["geometry"].boundary.intersects(geometry)
    ]

    if areas_gdf_select.empty:
        structure_at_boundary = True
    else:
        # check if hydroobject intersects structures within tolerance
        indices = list(struc_sindex.intersection(geometry.bounds))
        struc_series_select = struc_series.iloc[indices]
        struc_series_select = struc_series_select.loc[
            struc_series_select.distance(geometry) < tolerance
        ]
        if struc_series_select.empty:
            # if there are boundaries, there should be structures
            structure_at_boundary = False
        else:
            # if there are structures they should be close enough to the boundary
            # create series with intersections
            intersections_series = areas_gdf_select["geometry"].boundary.intersection(
                geometry
            )

            # if there is a structure close enough to all boundaries: True. Else: False
            structure_at_boundary = intersections_series.apply(
                lambda x: _structure_at_boundary(x, struc_series_select, distance)
            ).all()

    return structure_at_boundary


def _distant_to_others(row, gdf, sindex, distance):
    geometry = row["geometry"]

    # get a selection of other objects that intersect with
    indices = sindex.intersection(geometry.buffer(distance).bounds)
    indices = [i for i in indices if i != gdf.index.get_loc(row.name)]
    gdf_select = gdf.iloc[indices]
    distant_to_others = all(
        i.distance(geometry) > distance for i in gdf_select["geometry"]
    )
    return distant_to_others


def _no_struc_on_line(geometry, struc_series, sindex, tolerance):
    indices = list(sindex.intersection(geometry.bounds))
    if indices:
        struc_series_select = struc_series.iloc[indices]
        no_struc_on_line = all(
            _point_not_overlapping_line(i, geometry, tolerance)
            for i in struc_series_select
        )
    else:
        no_struc_on_line = True
    return no_struc_on_line


def _compare_longitudinal(
    row, parameter, compare_gdf, compare_parameter, direction, logical_operator
):
    branch_id = row["branch_id"]
    select_gdf = compare_gdf.loc[compare_gdf["branch_id"] == branch_id]
    right = None
    result = None

    if not select_gdf.empty:
        if direction == "downstream":
            # select downstream objects
            select_gdf = select_gdf[
                select_gdf["branch_offset"].gt(row["branch_offset"])
            ]
            # if any, select value from closest object
            if not select_gdf.empty:
                right = select_gdf.loc[select_gdf["branch_offset"].idxmin()][
                    compare_parameter
                ]
        elif direction == "upstream":
            # select downstream objects
            select_gdf = select_gdf[
                select_gdf["branch_offset"].lt(row["branch_offset"])
            ]
            # if any, select value from closest object
            if not select_gdf.empty:
                right = select_gdf.loc[select_gdf["branch_offset"].idxmax()][
                    compare_parameter
                ]

    # if there is a right, there is something to compare
    if right is not None:
        if logical_operator == "GT":
            result = row[parameter] > right
        elif logical_operator == "LT":
            result = row[parameter] < right
    return result


""" 
In this block we define topologic functions.
"""


def snaps_to_hydroobject(gdf, datamodel, method, tolerance=0.001, dtype=bool):
    """
    Check if geometries snap to HydroObject

    Parameters
    ----------
    gdf : ExtendedGeoDataframe
        ExtendedGeoDataFrame, typically a layer in a HyDAMO datamodel class
    datamodel : HyDAMO
        HyDAMO datamodel class
    method : str, options: 'intersecting', 'overal', 'centroid', 'ends'
        Method that can be used to deterine nearest hydrobject
    tolerance : numeric
        Tolerance used to snap to the hydroobjct
    dtype : dtype, optional
        Dtype to assign to the result series. The default is bool.

    Returns
    -------
    Pandas Series
        Default dtype is bool

    """

    branches = datamodel.hydroobject
    geometry.find_nearest_branch(
        branches=branches, geometries=gdf, method=method, maxdist=tolerance
    )
    series = ~gdf.branch_offset.isna()
    return series.astype(dtype)


def geometry_length(
    gdf, datamodel, length, statistic: Literal["min", "max"] = "min", dtype=bool
):
    """Check if geometrie length is longer/shorter than a value

    Parameters
    ----------
    gdf : ExtendedGeoDataframe
        ExtendedGeoDataFrame, typically a layer in a HyDAMO datamodel class
    datamodel : HyDAMO
        HyDAMO datamodel class
    length : numeric
        Length to compare the geometry length to
    statistic : str, options: 'min', 'max'
        Use length as minimal or maximal length. Default is 'min'
    dtype : dtype, optional
        Dtype to assign to the result series. The default is bool.

    Returns
    -------
    Pandas Series
        Default dtype is bool

    """
    if statistic == "min":
        series = gdf["geometry"].length >= length
    elif statistic == "max":
        series = gdf["geometry"].length <= length
    return series.astype(dtype)


def not_overlapping(gdf, datamodel, tolerance):
    """Check if an objects LineString geometry is not overlapping other object
    of the same layer.

    Parameters
    ----------
    gdf : ExtendedGeoDataframe
        ExtendedGeoDataFrame, typically a layer in a HyDAMO datamodel class
    datamodel : HyDAMO
        HyDAMO datamodel class
    tolerance : numeric
        Max tolerance for overlapping

    Returns
    -------
    Pandas Series
        Default dtype is bool

    """
    sindex = gdf.sindex
    if (gdf.geom_type == "LineString").all():
        return gdf.apply(
            lambda x: _not_overlapping_line(x, gdf, sindex, tolerance), axis=1
        )
    elif (gdf.geom_type == "Point").all():
        return gdf.apply(
            lambda x: _not_overlapping_point(x, gdf, sindex, tolerance), axis=1
        )
    else:
        raise TypeError(
            f"GeoDataFrame has invalid geometry types: {gdf.geom_type.unique()}. Implemented for this function: [Point, LineString]"
        )


def splitted_at_junction(gdf, datamodel, tolerance):
    """Check if line is splitted when it can considered to be a junction

    Parameters
    ----------
    gdf : ExtendedGeoDataframe
        ExtendedGeoDataFrame, HyDAMO hydroobject layer
    datamodel : HyDAMO
        HyDAMO datamodel class
    tolerance : numeric
        Max tolerance for junction nodes

    Returns
    -------
    Pandas Series
        Default dtype is bool

    """
    # get the nodes of the hydroobjects within tolerance
    nodes_series = _get_nodes(gdf, tolerance)

    # check for lines if there are nodes on segment outside tolerance of
    # the start-node and end-node.
    sindex = nodes_series.sindex
    return gdf.apply(
        (lambda x: _only_end_nodes(x, nodes_series, sindex, tolerance)), axis=1
    )


def structures_at_intersections(gdf, datamodel, structures, tolerance):
    """Check if there are structures at intersections

    Parameters
    ----------
    gdf : ExtendedGeoDataframe
        ExtendedGeoDataFrame, HyDAMO hydroobject layer
    datamodel : HyDAMO
        HyDAMO datamodel class
    structures: str or list
        HyDAMO structures to be expected at intersections
        ("stuw", "duikersifonhevel") Presented as a string or list
    tolerance : numeric
        Max tolerance for junction nodes

    Returns
    -------
    Pandas Series
        Default dtype is bool

    """
    # make a geodataframe from structures list
    struc_series = _layers_from_datamodel(structures, datamodel)
    # create spatial indices
    sindex = gdf.sindex
    struc_sindex = struc_series.sindex
    # return result
    return gdf.apply(
        lambda x: _structures_at_intersections(
            x, gdf, sindex, struc_series, struc_sindex, tolerance
        ),
        axis=1,
    )


def no_dangling_node(gdf, datamodel, tolerance):
    """Check if the end-node of a linestring object is not within tolerance of
    a start-node of another linestring object in the same layer.

    Parameters
    ----------
    gdf : ExtendedGeoDataframe
        ExtendedGeoDataFrame, HyDAMO hydroobject layer
    datamodel : HyDAMO
        HyDAMO datamodel class
    tolerance : numeric
        Max tolerance to determine if nodes are connected

    Returns
    -------
    Pandas Series
        Default dtype is bool

    """
    end_nodes_series = gdf["geometry"].apply(lambda x: Point(x.coords[-1]))
    series = gdf["geometry"].apply(lambda x: Point(x.coords[0]))
    sindex = series.sindex

    return end_nodes_series.apply(
        lambda x: _intersects_end_node(x, series, sindex, tolerance)
    )


def structures_at_boundaries(gdf, datamodel, areas, structures, tolerance, distance):
    """
    Check if there are structures near area (typically water-level areas) boundaries.

    Parameters
    ----------
    gdf : ExtendedGeoDataframe
        ExtendedGeoDataFrame, HyDAMO hydroobject layer
    datamodel : HyDAMO
        HyDAMO datamodel class
    areas : str
        HyDAMO datamodel class with areas ("peilgebiedenpraktijk")
    structures : str
        List with structure-types to be expected on the boundary
    tolerance : numeric
        Tolerance to dermine if a structure is on the hydroobject
    distance : numeric
        Max distance between structure and area-boundary

    Returns
    -------
    Pandas Series
        Default dtype is bool

    """

    areas_gdf = getattr(datamodel, areas)
    areas_sindex = areas_gdf.sindex

    struc_series = _layers_from_datamodel(structures, datamodel)
    struc_sindex = struc_series.sindex

    return gdf.apply(
        lambda x: _structures_at_boundaries(
            x, areas_gdf, areas_sindex, struc_series, struc_sindex, tolerance, distance
        ),
        axis=1,
    )


def distant_to_others(gdf, datamodel, distance):
    """
    Check if two objects are sufficient distant to other objects of the same layer.

    Parameters
    ----------
    gdf : ExtendedGeoDataframe
        ExtendedGeoDataFrame, HyDAMO hydroobject layer
    datamodel : HyDAMO
        HyDAMO datamodel class
    distance : numeric
        Max distance to other node

    Returns
    -------
    Pandas Series
        Default dtype is bool

    """

    sindex = gdf.sindex

    return gdf.apply(lambda x: _distant_to_others(x, gdf, sindex, distance), axis=1)


def structures_at_nodes(gdf, datamodel, structures, tolerance):
    """
    Check if structures are on boundary-nodes of lines (hydroobjects)

    Parameters
    ----------
    gdf : ExtendedGeoDataframe
        ExtendedGeoDataFrame, HyDAMO hydroobject layer
    datamodel : HyDAMO
        HyDAMO datamodel class
    structures : str
        List with structure-types to be expected on the boundary
    tolerance : numeric
        Tolerance to determine if a structure is on a node

    Returns
    -------
    Pandas Series
        Default dtype is bool

    """
    struc_series = _layers_from_datamodel(structures, datamodel)
    struc_sindex = struc_series.sindex

    return gdf["geometry"].apply(
        lambda x: _no_struc_on_line(x, struc_series, struc_sindex, tolerance)
    )


def compare_longitudinal(
    gdf,
    datamodel,
    parameter,
    compare_object,
    compare_parameter,
    direction,
    logical_operator,
):
    """
    Check if the value of a parameter in an object is lower/greater than an
    upstream/downstream value of another parameter from another object-layer.


    Parameters
    ----------
    gdf : ExtendedGeoDataframe
        ExtendedGeoDataFrame, HyDAMO hydroobject layer
    datamodel : HyDAMO
        HyDAMO datamodel class
    structures : str
        List with structure-types to be expected on the boundary
    tolerance : numeric
        Tolerance to determine if a structure is on a node

    Returns
    -------
    Pandas Series
        Default dtype is bool

    """
    branches = datamodel.hydroobject
    compare_gdf = getattr(datamodel, compare_object)

    # snap layers to to branches
    geometry.find_nearest_branch(
        branches=branches, geometries=compare_gdf, method="overall"
    )

    geometry.find_nearest_branch(branches=branches, geometries=gdf, method="overall")
    return gdf.apply(
        lambda x: _compare_longitudinal(
            x, parameter, compare_gdf, compare_parameter, direction, logical_operator
        ),
        axis=1,
    ).astype(bool)

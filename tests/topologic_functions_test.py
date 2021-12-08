from shapely.geometry import Point
import geopandas as gpd
from pathlib import Path

from hydamo_validation import topologic_functions
from hydamo_validation.datamodel import HyDAMO

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR

hydamo_version="2.2"
datamodel = HyDAMO(version=hydamo_version)

dataset_gpkg = DATA_DIR / "tasks" / "test_wrij" / "datasets" / "HyDAMO.gpkg"
stuw_gdf = gpd.read_file(dataset_gpkg, layer="Stuw")
hydroobject_gdf = gpd.read_file(dataset_gpkg, layer="Hydroobject")
duiker_gdf = gpd.read_file(dataset_gpkg, layer="Duikersifonhevel")
gemaal_gdf = gpd.read_file(dataset_gpkg, layer="Gemaal")
aagebied_gdf = gpd.read_file(dataset_gpkg, layer="Afvoergebiedaanvoergebied")

datamodel.set_data(stuw_gdf, "stuw")
datamodel.set_data(gemaal_gdf, "gemaal")
datamodel.set_data(hydroobject_gdf, "hydroobject")
datamodel.set_data(duiker_gdf, "duikersifonhevel")
datamodel.set_data(aagebied_gdf, "afvoergebiedaanvoergebied")



def test_snaps_to_hydroobject():
    tolerance = 5
    expected_result = [True, True, True, True, True]
    result = topologic_functions.snaps_to_hydroobject(datamodel.stuw,
                                                      datamodel,
                                                      method="overall",
                                                      tolerance=tolerance,
                                                      dtype=bool).to_list()
    assert result == expected_result


def test_length_hydroobject_min():
    expected_result = [True,
                       True,
                       True,
                       False,
                       True,
                       False,
                       False,
                       True,
                       False,
                       True,
                       True]
    result = topologic_functions.geometry_length(datamodel.hydroobject.loc[0:10],
                                                 datamodel=None,
                                                 length=100,
                                                 statistic="min").to_list()
    assert result == expected_result

def test_length_hydroobject_max():
    expected_result = [False,
                       False,
                       False,
                       True,
                       False,
                       True,
                       True,
                       False,
                       True,
                       False,
                       False]
    result = topologic_functions.geometry_length(datamodel.hydroobject.loc[0:10],
                                                 datamodel=None,
                                                 length=100,
                                                 statistic="max").to_list()
    assert result == expected_result


def test_not_overlapping():
    expected_result = [True, True, True, True, True, True, True, False, True, True]
    result = topologic_functions.not_overlapping(
        datamodel.duikersifonhevel,
        datamodel=None,
        tolerance=0.1
        )[:10].to_list()
    assert result == expected_result

def test_splitted_at_junction():
    expected_result = [True, True, True, True, True, True, True, False, True, True]
    result = topologic_functions.splitted_at_junction(
        datamodel.hydroobject,
        datamodel=None,
        tolerance=1
        )[-10:].to_list()
    assert result == expected_result


def test_structures_at_intersections():
    expected_result = [False, True, True, True, True, True, True, True, True, False]
    result = topologic_functions.structures_at_intersections(datamodel.hydroobject,
                                                             datamodel,
                                                             ["duikersifonhevel"],
                                                             tolerance=1)[-10:].to_list()
    assert result == expected_result


def test_no_dangling_node():
    expected_result = [True, True, False, True, True, True, True, True, False, False]
    result = topologic_functions.no_dangling_node(datamodel.hydroobject,
                                                  datamodel,
                                                  tolerance=1)[-10:].to_list()
    assert result == expected_result


def test_distance_to_others():
    expected_result = [True, False, True, False, True]
    result = topologic_functions.distant_to_others(datamodel.stuw,
                                                  datamodel=None,
                                                  distance=1).to_list()
    assert result == expected_result

def test_structures_at_boundaries():
    expected_result = [True, True, True, True, False, False, True, True, False, False]
    result = topologic_functions.structures_at_boundaries(datamodel.hydroobject,
                                                          datamodel=datamodel,
                                                          areas="afvoergebiedaanvoergebied",
                                                          structures=["gemaal",
                                                                      "stuw",
                                                                      "duikersifonhevel"],
                                                          tolerance=0.01,
                                                          distance=25)[:10].to_list()
    assert result == expected_result


def test_structures_at_nodes():
    expected_result = [True, True, True, True, True, True, True, True, True, True, True, False]
    result = topologic_functions.structures_at_nodes(datamodel.hydroobject,
                                                     datamodel=datamodel,
                                                     structures=["stuw"],
                                                     tolerance=1)[:12].to_list()
    assert result == expected_result


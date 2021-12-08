import geopandas as gpd
from hydamo_validation import general_functions
from hydamo_validation.datamodel import HyDAMO

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR

ahn_path = DATA_DIR / "dtm"

general_functions._set_coverage("ahn", ahn_path)
hydamo_version = "2.2"

dataset_gpkg = DATA_DIR / "tasks" / "test_wrij" / "datasets" / "HyDAMO.gpkg"
gdf = gpd.read_file(dataset_gpkg, layer="Stuw")
opening_gdf = gpd.read_file(dataset_gpkg, layer="Kunstwerkopening")

datamodel = HyDAMO(version=hydamo_version)


def test_coverage_not_found():
    try:
        general_functions._set_coverage("ahn", "test")
        assert False
    except FileNotFoundError:
        assert True


def test_sum_function_int():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    expected_result = [13.200000047683716, 10.1, 13.0]
    result = general_functions.sum(_gdf,
                                   ["kruinbreedte", 10]).to_list()
    assert result == expected_result


def test_sum_function_attr():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    _gdf["add"] = 10
    expected_result = [13.200000047683716, 10.1, 13.0]
    result = general_functions.sum(_gdf,
                                   ["kruinbreedte", "add"]).to_list()
    assert result == expected_result


def test_difference_function_int():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    expected_result = [-6.799999952316284, -9.9, -7.0]
    result = general_functions.difference(_gdf,
                                          left="kruinbreedte",
                                          right=10).to_list()
    assert result == expected_result


def test_difference_function_int_abs():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    expected_result = [6.799999952316284, 9.9, 7.0]
    result = general_functions.difference(_gdf,
                                          left="kruinbreedte",
                                          right=10,
                                          absolute=True).to_list()
    assert result == expected_result


def test_devide_function_attr():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    _gdf["devide"] = 10
    expected_result = [0.3200000047683716, 0.01, 0.3]
    result = general_functions.divide(_gdf,
                                      left="kruinbreedte",
                                      right="devide").to_list()
    assert result == expected_result


def test_devide_function_int():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    expected_result = [0.3200000047683716, 0.01, 0.3]
    result = general_functions.divide(_gdf,
                                      left="kruinbreedte",
                                      right=10).to_list()
    assert result == expected_result


def test_difference_function_attr():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    _gdf["subtract"] = 10
    expected_result = [-6.799999952316284, -9.9, -7.0]
    result = general_functions.difference(_gdf,
                                          left="kruinbreedte",
                                          right="subtract").to_list()
    assert result == expected_result


def test_buffer_function_int():
    radius = 30
    percentile = 95
    expected_result = [32.17, 32.1, 35.12, 32.1, 32.23]
    result = general_functions.buffer(gdf,
                                      radius,
                                      percentile,
                                      coverage="ahn").to_list()
    assert result == expected_result


def test_buffer_function_attr():
    radius = "buffer_radius"
    percentile = 95
    _gdf = gdf.copy()
    _gdf.loc[:, "buffer_radius"] = _gdf.loc[:, "kruinbreedte"] + 10
    expected_result = [31.26, 32.1, 34.78]
    result = general_functions.buffer(_gdf.loc[_gdf["buffer_radius"].notna()],
                                      radius,
                                      percentile,
                                      coverage="ahn").to_list()
    assert result == expected_result


def test_buffer_function_fill():
    radius = 2
    percentile = 95
    expected_result = [50.0, 31.34, 50.0, 31.34, 50.0]
    result = general_functions.buffer(gdf,
                                      radius,
                                      percentile,
                                      coverage="ahn",
                                      fill_value=50).to_list()
    assert result == expected_result


def test_object_relation_count():
    expected_result = [0.0, 1.0, 0.0, 0.0, 0.0]
    result = general_functions.object_relation(gdf,
                                               related_gdf=opening_gdf,
                                               code_relation="stuwid",
                                               statistic="count",
                                               related_parameter=None,
                                               fill_value=0).to_list()
    assert result == expected_result


def test_object_relation_majority():
    expected_result = [0, 'Onbekend', 0, 0, 0]
    result = general_functions.object_relation(gdf,
                                               related_gdf=opening_gdf,
                                               code_relation="stuwid",
                                               statistic="majority",
                                               related_parameter="vormopening",
                                               fill_value=0).to_list()
    assert result == expected_result

from shapely.geometry import Point
import geopandas as gpd
import pytest
from pathlib import Path

from hydamo_validation import logic_functions

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR

dataset_gpkg = DATA_DIR / "tasks" / "test_wrij" / "datasets" / "HyDAMO.gpkg"
gdf = gpd.read_file(dataset_gpkg, layer="Stuw")


def test_LE():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    expected_result = [False, True, True]
    result = logic_functions.LE(_gdf, left="kruinbreedte", right=3).to_list()
    assert result == expected_result


def test_LT():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    expected_result = [False, True, False]
    result = logic_functions.LT(_gdf, left="kruinbreedte", right=3).to_list()
    assert result == expected_result


def test_GE():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    expected_result = [True, False, True]
    result = logic_functions.GE(_gdf, left="kruinbreedte", right=3).to_list()
    assert result == expected_result


def test_GT():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    expected_result = [True, False, False]
    result = logic_functions.GT(_gdf, left="kruinbreedte", right=3).to_list()
    assert result == expected_result


def test_EQ():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    expected_result = [False, False, True]
    result = logic_functions.EQ(_gdf, left="kruinbreedte", right=3).to_list()
    assert result == expected_result


def test_BE():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    expected_result = [False, False, True]
    result = logic_functions.BE(
        _gdf, parameter="kruinbreedte", min=0.1, max=3.2
    ).to_list()
    assert result == expected_result


def test_BE_inclusive():
    _gdf = gdf.loc[gdf["kruinbreedte"].notna()].copy()
    expected_result = [False, True, True]
    result = logic_functions.BE(
        _gdf, parameter="kruinbreedte", min=0.1, max=3.2, inclusive=True
    ).to_list()
    assert result == expected_result


def test_ISIN():
    _gdf = gdf.loc[gdf["soortstuw"].notna()].copy()
    expected_result = [True, False, True]
    result = logic_functions.ISIN(
        _gdf, parameter="soortstuw", array=["overlaat"]
    ).to_list()
    assert result == expected_result


def test_NOTIN():
    _gdf = gdf.loc[gdf["soortstuw"].notna()].copy()
    expected_result = [False, True, False]
    result = logic_functions.NOTIN(
        _gdf, parameter="soortstuw", array=["overlaat"]
    ).to_list()
    assert result == expected_result


@pytest.mark.parametrize("groupers", [None, ["pompid", "afsluitmiddelid"]])
@pytest.mark.parametrize("priority", ["prioriteit", "priority"])
@pytest.mark.parametrize(
    "second_start, second_end, expected",
    [
        ("0201", "0401", [True, True]),
        ("0301", "0401", [True, False]),
        ("0101", "0101", [False, False]),
    ],
)
def test_consistent_period_groupers(groupers, priority, second_start, second_end, expected):
    object_column = "regelmiddelid" if groupers is None else groupers[1]
    sturing = gpd.GeoDataFrame(
        {
            "pompid": ["pomp", "pomp", None, None, None, None, None],
            object_column: [None, None, "afsluitmiddel", "afsluitmiddel", "ander", "afsluitmiddel", None],
            priority: [1, 1, 1, 1, 1, 2, 1],
            "beginperiode": ["0101", "0201", "0101", second_start, "0101", "0101", "0101"],
            "eindperiode": ["0101", "0401", "0101", second_end, "0401", "0401", "0401"],
        }
    )
    arguments = {"priority": priority}
    if groupers is not None:
        arguments["groupers"] = groupers

    result = logic_functions.consistent_period(sturing, **arguments)

    assert result.iloc[:-1].to_list() == [True, True, *expected, True, True]
    assert result.isna().iloc[-1]

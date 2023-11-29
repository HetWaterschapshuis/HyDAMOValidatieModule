from hydamo_validation import topologic_functions
import geopandas as gpd
import pytest

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR


@pytest.fixture
def gdf():
    gdf = gpd.read_file(DATA_DIR / "test_layers.gpkg", layer="not_overlapping")
    result = topologic_functions.not_overlapping(gdf, datamodel=None, tolerance=0.1)
    gdf.loc[:, ["test_result"]] = result
    gdf.to_file(DATA_DIR / "result_layers.gpkg", layer="not_overlapping")
    return gdf.set_index("beschrijving")


def test_enkel(gdf):
    assert (gdf.loc["enkel"]["valid"] == gdf.loc["enkel"]["test_result"]).all()


def test_aansluitend(gdf):
    assert (
        gdf.loc["aansluitend"]["valid"] == gdf.loc["aansluitend"]["test_result"]
    ).all()


def test_kruisend(gdf):
    assert (gdf.loc["kruisend"]["valid"] == gdf.loc["kruisend"]["test_result"]).all()


def test_parallel(gdf):
    assert (gdf.loc["parallel"]["valid"] == gdf.loc["parallel"]["test_result"]).all()


def test_aantakkend(gdf):
    assert (
        gdf.loc["aantakkend"]["valid"] == gdf.loc["aantakkend"]["test_result"]
    ).all()


def test_volledig_overlappend_1(gdf):
    assert (
        gdf.loc["volledig overlappend 1"]["valid"]
        == gdf.loc["volledig overlappend 1"]["test_result"]
    ).all()


def test_volledig_overlappend_2(gdf):
    assert (
        gdf.loc["volledig overlappend 2"]["valid"]
        == gdf.loc["volledig overlappend 2"]["test_result"]
    ).all()


def test_deels_overlappend(gdf):
    assert (
        gdf.loc["deels overlappend"]["valid"]
        == gdf.loc["deels overlappend"]["test_result"]
    ).all()


def test_buiten_tolerantie(gdf):
    assert (
        gdf.loc["buiten tolerantie"]["valid"]
        == gdf.loc["buiten tolerantie"]["test_result"]
    ).all()


def test_binnen_tolerantie(gdf):
    assert (
        gdf.loc["binnen tolerantie"]["valid"]
        == gdf.loc["binnen tolerantie"]["test_result"]
    ).all()


def test_complete_gdf(gdf):
    assert (gdf["valid"] == gdf["test_result"]).all()

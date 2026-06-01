"""
Test for the DataSets class (input dataset and all its layers). This uses an older 2.2 Wrij dataset and is a rather static test of the functioning of the DataSets class.
  - test_layers: are the layers in the input dataset as pre-defined?
  - test_read_layer: can it read a layer and return the pre-defined number of features?
  - test_read_layer_filter_status: can it read a layer and filter on statusobject and return the pre-defined number of features?
  - test_read_layer_not_available: can it evoke an error when requesting an incorrectly spelled layer?
"""

from hydamo_validation.datasets import DataSets

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR

dataset_path = DATA_DIR.joinpath(r"tasks/test_wrij/datasets")
datasets = DataSets(dataset_path)


def test_layers():
    expected_result = [
        "Afvoergebiedaanvoergebied",
        "Duikersifonhevel",
        "Hydroobject",
        "Regelmiddel",
        "Stuw",
        "Brug",
        "Bodemval",
        "Gemaal",
        "Kunstwerkopening",
        "Pomp",
    ]
    assert datasets.layers == expected_result


def test_read_layer():
    gdf, schema = datasets.read_layer(layer="Duikersifonhevel")
    assert len(gdf) == 55


def test_read_layer_filter_status():
    gdf, schema = datasets.read_layer(
        layer="Duikersifonhevel", status_object=["planvorming", "gerealiseerd"]
    )
    assert len(gdf) == 53


def test_read_layer_not_available():
    try:
        datasets.read_layer(layer="hydroobject")
        assert False
    except KeyError:
        assert True

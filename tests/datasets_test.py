from hydamo_validation.datasets import DataSets

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR


dataset_path = DATA_DIR.joinpath(r"tasks/test_wrij/datasets")
datasets = DataSets(dataset_path)


def test_layers():
    expected_result = ["Afvoergebiedaanvoergebied",
                       "Duikersifonhevel",
                       "Hydroobject",
                       "Regelmiddel",
                       "Stuw",
                       "Brug",
                       "Bodemval",
                       "Gemaal",
                       "Kunstwerkopening",
                       "Pomp"]
    assert datasets.layers == expected_result


def test_read_layer():
    gdf, schema = datasets.read_layer(layer="Duikersifonhevel")
    assert len(gdf) == 55


def test_read_layer_filter_status():
    gdf, schema = datasets.read_layer(
        layer="Duikersifonhevel", status_object=["gerealiseerd"]
        )
    assert len(gdf) == 52


def test_read_layer_not_available():
    try:
        datasets.read_layer(layer="hydroobject")
        assert False
    except KeyError:
        assert True

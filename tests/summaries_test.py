from hydamo_validation.summaries import LayersSummary, ResultSummary, OUTPUT_TYPES
from hydamo_validation.datasets import DataSets
from hydamo_validation import __version__
import geopandas as gpd
from pathlib import Path
import shutil

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR


dataset_path = DATA_DIR.joinpath(r"tasks/test_wrij/datasets")
exports_dir = Path(__file__).parent / "exports"
exports_dir.mkdir(exist_ok=True)

datasets = DataSets(dataset_path)
layers_summary = LayersSummary()
result_summary = ResultSummary()
result_summary.dataset_layers = datasets.layers

gdf, _  = datasets.read_layer("Hydroobject")

gdf.drop([i for i in gdf.columns if not i in ["nen3610id", "geometry"]],
         axis=1,
         inplace=True)

def test_layers_set_data():
    _gdf = gdf.copy()
    _gdf["syntax_1"] = ""
    _gdf["syntax_oordeel"] = True
    layers_summary.set_data(_gdf, "hydroobject", geo_type="LineString")
    assert layers_summary.hydroobject.equals(_gdf)

def test_layers_join_gdf():
    _gdf = gdf.copy()
    _gdf["general_1"] = 0.1
    _gdf["validate_1"] = True
    _gdf["rating"] = 10
    _gdf["tags"] = ""
    layers_summary.join_gdf(_gdf, "hydroobject")
    expected_columns = ["syntax_1",
                         "syntax_oordeel",
                         "general_1",
                         "validate_1",
                         "rating",
                         "tags",
                         "geometry"]
    assert all([i in expected_columns for i in layers_summary.hydroobject.columns])


def test_layers_export():
    result = exports_dir / "results"
    if result.exists():
        shutil.rmtree(result)
    result.mkdir(exist_ok=True)
    layers_summary.export(result, output_types=OUTPUT_TYPES)
    result_gpkg = result / "results.gpkg"
    if result_gpkg.exists():
        shutil.rmtree(result)
        assert True
    else:
        shutil.rmtree(result)
        assert False


def test_result_to_json():
    result_json = exports_dir / "validation_result.json"
    if result_json.exists():
        result_json.unlink()
    result_summary.to_json(exports_dir)
    if result_json.exists():
        assert True
        result_json.unlink()
    else:
        assert False
        result_json.unlink()


def test_result_to_dict():
    result = result_summary.to_dict()
    assert result["dataset_layers"] == datasets.layers


def test_result_to_all():
    result_json = exports_dir / "validation_result.json"
    if result_json.exists():
        result_json.unlink()
    result = result_summary.to_all(exports_dir)
    if (result_json.exists()) & (result["dataset_layers"] == datasets.layers):
        assert True
        result_json.unlink()
    else:
        assert False
        result_json.unlink()


def test_append_warning():
    result_summary.append_warning("my warning")
    assert result_summary.warnings == ["my warning"]


def test_append_error():
    result_summary.append_error("my error")
    assert result_summary.errors == ["my error"]

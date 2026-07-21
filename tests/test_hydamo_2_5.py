# %%
"""
Test voor datamodel.py
"""


from hydamo_validation.datamodel import HyDAMO
from hydamo_validation import validator
from pathlib import Path
import geopandas as gpd
import pytest
from shapely.geometry import Point

try:
    from .config import COVERAGE, DATA_DIR
except ImportError:
    from config import COVERAGE, DATA_DIR

hydamo_version = "2.5"
object_layers = [
    "admingrenswaterschap",
    "afsluitmiddel",
    "afvoergebiedaanvoergebied",
    "aquaduct",
    "beheergrenswaterschap",
    "bijzonderhydraulischobject",
    "bodemval",
    "brug",
    "doorstroomopening",
    "duikersifonhevel",
    "gemaal",
    "grondwaterinfolijn",
    "grondwaterinfopunt",
    "grondwaterkoppellijn",
    "grondwaterkoppelpunt",
    "hydrologischerandvoorwaarde",
    "hydroobject",
    "hydroobject_normgp",
    "kunstwerkopening",
    "lateraleknoop",
    "meetlocatie",
    "meetwaardeactiewaarde",
    "normgeparamprofiel",
    "normgeparamprofielwaarde",
    "peilafwijkinggebied",
    "peilbesluitgebied",
    "peilgebiedpraktijk",
    "peilgebiedvigerend",
    "pomp",
    "profielgroep",
    "profiellijn",
    "profielpunt",
    "reglementgrenswaterschap",
    "ruwheidprofiel",
    "sluis",
    "streefpeil",
    "sturing",
    "stuw",
    "vispassage",
    "vispassagevlak",
    "vuilvang",
    "zandvang",
]

ignored_layers = [
    "afvoeraanvoergebied",
    "imwa_geoobject",
    "leggerwatersysteem",
    "leggerwaterveiligheid",
    "waterbeheergebied",
]

directory = DATA_DIR / "tasks" / "test_synthetischedataset_hydamo_2_5"
dataset_gpkg = directory / "datasets" / "HyDAMO.gpkg"

exports_dir = Path(__file__).parent / "exports"
exports_dir.mkdir(exist_ok=True)

# fixture: in meerdere test hergebruikbare objecten aanmaken
@pytest.fixture
def datamodel():
    return HyDAMO(version=hydamo_version)


@pytest.fixture
def hydroobject_gdf():
    gdf = gpd.read_file(dataset_gpkg, layer="Hydroobject")
    gdf.rename(
        columns={
            "ruwheidswaardehoog": "ruwheidhoog",
            "ruwheidswaardelaag": "ruwheidlaag",
        },
        inplace=True,
    )
    return gdf


@pytest.fixture
def stuw_gdf():
    return gpd.read_file(dataset_gpkg, layer="Stuw")


@pytest.fixture(scope="module")
def validation_result():
    hydamo_validator = validator(
        output_types=[], coverages=COVERAGE, log_level="INFO"
    )
    return hydamo_validator(directory=directory, raise_error=True)


def test_version(datamodel):
    """Controleert dat het datamodel de juiste HyDAMO-versie gebruikt."""
    assert datamodel.version == hydamo_version


def test_layers(datamodel):
    """Controleert dat alle verwachte HyDAMO 2.5 lagen beschikbaar zijn."""
    assert datamodel.layers == object_layers


def test_ignored_layers(datamodel):
    """Controleert dat de verwachte niet-objectlagen worden genegeerd."""
    assert datamodel.ignored_layers == ignored_layers


def test_setting_data(datamodel, hydroobject_gdf):
    """Controleert dat hydroobject-data in het datamodel gezet kan worden."""
    datamodel.set_data(hydroobject_gdf, "hydroobject")
    assert not datamodel.hydroobject.empty


def test_typeerror_data(datamodel, hydroobject_gdf):
    """Controleert dat een verkeerd geometrietype wordt afgekeurd."""
    gdf = hydroobject_gdf.copy()
    gdf.loc[0, "geometry"] = Point(0, 0)
    with pytest.raises(TypeError):
        datamodel.set_data(gdf, "hydroobject")


def test_keyerror_missing_column(datamodel, hydroobject_gdf):
    """Controleert dat een ontbrekende verplichte kolom wordt afgekeurd."""
    gdf = hydroobject_gdf.copy()
    gdf.drop("categorieoppervlaktewater", axis=1, inplace=True)
    with pytest.raises(KeyError):
        datamodel.hydroobject._check_columns(gdf)


def test_snapping_data(datamodel, hydroobject_gdf, stuw_gdf):
    """Controleert dat stuwen op hydroobjecten gesnapt kunnen worden."""
    datamodel.set_data(hydroobject_gdf, "hydroobject", index_col=None)
    datamodel.set_data(stuw_gdf, "stuw", index_col=None)
    datamodel.stuw.snap_to_branch(datamodel.hydroobject, snap_method="overall")
    assert "branch_id" in datamodel.stuw.columns


def test_exporting_data(tmp_path, datamodel, hydroobject_gdf):
    """Controleert dat gevulde datamodel-data naar GeoPackage exporteert."""
    datamodel.set_data(hydroobject_gdf, "hydroobject", index_col=None)
    result = tmp_path.joinpath("datamodel_no_schema.gpkg")
    datamodel.to_geopackage(result)
    assert result.exists()
    result = tmp_path.joinpath("datamodel_schema.gpkg")
    datamodel.to_geopackage(result, use_schema=False)
    assert result.exists()


def test_validator_hydamo_2_5(validation_result):
    """Controleert dat de synthetische HyDAMO 2.5 dataset echt valideert."""
    datamodel, layer_summary, result_summary = validation_result

    assert datamodel.version == hydamo_version
    assert result_summary.success
    assert result_summary.status == "finished"
    assert result_summary.error == []
    assert result_summary.errors is None
    assert result_summary.error_layers == []

    assert "categorieoppervlaktewater" in layer_summary.hydroobject.columns
    assert "syntax_categorieoppervlaktewater" in layer_summary.hydroobject.columns
    assert "categorieoppwaterlichaam" not in layer_summary.hydroobject.columns
    assert "syntax_typeregelbaarheid" in layer_summary.stuw.columns

    assert "validate_000_capaciteit_gt_0" in layer_summary.pomp.columns
    assert layer_summary.pomp["validate_000_capaciteit_gt_0"].notna().any()

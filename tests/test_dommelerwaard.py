# %%
from hydamo_validation import validator
from pathlib import Path
import pandas as pd
import shutil

try:
    from .config import DATA_DIR
except ImportError:
    from config import DATA_DIR
coverage = {"AHN": DATA_DIR.joinpath(r"dtm")}
directory = DATA_DIR.joinpath(r"tasks/test_dommelerwaard")
exports_dir = Path(__file__).parent / "exports"
if exports_dir.exists():
    shutil.rmtree(exports_dir)
exports_dir.mkdir(exist_ok=True)

hydamo_validator = validator(
    output_types=["geopackage", "csv", "geojson"], coverages=coverage, log_level="INFO"
)


datamodel, layer_summary, result_summary = hydamo_validator(
    directory=directory, raise_error=False
)


def test_hydroobjects():
    EXPECTED_INDEX = pd.Index(
        [
            "nen3610id",
            "code",
            "categorieoppwaterlichaam",
            "geometry",
            "syntax_breedte",
            "syntax_categorieoppwaterlichaam",
            "syntax_code",
            "syntax_created_date",
            "syntax_detailniveaugeometrie",
            "syntax_draineert",
            "syntax_featuretype",
            "syntax_getijdebeinvloed",
            "syntax_globalid",
            "syntax_hydroobjectgroeperingid",
            "syntax_hyperlink",
            "syntax_krwstromendid",
            "syntax_last_edited_date",
            "syntax_lengte",
            "syntax_lvpublicatiedatum",
            "syntax_metadataid",
            "syntax_naam",
            "syntax_namespace",
            "syntax_nen3610id",
            "syntax_niveau",
            "syntax_objectid",
            "syntax_ontstaanswijze",
            "syntax_opmerking",
            "syntax_oppervlaktewaterlichaamid",
            "syntax_oppwatervolgnummer",
            "syntax_persistentie",
            "syntax_ruimtelijkeafbakeningbekend",
            "syntax_ruwheidhoog",
            "syntax_ruwheidlaag",
            "syntax_soortoppwaterkwaliteit",
            "syntax_soortoppwaterkwantiteit",
            "syntax_statusleggerwatersysteem",
            "syntax_statusobject",
            "syntax_typeruwheid",
            "syntax_vaarwegid",
            "syntax_geometry",
            "syntax_oordeel",
            "validate_000_lengte_gt_1m",
            "validate_001_goed_gesplitst",
            "validate_002_goede_kruising",
            "validate_003_geen_eindpunt",
            "validate_006_kunstwerk_op_peilscheiding",
            "validate_005_kunstwerk_tussen_hobject",
            "rating",
            "valid",
            "invalid",
            "invalid_critical",
            "invalid_non_critical",
            "ignored",
            "summary",
            "tags_assigned",
            "tags_invalid",
            "date_check",
        ]
    )

    assert layer_summary.hydroobject.columns.equals(EXPECTED_INDEX)


# %%

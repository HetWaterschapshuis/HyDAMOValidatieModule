# %%
"""Maak een CSV-overzicht van de uitvoerbaarheid van HyDAMO-validatieregels.

De analyse gebruikt dezelfde ``HyDAMO``-klasse als de validator. Daardoor worden
ook lagen die in ``datamodel.py`` bewust worden genegeerd als ontbrekend
gerapporteerd. Een regel is alleen uitvoerbaar wanneer alle benodigde lagen en
kolommen in dat datamodel aanwezig zijn. Resultaten van eerder uitgevoerde
general rules worden daarbij als beschikbare tussenresultaten meegenomen.

Het script combineert alle ``ValidationRules_#.#.json``-bestanden uit het
validatiehandboek met de HyDAMO-versies uit ``hydamo_version.enum`` van het
bijbehorende ``rules_#.#.json``-schema in het geïnstalleerde pakket
``hydamo_validation`` (installatie: ``pip install hydamo-validation``).
Een checkout van de repositories is niet nodig. Voer uit met:

    python overzicht_validatieregels.py

Per regelversie heet het detailrapport ``overzicht_regels_#_#.csv``; daarnaast ontstaat
``samenvatting_regels.csv``. Standaard worden de regelbestanden van GitHub
gedownload naar ``local/validation_rules`` en CSV's geschreven naar
``local/csv``, relatief aan de huidige werkmap. Ontbrekende mappen worden
aangemaakt. Met ``--rules-json`` gebruikt u bestaande lokale bestanden zonder
download. Zie ``python overzicht_validatieregels.py --help``.
"""

import argparse
import csv
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if (REPOSITORY_ROOT / "hydamo_validation").is_dir() and str(
    REPOSITORY_ROOT
) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hydamo_validation.datamodel import HyDAMO, SCHEMAS_DIR

# Centrale instellingen voor de rapportage. Beschikbare versies worden ontdekt.
RULES_SCHEMAS_PATH = SCHEMAS_DIR.parent / "rules"
VALIDATION_RULES_PATH = Path("local") / "validation_rules"
OUTPUT_DIRECTORY = Path("local") / "csv"
VALIDATION_RULES_API = (
    "https://api.github.com/repos/HetWaterschapshuis/"
    "HyDAMOValidatiehandboek/contents/validation_rules?ref=main"
)


def _download_json(url: str) -> Any:
    """Download JSON met een timeout en een herkenbare foutmelding."""
    request = Request(url, headers={"User-Agent": "HyDAMO-regeloverzicht"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except (URLError, TimeoutError, ValueError) as error:
        raise RuntimeError(f"Download mislukt: {url}: {error}") from error


def download_validation_rules(directory: Path) -> None:
    """Ververs alle versiebestanden uit het validatiehandboek op GitHub."""
    entries = _download_json(VALIDATION_RULES_API)
    files = [
        entry
        for entry in entries
        if entry.get("type") == "file"
        and re.fullmatch(r"ValidationRules_\d+(?:\.\d+)*\.json", entry["name"])
    ]
    if not files:
        raise FileNotFoundError("Geen ValidationRules_#.#.json gevonden op GitHub")
    directory.mkdir(parents=True, exist_ok=True)
    for entry in files:
        rules = _download_json(entry["download_url"])
        destination = directory / entry["name"]
        temporary = destination.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


COMPARISON_FUNCTIONS = {"LE", "LT", "GE", "GT", "EQ"}
TOPOLOGIC_FUNCTIONS = {
    "snaps_to_hydroobject",
    "geometry_length",
    "not_overlapping",
    "splitted_at_junction",
    "structures_at_intersections",
    "no_dangling_node",
    "structures_at_boundaries",
    "distant_to_others",
    "structures_at_nodes",
    "compare_longitudinal",
}
PERIOD_COLUMNS = [
    "pompid",
    "regelmiddelid",
    "prioriteit",
    "beginperiode",
    "eindperiode",
]


def _version_key(version: str) -> tuple[int, ...]:
    """Maak van een versie zoals ``1.5`` een sorteerbare waarde."""
    return tuple(int(part) for part in version.split("."))


def find_versioned_files(directory: Path, prefix: str) -> list[tuple[str, Path]]:
    """Vind versiebestanden en sorteer numeriek op versie."""
    candidates: list[tuple[str, Path]] = []
    for path in directory.glob(f"{prefix}_*.json"):
        match = re.fullmatch(rf"{re.escape(prefix)}_(\d+(?:\.\d+)*)", path.stem)
        if match:
            candidates.append((match.group(1), path))
    if not candidates:
        raise FileNotFoundError(f"Geen {prefix}_#.#.json gevonden in: {directory}")
    return sorted(candidates, key=lambda item: _version_key(item[0]))


def supported_hydamo_versions(rules_schema_json: Path) -> list[str]:
    """Lees de toegestane HyDAMO-versies uit een regelschema."""
    schema = json.loads(rules_schema_json.read_text(encoding="utf-8"))
    versions = schema["properties"]["hydamo_version"]["enum"]
    return sorted(versions, key=_version_key)


@dataclass
class Dependencies:
    """Lagen, datamodelkolommen en general-rule-tussenresultaten van één regel."""

    layers: list[str] = field(default_factory=list)
    columns: list[tuple[str, str]] = field(default_factory=list)

    def add_layer(self, layer: str) -> None:
        if layer not in self.layers:
            self.layers.append(layer)

    def add_column(self, layer: str, column: Any) -> None:
        if not isinstance(column, str):
            return
        # ``geometry.z`` is een afgeleid geometrie-attribuut; de echte
        # datamodelkolom is ``geometry``.
        column = column.split(".", maxsplit=1)[0]
        item = (layer, column)
        if item not in self.columns:
            self.columns.append(item)


def _function_and_arguments(rule_part: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Haal de ene functie met bijbehorende argumenten uit een regelonderdeel."""
    function = next(iter(rule_part))
    return function, rule_part[function]


def _add_comparison_columns(
    dependencies: Dependencies, layer: str, arguments: dict[str, Any]
) -> None:
    dependencies.add_column(layer, arguments.get("left"))
    dependencies.add_column(layer, arguments.get("right"))


def _add_function_dependencies(
    dependencies: Dependencies,
    layer: str,
    function: str,
    arguments: dict[str, Any],
) -> None:
    """Leg expliciete én impliciete datamodelafhankelijkheden vast.

    Dit volgt de implementaties in ``logical_validation.py``,
    ``functions/general.py``, ``functions/logic.py`` en
    ``functions/topologic.py``. Bijvoorbeeld: ``object_relation`` gebruikt
    naast de genoemde gerelateerde laag ook ``<hoofdlaag>id`` en ``globalid``.
    """
    dependencies.add_layer(layer)

    if function in COMPARISON_FUNCTIONS:
        _add_comparison_columns(dependencies, layer, arguments)
    elif function in {"BE", "ISIN", "NOTIN", "NOTNA"}:
        dependencies.add_column(layer, arguments.get("parameter"))
    elif function in {"difference", "divide", "multiply"}:
        _add_comparison_columns(dependencies, layer, arguments)
    elif function == "sum":
        for value in arguments.get("array", []):
            dependencies.add_column(layer, value)
    elif function == "buffer":
        dependencies.add_column(layer, "geometry")
        dependencies.add_column(layer, arguments.get("radius"))
    elif function == "object_relation":
        related_layer = arguments["related_object"]
        dependencies.add_layer(related_layer)
        dependencies.add_column(layer, "globalid")
        dependencies.add_column(related_layer, f"{layer}id")
        dependencies.add_column(related_layer, arguments.get("related_parameter"))
    elif function == "join_parameter":
        join_layer = arguments["join_object"]
        dependencies.add_layer(join_layer)
        dependencies.add_column(layer, f"{join_layer}id")
        dependencies.add_column(join_layer, "globalid")
        dependencies.add_column(join_layer, arguments["join_parameter"])
    elif function == "join_object_exists":
        join_layer = arguments["join_object"]
        dependencies.add_layer(join_layer)
        dependencies.add_column(layer, f"{join_layer}id")
        dependencies.add_column(join_layer, "globalid")
    elif function == "consistent_period":
        # De huidige functie kent geen regelconfiguratie voor deze parameters.
        for column in PERIOD_COLUMNS:
            dependencies.add_column(layer, column)

    if function not in TOPOLOGIC_FUNCTIONS:
        return

    # Alle topologische functies werken op de geometrie van de hoofdlaag.
    dependencies.add_column(layer, "geometry")

    if function == "snaps_to_hydroobject":
        dependencies.add_layer("hydroobject")
        dependencies.add_column("hydroobject", "geometry")
    elif function in {"structures_at_intersections", "structures_at_nodes"}:
        for structure in arguments["structures"]:
            dependencies.add_layer(structure)
            dependencies.add_column(structure, "geometry")
    elif function == "structures_at_boundaries":
        area_layer = arguments["areas"]
        dependencies.add_layer(area_layer)
        dependencies.add_column(area_layer, "geometry")
        for structure in arguments["structures"]:
            dependencies.add_layer(structure)
            dependencies.add_column(structure, "geometry")
    elif function == "compare_longitudinal":
        compare_layer = arguments["compare_object"]
        dependencies.add_layer("hydroobject")
        dependencies.add_column("hydroobject", "geometry")
        dependencies.add_layer(compare_layer)
        dependencies.add_column(compare_layer, "geometry")
        dependencies.add_column(layer, arguments["parameter"])
        dependencies.add_column(compare_layer, arguments["compare_parameter"])


def _dependencies_for_rule(rule: dict[str, Any], layer: str) -> Dependencies:
    dependencies = Dependencies()
    function, arguments = _function_and_arguments(rule["function"])
    _add_function_dependencies(dependencies, layer, function, arguments)

    # Een filter wordt vóór de eigenlijke validatiefunctie uitgevoerd en kan
    # dus eveneens verhinderen dat de regel kan starten.
    if "filter" in rule:
        filter_function, filter_arguments = _function_and_arguments(rule["filter"])
        _add_function_dependencies(
            dependencies, layer, filter_function, filter_arguments
        )
    return dependencies


def _format(values: Iterable[str]) -> str:
    return " | ".join(values) or "-"


def _analyse_dependencies(
    dependencies: Dependencies,
    model_columns: dict[str, set[str]],
    schema_layers: set[str],
    available_general_results: dict[str, set[str]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Bepaal aanwezige en ontbrekende lagen/kolommen voor een regel."""
    missing_layers: list[str] = []
    for layer in dependencies.layers:
        if layer not in model_columns:
            reason = (
                "genegeerd door datamodel.py"
                if layer in schema_layers
                else "ontbreekt in HyDAMO-schema"
            )
            missing_layers.append(f"{layer} ({reason})")

    datamodel_columns: list[str] = []
    general_results: list[str] = []
    missing_columns: list[str] = []
    for layer, column in dependencies.columns:
        # ``logical_validation.execute`` schrijft general-rule-resultaten terug
        # naar het datamodel. Daarom zijn zij ook beschikbaar voor een later
        # verwerkt object (bijvoorbeeld profiellijn -> duikersifonhevel).
        if column in available_general_results.get(layer, set()):
            general_results.append(f"{layer}.{column}")
        elif layer in model_columns and column in model_columns[layer]:
            datamodel_columns.append(f"{layer}.{column}")
        elif layer in model_columns:
            missing_columns.append(f"{layer}.{column}")

    return missing_layers, datamodel_columns, general_results, missing_columns


def _row(
    *,
    hydamo_version: str,
    rules_schema_version: str,
    rules_version: str,
    layer: str,
    rule_kind: str,
    rule: dict[str, Any],
    model_columns: dict[str, set[str]],
    schema_layers: set[str],
    available_general_results: dict[str, set[str]],
) -> dict[str, str | int | bool]:
    dependencies = _dependencies_for_rule(rule, layer)
    missing_layers, datamodel_columns, general_results, missing_columns = (
        _analyse_dependencies(
            dependencies,
            model_columns,
            schema_layers,
            available_general_results,
        )
    )
    executable = not missing_layers and not missing_columns
    active = rule.get("active", True)
    conclusion = "Uitvoerbaar" if executable else "Niet uitvoerbaar"
    if not active:
        conclusion = f"Inactief; {conclusion.lower()}"

    function, _ = _function_and_arguments(rule["function"])
    return {
        "hydamo_versie": hydamo_version,
        "regels_versie": rules_schema_version,
        "hydamo_versie_regels_json": rules_version,
        "regelsoort": rule_kind,
        "laag": layer,
        "regel_id": rule["id"],
        "regelnaam": rule.get("name", rule.get("result_variable", "-")),
        "actief": active,
        "validatietype": rule.get("type", "general"),
        "functie": function,
        "benodigde_lagen": _format(dependencies.layers),
        "ontbrekende_lagen": _format(missing_layers),
        "gebruikte_datamodelkolommen": _format(datamodel_columns),
        "tussenresultaten_general_rules": _format(general_results),
        "ontbrekende_kolommen": _format(missing_columns),
        "uitvoerbaar": executable,
        "conclusie": conclusion,
    }


def analyse_version(
    hydamo_version: str,
    rules: dict[str, Any],
    rules_schema_version: str,
) -> list[dict[str, str | int | bool]]:
    """Analyseer regels en schrijf één Excel-vriendelijke CSV met één rij per regel."""
    schema_file = SCHEMAS_DIR / f"HyDAMO_{hydamo_version}.json"
    if not schema_file.is_file():
        raise FileNotFoundError(f"HyDAMO-schema bestaat niet: {schema_file}")

    # ``HyDAMO`` is de bron van waarheid voor de lagen en kolommen die de
    # validator werkelijk beschikbaar stelt.
    datamodel = HyDAMO(
        version=hydamo_version,
        schemas_path=schema_file.parent,
    )
    model_columns = {
        layer: {field["id"] for field in datamodel.validation_schemas[layer]}
        for layer in datamodel.layers
    }
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    schema_layers = set(schema["definitions"])

    rows: list[dict[str, str | int | bool]] = []
    rules_version = str(rules.get("hydamo_version", "onbekend"))
    available_general_results: dict[str, set[str]] = {}
    for object_rules in rules["objects"]:
        layer = object_rules["object"]
        available_general_results.setdefault(layer, set())

        for rule in sorted(
            object_rules.get("general_rules", []), key=lambda item: item["id"]
        ):
            rows.append(
                _row(
                    hydamo_version=hydamo_version,
                    rules_schema_version=rules_schema_version,
                    rules_version=rules_version,
                    layer=layer,
                    rule_kind="general_rule",
                    rule=rule,
                    model_columns=model_columns,
                    schema_layers=schema_layers,
                    available_general_results=available_general_results,
                )
            )
            # De validator schrijft dit resultaat na uitvoering terug naar de
            # hoofdlaag. Voeg het uitsluitend toe wanneer de producerende
            # general rule zelf kan draaien; anders bestaat het resultaat niet.
            if rows[-1]["uitvoerbaar"]:
                available_general_results[layer].add(rule["result_variable"])

        for rule in sorted(
            object_rules["validation_rules"], key=lambda item: item["id"]
        ):
            rows.append(
                _row(
                    hydamo_version=hydamo_version,
                    rules_schema_version=rules_schema_version,
                    rules_version=rules_version,
                    layer=layer,
                    rule_kind="validatieregel",
                    rule=rule,
                    model_columns=model_columns,
                    schema_layers=schema_layers,
                    available_general_results=available_general_results,
                )
            )

    return rows


def _write_csv(rows: list[dict[str, str | int | bool]], output_csv: Path) -> None:
    """Schrijf rijen als puntkomma-CSV met UTF-8-BOM voor Excel."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(
    rows: list[dict[str, str | int | bool]],
    hydamo_versions: list[str],
    rules_schema_version: str,
) -> list[dict[str, str | int]]:
    """Tel per HyDAMO-versie de uitvoerbare en niet-uitvoerbare regels."""
    summary: list[dict[str, str | int]] = []
    for version in hydamo_versions:
        version_rows = [row for row in rows if row["hydamo_versie"] == version]
        executable = sum(row["uitvoerbaar"] is True for row in version_rows)
        general_rules = [
            row for row in version_rows if row["regelsoort"] == "general_rule"
        ]
        validation_rules = [
            row for row in version_rows if row["regelsoort"] == "validatieregel"
        ]
        summary.append(
            {
                "regels_versie": rules_schema_version,
                "hydamo_versie": version,
                "totaal_regels": len(version_rows),
                "uitvoerbare_regels": executable,
                "niet_uitvoerbare_regels": len(version_rows) - executable,
                "totaal_general_rules": len(general_rules),
                "totaal_validatieregels": len(validation_rules),
            }
        )
    return summary


def create_reports(
    validation_rules_json: Path,
    output_directory: Path,
) -> tuple[list[Path], Path, list[str]]:
    """Combineer elk regelbestand alleen met de ondersteunde HyDAMO-versies."""
    if validation_rules_json.is_dir():
        rules_files = find_versioned_files(validation_rules_json, "ValidationRules")
    else:
        if not validation_rules_json.is_file():
            raise FileNotFoundError(
                f"Regelbestand of map bestaat niet: {validation_rules_json}"
            )
        rules = json.loads(validation_rules_json.read_text(encoding="utf-8"))
        match = re.fullmatch(
            r"ValidationRules_(\d+(?:\.\d+)*)", validation_rules_json.stem
        )
        version = match.group(1) if match else str(rules["schema"])
        rules_files = [(version, validation_rules_json)]

    analysed_versions: set[str] = set()
    overviews: list[Path] = []
    summary: list[dict[str, str | int | bool]] = []
    for rules_version, rules_file in rules_files:
        rules = json.loads(rules_file.read_text(encoding="utf-8"))
        schema_version = str(rules["schema"])
        supported = supported_hydamo_versions(
            RULES_SCHEMAS_PATH / f"rules_{schema_version}.json"
        )
        analysed_versions.update(supported)
        rows = [
            row
            for hydamo_version in supported
            for row in analyse_version(hydamo_version, rules, rules_version)
        ]
        version_summary = _summary_rows(rows, supported, rules_version)
        for row in [*rows, *version_summary]:
            row["regelschema_versie"] = schema_version
            row["combinatie_ondersteund_door_regelschema"] = (
                row["hydamo_versie"] in supported
            )
        version_label = rules_version.replace(".", "_")
        overview_csv = output_directory / f"overzicht_regels_{version_label}.csv"
        _write_csv(rows, overview_csv)
        overviews.append(overview_csv)
        summary.extend(version_summary)

    summary_csv = output_directory / "samenvatting_regels.csv"
    _write_csv(summary, summary_csv)
    return overviews, summary_csv, sorted(analysed_versions, key=_version_key)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rules-json",
        type=Path,
        default=None,
        help=(
            "Bestaande map met ValidationRules_#.#.json-bestanden, of één regelbestand. "
            "Standaard: download van GitHub naar local/validation_rules."
        ),
    )
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    if arguments.rules_json is None:
        download_validation_rules(VALIDATION_RULES_PATH)
        arguments.rules_json = VALIDATION_RULES_PATH
    overview_csvs, summary_csv, hydamo_versions = create_reports(
        validation_rules_json=arguments.rules_json,
        output_directory=arguments.output_directory,
    )
    print(
        f"HyDAMO-versies geanalyseerd: {', '.join(hydamo_versions)}\n"
        "Overzichten:\n"
        + "\n".join(str(path.resolve()) for path in overview_csvs)
        + "\n"
        f"Samenvatting: {summary_csv.resolve()}"
    )

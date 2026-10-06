"""Vergelijk validatieregels met HyDAMO-modellen zonder datasets te valideren.

De analyse leest lokale schema's en geeft resultaten terug zonder downloads,
console-uitvoer of rapportbestanden. Afhankelijkheden blijven lijsten; de
CSV-presentatie wordt door het aanroepende script verzorgd.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydamo_validation.datamodel import HyDAMO, SCHEMAS_DIR

RULES_SCHEMAS_PATH = SCHEMAS_DIR.parent / "rules"


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


def _analyse_dependencies(
    dependencies: Dependencies,
    model_columns: dict[str, set[str]],
    schema_layers: set[str],
    available_general_results: dict[str, set[str]],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Bepaal gebruikte lagen/kolommen en welke daarvan ontbreken."""
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
        else:
            datamodel_columns.append(f"{layer}.{column}")
            # Bij een ontbrekende laag wordt alleen de laag als ontbrekend
            # gerapporteerd, maar blijven alle gebruikte kolommen zichtbaar.
            if layer in model_columns and column not in model_columns[layer]:
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
) -> dict[str, str | int | bool | list[str]]:
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
        "benodigde_lagen": dependencies.layers,
        "ontbrekende_lagen": missing_layers,
        "gebruikte_datamodelkolommen": datamodel_columns,
        "tussenresultaten_general_rules": general_results,
        "ontbrekende_kolommen": missing_columns,
        "uitvoerbaar": executable,
        "conclusie": conclusion,
    }


def analyse_version(
    hydamo_version: str,
    rules: dict[str, Any],
    rules_schema_version: str,
    *,
    schemas_path: Path = SCHEMAS_DIR,
) -> list[dict[str, str | int | bool | list[str]]]:
    """Vergelijk regels met één HyDAMO-versie zonder uitvoer te schrijven.

    Parameters
    ----------
    hydamo_version : str
        Te onderzoeken versie, ook als het regelschema deze nog niet ondersteunt.
    rules : dict
        Ingelezen ValidationRules-JSON; wordt niet gewijzigd.
    rules_schema_version : str
        Regelbestandversie voor het veld ``regels_versie``. De parameternaam is
        behouden uit het script; het JSON-veld ``schema`` kan hiervan afwijken.
    schemas_path : Path
        Map met ``HyDAMO_<versie>.json``. Standaard de schema's van het pakket.

    Returns
    -------
    list of dict
        Eén resultaat per regel, met afhankelijkheden als lijsten en
        ``uitvoerbaar`` als boolean. Gebruikte datamodelkolommen bevatten ook
        ontbrekende kolommen; beschikbare general-rule-resultaten staan apart.
        Uitvoerbaarheid betreft de afhankelijkheden, niet de aangeleverde data.
    """
    schema_file = schemas_path / f"HyDAMO_{hydamo_version}.json"
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

    rows: list[dict[str, str | int | bool | list[str]]] = []
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


def analyse_versions(
    rules: dict[str, Any],
    rules_version: str,
    *,
    hydamo_versions: list[str] | None = None,
    schemas_path: Path = SCHEMAS_DIR,
    rules_schemas_path: Path = RULES_SCHEMAS_PATH,
) -> list[dict[str, str | int | bool | list[str]]]:
    """Analyseer dezelfde regels voor meerdere modellen zonder rapportuitvoer.

    Parameters
    ----------
    rules : dict
        Ingelezen ValidationRules-JSON, inclusief ``schema`` en ``objects``.
    rules_version : str
        Versie van het regelbestand voor het resultaatveld ``regels_versie``.
    hydamo_versions : list of str, optional
        Expliciete versies; standaard ``hydamo_version.enum`` uit het regelschema.
        Expliciete versies worden gesorteerd en ontdubbeld. Een lege lijst is
        ongeldig. Voor alle lokale versies kan de aanroeper
        ``find_versioned_files(schemas_path, "HyDAMO")`` gebruiken.
    schemas_path : Path
        Map met HyDAMO-schema's, ook voor onderzoek naar nieuwe modellen.
    rules_schemas_path : Path
        Map met regelschema's. Ondersteuning wordt bepaald door het schema dat
        ``rules["schema"]`` aanwijst, onafhankelijk van ``schemas_path``.

    Returns
    -------
    list of dict
        Resultaten van ``analyse_version``, aangevuld met ``regelschema_versie``
        en de boolean ``combinatie_ondersteund_door_regelschema``. Een expliciet
        gekozen, niet-ondersteunde combinatie wordt wel geanalyseerd; ondersteuning
        en uitvoerbaarheid zijn afzonderlijke uitkomsten. Ontbrekende schema's
        veroorzaken een FileNotFoundError en worden niet stilzwijgend overgeslagen.
    """
    schema_version = str(rules["schema"])
    supported = supported_hydamo_versions(
        rules_schemas_path / f"rules_{schema_version}.json"
    )
    if hydamo_versions is None:
        selected = supported
    else:
        if not hydamo_versions:
            raise ValueError("Kies ten minste één HyDAMO-versie")
        selected = sorted(set(hydamo_versions), key=_version_key)

    rows = [
        row
        for hydamo_version in selected
        for row in analyse_version(
            hydamo_version, rules, rules_version, schemas_path=schemas_path
        )
    ]
    for row in rows:
        row["regelschema_versie"] = schema_version
        row["combinatie_ondersteund_door_regelschema"] = (
            row["hydamo_versie"] in supported
        )
    return rows

# %%
"""Vergelijk validatieregels met HyDAMO-modellen en schrijf CSV-overzichten.

De analyse gebruikt dezelfde ``HyDAMO``-klasse als de validator. Daardoor worden
ook lagen die in ``datamodel.py`` bewust worden genegeerd als ontbrekend
gerapporteerd. Een regel is alleen uitvoerbaar wanneer alle benodigde lagen en
kolommen in dat datamodel aanwezig zijn. Resultaten van eerder uitgevoerde
general rules worden daarbij als beschikbare tussenresultaten meegenomen.
Gebruikte datamodelkolommen bevatten ook ontbrekende kolommen. Er wordt geen
aangeleverde dataset gecontroleerd en er worden geen validatiefuncties uitgevoerd.

Het script combineert alle ``ValidationRules_#.#.json``-bestanden uit het
validatiehandboek met de HyDAMO-versies uit ``hydamo_version.enum`` van het
bijbehorende ``rules_#.#.json``-schema in ``hydamo_validation``. Met
``--hydamo-version`` (herhaalbaar) en ``--schemas-path`` kunt u ook een nieuw
datamodel onderzoeken dat nog niet door het regelschema wordt ondersteund.
Voer vanuit de repositoryroot uit na de ontwikkelinstallatie met UV:

    uv run --no-sync python scripts/overzicht_validatieregels.py

Per regelversie heet het detailrapport ``overzicht_regels_#_#.csv``; daarnaast ontstaat
``samenvatting_regels.csv``. Standaard worden de regelbestanden van GitHub
gedownload naar ``local/validation_rules`` en CSV's geschreven naar
``local/csv``, relatief aan de huidige werkmap. Ontbrekende mappen worden
aangemaakt. Met ``--rules-json`` gebruikt u bestaande lokale bestanden zonder
download. Zie ``python scripts/overzicht_validatieregels.py --help`` en de
handleiding ``docs/guides/rule_analysis.md`` voor de betekenis van de CSV-velden.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if (REPOSITORY_ROOT / "hydamo_validation").is_dir() and str(
    REPOSITORY_ROOT
) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hydamo_validation.datamodel import SCHEMAS_DIR
from hydamo_validation.rule_analysis import (
    _version_key,
    analyse_versions,
    find_versioned_files,
)

# Centrale instellingen voor de rapportage. Beschikbare versies worden ontdekt.
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


def _write_csv(
    rows: list[dict[str, str | int | bool | list[str]]], output_csv: Path
) -> None:
    """Schrijf rijen als puntkomma-CSV met UTF-8-BOM voor Excel."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(
            {
                key: (" | ".join(value) or "-") if isinstance(value, list) else value
                for key, value in row.items()
            }
            for row in rows
        )


def _summary_rows(
    rows: list[dict[str, str | int | bool | list[str]]],
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
    *,
    hydamo_versions: list[str] | None = None,
    schemas_path: Path = SCHEMAS_DIR,
) -> tuple[list[Path], Path, list[str]]:
    """Rapporteer ondersteunde of expliciet gekozen HyDAMO-versies per regelbestand."""
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
        rows = analyse_versions(
            rules,
            rules_version,
            hydamo_versions=hydamo_versions,
            schemas_path=schemas_path,
        )
        support_by_version = {
            row["hydamo_versie"]: row["combinatie_ondersteund_door_regelschema"]
            for row in rows
        }
        selected = sorted(support_by_version, key=_version_key)
        analysed_versions.update(selected)
        version_summary = _summary_rows(rows, selected, rules_version)
        for row in version_summary:
            row["regelschema_versie"] = schema_version
            row["combinatie_ondersteund_door_regelschema"] = support_by_version[
                row["hydamo_versie"]
            ]
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
    parser.add_argument(
        "--hydamo-version",
        action="append",
        dest="hydamo_versions",
        help=(
            "Te onderzoeken HyDAMO-versie (herhaalbaar), ook buiten het regelschema. "
            "Standaard: alle versies die het bijbehorende regelschema ondersteunt."
        ),
    )
    parser.add_argument(
        "--schemas-path",
        type=Path,
        default=SCHEMAS_DIR,
        help="Map met HyDAMO_<versie>.json; standaard de HyDAMO-schema's van het pakket.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    if arguments.rules_json is None:
        download_validation_rules(VALIDATION_RULES_PATH)
        arguments.rules_json = VALIDATION_RULES_PATH
    overview_csvs, summary_csv, hydamo_versions = create_reports(
        validation_rules_json=arguments.rules_json,
        output_directory=arguments.output_directory,
        hydamo_versions=arguments.hydamo_versions,
        schemas_path=arguments.schemas_path,
    )
    print(
        f"HyDAMO-versies geanalyseerd: {', '.join(hydamo_versions)}\n"
        "Overzichten:\n"
        + "\n".join(str(path.resolve()) for path in overview_csvs)
        + "\n"
        f"Samenvatting: {summary_csv.resolve()}"
    )

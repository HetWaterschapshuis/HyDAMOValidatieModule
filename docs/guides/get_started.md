# Installatie en gebruik

De module ondersteunt **Python 3.12** (`>=3.12,<3.13` in `pyproject.toml`). Gebruik [uv](https://docs.astral.sh/uv/getting-started/installation/) om de Python-omgeving te beheren. Voor het aanpassen van de broncode volg je de [ontwikkelaarsinstallatie](contribute.md#ontwikkelomgeving).

## Een release installeren

Kies een [GitHub Release](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/releases) en download `requirements.lock` uit de **Assets** van die release. Dit bestand legt de versies van de runtime-dependencies vast; de module zelf installeer je apart. Gebruik steeds de packageversie die bij dezelfde tag hoort, bijvoorbeeld `vX.Y.Z`. Gebruik hiervoor geen lockbestand uit de actuele `main`.

Open een lege werkmap. Vervang in dit PowerShell-voorbeeld `X.Y.Z` door de gekozen versie:

```powershell
$version = "X.Y.Z"
$tag = "v$version"
Invoke-WebRequest "https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/releases/download/$tag/requirements.lock" -OutFile requirements.lock
uv venv --python 3.12
uv pip sync requirements.lock
uv pip install --no-deps "hydamo-validation==$version"
uv pip check
```

Op Linux en macOS:

```sh
version=X.Y.Z
curl --fail --location --output requirements.lock "https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/releases/download/v${version}/requirements.lock"
uv venv --python 3.12
uv pip sync requirements.lock
uv pip install --no-deps "hydamo-validation==${version}"
uv pip check
```

`uv pip sync` brengt de omgeving in overeenstemming met het lockbestand. Installeer de module daarna met `--no-deps`, zodat de vastgelegde dependencyversies behouden blijven. Na opnieuw synchroniseren moet je de module opnieuw installeren.

Bij een oudere release zonder asset kun je het lockbestand uit die exacte tag gebruiken, als het daar aanwezig is. De gekozen packageversie moet ook op PyPI beschikbaar zijn.

De releaseworkflow controleert het runtime-lockbestand met Python 3.12 op Windows x64, Linux x64 en macOS ARM64. Andere Python-versies en platforms vallen buiten die controle. De geospatial packages stellen ook eisen aan het besturingssysteem; de [gepinde pyproj 3.7.2-wheel](https://pypi.org/project/pyproj/3.7.2/#files) voor macOS ARM64 vereist bijvoorbeeld macOS 14 of nieuwer.

### Installeren zonder lockbestand

Voor gebruik zonder vastgelegde dependencyversies kun je de module rechtstreeks installeren:

```sh
uv venv --python 3.12
uv pip install hydamo-validation
uv pip check
```

Dit installeert de beschikbare release en laat uv passende dependencyversies kiezen. Gebruik de installatie met het bijbehorende lockbestand wanneer je een release reproduceerbaar wilt uitvoeren.

### Omgeving activeren

Activeer de omgeving voordat je de onderstaande Python- en CLI-commando's uitvoert.

| Shell | Commando |
| --- | --- |
| Windows-opdrachtprompt (cmd) | `.venv\Scripts\activate.bat` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Linux/macOS | `source .venv/bin/activate` |

Zonder activatie kun je de interpreter rechtstreeks aanroepen: `.venv\Scripts\python.exe` op Windows of `.venv/bin/python` op Linux/macOS.

## Invoergegevens

Een projectmap bevat een submap `datasets` met één of meer GeoPackages en het bestand `validationrules.json`. Deze namen liggen vast. Gebruik de bestandsnaam in kleine letters, zodat de validatie ook op een hoofdlettergevoelig bestandssysteem werkt.

```text
project/
├── datasets/
│   └── hydamo.gpkg
└── validationrules.json
```

Het regelbestand bevat onder meer `schema` (de versie van het regelschema), `hydamo_version` (de versie van het datamodel) en `objects` (de uit te voeren regels per objectlaag). De module controleert dit bestand tegen het meegeleverde JSON-schema. Gebruik een regelbestand dat bij de moduleversie en het datamodel past. Zie [datamodel en regelschema's](../reference/hydamo.md) voor de technische samenhang.

De [modulepagina in het handboek](https://hetwaterschapshuis.github.io/HyDAMOValidatiehandboek/HyDAMO%20Validatietool/Aan%20de%20slag/04%20De%20Validatiemodule/#2-invoergegevens-voorbereiden) beschrijft het klaarzetten van een voorbeeldset. De repository bevat ook `tests/data/tasks/test_wrij` en AHN-testgegevens in `tests/data/dtm`. Gebruik voorbeeldgegevens en regels uit dezelfde checkout of releasetag.

### Rastergegevens en CRS

Regels die een coverage gebruiken, hebben de bijbehorende rastergegevens nodig. De coverage mag buiten de projectmap staan. De sleutel is hoofdlettergevoelig en moet overeenkomen met `coverage` in het regelbestand, bijvoorbeeld `AHN`.

De huidige bufferfunctie verwacht in de coverage-map `index.shp` met de bijbehorende shapefile-bestanden en een veld `bladnr`. De rasterbestanden heten `<BLADNR>_CM.tif`, met het bladnummer in hoofdletters. Neem de volledige index en de benodigde rastertegels over.

Het datamodel gebruikt RD New (`EPSG:28992`), met afstanden in meters. Gegevens met een bekend ander CRS worden bij opname in het datamodel geherprojecteerd. Controleer zelf of het bron-CRS klopt en of coverage en index hetzelfde CRS gebruiken; de bufferfunctie herprojecteert deze bestanden niet.

## Commandline

Beide entrypoints voeren dezelfde validatie uit:

```sh
python -m hydamo_validation "pad/naar/project" --coverages "AHN=pad/naar/ahn"
hydamo-validate "pad/naar/project" --coverages "AHN=pad/naar/ahn"
```

Vanuit de repositoryroot werkt ook `python hydamo_validation "pad/naar/project"`, wanneer de module in de actieve omgeving is geïnstalleerd. Zet paden met spaties tussen aanhalingstekens.

| Argument | Betekenis | Standaard |
| --- | --- | --- |
| `directory` | Projectmap met `datasets` en `validationrules.json`. | Verplicht |
| `--coverages KEY=VALUE` | Coverage-map; herhaal deze optie voor meerdere coverages. | Geen coverages |
| `--output-types geopackage csv geojson` | Eén of meer uitvoerformaten. | `geopackage` |
| `--log-level INFO` of `--log-level DEBUG` | Detailniveau van de logging. | `INFO` |
| `--raise-error` | Geef technische exceptions door met een volledige traceback. | Uitgeschakeld |

Elke coverage bevat een niet-lege sleutel en waarde, gescheiden door de eerste `=`. Bij herhaling van dezelfde sleutel wordt de laatste waarde gebruikt. Bijvoorbeeld:

```sh
python -m hydamo_validation "pad/naar/project" --coverages "AHN=pad/naar/ahn" --coverages "DTM=pad/naar/dtm" --output-types geopackage csv geojson --log-level DEBUG
python -m hydamo_validation --help
```

De CLI toont de projectmap, technische status, doorlooptijd en resultaatlocatie.

| Exitcode | Betekenis |
| --- | --- |
| `0` | De validatie is technisch afgerond. Objecten kunnen wel zijn afgekeurd door validatieregels. |
| `1` | Een technische fout heeft de uitvoering onderbroken of een regel kon niet worden uitgevoerd. Er kunnen gedeeltelijke resultaten zijn. |
| `2` | Argumenten ontbreken of zijn ongeldig. |

## Python-interface

`validator` configureert de uitvoerformaten, logging en coverages. De geretourneerde functie voert de validatie uit:

```python
from pathlib import Path

from hydamo_validation import validator

run_validation = validator(
    output_types=["geopackage"],
    log_level="INFO",
    coverages={"AHN": Path("pad/naar/ahn")},
)

result = run_validation(directory=Path("pad/naar/project"), raise_error=False)
if result is None:
    raise RuntimeError("Validatie afgebroken; controleer validator.log.")

datamodel, layer_summary, result_summary = result
if not result_summary.success or result_summary.error or result_summary.errors:
    raise RuntimeError("Technische validatiefout; controleer de samenvatting.")
```

- `datamodel` is het gevulde `HyDAMO`-datamodel.
- `layer_summary` is een `LayersSummary` met de resultaten per objectlaag.
- `result_summary` is een `ResultSummary` met de uitvoeringsstatus en meldingen.

Bij `raise_error=False` retourneert een afgebroken uitvoering `None`. Technische fouten per regel worden geregistreerd in `errors`; de uitvoering kan dan doorgaan. Met `raise_error=True` worden exceptions doorgegeven. Een afgekeurd object is geen Python-exception.

**Verschil in standaarduitvoer:** de CLI schrijft standaard alleen GeoPackage. `validator()` vanuit Python schrijft standaard alle drie de formaten. Geef `output_types` expliciet op voor dezelfde uitvoer. Zie de [API-reference](../reference/validator.md) voor parameters en resultaatobjecten.

## Uitvoer en fouten

De module schrijft naar `results` in de projectmap. **Bij een nieuwe validatie wordt deze map verwijderd en opnieuw aangemaakt.** Bewaar eerdere resultaten buiten deze map.

| Bestand of map | Inhoud |
| --- | --- |
| `results/results.gpkg` | Niet-lege resultaatlagen in één GeoPackage, als `geopackage` is gekozen. |
| `results/csv/<laag>.csv` | Eén CSV per niet-lege resultaatlaag, zonder geometrie, als `csv` is gekozen. |
| `results/geojson/<laag>.geojson` | Eén bestand per niet-lege resultaatlaag, bij een bekend CRS omgezet naar WGS84 (`EPSG:4326`), als `geojson` is gekozen. |
| `results/validation_result.json` | Samenvatting met moduleversie, status, lagen en meldingen. |
| `results/validator.log` | Voortgang en logging van de validator. |

Validatiefouten betekenen dat gegevens niet aan een regel voldoen. Technische fouten betekenen dat een controle niet kon worden uitgevoerd, bijvoorbeeld door een ontbrekende coverage of kolom.

Controleer in de samenvatting altijd `success`, `status`, `error` en `errors`. Ook in de huidige Python-interface kan `success: true` met `status: "finished"` samengaan met technische regelfouten in `errors`. De huidige CLI neemt die fouten mee in exitcode `1`. De aanwezigheid van uitvoerbestanden of alleen `success: true` bewijst dus niet dat alle regels zijn uitgevoerd.

Voor de betekenis van resultaatkolommen, syntaxfoutcodes en beoordelingen verwijst het [handboek naar de interpretatie van resultaten](https://hetwaterschapshuis.github.io/HyDAMOValidatiehandboek/HyDAMO%20Validatietool/Aan%20de%20slag/05%20Resultaten%20interpreteren/).

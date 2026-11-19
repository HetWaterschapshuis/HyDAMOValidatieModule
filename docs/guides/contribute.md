# Ontwikkelen en bijdragen

Bijdragen van ontwikkelaars en leveranciers zijn welkom. Je kunt fouten oplossen, validatiefuncties uitbreiden, tests toevoegen of de documentatie verbeteren. Sluit aan bij de bestaande code en houd wijzigingen gericht, zodat ze goed te beoordelen zijn.

## Ontwikkelomgeving

Gebruik Python 3.12 en [uv](https://docs.astral.sh/uv/getting-started/installation/). Maak een fork van de [repository](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule) en clone die lokaal. Werk vanuit de repositoryroot op een eigen branch.

```sh
uv venv --python 3.12
uv pip sync requirements-dev.lock
uv pip install --no-deps --editable .
uv pip check
```

Activeer vervolgens de omgeving zoals beschreven bij [installatie en gebruik](get_started.md#omgeving-activeren). De editable installatie gebruikt de broncode uit je checkout. Installeer als ontwikkelaar dus geen losse PyPI-release over deze installatie heen.

| Bestand | Doel |
| --- | --- |
| `pyproject.toml` | Python-versie, dependencies, extras en packageconfiguratie. |
| `requirements.lock` | Vastgelegde runtime-dependencies voor releasegebruik en releasecontroles. |
| `requirements-dev.lock` | Ontwikkelomgeving, inclusief pytest, coverage en notebooks. |
| `requirements-docs.lock` | Documentatieomgeving met MkDocs, Material en mkdocstrings. |
| `requirements_docker.lock` | Omgeving voor de Docker-helperapplicatie. |

De lockbestanden bevatten dependencies; de module zelf installeer je daarna apart. Opnieuw synchroniseren verwijdert de editable installatie. Installeer deze dan opnieuw met `--no-deps --editable .`. Gebruik het lockbestand dat bij je checkout hoort.

## Code aanpassen en testen

Packagecode staat in `hydamo_validation/`, tests in `tests/` en voorbeelden in `notebooks/`. Volg de bestaande module-indeling:

- `functions/general.py`: berekeningen en tussenresultaten voor algemene regels.
- `functions/logic.py`: logische controles op attributen en relaties.
- `functions/topologic.py`: ruimtelijke controles met het datamodel.
- `schemas/`: de ondersteunde datamodellen en regelschema's.

Gebruik GeoPandas en de bestaande helpers waar die passen. Documenteer nieuwe functies met docstrings in de bestaande NumPy-stijl. Controleer bij een nieuwe of gewijzigde validatiefunctie ook de bijbehorende JSON-schemadefinitie, regelconfiguratie en [API-reference](../reference/general_functions.md). Een Python-functie is pas via een regelbestand bruikbaar als het regelschema de configuratie toelaat.

Voeg gerichte tests toe voor het zichtbare gedrag, met een normaal geval en een relevante fout of grenswaarde. Gebruik kleine synthetische datasets of bestaande fixtures. Een losse berekening kun je bijvoorbeeld zo controleren:

```python
import geopandas as gpd

from hydamo_validation import general_functions


def test_multiply():
    gdf = gpd.GeoDataFrame({"left": [1, 2, 3], "right": [4, 5, 6]})
    result = general_functions.multiply(gdf, left="left", right="right")
    assert result.to_list() == [4, 10, 18]
```

Voer eerst de betrokken tests uit en daarna de volledige suite:

```sh
python -m pytest tests/test_general_functions.py
python -m pytest
python -m pytest --cov=hydamo_validation --cov-report=term-missing tests/
```

De laatste opdracht toont ook de coverage. Sommige integratietests gebruiken lokale GeoPackages en AHN-testbestanden uit `tests/data` en schrijven resultaten. Gebruik een ontwikkelcheckout en vervang geen referentiedata om tests te laten slagen. Voor releasecontroles gebruik je de [runtimeversies van de release](release.md#2-release-testen).

Gebruik vóór ondersteuning van een nieuw datamodel de [regelvergelijking](rule_analysis.md) om ontbrekende lagen en kolommen vast te stellen.

## Documentatie

Maak voor documentatiewerk een aparte omgeving, zodat het synchroniseren van documentatie-dependencies de ontwikkelomgeving niet vervangt:

```sh
uv venv .venv-docs --python 3.12
uv pip sync --python .venv-docs requirements-docs.lock
uv pip install --python .venv-docs --no-deps --editable .
```

Gebruik op Windows:

```powershell
.\.venv-docs\Scripts\python.exe -m mkdocs serve
.\.venv-docs\Scripts\python.exe -m mkdocs build --strict
```

Je kunt de server op Windows ook starten met `.\mkdocs_serve.ps1` vanuit de repositoryroot. Het script gebruikt `.venv-docs` en geeft extra argumenten door aan `mkdocs serve`, bijvoorbeeld `.\mkdocs_serve.ps1 --dev-addr 127.0.0.1:8001`.

Als PowerShell het uitvoeren van scripts blokkeert, gebruik dan `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\mkdocs_serve.ps1`. Deze instelling geldt alleen voor dat proces.

Op Linux/macOS:

```sh
.venv-docs/bin/python -m mkdocs serve
.venv-docs/bin/python -m mkdocs build --strict
```

Pagina's staan in `docs/`; navigatie en instellingen staan in `mkdocs.yml`. De API-reference gebruikt mkdocstrings voor signatures en standaardwaarden uit de broncode, met Nederlandstalige toelichting op de referentiepagina's. De CSS en logo's volgen de huisstijl van het handboek en staan lokaal in `docs/stylesheets/` en `docs/assets/`. De footer staat in `overrides/partials/copyright.html`.

### Documentatie publiceren

De workflow **Publish documentation** (`.github/workflows/docs.yml`) bouwt pull requests naar `main` met `mkdocs build --strict`. Na het mergen van een pull request publiceert de workflow de documentatie uit `main` op [GitHub Pages](https://hetwaterschapshuis.github.io/HyDAMOValidatieModule/). Het sluiten van een pull request zonder merge publiceert niets.

Beheerders kunnen via **Actions > Publish documentation > Run workflow** een branch kiezen. Een handmatige run vanaf `main` publiceert de site; een andere branch bouwt alleen een documentatie-artifact. De GitHub CLI kan dezelfde workflow starten:

```sh
gh workflow run docs.yml --ref main
```

Bij de eerste inrichting kies je **GitHub Actions** onder **Settings > Pages > Build and deployment > Source**. De omgeving `github-pages` moet publicatie vanaf `main` toestaan.

## Voorwaarden voor codebijdragen

De module is beschikbaar onder de [MIT-licentie](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/blob/main/LICENSE). Voor het aanleveren van nieuwe code gelden de onderstaande voorwaarden van Het Waterschapshuis.

- Leveranciers dragen de auteursrechten op nieuw ontwikkelde code over aan Het Waterschapshuis en verklaren dat zij bevoegd zijn die code te leveren.
- De overdracht geldt alleen voor code waarvoor de leverancier zelf de rechten kan overdragen.
- De code wordt vervolgens door Het Waterschapshuis onder de MIT-licentie beschikbaar gesteld, dus is door iedereen vervolgens weer vrij te gebruiken.
- Voor het aanleveren, opnemen, gebruiken, wijzigen en verspreiden van bijdragen wordt geen vergoeding betaald aan leveranciers.
- Nieuwe externe packages worden alleen na expliciete beoordeling geaccepteerd.
- Leveranciers en ontwikkelaars worden opgenomen in de credits.

## Pull request indienen

Commit de wijziging op je eigen branch en open een pull request naar `main`. Beschrijf het probleem, de gewijzigde werking en de uitgevoerde controles. Werk de betrokken documentatie bij en verwijs waar nodig naar de [voorwaarden voor codebijdragen](#voorwaarden-voor-codebijdragen).

Een maintainer beoordeelt de bijdrage vóór opname. Geslaagde tests vervangen die beoordeling niet.

## Credits en contact

De bestaande geregistreerde bijdragers zijn Het Waterschapshuis, D2Hydro, HKV en HydroConsult. De maintainer is Philip Hansmann, bereikbaar via [hydamo-validatietool@hetwaterschapshuis.nl](mailto:hydamo-validatietool@hetwaterschapshuis.nl).

---
hide:
  - navigation
---

# Ontwikkelen en bijdragen

Bijdragen aan de HyDAMO-validatiemodule zijn welkom. Je kunt fouten oplossen, validatiefuncties uitbreiden, tests toevoegen of documentatie verbeteren. Houd wijzigingen gericht en volg de bestaande codeconventies en lees eerst [voorwaarden voor codebijdragen](#voorwaarden-voor-codebijdragen).

## Ontwikkelomgeving

Gebruik Python 3.12 en [uv](https://docs.astral.sh/uv/getting-started/installation/). Maak een fork van de [repository](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule), clone deze lokaal en werk op een eigen branch.

```sh
uv venv --python 3.12
uv pip sync requirements-dev.lock
uv pip install --no-deps --editable .
uv pip check
```

Activeer de omgeving met `.venv\Scripts\activate.bat` (cmd), `.\.venv\Scripts\Activate.ps1` (PowerShell) of `source .venv/bin/activate` (Linux/macOS).

De belangrijkste configuratiebestanden:

| Bestand | Doel |
| --- | --- |
| `pyproject.toml` | Packageconfiguratie en dependencies |
| `requirements.lock` | Runtime-dependencies voor releases |
| `requirements-dev.lock` | Ontwikkelomgeving en tests |
| `requirements-docs.lock` | MkDocs en documentatie |
| `requirements_docker.lock` | Docker-helperapplicatie |

De lockbestanden leggen dependencyversies vast. De module installeer je afzonderlijk. Na `uv pip sync` moet je de editable installatie opnieuw uitvoeren met `uv pip install --no-deps --editable .`.

## Code aanpassen en testen

De broncode staat in `hydamo_validation/`, tests in `tests/` en voorbeelden in `notebooks/`.

De validatiefuncties zijn onderverdeeld in:

- `functions/general.py`: algemene berekeningen en relaties.
- `functions/logic.py`: logische controles.
- `functions/topologic.py`: ruimtelijke controles.
- `schemas/`: datamodellen en regelschema's.

Gebruik bestaande helpers en GeoPandas waar mogelijk. Schrijf docstrings in NumPy-stijl. Controleer bij nieuwe functies ook de regelconfiguratie en JSON-schema's. Een functie kan pas vanuit een regelbestand worden aangeroepen wanneer het schema de configuratie ondersteunt.

Voeg tests toe voor normaal gedrag en relevante fout- of grensgevallen. Een eenvoudig voorbeeld:

```python
import geopandas as gpd
from hydamo_validation import general_functions

def test_multiply():
    gdf = gpd.GeoDataFrame({"left": [1, 2, 3], "right": [4, 5, 6]})
    result = general_functions.multiply(gdf, left="left", right="right")
    assert result.to_list() == [4, 10, 18]
```

Voer eerst de relevante tests uit en daarna de volledige suite met coverage:

```sh
python -m pytest tests/test_general_functions.py
python -m pytest --cov=hydamo_validation --cov-report=term-missing tests/
```

Sommige tests gebruiken GeoPackages en AHN-gegevens uit `tests/data`. Wijzig geen referentiedata om tests te laten slagen.

Gebruik bij ondersteuning van een nieuw datamodel eerst de [regelvergelijking](rule_analysis.md). Voor releases volg je de [releaseprocedure](release.md).

## Documentatie bijwerken

De pagina's staan in `docs/`, de configuratie in `mkdocs.yml`. De documentatie gebruikt MkDocs Material en mkdocstrings. De huisstijl staat in `docs/stylesheets/`, `docs/assets/` en `overrides/`.

Voor het lokaal bouwen van documentatie maak je een afzonderlijke omgeving:

```sh
uv venv .venv-docs --python 3.12
uv pip sync --python .venv-docs requirements-docs.lock
uv pip install --python .venv-docs --no-deps --editable .
```

De documentatie kun je lokaal (op Windows) hosten met:

```powershell
.\mkdocs_serve.ps1
```

## Voorwaarden voor codebijdragen

De module wordt beheerd door Het Waterschapshuis en is beschikbaar onder de [MIT-licentie](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/blob/main/LICENSE). Voor nieuwe codebijdragen gelden de volgende voorwaarden:

- Leveranciers dragen de auteursrechten op nieuw ontwikkelde code over aan Het Waterschapshuis, voor zover zij daartoe bevoegd zijn.
- Het Waterschapshuis publiceert de bijdragen onder de MIT-licentie.
- Voor het aanleveren, opnemen, gebruiken, wijzigen en verspreiden van bijdragen wordt geen vergoeding betaald.
- Nieuwe externe dependencies worden alleen na expliciete beoordeling geaccepteerd.
- Ontwikkelaars en leveranciers worden opgenomen in de credits.

## Pull request indienen

Commit je wijzigingen op een eigen branch en open een pull request naar `main`. Beschrijf de wijzigingen, uitgevoerde tests en eventuele gevolgen voor bestaande functionaliteit. Werk waar nodig de documentatie bij.

Een maintainer beoordeelt de bijdrage voordat deze wordt opgenomen.

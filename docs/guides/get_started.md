# Installatie en gebruik

Het [HyDAMO-validatiehandboek](https://hetwaterschapshuis.github.io/HyDAMOValidatiehandboek/HyDAMO%20Validatietool/Aan%20de%20slag/04%20De%20Validatiemodule/){:target="_blank"} beschrijft het installeren en uitvoeren van de validatiemodule, inclusief voorbeeldgegevens, commandline, Python-interface en validatieresultaten.

Deze pagina bevat aanvullende technische informatie voor ontwikkelaars en geautomatiseerd gebruik.

## Reproduceerbare installatie

Gebruik Python 3.12 en download `requirements.lock` uit de **Assets** van de gewenste [GitHub Release](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/releases). Dit bestand legt de exacte versies van alle afhankelijkheden vast. Zo kun je dezelfde softwareomgeving opnieuw opbouwen, bijvoorbeeld om validatieresultaten te reproduceren of technische fouten te onderzoeken.

Installeer de bijbehorende packageversie (`X.Y.Z`):

```sh
uv venv --python 3.12
uv pip sync requirements.lock
uv pip install --no-deps hydamo-validation==X.Y.Z
uv pip check
```

Gebruik altijd het lockbestand en de packageversie van dezelfde release. `uv pip sync` verwijdert eerder geïnstalleerde packages die niet in het lockbestand staan, waaronder de validatiemodule zelf. Installeer deze daarom na het synchroniseren.

Voor het aanpassen van de broncode zie [Ontwikkelen en bijdragen](contribute.md).

## Rastergegevens en CRS

De module gebruikt RD New (`EPSG:28992`). Gegevens met een bekend ander CRS worden bij het inlezen in het datamodel geherprojecteerd.

Voor rastercoverages gelden aanvullende eisen:

- De coverage-map bevat `index.shp` met het veld `bladnr` en bijbehorende rasterbestanden `<BLADNR>_CM.tif`.
- De index en rasters moeten hetzelfde CRS gebruiken; de bufferfunctie herprojecteert deze niet.
- Coverage-sleutels zijn hoofdlettergevoelig en moeten overeenkomen met de regelconfiguratie.

Zie de [API-reference van de algemene functies](../reference/general_functions.md) voor de werking van de rasterfuncties.

## Commandline en Python-interface

Naast `python -m hydamo_validation` is het commando `hydamo-validate` beschikbaar. Beide gebruiken dezelfde validator.

De commandline retourneert de volgende exitcodes:

| Code | Betekenis |
| --- | --- |
| `0` | Technisch afgerond; validatieregels kunnen wel afgekeurde objecten opleveren. |
| `1` | Technische fout tijdens de uitvoering. |
| `2` | Ongeldige of ontbrekende argumenten. |

De CLI schrijft standaard een GeoPackage. De Python-interface `validator()` schrijft standaard GeoPackage, CSV en GeoJSON. Geef `output_types` expliciet op wanneer dezelfde uitvoer gewenst is.

Bij gebruik vanuit Python retourneert `validator()` een functie die drie objecten oplevert: `datamodel`, `layer_summary` en `result_summary`. Bij een afgebroken uitvoering met `raise_error=False` kan de retourwaarde `None` zijn.

Controleer voor geautomatiseerde verwerking ook `result_summary.errors`: `success=True` betekent niet noodzakelijk dat iedere validatieregel technisch is uitgevoerd.

Zie de [API-reference](../reference/validator.md) voor parameters, resultaatobjecten en foutafhandeling.

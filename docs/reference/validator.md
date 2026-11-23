# Validator en resultaten

De publieke ingang is `from hydamo_validation import validator`. Deze functie retourneert een vooraf geconfigureerde validatiefunctie. Zie [installatie en gebruik](../guides/get_started.md#commandline-en-python-interface) voor het gebruik vanuit Python.

## Validatie configureren

| Parameter | Betekenis |
| --- | --- |
| `output_types` | Lijst met `geopackage`, `geojson` en/of `csv`. Vanuit Python zijn standaard alle drie geselecteerd; de CLI kiest standaard alleen `geopackage`. |
| `log_level` | `INFO` of `DEBUG`; standaard `INFO`. |
| `coverages` | Dictionary met coverage-sleutels en paden. Sleutels moeten exact overeenkomen met de regelconfiguratie. Standaard leeg. |

::: hydamo_validation.validator.validator

Roep de geretourneerde functie aan met `directory` (projectmap als `str` of `Path`) en optioneel `raise_error` (standaard `False`). Bij afronding retourneert ze een tuple `(HyDAMO, LayersSummary, ResultSummary)`. Bij een afgebroken uitvoering zonder doorgegeven exception retourneert ze `None`. Met `raise_error=True` worden technische exceptions doorgegeven.

## Uitvoeringsstappen

1. De validator maakt `results` opnieuw aan, configureert logging en leest `validationrules.json`.
2. `read_validation_rules` controleert de regelset tegen het schema dat het veld `schema` aanwijst.
3. `HyDAMO` initialiseert het datamodel voor `hydamo_version`.
4. De syntaxvalidatie controleert lagen en velden. Alleen de resterende valide gegevens gaan door naar het datamodel voor de logische en topologische controles.
5. Algemene regels berekenen tussenresultaten; logische en topologische regels beoordelen objecten.
6. De validator exporteert resultaatlagen en schrijft de samenvatting.

De bronbestanden worden hierbij niet aangepast. Conversies, ontbrekende waarden en ongeldige geometrieën kunnen wel bepalen welke gegevens in het interne datamodel terechtkomen. De syntaxresultaten leggen deze beoordeling vast.

## Regelbestand controleren

`read_validation_rules` accepteert een `Path` naar het JSON-bestand en optioneel een `ResultSummary` om foutinformatie vast te leggen. De functie retourneert de ingelezen dictionary. Ongeldige JSON, een ontbrekend regelschema of een afwijking van dat schema veroorzaakt een exception.

::: hydamo_validation.validator.read_validation_rules

## Resultaten per laag

Een `LayersSummary` bewaart resultaatlagen als attributen, bijvoorbeeld `layer_summary.brug`. Dit zijn GeoDataFrames met syntaxresultaten, regelresultaten en samenvattingskolommen. Een laag zonder resultaten hoeft niet als attribuut aanwezig te zijn.

| Methode | Werking |
| --- | --- |
| `set_data(gdf, layer, geo_type)` | Registreert een resultaatlaag en het geometrietype volgens Fiona. |
| `join_gdf(gdf, layer)` | Voegt regelresultaten op index samen met een bestaande resultaatlaag. |
| `export(results_path, output_types)` | Schrijft niet-lege lagen en retourneert de namen van geëxporteerde lagen. |

Onder `results_path` verschijnen `results.gpkg`, `csv/<laag>.csv` en/of `geojson/<laag>.geojson`, afhankelijk van `output_types`. CSV bevat geen geometrie; GeoJSON wordt bij een bekend CRS omgezet naar WGS84 (`EPSG:4326`).

::: hydamo_validation.summaries.LayersSummary
    options:
      merge_init_into_class: true
      members: [set_data, join_gdf, export]

## Uitvoeringssamenvatting

`ResultSummary` beschrijft de uitvoering, los van het oordeel over afzonderlijke objecten.

| Attribuut | Betekenis |
| --- | --- |
| `success`, `status` | Of het einde van de uitvoering is bereikt en wat de laatst bereikte stap is. Controleer daarnaast altijd `error` en `errors`. |
| `module_version`, `date_check`, `duration` | Moduleversie, starttijd en doorlooptijd in seconden. |
| `dataset_layers`, `result_layers` | Aangeboden lagen en geëxporteerde resultaatlagen. |
| `missing_layers`, `error_layers` | Ontbrekende datamodellagen en aangeboden lagen zonder geëxporteerd resultaat. |
| `syntax_result`, `validation_result` | Lagen waarvoor syntaxresultaten respectievelijk validatieresultaten zijn geregistreerd. |
| `error` | Foutinformatie bij een afgebroken uitvoering. |
| `errors` | Technische fouten per regel. Kan ook gevuld zijn bij `success=True`. |
| `warnings` | Waarschuwingen, bijvoorbeeld over overgeslagen objecten met ontbrekende invoer. |
| `log` | Logregels die de validator vóór het schrijven van de samenvatting toevoegt. |

`to_dict()` retourneert de attributen zonder waarden die `None` zijn. `to_json(results_path)` schrijft deze naar `validation_result.json`. `to_all(results_path)` doet beide. `append_error(message)` en `append_warning(message)` voegen meldingen toe aan `errors` en `warnings`.

::: hydamo_validation.summaries.ResultSummary
    options:
      merge_init_into_class: true
      members: [to_dict, to_json, to_all, append_error, append_warning]

# Algemene functies

Algemene functies berekenen tussenresultaten voor `general_rules`. Ze ontvangen een GeoDataFrame (`gdf`) en retourneren een pandas Series. De regeluitvoering voegt het resultaat toe onder `result_variable`, zodat volgende regels die kolom kunnen gebruiken.

De functies staan in `hydamo_validation.functions.general` en zijn ook beschikbaar via `from hydamo_validation import general_functions`.

## Berekeningen en relaties

| Functie | Parameters en resultaat |
| --- | --- |
| `sum` | Telt de kolommen en numerieke constanten in `array` bij elkaar op. |
| `difference` | Berekent `left - right`; beide kunnen een kolomnaam of waarde zijn. Met `absolute=True` wordt de absolute waarde van het verschil genomen. |
| `divide` | Berekent `left / right` met kolomnamen of numerieke constanten. |
| `multiply` | Berekent `left * right` met kolomnamen of numerieke constanten. |
| `join_parameter` | Koppelt `join_parameter` uit `join_gdf` via `<join_object>id` in `gdf` aan `globalid` in de gerelateerde laag. `fill_value` vult ontbrekende resultaten als deze is opgegeven. |
| `object_relation` | Groepeert `related_gdf` op `code_relation` en koppelt aan `globalid` in `gdf`. `statistic` kiest `min`, `max`, `sum` of `count`; `related_parameter` kiest de bronkolom. Bij `count` is die bronkolom niet nodig. `fill_value` vult ontbrekende resultaten. |

In regelconfiguraties levert de uitvoeringsmodule `join_gdf` aan op basis van `join_object`. Bij `object_relation` leidt ze `related_gdf` en de relatiesleutel af uit `related_object` en de hoofdlaag. Bij een directe Python-aanroep geef je deze GeoDataFrames en sleutels zelf mee.

## Hoogte uit een coverage

`buffer` berekent `percentile` van rasterwaarden binnen een buffer rond de geometrie. `radius` is een constante of kolomnaam; de minimale straal is 0,5 in de eenheid van het CRS. `coverage` kiest een geregistreerde coverage. De standaard is `"ahn"`; gebruik de exacte sleutel uit je configuratie, bijvoorbeeld `"AHN"`.

De functie gebruikt `index.shp` en `<BLADNR>_CM.tif`, verwerkt de rasterschaalfactor en rondt af op twee decimalen. Zonder resultaat blijft de waarde ontbrekend, tenzij `fill_value` is opgegeven. Zie [rastergegevens en CRS](../guides/get_started.md#rastergegevens-en-crs) voor het klaarzetten van de coverage.

## API-reference

::: hydamo_validation.functions.general
    options:
      members: [sum, difference, divide, multiply, buffer, join_parameter, object_relation]


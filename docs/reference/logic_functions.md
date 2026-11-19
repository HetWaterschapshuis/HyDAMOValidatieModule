# Logische functies

Logische functies beoordelen attributen en relaties. Ze ontvangen een GeoDataFrame (`gdf`) en retourneren een pandas Series met beoordelingen per object. De vergelijkingsfuncties gebruiken standaard `dtype=bool`. `consistent_period` kan ontbrekende beoordelingen teruggeven voor rijen zonder objectgroep.

De functies staan in `hydamo_validation.functions.logic` en zijn ook beschikbaar via `from hydamo_validation import logic_functions`. Een regel met `type: "logic"` selecteert de functie en parameters via `function` in het regelbestand. Dezelfde functies kunnen een regelfilter vormen.

| Functie | Controle en parameters |
| --- | --- |
| `LE`, `LT` | Vergelijken `left` met `right`: kleiner dan of gelijk aan, respectievelijk kleiner dan. Een operand kan een kolomnaam of constante zijn. |
| `GE`, `GT` | Vergelijken `left` met `right`: groter dan of gelijk aan, respectievelijk groter dan. |
| `EQ` | Controleert of `left` en `right` gelijk zijn. |
| `BE` | Controleert of de kolom `parameter` tussen `min` en `max` ligt. Grenswaarden tellen alleen mee bij `inclusive=True`. |
| `ISIN`, `NOTIN` | Controleert of waarden uit `parameter` wel of niet in `array` voorkomen. |
| `NOTNA` | Controleert of `parameter` een niet-ontbrekende waarde heeft. |
| `join_object_exists` | Controleert of `<join_object>id` uit `gdf` bestaat in `globalid` van `join_gdf`. De regeluitvoering levert de gerelateerde laag aan. |
| `consistent_period` | Controleert overlap en tussenruimten in perioden, gegroepeerd op objectverwijzing en prioriteit. |

## Perioden controleren

Bij `consistent_period` bepaalt `groupers` de objectverwijzingen en `priority` de prioriteitskolom. Standaard zijn dit `["pompid", "regelmiddelid"]` en `prioriteit`. De begin- en einddatum staan standaard in `beginperiode` en `eindperiode` en worden gelezen met `date_format="%d%m"`.

Een periode waarvan de begindatum na de einddatum ligt, loopt door in het volgende jaar. `max_gap` is de maximaal toegestane tussenruimte in dagen, standaard `1`. Perioden worden per object en prioriteit gecontroleerd. Rijen zonder een waarde voor de objectgroep kunnen een ontbrekend resultaat houden.

## Voorbeeld

```python
import geopandas as gpd

from hydamo_validation import logic_functions

gdf = gpd.GeoDataFrame({"lengte": [0.5, 2.0]})
result = logic_functions.GE(gdf, left="lengte", right=1.0)
assert result.to_list() == [False, True]
```

## API-reference

::: hydamo_validation.functions.logic
    options:
      members: [LE, LT, GT, GE, EQ, BE, ISIN, NOTIN, NOTNA, join_object_exists, consistent_period]


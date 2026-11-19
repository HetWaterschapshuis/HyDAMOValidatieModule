# HyDAMO-datamodel

`HyDAMO` bouwt voor elke objectlaag een `ExtendedGeoDataFrame` op uit een meegeleverd JSON-schema. Het schema beschrijft onder meer velden, gegevenstypen, verplichte waarden, domeinen en geometrieën.

## Datamodel en regelschema's

De modelschema's staan in `hydamo_validation/schemas/hydamo/HyDAMO_<versie>.json`. De huidige checkout bevat versies 2.2 tot en met 2.6. De regelschema's staan afzonderlijk in `hydamo_validation/schemas/rules/rules_<versie>.json`.

In `validationrules.json` kiest `schema` het regelschema en kiest `hydamo_version` het datamodel. De toegestane combinaties staan in `hydamo_version.enum` van het regelschema. Een aanwezig modelschema betekent dus niet dat elke regelset die versie ondersteunt. De [regelvergelijking](../guides/rule_analysis.md) onderzoekt daarnaast of de benodigde lagen en kolommen beschikbaar zijn.

## HyDAMO gebruiken

| Parameter of attribuut | Betekenis |
| --- | --- |
| `version` | Versie van het modelschema. De klasse gebruikt standaard `"2.2"`; de validator neemt de versie uit het regelbestand. |
| `schemas_path` | Map met modelschema's; standaard de meegeleverde schema's. |
| `ignored_layers` | Lagen die de klasse overslaat. De actuele standaardlijst staat in de signature. |
| `layers` | Lagen die na het overslaan beschikbaar zijn als attributen. |
| `data_layers` | Beschikbare lagen die daadwerkelijk gegevens bevatten. |
| `validation_schemas` | Velddefinities per laag voor de syntaxvalidatie. |

```python
from hydamo_validation.datamodel import HyDAMO

datamodel = HyDAMO(version="2.5")
print(datamodel.layers)
print(datamodel.brug.required_columns)
```

`init_datamodel()` leest het schema en initialiseert de lagen. `get(layer, global_id)` zoekt een object op via `globalid`. `set_data(gdf, layer, ...)` vult een laag en controleert standaard de verplichte kolommen en het geometrietype. `index_col` kiest een indexkolom; standaard blijft de invoerindex behouden.

`from_geopackage(file_path, version, ...)` leest een GeoPackage naar een nieuw model en neemt alleen herkende laagnamen over. `to_geopackage(file_path, use_schema=True)` schrijft het model en voegt de beschikbare stijlen toe. Met `use_schema=True` worden kolommen beperkt tot de schemadefinitie en worden ontbrekende schemakolommen toegevoegd. Dit is een modelexport; gebruik de validator voor een validatie met resultaatkolommen.

::: hydamo_validation.datamodel.HyDAMO
    options:
      merge_init_into_class: true
      members: [data_layers, init_datamodel, get, set_data, from_geopackage, to_geopackage]

## ExtendedGeoDataFrame

Deze uitbreiding van `GeoDataFrame` bewaart `required_columns`, `geotype`, `layer_name` en `validation_schema`. Bij initialisatie worden ontbrekende verplichte kolommen toegevoegd. `set_data` normaliseert kolomnamen en controleert standaard kolommen en geometrieën; een afwijking veroorzaakt een `KeyError` of `TypeError`.

Het interne model gebruikt `EPSG:28992`. Bij `set_data` worden gegevens met een bekend ander CRS geherprojecteerd. Controleer invoer zonder CRS vooraf: de klasse kan bij initialisatie RD New toekennen, maar bepaalt daarmee niet of dat het werkelijke bron-CRS is. `set_data` kan ook de kolomnamen en het CRS van de aangeleverde GeoDataFrame aanpassen; geef een kopie mee als je die wilt behouden.

`delete_all()` verwijdert alle rijen. `snap_to_branch(branches, snap_method, maxdist=5)` voegt informatie toe over de dichtstbijzijnde tak. Methoden zijn `intersecting`, `overall`, `centroid` en `ends`; de afstand is in de eenheid van het CRS.

::: hydamo_validation.datamodel.ExtendedGeoDataFrame
    options:
      merge_init_into_class: true
      members: [validation_schema, set_data, delete_all, snap_to_branch]


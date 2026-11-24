# GeoPackages inlezen

`DataSets` inventariseert de `*.gpkg`-bestanden direct in `dataset_dir`. De klasse bewaart per bestand de beschikbare lagen en Fiona-schema's in `properties`. De property `layers` geeft alle gevonden laagnamen terug.

## Een laag lezen

`read_layer` retourneert `(gdf, schema)`: een GeoDataFrame en het bijbehorende Fiona-schema. De laagnaam is hoofdlettergevoelig. De kolomnamen worden na inlezen omgezet naar kleine letters. Als dezelfde laagnaam in meerdere bestanden voorkomt, gebruikt de methode het eerste gevonden bestand.

| Parameter | Betekenis |
| --- | --- |
| `layer` | Naam van de in te lezen GeoPackage-laag. |
| `result_summary` | `ResultSummary` waaraan bij een leesfout een waarschuwing wordt toegevoegd; de exception wordt daarna doorgegeven. |
| `status_object` | Optionele lijst toegestane waarden voor de kolom `statusobject`. Ontbrekende waarden blijven behouden. Zonder deze kolom wordt niet gefilterd. |

Een onbekende laag veroorzaakt een `KeyError`. Voor invoerfouten tijdens het lezen wordt de oorspronkelijke exception doorgegeven.

```python
from pathlib import Path

from hydamo_validation.datasets import DataSets
from hydamo_validation.summaries import ResultSummary

datasets = DataSets(Path("pad/naar/project/datasets"))
if datasets.layers:
    gdf, schema = datasets.read_layer(
        datasets.layers[0], result_summary=ResultSummary()
    )
```

::: hydamo_validation.datasets.DataSets
    options:
      merge_init_into_class: true
      members: [layers, read_layer]


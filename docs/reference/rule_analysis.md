# API voor regelanalyse

`hydamo_validation.rule_analysis` vergelijkt afhankelijkheden van regels met datamodellen. De functies downloaden geen regels, schrijven geen rapporten en voeren geen validatiefuncties uit. Zie [regels vergelijken met datamodellen](../guides/rule_analysis.md) voor de CLI en de betekenis van alle resultaatvelden.

## Meerdere versies analyseren

`analyse_versions` ontvangt de ingelezen regelset (`rules`) en de regelbestandversie (`rules_version`). Zonder `hydamo_versions` worden de versies uit het bijbehorende regelschema gebruikt. Een expliciete lijst wordt gesorteerd en ontdubbeld; een lege lijst veroorzaakt een `ValueError`.

`schemas_path` kiest de map met modelschema's. `rules_schemas_path` kiest afzonderlijk de map met regelschema's. Een ontbrekend schema veroorzaakt een `FileNotFoundError`.

Het resultaat is een lijst dictionaries met regelidentificatie, afhankelijkheden als lijsten en `uitvoerbaar` als boolean. `regelschema_versie` en `combinatie_ondersteund_door_regelschema` maken onderscheid tussen de aangegeven ondersteuning en de beschikbaarheid van afhankelijkheden.

```python
import json
from pathlib import Path

from hydamo_validation.datamodel import SCHEMAS_DIR
from hydamo_validation.rule_analysis import analyse_versions, find_versioned_files

rules = json.loads(
    Path("local/validation_rules/ValidationRules_1.5.json").read_text(encoding="utf-8")
)
versions = [version for version, _ in find_versioned_files(SCHEMAS_DIR, "HyDAMO")]
results = analyse_versions(rules, "1.5", hydamo_versions=versions)
```

Dit voorbeeld onderzoekt alle meegeleverde modelversies, ook combinaties die het regelschema niet ondersteunt. De invoerdictionary wordt niet gewijzigd.

::: hydamo_validation.rule_analysis.analyse_versions

## Eén versie analyseren

`analyse_version` vergelijkt `rules` met één `hydamo_version`. De parameter `rules_schema_version` bevat hier de **regelbestandversie** voor het resultaatveld `regels_versie`; het JSON-veld `schema` kan daarvan afwijken. `schemas_path` kiest de map met `HyDAMO_<versie>.json`.

Deze functie retourneert afhankelijkheidsresultaten zonder de twee aanvullende velden voor regelschema-ondersteuning die `analyse_versions` toevoegt. Ze berekent geen algemene tussenresultaten, maar houdt wel bij of hun afhankelijkheden beschikbaar zijn voor volgende regels.

::: hydamo_validation.rule_analysis.analyse_version

## Versiebestanden vinden

`find_versioned_files(directory, prefix)` vindt bestanden met de naam `<prefix>_<versie>.json` en retourneert numeriek gesorteerde tuples `(versie, Path)`. Voorbeelden van prefixes zijn `HyDAMO` en `ValidationRules`.

::: hydamo_validation.rule_analysis.find_versioned_files

# Validatieregels vergelijken met datamodellen

Met `scripts/overzicht_validatieregels.py` controleer je of een HyDAMO-datamodel de lagen en kolommen bevat die validatieregels nodig hebben. Gebruik dit bij onderzoek naar een nieuw of gewijzigd datamodel.

De analyse gebruikt dezelfde `HyDAMO`-klasse als de validator, inclusief de genegeerde lagen. Ze leest geen aangeleverde dataset en voert geen validatiefuncties uit. Een regel is hier *uitvoerbaar* wanneer de gevonden afhankelijkheden beschikbaar zijn. Dat bewijst niet dat de functie correct op gegevens werkt.

## Script uitvoeren

Volg de [ontwikkelaarsinstallatie](contribute.md#ontwikkelomgeving), inclusief de editable installatie, en voer opdrachten uit vanuit de repositoryroot:

```sh
python scripts/overzicht_validatieregels.py
```

Standaard downloadt het script `ValidationRules_#.#.json`-bestanden uit `main` van het validatiehandboek naar `local/validation_rules`. Dit vereist netwerktoegang en vervangt gelijknamige lokale regelbestanden. Rapporten komen in `local/csv`; gelijknamige rapporten worden vervangen. Ontbrekende mappen worden aangemaakt. Relatieve paden gelden vanaf de huidige werkmap.

Kies met `--branch` een andere branch van het handboek:

```sh
python scripts/overzicht_validatieregels.py --branch fix/validation-rules
```

Bij uitvoering vanuit een IDE kun je ook `VALIDATION_RULES_BRANCH` bovenaan het script instellen. De CLI-optie gaat voor. Bij `--rules-json` wordt `--branch` genegeerd.

Gebruik voor een reproduceerbare vergelijking dezelfde lokale regelbestanden vóór en na een wijziging. Deze opdracht downloadt niets:

```sh
python scripts/overzicht_validatieregels.py --rules-json local/validation_rules --output-directory local/csv
```

`--rules-json` accepteert ook één bestand, zoals `local/validation_rules/ValidationRules_1.5.json`. Met `--help` bekijk je alle opties.

### Meldingen over afhankelijkheden

Het script toont ontbrekende lagen en kolommen per regel, met bestandsnaam, HyDAMO-versie, objectlaag, regelsoort, ID en naam. Algemene en inactieve regels tellen ook mee. De CSV-bestanden bevatten alle regels, ook zonder ontbrekende afhankelijkheden.

Verberg specifieke kolommen alleen in de console met `IGNORED_LOG_COLUMNS` in het script, bijvoorbeeld `["regelmiddel.maximalehoogteopening"]`, of met de herhaalbare optie:

```sh
python scripts/overzicht_validatieregels.py --rules-json local/validation_rules --ignore-log-column regelmiddel.maximalehoogteopening
```

De vergelijking gebruikt de exacte verwijzing `laag.kolom`. Andere ontbrekende kolommen en ontbrekende lagen blijven zichtbaar. Een regel met uitsluitend verborgen ontbrekende kolommen wordt niet getoond. Analyse, CSV-details en tellingen blijven gelijk.

## Datamodelversies kiezen

Standaard analyseert het script elke regelset tegen de versies uit `hydamo_version.enum` van het bijbehorende `rules_<schema>.json`. Het veld `schema` in de regelset bepaalt welk regelschema wordt gebruikt. Versies worden numeriek gesorteerd; een ontbrekend schema veroorzaakt een fout.

Kies versies expliciet met de herhaalbare optie `--hydamo-version`:

```sh
python scripts/overzicht_validatieregels.py --rules-json local/validation_rules --hydamo-version 2.4 --hydamo-version 2.5
```

Dubbele versies worden eenmaal geanalyseerd. De selectie geldt voor elk regelbestand, ook wanneer het regelschema die combinatie niet ondersteunt.

Voor een experimenteel model plaats je `HyDAMO_<versie>.json` in een aparte map. Bijvoorbeeld, met `HyDAMO_2.7.json` in `local/schemas`:

```sh
python scripts/overzicht_validatieregels.py --rules-json local/validation_rules/ValidationRules_1.5.json --hydamo-version 2.7 --schemas-path local/schemas --output-directory local/csv_experimental
```

Dit voorbeeld vereist een zelf aangeleverd schema dat de huidige `HyDAMO`-klasse kan lezen. Alle geselecteerde modelschema's moeten in `--schemas-path` staan; er is geen terugval op de meegeleverde schema's. Deze optie wijzigt de regelschema's niet en selecteert op zichzelf geen extra versies. Gebruik daarvoor `--hydamo-version`.

De rapporten onderscheiden **ondersteuning volgens het regelschema** en **beschikbaarheid van afhankelijkheden**. Een niet-ondersteunde combinatie kan uitvoerbare regels bevatten. Je hoeft de officiële versielijst niet te wijzigen om een nieuw model te onderzoeken.

## Detailrapporten lezen

`overzicht_regels_<versie>.csv` bevat één rij per regel en geanalyseerde HyDAMO-versie. Algemene regels en validatieregels zijn opgenomen. Punten in de bestandsversie worden underscores, bijvoorbeeld `overzicht_regels_1_5.csv`.

| CSV-veld | Betekenis |
| --- | --- |
| `hydamo_versie` | Geanalyseerde datamodelversie. |
| `regels_versie` | Versie uit `ValidationRules_<versie>.json`; bij andere bestandsnamen wordt het JSON-veld `schema` gebruikt. |
| `hydamo_versie_regels_json` | Datamodelversie in de regelset; deze beperkt de geanalyseerde versies niet. |
| `regelschema_versie` | Versie uit het JSON-veld `schema`. |
| `combinatie_ondersteund_door_regelschema` | Of het regelschema de geanalyseerde datamodelversie toestaat. |
| `regelsoort` | `general_rule` of `validatieregel`. |
| `laag` | Objectlaag waarop de regel wordt toegepast. |
| `regel_id` | ID binnen de objectlaag en regelsoort. Samen met de versievelden identificeert dit de rij. |
| `regelnaam` | Regelnaam, of de resultaatvariabele als een naam ontbreekt. |
| `actief` | Ingestelde activering; standaard `True`. Ook inactieve regels worden geanalyseerd. |
| `validatietype` | Ingesteld type, of `general` als dit ontbreekt. |
| `functie` | Hoofdfunctie van de regel. Afhankelijkheden van het filter worden ook onderzocht. |
| `benodigde_lagen` | Alle gebruikte lagen, inclusief ontbrekende lagen. |
| `ontbrekende_lagen` | Niet-beschikbare lagen, met reden: afwezig in het schema of genegeerd door het datamodel. |
| `gebruikte_datamodelkolommen` | Verwijzingen naar datamodelkolommen, inclusief ontbrekende kolommen en kolommen van ontbrekende lagen. Bevat ook impliciete relatiesleutels; `geometry.z` wordt `geometry`. Beschikbare algemene tussenresultaten staan apart. |
| `tussenresultaten_general_rules` | Verwijzingen naar resultaten van eerdere algemene regels waarvan de afhankelijkheden beschikbaar zijn. |
| `ontbrekende_kolommen` | Verwijzingen die in een bestaande laag ontbreken en niet als beschikbaar tussenresultaat zijn gevonden. Bij een ontbrekende laag staat die onder `ontbrekende_lagen`. |
| `uitvoerbaar` | `True` wanneer de analyse geen ontbrekende lagen of kolommen vindt. |
| `conclusie` | Leesbare conclusie, met vermelding als de regel inactief is. |

*Gebruikt* betekent dat de regel naar een kolom verwijst, ook als die ontbreekt. Algemene brugregel 1 uit regelbestand 1.5 gebruikt bijvoorbeeld `brug.globalid`, `kunstwerkopening.laagstedoorstroomhoogte` en `kunstwerkopening.brugid`. Bij HyDAMO 2.4 staan alle drie bij `gebruikte_datamodelkolommen`; `kunstwerkopening.brugid` staat ook bij `ontbrekende_kolommen`. De regel is dan niet uitvoerbaar.

Algemene regels worden per object in ID-volgorde onderzocht, vóór de validatieregels. De objectvolgorde uit het regelbestand blijft behouden. Een algemeen tussenresultaat wordt pas beschikbaar voor volgende regels als de producerende regel alle afhankelijkheden heeft. Een niet-beschikbaar tussenresultaat kan daarom bij een latere regel als ontbrekende kolom verschijnen. Er worden geen waarden berekend.

Voor `consistent_period` volgen de afhankelijkheden uit `groupers`, `priority`, `start_date` en `end_date`. Standaard zijn dit `["pompid", "regelmiddelid"]`, `prioriteit`, `beginperiode` en `eindperiode`. De controle groepeert perioden per objectverwijzing en prioriteit.

## Samenvatting lezen

`samenvatting_regels.csv` bevat één rij per regelbestandversie en geanalyseerde datamodelversie. De velden `regels_versie`, `hydamo_versie`, `regelschema_versie` en `combinatie_ondersteund_door_regelschema` hebben dezelfde betekenis als hierboven.

| Telling | Betekenis |
| --- | --- |
| `totaal_regels` | Alle onderzochte algemene regels en validatieregels. |
| `uitvoerbare_regels` | Regels waarvan alle afhankelijkheden beschikbaar zijn. |
| `niet_uitvoerbare_regels` | Regels met ontbrekende afhankelijkheden. |
| `totaal_general_rules` | Aantal algemene regels. |
| `totaal_validatieregels` | Aantal validatieregels. |

Deze tellingen omvatten inactieve regels. Het zijn geen aantallen gevalideerde objecten of geslaagde functie-uitvoeringen.

Alle CSV-bestanden gebruiken puntkomma's en UTF-8 met een BOM voor Excel. Meerdere waarden in een cel zijn gescheiden door ` | `; een lege lijst wordt `-`.

## Gedeelde analyse en regressiecontroles

Het script gebruikt `hydamo_validation.rule_analysis` voor afhankelijkheden, versieselectie en modelvergelijking. Downloads en CSV-uitvoer blijven in het script. De Python-interface retourneert lijsten en booleans direct, zodat aanroepers geen CSV-tekst hoeven te ontleden.

De bestaande tests in `tests/test_overzicht_validatieregels.py` controleren onder meer de analyse en rapportage. Een volledige regressiematrix van vaste regelbestanden tegen alle modelversies met expliciete verwachtingen is nog niet geïmplementeerd. De bestaande functie- en integratietests blijven nodig om de uitvoering op gegevens te controleren.

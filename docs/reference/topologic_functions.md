# Topologische functies

Topologische functies controleren ruimtelijke samenhang. Ze ontvangen een objectlaag (`gdf`) en het `HyDAMO`-datamodel (`datamodel`) en retourneren een pandas Series met beoordelingen. De overige parameters komen uit de regelconfiguratie.

De functies staan in `hydamo_validation.functions.topologic` en zijn ook beschikbaar via `from hydamo_validation import topologic_functions`. Gebruik geometrieën met een passend, gelijk CRS; het datamodel gebruikt RD New (`EPSG:28992`). Afstanden en toleranties zijn dan in meters.

| Functie | Controle en parameters |
| --- | --- |
| `snaps_to_hydroobject` | Controleert aansluiting op `datamodel.hydroobject`. `method` kiest `intersecting`, `overall`, `centroid` of `ends`; `tolerance` bepaalt de zoekafstand bij de laatste drie methoden. Voegt ook `branch_id` en `branch_offset` aan de invoerlaag toe. |
| `geometry_length` | Vergelijkt de geometrische lengte met `length`. `statistic="min"` toetst op minimaal deze lengte; `"max"` toetst op maximaal deze lengte. De grens telt mee. |
| `not_overlapping` | Controleert overlap met andere objecten in dezelfde laag binnen `tolerance`. Ondersteunt lagen met uitsluitend `Point` of uitsluitend `LineString`; andere combinaties veroorzaken een `TypeError`. |
| `splitted_at_junction` | Controleert of knooppunten van lijnen alleen bij lijnuiteinden liggen, met `tolerance` als afstandsgrens. |
| `structures_at_intersections` | Controleert of kruisingen van hydroobjecten een kunstwerk uit `structures` binnen `tolerance` hebben. `structures` is een lijst met laagnamen. |
| `no_dangling_node` | Controleert of het eindpunt van elke lijn binnen `tolerance` aansluit op een beginpunt uit dezelfde laag. |
| `structures_at_boundaries` | Controleert of kruisingen met grenzen van de laag `areas` een kunstwerk uit `structures` hebben. `tolerance` bepaalt de aansluiting op het hydroobject; `distance` de maximale afstand tot de gebiedsgrens. |
| `distant_to_others` | Controleert of de afstand tot andere objecten in dezelfde laag groter is dan `distance`. |
| `structures_at_nodes` | Controleert of kunstwerken uit `structures` niet op het binnenste van een lijn liggen en zo nodig bij een knooppunt zijn gesplitst. Gebruikt `tolerance`. |
| `compare_longitudinal` | Vergelijkt `parameter` met `compare_parameter` van het dichtstbijzijnde object uit `compare_object` op hetzelfde hydroobject. `direction` is `upstream` of `downstream`; `logical_operator` is `GT` of `LT`. |

Bij functies met `structures` geef je een lijst met laagnamen mee, ook als die lijst één naam bevat. Functies die objecten aan hydroobjecten koppelen, kunnen `branch_id` en `branch_offset` toevoegen aan de gebruikte lagen. De inhoudelijke regels en drempelwaarden staan in het [HyDAMO Validatiehandboek](https://hetwaterschapshuis.github.io/HyDAMOValidatiehandboek/HyDAMO%20Validatietool/Introductie%20tool/02%20Validatieregels/).

## API-reference

::: hydamo_validation.functions.topologic
    options:
      members: [snaps_to_hydroobject, geometry_length, not_overlapping, splitted_at_junction, structures_at_intersections, no_dangling_node, structures_at_boundaries, distant_to_others, structures_at_nodes, compare_longitudinal]


# Compare validation rules with datamodels

Use `scripts/overzicht_validatieregels.py` to check whether validation rules have
the layers and columns they need in a HyDAMO datamodel. This helps developers
assess a new or changed model before implementing support for it.

The analysis uses the same `HyDAMO` class as the validator, including its ignored
layers. It compares dependencies; it does not read a supplied dataset or execute
validation functions. An executable rule has the dependencies identified by the
analysis. This does not establish that its implementation runs correctly on data.

## Run the script

Complete the [development installation](contribute.md#setup-environment) with UV,
including `uv pip install -e .`, and run the following commands from the repository
root. The editable installation uses the code and schemas in your checkout.

```sh
uv run --no-sync python scripts/overzicht_validatieregels.py
```

By default, the script downloads versioned `ValidationRules_#.#.json` files from
the `main` branch of the HyDAMO validation handbook into `local/validation_rules`.
This requires network access and replaces matching local rule files. Reports are
written to `local/csv`; matching report files are replaced. Directories are created
when needed, and relative paths are resolved from the current working directory.

For a reproducible comparison, use the same local rule files before and after a
change. This command uses existing files without downloading them:

```sh
uv run --no-sync python scripts/overzicht_validatieregels.py --rules-json local/validation_rules --output-directory local/csv
```

`--rules-json` also accepts a single file, for example
`local/validation_rules/ValidationRules_1.5.json`. Use `--help` to list the options.

## Select datamodel versions

Normally, each rule file is analysed against the versions listed in
`hydamo_version.enum` of its corresponding `rules_<schema>.json` in the package.
The rule file's `schema` field selects that rules schema. Versions are sorted
numerically; a missing schema causes an error rather than being skipped.

To select versions explicitly, repeat `--hydamo-version`:

```sh
uv run --no-sync python scripts/overzicht_validatieregels.py --rules-json local/validation_rules --hydamo-version 2.4 --hydamo-version 2.5
```

Duplicate versions are analysed once. The explicit selection applies to every
selected rule file, even if its rules schema does not support that version.

For an experimental model, put its schema in a separate directory using the name
`HyDAMO_<version>.json`. For example, with an experimental `HyDAMO_2.6.json` in
`local/schemas`:

```sh
uv run --no-sync python scripts/overzicht_validatieregels.py --rules-json local/validation_rules/ValidationRules_1.5.json --hydamo-version 2.6 --schemas-path local/schemas --output-directory local/csv_experimental
```

This example does not imply that HyDAMO 2.6 is supported. The supplied schema must
be readable by the current `HyDAMO` class. All selected model schemas must exist in
`--schemas-path`; there is no fallback to packaged model schemas. The option does
not change the rules schemas used to determine support. A custom schema path alone
does not select additional versions; use `--hydamo-version` for that.

The reports distinguish **support according to the rules schema** from **dependency
availability**. An unsupported combination can still have executable rules. You do
not need to change the official version enumeration to investigate a new model.

## Read the detail reports

`overzicht_regels_<version>.csv` contains one row per rule and analysed HyDAMO
version. Both general rules and validation rules are included. Dots in the rule
file version become underscores, for example `overzicht_regels_1_5.csv`.

| CSV field | Meaning |
| --- | --- |
| `hydamo_versie` | Datamodel version being analysed. |
| `regels_versie` | Rule file version from `ValidationRules_<version>.json`; for other filenames, the JSON `schema` value is used. |
| `hydamo_versie_regels_json` | HyDAMO version declared inside the rule file; this does not restrict the versions analysed. |
| `regelschema_versie` | JSON schema version declared by the rule file's `schema` field. |
| `combinatie_ondersteund_door_regelschema` | Whether the analysed model version is listed in that rules schema. |
| `regelsoort` | `general_rule` or `validatieregel`. |
| `laag` | Main object layer to which the rule applies. |
| `regel_id` | Rule ID within its object layer and rule kind. Together with the version fields these identify the row. |
| `regelnaam` | Rule name, or its result variable if no name is supplied. |
| `actief` | Configured activation, defaulting to `True`. This is separate from dependency availability; inactive rules are also analysed. |
| `validatietype` | Configured validation type, or `general` when absent. |
| `functie` | Main function configured by the rule. Dependencies from its filter are also analysed. |
| `benodigde_lagen` | All layers used by the rule, including missing layers. |
| `ontbrekende_lagen` | Required layers unavailable through `HyDAMO`, with a reason: absent from the schema or ignored by the datamodel. |
| `gebruikte_datamodelkolommen` | Column references used by the rule, including missing columns and columns of missing layers. References resolved to available general-rule results are listed separately. Includes implicit dependencies such as relationship keys; `geometry.z` is reported as `geometry`. |
| `tussenresultaten_general_rules` | References resolved to results of earlier general rules whose dependencies are available. |
| `ontbrekende_kolommen` | Required column references unavailable in an existing model layer and not resolved to an available general-rule result. If the entire layer is missing, it is reported under `ontbrekende_lagen` instead. |
| `uitvoerbaar` | `True` when no required layers or columns are missing according to the dependency analysis. |
| `conclusie` | Readable conclusion, including an indication when the rule is inactive. |

“Used” means referenced by the rule, not successfully found or actually executed.
For example, bridge general rule 1 from rule file 1.5 needs
`brug.globalid`, `kunstwerkopening.laagstedoorstroomhoogte` and
`kunstwerkopening.brugid`. For HyDAMO 2.4, all three appear under
`gebruikte_datamodelkolommen`, while `kunstwerkopening.brugid` also appears under
`ontbrekende_kolommen`. The rule is not executable.

General rules are analysed in ID order before the validation rules of each object.
Objects retain their order in the rule file. A general-rule result becomes
available to subsequent rules only if its producing rule has the required
dependencies. An unavailable intermediate result can therefore appear as a missing
column for a later rule. No general-rule values are calculated.

## Read the summary

`samenvatting_regels.csv` contains one row per rule file version and analysed model
version. It includes `regels_versie`, `hydamo_versie`, `regelschema_versie` and
`combinatie_ondersteund_door_regelschema`, with the meanings described above.

| Count | Meaning |
| --- | --- |
| `totaal_regels` | All analysed general rules and validation rules. |
| `uitvoerbare_regels` | Rules with all required dependencies available. |
| `niet_uitvoerbare_regels` | Rules with missing dependencies. |
| `totaal_general_rules` | Number of analysed general rules. |
| `totaal_validatieregels` | Number of analysed validation rules. |

These counts include inactive rules. They do not count validated data objects or
successful function executions.

All CSV files use semicolons and UTF-8 with a BOM for Excel. Within a cell, multiple
values are separated by ` | ` and an empty list is shown as `-`.

## Shared analysis and future regression checks

The script uses `hydamo_validation.rule_analysis` for dependencies, version
selection and model comparison. Downloads and CSV output remain in the script.
The [Python interface](../reference/rule_analysis.md) returns dependency lists and
booleans directly, so callers do not need to parse CSV text.

Two uses share this analysis: a developer can investigate new models with the
script, and future regression tests can check changes against fixed local rule
files and all model versions. The latter would compare results with explicit
expectations for supported combinations and known incompatibilities. Pytest
integration and those expectations have not yet been implemented. Executing
validation functions on data remains the responsibility of the existing function
and integration tests.

# Rule analysis

See the [rule comparison guide](../guides/rule_analysis.md) for CSV meanings and
command-line use. These functions analyse dependencies without downloading rules,
writing reports or executing validation functions.

To analyse a local rule file against every model version present in the package:

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

Omit `hydamo_versions` to use the versions allowed by the rules schema. Each result
contains the rule identity, dependency lists, `uitvoerbaar` and
`combinatie_ondersteund_door_regelschema`. Use the latter to distinguish declared
support from dependency availability. This interface can be called by future
tests using fixed local input; it does not implement test expectations itself.

::: hydamo_validation.rule_analysis.analyse_versions

::: hydamo_validation.rule_analysis.analyse_version

::: hydamo_validation.rule_analysis.find_versioned_files

# The HyDAMO Validation Module: hydamo_validation

Validation Module for HyDAMO data.

## Installation

Python 3.12 is required (see `requires-python` in `pyproject.toml`).
```
pip install hydamo-validation
```
Alternatively, install with uv

```
uv venv --python 3.12
uv pip install hydamo-validation
```

For a reproducible installation, download `requirements.lock` from the **Assets**
of the selected [GitHub Release](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/releases).
Use the lockfile from that release, rather than `main`, and replace `X.Y.Z` with
the version of the selected tag (`vX.Y.Z`):

```
uv venv --python 3.12
uv pip sync requirements.lock
uv pip install --no-deps hydamo-validation==X.Y.Z
uv pip check
```

See the [installation guide](docs/guides/get_started.md) for download commands
and platform requirements.

## Command line

Activate the environment (`.venv\Scripts\activate.bat` in Windows Command Prompt,
or `source .venv/bin/activate` on Linux/macOS), then run:

```
hydamo-validate "your/directory" --coverages "AHN=your/local/ahn/dir"
python -m hydamo_validation "your/directory" --coverages "AHN=your/local/ahn/dir"
```

Both commands validate the same project and write to its `results` directory.
Use `--help` for options. Exit codes are `0` for a technically completed run
(including failed validation rules), `1` for a technical failure, and `2` for
invalid arguments. See the [CLI guide](docs/guides/get_started.md#command-line-validation)
for details.

## Specify a coverage directory
To get the validator running you need some AHN data. You can find these in the [data directory](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/tree/ee9ea1efed385deb692b89057e9c97114fd8c3be/tests/data/dtm) of this directory. We assume you copy this to `your/local/ahn/dir`. Now specify your coverage and init the validator in a python-script:

```
from hydamo_validation import validator
from pathlib import Path

coverage = {"AHN": Path("your/local/ahn/dir")}

hydamo_validator = validator(
    output_types=["geopackage", "csv", "geojson"], coverages=coverage, log_level="INFO"
)

```

With this validator you can validate a directory that directory should have the following structure. The name of `datasets` directory and `ValidatorRules.json` are mandatory. Within datasets you can put one or more geopackages with `HyDAMO` layers.

```
your/directory/
├─ datasets/
│  ├─ hydamo.gpkg
├─ ValidationRules.json
```

Now you can validate the `HyDAMO` layers inside `your/ directory` by:
```

directory = Path("your/directory")

datamodel, layer_summary, result_summary = hydamo_validator(
    directory=directory, raise_error=False
)
```

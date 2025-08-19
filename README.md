# The HyDAMO Validation Module: hydamo_validation

Validation Module for HyDAMO data.

## Installation

### Python installation
Make sure you have an Miniconda or Anaconda installation. You can download these here:
 - https://www.anaconda.com/products/individual
 - https://docs.conda.io/en/latest/miniconda.html

During installation, tick the box "Add Anaconda to PATH", ignore the red remarks

### Create the `validatietool` environment
Use the `env/environment.yml` in the repository to create the conda environment: `validatietool`

```
conda env create -f environment.yml
```

After installation you can activate your environment in command prompt

```
conda activate validatietool
```

### Install hydamo_validation
Simply install the module in the activated environment:

```
pip install hydamo_validation
```

### Develop-install hydamo_validation
Download or clone the repository. Now simply install the module in the activated environment:

```
pip install .
```

## Run in Python

### Specify a coverage directory
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

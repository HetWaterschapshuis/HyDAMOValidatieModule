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

## Run local

### Specify a coverage directory
To get the validator running you need some AHN data. You can find these in the [data directory](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/tree/ee9ea1efed385deb692b89057e9c97114fd8c3be/tests/data/dtm) of this directory. We assume you copy this to `your/local/ahn/dir`. Now specify your coverage and init the validator:

```
from hydamo_validation import validator
from pathlib import Path

coverage = {"AHN": Path("your/local/ahn/dir")}

hydamo_validator = validator(
    output_types=["geopackage", "csv", "geojson"], coverages=coverage, log_level="INFO"
)

```



DATA_DIR = Path(__file__).parents[1].joinpath("tests", "data")
ISSUES_DIR = Path(r"d:\projecten\D2401.ValidatieModule\01.Issues")
ISSUE_CODE = "HYV-214"


coverage = {"AHN": DATA_DIR.joinpath(r"dtm")}
directory = ISSUES_DIR / ISSUE_CODE

# %%

hydamo_validator = validator(
    output_types=["geopackage"], coverages=coverage, log_level="INFO"
)

# %%
datamodel, layer_summary, result_summary = hydamo_validator(
    directory=directory, raise_error=True
)

A working example with data can be found in `notebooks/test_wrij.ipynb`. In the activated environment launch jupyter notebook by:

```
jupyter notebook
```

Select `test_wrij.ipynb` read and run it.

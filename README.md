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

## Run an example
A working example with data can be found in `notebooks/test_wrij.ipynb`. In the activated environment launch jupyter notebook by:

```
jupyter notebook
```

Select `test_wrij.ipynb` read and run it.

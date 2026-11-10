# Installation

The module supports Python 3.12 (`>=3.12,<3.13` in `pyproject.toml`). Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the environment.

## Install a release reproducibly

Select a [GitHub Release](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/releases)
and download `requirements.lock` from its **Assets**. The lockfile pins the runtime
dependencies; the module itself is installed separately. Always use the package
version matching the release tag, not the lockfile from the current `main` branch.

For example, in Windows PowerShell, replace `X.Y.Z` with the selected version:

```powershell
$version = "X.Y.Z"
$tag = "v$version"
Invoke-WebRequest "https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/releases/download/$tag/requirements.lock" -OutFile requirements.lock
uv venv --python 3.12
uv pip sync requirements.lock
uv pip install --no-deps "hydamo-validation==$version"
uv pip check
```

On Linux or macOS:

```sh
version=X.Y.Z
curl --fail --location --output requirements.lock "https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/releases/download/v${version}/requirements.lock"
uv venv --python 3.12
uv pip sync requirements.lock
uv pip install --no-deps "hydamo-validation==${version}"
uv pip check
```

[`uv pip sync`](https://docs.astral.sh/uv/pip/compile/#syncing-an-environment)
makes the environment match the lockfile. Install the module afterwards with
`--no-deps` to retain those dependency versions. Running `sync` again removes the
module, so repeat the `install --no-deps` step afterwards.

`requirements.lock` remains in the repository. For an older release without an
asset, obtain the file from that exact tag, if present. A package version must
also be available on PyPI before the installation command can succeed.

The runtime lockfile is checked with Python 3.12 on Windows x64, Linux x64 and
macOS ARM64 by the release workflow. The pinned geospatial packages require a
recent operating system: for example,
[pyproj 3.7.2 wheels](https://pypi.org/project/pyproj/3.7.2/#files) require macOS
14 or newer on ARM64. Other Python versions and platforms are not covered by
these checks.

## Activate the environment

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```sh
source .venv/bin/activate
```

## Command line validation

A project directory contains a `datasets` subdirectory with one or more
GeoPackages and a `validationrules.json` file. The validator writes its output to
the project's `results` directory, replacing previous results as before.

```text
project/
  datasets/
    hydamo.gpkg
  validationrules.json
```

Both entrypoints use the same validator:

```sh
hydamo-validate "path/to/project" --coverages "AHN=path/to/ahn"
python -m hydamo_validation "path/to/project" --coverages "AHN=path/to/ahn"
```

From the repository root, with the module installed in the active environment,
you can also run `python hydamo_validation "path/to/project"`.

Use quotes around paths containing spaces. Options:

| Argument | Meaning | Default |
| --- | --- | --- |
| `directory` | Project directory | Required |
| `--log-level INFO` or `DEBUG` | Logging detail | `INFO` |
| `--output-types geopackage csv geojson` | One or more output formats | `geopackage` |
| `--coverages KEY=VALUE` | Raster coverage location; repeat for multiple coverages | No coverages |
| `--raise-error` | Propagate technical exceptions with a full traceback | Disabled |

Each coverage requires a non-empty key and value, separated by the first `=`.
For example, `--coverages "AHN=C:/data/ahn" --coverages "DTM=C:/data/dtm"`.
Use `hydamo-validate --help` or `python -m hydamo_validation --help` for help.

The CLI prints the project directory, technical status, elapsed time and results
location. Existing logging and result files, including `validator.log` and
`validation_result.json`, remain available in `results`.

| Exit code | Meaning |
| --- | --- |
| `0` | Validation completed technically, even if objects fail validation rules. Inspect the result files for data quality. |
| `1` | A technical error interrupted validation or prevented a rule from executing. Partial results may be available. |
| `2` | Invalid or missing command line arguments. |

Technical errors appear as concise error messages. Add `--raise-error` when a
full traceback is needed for troubleshooting.

## Run an example in Python

A working example is available in `notebooks/test_wrij.ipynb` in the repository.
For a development installation and notebook dependencies, see the
[contribution guide](contribute.md).

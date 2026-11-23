# HyDAMO-validatiemodule

De Python-module `hydamo_validation` valideert GeoPackages met HyDAMO-gegevens aan de hand van het datamodel en configureerbare logische en topologische regels.

- [Documentatie van de module](https://hetwaterschapshuis.github.io/HyDAMOValidatieModule/): installatie, Python-interface, API-reference en ontwikkeling.
- [HyDAMO Validatiehandboek](https://hetwaterschapshuis.github.io/HyDAMOValidatiehandboek/): de validatietool, validatieregels en interpretatie van resultaten.

## Installatie en gebruik

De module ondersteunt Python 3.12 (`>=3.12,<3.13` in `pyproject.toml`). Voor een installatie met [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uv venv --python 3.12
uv pip install hydamo-validation
uv pip check
```

Voor een reproduceerbare installatie kies je de packageversie en `requirements.lock` uit dezelfde GitHub Release. Volg hiervoor de [installatiehandleiding](docs/guides/get_started.md#reproduceerbare-installatie).

Activeer de omgeving met `.venv\Scripts\activate.bat` (cmd), `.\.venv\Scripts\Activate.ps1` (PowerShell) of `source .venv/bin/activate` (Linux/macOS). Een project heeft deze structuur:

```text
project/
├── datasets/
│   └── hydamo.gpkg
└── validationrules.json
```

Voer de validatie uit met een van beide entrypoints:

```sh
python -m hydamo_validation "pad/naar/project" --coverages "AHN=pad/naar/ahn"
hydamo-validate "pad/naar/project" --coverages "AHN=pad/naar/ahn"
```

De module schrijft naar `results` in de projectmap en vervangt daarbij eerdere resultaten. Exitcode `0` betekent dat de validatie technisch is afgerond; gegevens kunnen wel zijn afgekeurd. Exitcode `1` meldt een technische fout en `2` ongeldige argumenten. Zie [installatie en gebruik](docs/guides/get_started.md) en de [API-reference](docs/reference/validator.md) voor technische details en foutafhandeling.

## Bijdragen en credits

Bijdragen zijn welkom. Zie [ontwikkelen en bijdragen](docs/guides/contribute.md) voor de ontwikkelomgeving, tests en voorwaarden voor codebijdragen.

De module is beschikbaar onder de [MIT-licentie](LICENSE). De geregistreerde bijdragers zijn Het Waterschapshuis, D2Hydro, HKV en HydroConsult. De maintainer is Philip Hansmann, bereikbaar via [hydamo-validatietool@hetwaterschapshuis.nl](mailto:hydamo-validatietool@hetwaterschapshuis.nl).

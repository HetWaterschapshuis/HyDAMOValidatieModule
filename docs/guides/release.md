# Releaseprocedure

Deze procedure is bedoeld voor beheerders en volgt op [ontwikkelen en bijdragen](contribute.md). Bereid wijzigingen voor in een pull request. Laat ze beoordelen en opnemen in `main` voordat je een release publiceert.

## 1. Versie en wijzigingen voorbereiden

Kies een versienummer volgens semantic versioning:

- **MAJOR**: een brekende wijziging, bijvoorbeeld `1.0.0` naar `2.0.0`.
- **MINOR**: nieuwe functionaliteit die achterwaarts compatibel is.
- **PATCH**: een bugfix of kleine compatibele wijziging.

Werk `__version__` in `hydamo_validation/__init__.py` bij. De packageconfiguratie leest dit versienummer automatisch. Beschrijf de wijzigingen in de [changelog](../changelog.md) en controleer de installatie-instructies en API-reference.

Het gecommitteerde `requirements.lock` wordt bij publicatie niet opnieuw gegenereerd. Als dependencies moeten wijzigen, werk het lockbestand vooraf bij in de pull request en controleer de installatie op de ondersteunde platforms. Compileer runtime-lockbestanden met Python 3.12 en `--universal`:

```sh
uv pip compile pyproject.toml --python 3.12 --universal --output-file requirements.lock
```

Doe dit alleen bij een bedoelde dependencywijziging. Voor bijdragen gelden de [voorwaarden voor codebijdragen](contribute.md#voorwaarden-voor-codebijdragen).

## 2. Release testen

Test met de runtimeversies uit `requirements.lock`. Gebruik een schone checkout van de releasecommit en een aparte omgeving:

```sh
uv venv --python 3.12
uv pip sync requirements.lock
uv pip install --constraint requirements.lock --constraint requirements-dev.lock pytest
uv pip install --no-deps .
uv pip check
```

Activeer de omgeving en voer de volledige suite uit:

```sh
python -m pytest
```

Controleer ook de documentatie met de [documentatieomgeving](contribute.md#documentatie) en `python -m mkdocs build --strict`. Publiceer pas nadat de controles slagen en de wijzigingen zijn beoordeeld.

## 3. GitHub Release publiceren

1. Open **Releases > Draft a new release** in de repository.
2. Kies de geteste commit op `main` als doel.
3. Maak een tag met exact `v` gevolgd door `__version__`, bijvoorbeeld `v1.7.0`. Gebruik geen punt na de `v`.
4. Gebruik de tag als releasetitel en beschrijf nieuwe functionaliteit, bugfixes en eventuele brekende wijzigingen.
5. Controleer de gegevens en publiceer de release.

De workflow **Test release and publish lockfile** (`.github/workflows/release.yml`) checkt de releasetag uit, vergelijkt de packageversie met de tag en test met Python 3.12 op Windows x64, Linux x64 en macOS ARM64. De workflow draait deze tests:

```sh
python -m pytest tests/test_cli.py tests/test_init.py tests/test_validationrules.py tests/test_logic_functions.py tests/test_overzicht_validatieregels.py
```

Deze controles draaien ook op pull requests naar `main`. De volledige suite blijft onderdeel van de handmatige releasecontrole.

Pas wanneer alle platformcontroles slagen, uploadt de workflow het ongewijzigde `requirements.lock` uit de tag als release-asset. Controleer de workflow en het asset. Bij een fout ontbreekt het asset; onderzoek de oorzaak voordat je de workflow opnieuw uitvoert. Als de code moet veranderen, gebruik dan een nieuwe releasecommit en tag, zodat tag, package en lockbestand bij elkaar blijven horen.

## 4. Package op PyPI publiceren

De releaseworkflow publiceert het package niet op PyPI. Gebruik een schone checkout van dezelfde geteste releasetag en de releaseomgeving van stap 2. Zorg dat `dist/` leeg is of gebruik een nieuwe checkout.

Installeer Twine met de bestaande constraints:

```sh
uv pip install --constraint requirements.lock --constraint requirements-dev.lock twine
uv pip check
```

Bouw en controleer de brondistributie en upload die vervolgens:

```sh
python setup.py sdist
python -m twine check dist/*
python -m twine upload dist/*
```

Gebruik voor de upload de PyPI-inlogmethode van de beheerder. Zet tokens of wachtwoorden niet in de documentatie of repository.

Controleer na publicatie of de versie beschikbaar is op [PyPI](https://pypi.org/project/hydamo-validation/). Test vervolgens de [versiespecifieke installatie](get_started.md#een-release-installeren) met het lockbestand uit dezelfde GitHub Release in een nieuwe omgeving.

## Documentatiepublicatie

Documentatie wordt los van de package-release gepubliceerd. De workflow **Publish documentation** publiceert vanuit `main` na een gemergede pull request of een handmatige run op `main`. Een releasetag start deze workflow niet. Zie [documentatie publiceren](contribute.md#documentatie-publiceren) voor de inrichting en handmatige uitvoering.

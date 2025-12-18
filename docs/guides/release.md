# Release

Deze handleiding volgt op de handleiding `contribute` en beschrijft de release-procedure vanuit de main branch.

## 1. Verander het versienummer van de module

Kies het nieuwe versienummer volgens _semantic versioning_:

- **MAJOR**: brekende wijzigingen (bijv. `1.0.0` → `2.0.0`)
- **MINOR**: nieuwe functionaliteit, achterwaarts compatibel (bijv. `1.3.0` → `1.4.0`)
- **PATCH**: bugfixes, kleine wijzigingen (bijv. `1.3.2` → `1.3.3`)

Voorbeeld: huidige versie is `1.4.1`, je kiest `1.4.2` omdat we bestaande functionaliteit hebben verbeterd.

Zet het versienummer in `hydamo_validation/__init__.py` achter [`__version__`](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/blob/main/hydamo_validation/__init__.py#L4).

Zorg dat alle wijzigingen via Pull Requests zijn gecommit in de main branch


## 2. Testen
Nadat álle code voor een nieuwe release is gecommit in de main branch, draai nog 1x alle tests met pytest. In de dev-environment (zie `env/dev_environment.yml`) is pytest beschikbaar. Run: 

`pytest --cov=hydamo_validation tests/`

Publiceer alleen als alle tests slagen:
![](images/test.png "pytest")

## 3. Aanmaken van een release
Zorg dat alle wijzigingen via Pull Requests zijn gecommit in de main branch

1. Ga naar de GitHub-pagina van de repository.
2. Klik op Releases (rechts in de sidebar of onder het tabje “Code”).
3. Klik op `Draft a new release` (of New release).
4. Kies `Tag: Select tag` en kies `Create new tag`. Maak een annotated tag gelijk aan versienummer, dus `v.X.Y.Z`, bijvoorbeeld `v1.4.2`
5. Laat `Target: main`
6. Zet de tag in de `Release title`, dus weer `vX.Y.Z`, dus `v1.4.2` in het voorbeeld hierboven
7. `Release notes`, klik eventueel `Generate release notes` en/of beschrijf de belangrijkste wijzigingen:
    * Nieuwe features
    * Bugfixes
    * Breaking changes
8. Klik op `Publish release`

Je ziet nu een nieuwe release beschikbaar in GitHub met een `afdruk` van de code uit de main branch

## 4. Publiceren op PyPi

Publiceren doe je in 2 stappen:
1. Bouw distributies met `python setup.py sdist` 
2. Upload naar PyPI `twine upload dist/* -p jouw_eigen_twine_password`

De laatste release moet nu ook beschikbaar zijn op https://pypi.org/project/hydamo-validation/ en wordt vanaf nu geinstalleerd met `pip install hydamo-validation`
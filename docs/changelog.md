# Changelog

Deze pagina bevat de bestaande versiehistorie. Zie [GitHub Releases](https://github.com/HetWaterschapshuis/HyDAMOValidatieModule/releases) voor latere releases en de bijbehorende release notes. De [releaseprocedure](guides/release.md) beschrijft de controles en publicatie.

## 1.1.1

### Toegevoegd

- `changelog.md` in de documentatie.
- `validation.log` in `data_dir`.
- Uitgebreidere logging: vermelding van de uitgevoerde logische regel.

### Gewijzigd

- Specificatie van Python- en packageversies in de `environment.yml`-bestanden: HYV-167.

### Opgelost

- De logische validatie filtert de regelset vóór uitvoering. Regels worden alleen uitgevoerd op aanwezige lagen, om crashes te voorkomen: HYV-187.

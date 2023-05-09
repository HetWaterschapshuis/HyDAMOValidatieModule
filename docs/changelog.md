## 1.1.1
### Added  
* changelog.md in documentation 
* validation.log in data_dir
* more extensive logging: info which logical-rule is executed

### Changed
* Specification of python and package versions in environment.yml's: HYV-167

### Fixed
* logical-validation: validation-ruleset is cleaned prior to iteration of rules; only rules will be executed on existing layers (avoiding crashes): HYV-187

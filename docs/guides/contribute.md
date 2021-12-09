# Contribute

## Installation
To contribute, you'll need a few more packages:

- `black`: to style your code

- `pytest`: for writing a test for your new code and check if nothing else is damaged

### Create `validatietool` environment for development
Use the `env/dev_environment.yml` in the repository to create the conda environment `validatietool` with all required packages

```
conda env create -f dev_environment.yml
```

After installation you can activate your environment in command prompt

### Get a copy
Fork the respository to your own GitHub account:
1. Click `fork` in the upper-right of the rository.
2. Select your own github account

The repository is now available on your own github account:

![](images/fork.gif "Fork repository")


### Install hydamo_validation in develop-mode
Clone the repository using git of github desktop. Install the module in the activated environment in develop-mode:

```
pip install . -e
```


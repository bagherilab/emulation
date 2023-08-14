# Hyperparameter Selection and Permutation Testing


[![Build Status](https://github.com/bagherilab/emulation_permutation/workflows/build/badge.svg)](https://github.com/bagherilab/emulation_permutation/actions?query=workflow%3Abuild)
[![Codecov](https://img.shields.io/codecov/c/gh/bagherilab/emulation_permutation?token=HYF4KEB84L)](https://codecov.io/gh/bagherilab/emulation_permutation)
[![Lint Status](https://github.com/bagherilab/emulation_permutation/workflows/lint/badge.svg)](https://github.com/bagherilab/emulation_permutation/actions?query=workflow%3Alint)
[![Documentation](https://github.com/bagherilab/emulation_permutation/workflows/documentation/badge.svg)](https://bagherilab.github.io/emulation_permutation/)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


[Description](#description) | [Installation](#installation) | [Usage](#usage)

## Description

Emulation is an automated machine learning tool for hyperparameter selection and permutation testing. 
The project was developed as part of the research described in the manuscript "Incorporating temporal information during feature engineering bolsters emulation of spatio-temporal emergence".

## Installation

Package and dependency management for this project is done with Poetry. 
To install dependencies, navigate to the project folder in the command line and run:

```bash
$ poetry install
```

If you do not have poetry installed, refer to the documantation they provide [here](https://python-poetry.org).

## Usage

Once dependencies are installed, add your data file (currently only `csv` files are supported) to the data folder. 
Next, there are several config files that inform the program on operating details.
All config files are located inside of the `src/conf` directory.

### Main config

The `config.yaml` file outlines high-level experimental details, incluing:
- The Sobol power with which to sample hyperparameters (`sobol_power`)
- A column of the data that should be used for stratified splitting and K-fold (`stratify`, can be left blank)
- Whether the experiment is a quantity experiment to test the effects of different amounts of training data (`quantity_experiment`)
- Whether or not the data should be cleaned of `NaN` and `inf` values (`clean_data`)

### Experiment configs

Inside the `cs` directory, config files can be specified for any experiment the user wants run. 
Examples can be found in the directory, but they must include:
- A list of models to run (`defaults`)
- At least one experiment in the format

  ```yaml
  [experiment name]:
    files:
      data: [name of csv data file]
    paths:
      log: ${hydra:runtime.cwd}/logs/[path to log save location]
      data: ${hydra:runtime.cwd}/data/[folder that contains experiment data]
      results: ${hydra:runtime.cwd}/results/[path to result save location
  ```

- List of features to train the models on, as well as a list of responses to predict (`data`, `features`, `response`)

### Model configs

Inside the `cs/models` directory, model configs can be used to specify the hyperparameters that should be searched over. 
Each model can have `continuous`, `discrete`, and `static` hyperparameters.

Once config files have been updated, start the Poetry virtual environment:

```bash
$ poetry shell
```

Finally, experiments can be run manually by specifying the experimental config:

```bash
$ python src/config.py cs=[config file]
```

Alternatively, experiment files can be specified in the `run.sh` bash script to be run in batches.



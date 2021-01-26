# Metadamage - Ancient Damage Estimation for Metagenomics

![PyPI](https://img.shields.io/pypi/v/metadamage) ![PyPI - License](https://img.shields.io/pypi/l/metadamage)

Personal recommendations for this project:

- Python Version: [Pyenv](https://github.com/pyenv/pyenv)
- Virtual Environment:  [Virtualenv](https://github.com/pyenv/pyenv-virtualenv)
- Package Management: [Poetry](https://python-poetry.org/)

Requires a decent Python version (>=3.8) installed. See more below for further installation hints.


## Installation:

With Poetry:
```console
$ poetry add metadamage
```
or, if you prefer regular pip:
```console
$ pip install metadamage
```

## Update:

With Poetry:
```console
$ poetry update metadamage
```
or, if you prefer regular pip:
```console
$ pip install metadamage --upgrade
```


## Metadamage CLI

For help on the CLI interface, `metadamage` provides a `help` function:

```console
$ metadamage --help
```

Example for running `metadamage` on a single file:
```console
$ metadamage --verbose --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt
```

`metadamage` also allows to run on a multiple files:
```console
$ metadamage --verbose --max-fits 10 --max-cores 2 ./data/input/*.txt
```

## Metadamage CLI Options

The `metadamage` CLI has the following options.

- Maximum values
  - `--max-fits`: Maximum number of fits to do. Default is None, i.e. fit everything.
  - `--max-plots`: Maximum number of plots to do. Default is None, i.e. plot everything that has to be fitted. Is always less than (or equal to) `max-fits`.
  - `--max-cores`: Maximum number of cores to use while fitting. Default is 1.
  - `--max-position`: Maximum position in the sequence to include. Default is +/- 15 (forward/reverse).

- Minimum values or cuts/thresholds for plots
  - `--min-damage`: Minimum threshold of damage (![D_\mathrm{max}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+D_%5Cmathrm%7Bmax%7D)
) in the fit to be included in the plot. Default is None, i.e. to not make any cuts.
  - `--min-sigma`: Minimum threshold of sigma (![n_\sigma](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+n_%5Csigma)
) in the fit to be included in the plot. Default is None, i.e. to not make any cuts.
  - `--min-alignments`: Minimum number of alignments (![N_\mathrm{alignments}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+N_%5Cmathrm%7Balignments%7D)
) in the sequence to be included in the plot. Default is None, i.e. to not make any cuts.

- Other:
  - `--sort-by`:     [`alignments`|`damage`|`sigma`]. Which order to sort the plots by when only plotting e.g. the top 10 plots. Default is `alignments` (which is also independent on the fit).

- Boolean Flags
  - `--verbose`: Prints out more information during the execution.
  - `--force-reload-files`: Force reload the original datafile (and thus not load the autogenerated parquet file).
  - `--force-fits`: Force remake the fits, even though the fit results already exist.
  - `--force-plots`: Force recreate  the plots, even though the plots already exist.
  - `--version`: Print the current version of the program and exit.


<!-- [tex-image-link-generator](https://tex-image-link-generator.herokuapp.com/) -->
<!-- https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b -->




<!-- poe release -->


<!--
## Anaconda

If first time using:

###### 1a)
```console
$ git clone https://github.com/ChristianMichelsen/metadamage
$ cd metadamage
$ conda env create -f environment.yaml
```

or, if already installed:
###### 1b)
```console
$ conda env update --file environment.yaml
```

Afterwards remember to activcate the new environment:
###### 2)
```console
$ conda activate metadamage
``` -->

<!--
git clone https://github.com/pyro-ppl/numpyro.git
# install jax/jaxlib first for CUDA support
pip install -e .[dev]  # contains additional dependencies for NumPyro development -->


<!-- poetry add git+https://github.com/christianmichelsen/metadamage.git -->
<!-- poetry add git+https://github.com/christianmichelsen/metadamage.git#develop -->
<!-- poetry add ./my-package/ -->



<!--
Make sure you have a local Python environment. Personally, I recommend using Pyenv for installing Python versions and Pyenv-Virtualenv for easy managing of virtuel environments. See e.g. [this](https://github.com/pyenv/pyenv-installer#pyenv-installer) for easy installation of both.

Make sure you have a decent Python version (>=3.8) installed:

pyenv install 3.8.7
pyenv virtualenv 3.8.7 metadamage38
pyenv activate metadamage38
pyenv local metadamage38 # in dir

create a new dir:

mkdir metadamage
cd metadamage

pyenv local metadamage38 -->

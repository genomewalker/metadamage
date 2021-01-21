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
  - `--max-fits`:      asdsasadasdad
  - `--max-plots`:     asdsasadasdad
  - `--max-cores`:     asdsasadasdad
  - `--max-position`:     asdsasadasdad

- Minimum values or cuts/thresholds
  - `--min-damage`:     asdsasadasdad
  - `--min-sigma`:     asdsasadasdad
  - `--min-alignments`:     asdsasadasdad

- Other:
  - `--sort-by`:     [`alignments`|`damage`|`sigma`]

- Boolean Flags
  - `--verbose`:     asdsasadasdad
  - `--force-reload-files`:     asdsasadasdad
  - `--force-plots`:     asdsasadasdad
  - `--force-fits`:     asdsasadasdad
  - `--version`:     asdsasadasdad


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


<!-- poetry add git+https://github.com/sdispater/pendulum.git -->
<!-- poetry add git+https://github.com/sdispater/pendulum.git#develop -->
<!-- poetry add ./my-package/ -->



<!--
Make sure you have a local Python environment. Personally, I recommend using Pyenv for installing Python versions and Pyenv-Virtualenv for easy managing of virtuel environments. See e.g. [this](https://github.com/pyenv/pyenv-installer#pyenv-installer) for easy installation of both.

Make sure you have a decent Python version (>=3.8) installed:

pyenv install 3.8.7
pyenv virtualenv 3.8.7 metadamage38
pyenv activate metadamage38

create a new dir:

mkdir metadamage
cd metadamage

pyenv local metadamage38 -->

# Metadamage - Ancient Damage Estimation for Metagenomics

[![PyPI](https://img.shields.io/pypi/v/metadamage)](https://pypi.org/project/metadamage) ![PyPI - License](https://img.shields.io/pypi/l/metadamage)

---

#### Work in progress. Please contact christianmichelsen@gmail.com for further information.

---


Personal recommendations for this project:

- Python Version: [Pyenv](https://github.com/pyenv/pyenv)
- Virtual Environment:  [Virtualenv](https://github.com/pyenv/pyenv-virtualenv)
- Package Management: [Poetry](https://python-poetry.org/)

This project requires a recent Python version (>=3.8) installed. See more [below](#setup).


## <a name="install"></a> Installation:

With Poetry:
```console
$ poetry add metadamage
```
or, if you prefer regular pip:
```console
$ pip install metadamage
```

## <a name="cli"></a> Metadamage CLI:

For help on the CLI interface, `metadamage` provides a `help` function:

```console
$ metadamage --help
```

Example for running `metadamage` on a single file:
```console
$ metadamage --max-fits 10 --max-cores 2 ./data/input/data_ancient.txt
```

`metadamage` also allows to run on a multiple files:
```console
$ metadamage --max-fits 10 --max-cores 2 ./data/input/*.txt
```

## <a name="options"></a> Metadamage CLI Options:

The `metadamage` CLI has the following options.

- Maximum values
  - `--max-fits`: Maximum number of fits to do. Default is None, i.e. fit everything.
  - `--max-plots`: Maximum number of plots to do. Default is 0, i.e. to not plot anything. If -1 plot everything that is fitted. Is always less than (or equal to) `max-fits`.
  - `--max-cores`: Maximum number of cores to use while fitting. Default is 1.
  - `--max-position`: Maximum position in the sequence to include. Default is +/- 15 (forward/reverse).

- Minimum values or cuts/thresholds for plots
  - `--min-damage`: Minimum threshold of damage (![D_\mathrm{max}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+D_%5Cmathrm%7Bmax%7D)
) in the fit to be included in the plot. Default is None, i.e. to not make any cuts.
  - `--min-sigma`: Minimum threshold of sigma (![n_\sigma](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+n_%5Csigma)
) in the fit to be included in the plot. Default is None, i.e. to not make any cuts.
  - `--min-alignments`: Minimum number of alignments (![N_\mathrm{alignments}](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+N_%5Cmathrm%7Balignments%7D)
) in the sequence to be included in the plot. Default is 10.

- Other:
  - `--sort-by`:     [`alignments`|`damage`|`sigma`]. Which order to sort the plots by when only plotting e.g. the top 10 plots. Default is `alignments` (which is also independent on the fit).
  - `--substitution-bases-forward`:  Which substitution to check for damage in the forward region. Do not change this value except for control checks. Default is `CT`.
  - `--substitution-bases-reverse`:  Which substitution to check for damage in the reverse region. Do not change this value except for control checks. Default is `GA`.

- Boolean Flags
  - `--force-reload-files`: Force reload the original datafile (and thus not load the autogenerated parquet file).
  - `--force-fits`: Force remake the fits, even though the fit results already exist.
  - `--force-plots`: Force recreate  the plots, even though the plots already exist.
  - `--version`: Print the current version of the program and exit.


<!-- [tex-image-link-generator](https://tex-image-link-generator.herokuapp.com/) -->
<!-- https://gist.github.com/a-rodin/fef3f543412d6e1ec5b6cf55bf197d7b -->

---

## <a name="setup"></a> Setup Local Python Environment:

Make sure you have a local Python environment. Personally, I recommend using Pyenv for installing Python versions and Pyenv-Virtualenv for easy managing of virtuel environments. See e.g. [this](https://github.com/pyenv/pyenv-installer#pyenv-installer) for easy installation of both.

Make sure you have a decent Python version (>=3.8) installed:
```console
$ pyenv install 3.8.7
```

Now we set up a virtual environment, such that changes you do in this environment does not affect your other Python projects:
```console
$ pyenv virtualenv 3.8.7 metadamage38
$ pyenv activate metadamage38
```

We now use Poetry to setup a new project which uses metadamage. Follow the interactive guide:
```console
$ poetry new metadamage-folder
$ cd metadamage-folder
```

Instead of activating the environment manually after every new login, we can tell `pyenv` to remember it for us:
```console
$ pyenv local metadamage38
```

We now have a working local, virtual Python environment where the packages are managed by Poetry, so we can now add metadamamage to our project:
```console
$ poetry add metadamage
```

At this point you should log out of your terminal and log in again for reloading everything. Now if you just write:

```console
$ metadamage
```

you should see the following:

```console
$ metadamage
Usage: metadamage [OPTIONS] FILENAMES...
Try 'metadamage --help' for help.

Error: Missing argument 'FILENAMES...'.
```

which shows that it is working and installed. You can now use `metadamage --help` for more help (together with the variable explanations [above](#options)).

---

## <a name="update"></a> Update:

With Poetry:
```console
$ poetry update metadamage
```
or, if you prefer regular pip:
```console
$ pip install metadamage --upgrade
```

---

## Conda:

If you prefer using Conda, you can also install `metadamage` (via pip). First create a folder:
```console
$ mkdir metadamage-conda
$ cd metadamage-conda
```

To install `metadamage`:
```console
$ wget https://raw.githubusercontent.com/ChristianMichelsen/metadamage/main/environment.yaml
$ conda env create -f environment.yaml
```

To update it to a new, released version of `metadamage`:
```console
$ wget https://raw.githubusercontent.com/ChristianMichelsen/metadamage/main/environment.yaml
$ conda env update --file environment.yaml
```

Finally remember to activate the environment:
```console
$ conda activate metadamage
```

<!-- poetry add git+https://github.com/christianmichelsen/metadamage.git -->
<!-- poetry add git+https://github.com/christianmichelsen/metadamage.git#develop -->

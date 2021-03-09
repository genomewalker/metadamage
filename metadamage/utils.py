# Scientific Library
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm as sp_norm
from scipy.stats.distributions import chi2 as sp_chi2

# Standard Library
from dataclasses import asdict, dataclass, field
from enum import Enum
from importlib.metadata import version
import importlib.resources as importlib_resources
import logging
from pathlib import Path
import shutil
from typing import List, Optional, Union

# Third Party
from PyPDF2 import PdfFileReader
from click_help_colors import HelpColorsCommand, HelpColorsGroup
import dill
from joblib import Parallel
from psutil import cpu_count
import platform

# First Party
from metadamage.progressbar import console, progress


# def __post_init__

#%%


logger = logging.getLogger(__name__)


#%%


@dataclass
class Config:
    max_fits: Optional[int]
    max_cores: int
    max_position: Optional[int]
    #
    min_alignments: int
    min_y_sum: int
    #
    substitution_bases_forward: str
    substitution_bases_reverse: str
    #
    force_fits: bool
    version: str
    #
    filename: Optional[str] = None
    name: Optional[str] = None
    filename_out: Optional[str] = None

    N_fits: Optional[int] = None

    num_cores: int = field(init=False)

    def __post_init__(self):
        self._set_num_cores()

    def _set_num_cores(self):
        available_cores = cpu_count(logical=True)
        if self.max_cores > available_cores:
            self.num_cores = available_cores - 1
            logger.info(
                f"'max_cores' is set to a value larger than the maximum available"
                f"so clipping to {self.num_cores} (available-1) cores"
            )
        elif self.max_cores < 0:
            self.num_cores = available_cores - abs(self.max_cores)
            logger.info(
                f"'max-cores' is set to a negative value"
                f"so using {self.num_cores} (available-max_cores) cores"
            )
        else:
            self.num_cores = self.max_cores

    def add_filename(self, filename):
        self.filename = filename
        self.name = extract_name(filename)
        self.filename_out = f"./data/out/{self.name}.hdf5"

    def set_number_of_fits(self, df):
        self.N_taxids = len(pd.unique(df.taxid))

        if self.max_fits is not None and self.max_fits > 0:
            self.N_fits = min(self.max_fits, self.N_taxids)

        # use all TaxIDs available
        else:
            self.N_fits = self.N_taxids

        logger.info(f"Setting number_of_fits to {self.N_fits}")

    def to_dict(self):
        return asdict(self)


class SubstitutionBases(str, Enum):
    AC = "AC"
    AG = "AG"
    AT = "AT"

    CA = "CA"
    CG = "CG"
    CT = "CT"

    GA = "GA"
    GC = "GC"
    GT = "GT"

    TA = "TA"
    TC = "TC"
    TG = "TG"


#%%


def find_style_file():
    with importlib_resources.path("metadamage", "style.mplstyle") as path:
        return path


def is_ipython():
    try:
        return __IPYTHON__
    except NameError:
        return False


def extract_name(filename, max_length=60):
    name = Path(filename).stem.split(".")[0]
    if len(name) > max_length:
        name = name[:max_length] + "..."
    logger.info(f"Running new file: {name}")
    return name


def file_is_valid(file):
    if Path(file).exists() and Path(file).stat().st_size > 0:
        return True
    return False


def delete_folder(path):
    try:
        shutil.rmtree(path)
    except OSError:
        logger.exception(f"Could not delete folder, {path}")


def init_parent_folder(filename):
    if isinstance(filename, str):
        filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)


def is_forward(df):
    return df["strand"] == "5'"


def get_forward(df):
    return df[is_forward(df)]


def get_reverse(df):
    return df[~is_forward(df)]


def get_specific_taxid(df, taxid):
    if taxid == -1:
        taxid = df.taxid.iloc[0]
    return df.query("taxid == @taxid")


def load_dill(filename):
    with open(filename, "rb") as file:
        return dill.load(file)


def save_dill(filename, x):
    init_parent_folder(filename)
    with open(filename, "wb") as file:
        dill.dump(x, file)


def save_to_hdf5(filename, key, value):
    with pd.HDFStore(filename, mode="a") as store:
        store.put(key, value, data_columns=True, format="Table")


def save_metadata_to_hdf5(filename, key, value, metadata):
    with pd.HDFStore(filename, mode="a") as store:
        store.get_storer(key).attrs.metadata = metadata


def load_from_hdf5(filename, key):
    with pd.HDFStore(filename, mode="r") as store:
        df = store.get(key)
    return df


def load_metadata_from_hdf5(filename, key):
    with pd.HDFStore(filename, mode="r") as store:
        metadata = store.get_storer(key).attrs.metadata
    return metadata


def get_hdf5_keys(filename, ignore_subgroups=False):
    with pd.HDFStore(filename, mode="r") as store:
        keys = store.keys()

    if ignore_subgroups:
        keys = list(set([key.split("/")[1] for key in keys]))
        return keys
    else:
        raise AssertionError(f"ignore_subgroups=False not implemented yet.")


#%%


def metadata_is_similar(cfg, key, include=None, exclude=None):
    """ Compares the metadata in the hdf5 file (cfg.filename_out) with that in cfg"""

    if include is not None and exclude is not None:
        raise AssertionError(f"Cannot both include and exclude")

    metadata_file = load_metadata_from_hdf5(filename=cfg.filename_out, key=key)
    metadata_cfg = cfg.to_dict()

    # the metadata is not similar if it contains different keys
    if set(metadata_file.keys()) != set(metadata_cfg.keys()):
        is_similar = False
        return is_similar

    if isinstance(include, (list, tuple)) and exclude is None:
        # include = ['max_fits', 'max_position']
        # exclude = None
        is_similar = all([metadata_file[key] == metadata_cfg[key] for key in include])
        return is_similar

    elif isinstance(exclude, (list, tuple)) and include is None:
        # include = None
        # exclude = ['max_cores', 'num_cores']
        all_keys = metadata_file.keys()
        is_similar = all(
            [
                metadata_file[key] == metadata_cfg[key]
                for key in all_keys
                if key not in exclude
            ]
        )
        return is_similar

    elif include is None and exclude is None:
        # include = None
        # exclude = None
        all_keys = metadata_file.keys()
        is_similar = all({metadata_file[key] == metadata_cfg[key] for key in all_keys})
        return is_similar

    else:
        raise AssertionError("Did not except to get here")


#%%


def avoid_fontconfig_warning():
    # Standard Library
    import os

    os.environ["LANG"] = "en_US.UTF-8"
    os.environ["LC_CTYPE"] = "en_US.UTF-8"
    os.environ["LC_ALL"] = "en_US.UTF-8"


def human_format(num, digits=3, mode="eng"):
    num = float(f"{num:.{digits}g}")
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    if mode == "eng" or mode == "SI":
        translate = ["", "k", "M", "G", "T"]
    elif mode == "scientific" or mode == "latex":
        translate = ["", r"\cdot 10^3", r"\cdot 10^6", r"\cdot 10^9", r"\cdot 10^12"]
    else:
        raise AssertionError(f"'mode' has to be 'eng' or 'scientific', not {mode}.")

    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), translate[magnitude]
    )


def file_exists(filename, forced=False):
    return Path(filename).exists() and not forced


def is_pdf_valid(filename, forced=False, N_pages=None):
    try:
        if file_exists(filename, forced=forced):
            pdf_reader = PdfFileReader(filename)
            if N_pages is None:
                return True
            if N_pages == pdf_reader.numPages:
                return True
    except:
        pass
    return False


def is_macbook():
    return platform.system() == "Darwin"


#%%


# def get_percentile_as_lim(x, percentile_max=99):
#     # percentile_max = 99.5
#     percentile_min = 100 - percentile_max

#     if x.min() == 0:
#         return (0, np.percentile(x, percentile_max))
#     else:
#         return (np.percentile(x, percentile_min), np.percentile(x, percentile_max))


#%%


# def get_num_cores(cfg):

#     if cfg.num_cores > 0:
#         num_cores = cfg.num_cores
#     else:
#         N_cores = cpu_count(logical=True)
#         num_cores = N_cores - abs(cfg.num_cores)
#         if num_cores < 1:
#             num_cores = 1
#     return num_cores


#%%


def get_sorted_and_cutted_df(df, df_results, cfg):

    min_damage = cfg.min_damage if cfg.min_damage else -np.inf
    min_sigma = cfg.min_sigma if cfg.min_sigma else -np.inf
    min_alignments = cfg.min_alignments if cfg.min_alignments else -np.inf

    query = (
        f"D_max >= {min_damage} "
        + f"and n_sigma >= {min_sigma} "
        + f"and N_alignments >= {min_alignments}"
    )

    # cut away fits and TaxIDs which does not satisfy cut criteria
    df_results_cutted = df_results.query(query)
    if len(df_results_cutted) == 0:
        logger.warning(
            f"{cfg.name} did not have any fits that matched the requirements. "
            f"Skipping for now"
        )
        return None

    d_sort_by = {
        "alignments": "N_alignments",
        "damage": "D_max",
        "sigma": "n_sigma",
    }
    sort_by = d_sort_by[cfg.sort_by.lower()]

    # sort the TaxIDs
    df_results_cutted_ordered = df_results_cutted.sort_values(sort_by, ascending=False)

    taxids = df_results_cutted_ordered.index

    # get the number_of_plots in the top
    taxids_top = taxids[: cfg.number_of_plots]

    # the actual dataframe, unrelated to the fits
    df_plot = df.query("taxid in @taxids_top")
    # the actual dataframe, unrelated to the fits, now sorted
    # df_plot_sorted = df_plot.sort_values(sort_by, ascending=False)
    df_plot_sorted = pd.concat(
        [df_plot.query(f"taxid == {taxid}") for taxid in taxids_top]
    )

    return df_plot_sorted


#%%

#%%


def is_df_accepted(df, cfg):
    if len(df) > 0:
        return True

    logger.warning(
        f"{cfg.name}: Length of dataframe was 0. Stopping any further operations on this file."
        f"This might be due to a quite restrictive cut at the moment."
        f"requiring that both C and G are present in the read."
    )
    return False


#%%


def initial_print(filenames, cfg):

    console.print("\n")
    console.rule("[bold red]Initialization")
    console.print(
        f"\nRunning [bold green underline]metadamage[/bold green underline] "
        f"on {len(filenames)} file(s) using the following configuration: \n"
    )
    console.print(cfg)
    console.print("")

    console.rule("[bold red]Main")
    console.print("")


#%%

# filename: Optional[str] = None
# %%


def normalize_header(cell):
    # Standard Library
    import re

    cell = re.sub(r'[-:;/\\,. \(\)#\[\]{}\$\^\n\r\xa0*><&!"\'+=%]', "_", cell)
    cell = re.sub("__+", "_", cell)
    cell = cell.strip("_")
    cell = cell.upper()
    cell = cell or "BLANK"
    return cell


#%%


def fix_latex_warnings_in_string(s):
    # https://matplotlib.org/stable/tutorials/text/usetex.html

    # fix LaTeX errors:
    replacements = [
        (r"_", r"\_"),
        (r"&", r"\&"),
        (r"#", r"\#"),
        (r"%", r"\%"),
        (r"$", r"\$"),
    ]
    # fix bad root title
    replacements.append(("root, no rank", "root"))
    for replacement in replacements:
        s = s.replace(replacement[0], replacement[1])
    return s

# Scientific Library
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm as sp_norm
from scipy.stats.distributions import chi2 as sp_chi2

# Standard Library
from dataclasses import dataclass, field
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

# First Party
from metadamage.progressbar import console, progress


# def __post_init__

#%%


logger = logging.getLogger(__name__)


#%%


@dataclass
class Config:
    # filenames: List[Path]
    max_fits: Optional[int]
    max_plots: Optional[int]
    max_cores: int
    max_position: Optional[int]
    #
    min_damage: Optional[int]
    min_sigma: Optional[int]
    min_alignments: int
    #
    sort_by: str
    substitution_bases_forward: str
    substitution_bases_reverse: str
    #
    # verbose: bool
    #
    force_reload_files: bool
    force_fits: bool
    force_plots: bool
    version: str
    #
    filename: Optional[str] = None
    name: Optional[str] = None
    number_of_fits: Optional[int] = None

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

    def set_number_of_fits(self, df):
        N_all_taxids = len(pd.unique(df.taxid))

        if self.max_fits is not None and self.max_fits > 0:
            self.number_of_fits = min(self.max_fits, N_all_taxids)
        else:  # use all TaxIDs available
            self.number_of_fits = N_all_taxids
        self.set_number_of_plots()
        logger.info(f"Setting number_of_fits to {self.number_of_fits}")

    def set_number_of_plots(self):
        if self.max_plots is None:
            self.number_of_plots = self.number_of_fits
        else:
            self.number_of_plots = self.max_plots

        # do not allow number of plots to be larger than number of fits
        if (self.number_of_fits is not None) and (
            self.number_of_plots > self.number_of_fits
        ):
            self.number_of_plots = self.number_of_fits

    @property
    def do_make_fits(self):
        if self.max_fits is None or self.max_fits > 0:
            return True
        return False

    @property
    def do_make_plots(self):
        if self.max_plots is None or self.max_plots > 0:
            return True
        return False

    # FILENAMES BASED ON CFG

    def _get_substitution_bases_name(self):
        if (
            self.substitution_bases_forward == "CT"
            and self.substitution_bases_reverse == "GA"
        ):
            return ""
        else:
            return (
                f"__{self.substitution_bases_forward}"
                f"__{self.substitution_bases_reverse}"
            )

    @property
    def filename_parquet(self):
        return (
            f"./data/parquet/{self.name}"
            + self._get_substitution_bases_name()
            + ".parquet"
        )

    @property
    def filename_fit_results(self):
        return (
            f"./data/fits/{self.name}"
            f"__number_of_fits__{self.number_of_fits}"
            + self._get_substitution_bases_name()
            + ".csv"
        )

    @property
    def filename_plot_error_rates(self):
        return (
            f"./figures/error_rates__{self.name}"
            f"__sort_by__{self.sort_by}"
            f"__number_of_plots__{self.number_of_plots}"
            + self._get_substitution_bases_name()
            + f".pdf"
        )

    @property
    def filename_plot_fit_results(self):
        return (
            f"./figures/all_fit_results"
            f"__number_of_fits__{self.number_of_fits}"
            + self._get_substitution_bases_name()
            + ".pdf"
        )


class SortBy(str, Enum):
    alignments = "alignments"
    damage = "damage"
    sigma = "sigma"


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


class AllFiledWereBad(Exception):
    pass


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


# def get_percentile_as_lim(x, percentile_max=99):
#     # percentile_max = 99.5
#     percentile_min = 100 - percentile_max

#     if x.min() == 0:
#         return (0, np.percentile(x, percentile_max))
#     else:
#         return (np.percentile(x, percentile_min), np.percentile(x, percentile_max))


#%%


def get_num_cores(cfg):

    if cfg.num_cores > 0:
        num_cores = cfg.num_cores
    else:
        N_cores = cpu_count(logical=True)
        num_cores = N_cores - abs(cfg.num_cores)
        if num_cores < 1:
            num_cores = 1
    return num_cores


#%%


def get_sorted_and_cutted_df(cfg, df, df_results):

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

import numpy as np
from scipy.stats.distributions import chi2 as sp_chi2
from scipy.stats import norm as sp_norm
from pathlib import Path
from scipy import stats
import dill
import shutil
from PyPDF2 import PdfFileReader
from tqdm.auto import tqdm
from joblib import Parallel
import pandas as pd
import importlib.resources as importlib_resources
from importlib.metadata import version

# def __post_init__

#%%


from psutil import cpu_count

from typing import Optional, List, Union
from pathlib import Path
from click_help_colors import HelpColorsGroup, HelpColorsCommand
from rich.console import Console
from dataclasses import dataclass, field


@dataclass
class Config:
    # filenames: List[Path]
    max_fits: Optional[int]
    max_plots: Optional[int]
    max_cores: int
    max_position: Optional[int]
    verbose: bool
    force_reload_files: bool
    force_plots: bool
    force_fits: bool
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
            if self.verbose:
                print(
                    f"'max-cores' is set to a value larger than the maximum available",
                    f"so clipping to {self.num_cores} (available-1) cores",
                )
        elif self.max_cores < 0:
            self.num_cores = available_cores - self.max_cores
            if self.verbose:
                print(
                    f"'max-cores' is set to a negative value",
                    f"so using {self.num_cores} (available-max_cores) cores",
                )
        else:
            self.num_cores = self.max_cores

    def set_number_of_fits(self, df):
        N_all_taxids = len(pd.unique(df.taxid))

        if self.max_fits is not None and self.max_fits > 0:
            self.number_of_fits = min(self.max_fits, N_all_taxids)
        else:  # use all TaxIDs available
            self.number_of_fits = N_all_taxids
        if self.verbose:
            print(f"Setting number_of_fits to {self.number_of_fits}")

    @property
    def make_fits(self):
        if self.max_fits is None or self.max_fits > 0:
            return True
        return False

    @property
    def make_plots(self):
        if self.max_plots is None or self.max_plots > 0:
            return True
        return False


#%%


def find_style_file():
    with importlib_resources.path("metadamage", "style.mplstyle") as path:
        return path


def is_ipython():
    try:
        return __IPYTHON__
    except NameError:
        return False


def extract_name(filename):
    return Path(filename).stem.split(".")[0]


def file_is_valid(file):
    if Path(file).exists() and Path(file).stat().st_size > 0:
        return True
    return False


def delete_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def clean_up_after_dask():
    delete_folder("./dask-worker-space")


def is_forward(df):
    return df["direction"] == "5'"


def get_forward(df):
    return df[is_forward(df)]


def get_reverse(df):
    return df[~is_forward(df)]


def is_reverse_direction(direction):
    matches = ["reverse", "3", "-"]
    if any(match in direction.lower() for match in matches):
        return True
    else:
        return False


def is_forward_direction(direction):
    return not is_reverse_direction(direction)


def init_parent_folder(filename):
    if isinstance(filename, str):
        filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)


def load_dill(filename):
    with open(filename, "rb") as file:
        return dill.load(file)


def save_dill(filename, x):
    init_parent_folder(filename)
    with open(filename, "wb") as file:
        dill.dump(x, file)


def avoid_fontconfig_warning():
    import os

    os.environ["LANG"] = "en_US.UTF-8"
    os.environ["LC_CTYPE"] = "en_US.UTF-8"
    os.environ["LC_ALL"] = "en_US.UTF-8"


def string_pad_left_and_right(s, left=0, right=0, char=" "):
    N = len(s)
    s = s.ljust(N + right, char)
    N = len(s)
    s = s.rjust(N + left, char)
    return s


# def compute_fraction_and_uncertainty(x, N):
#     f = x / N
#     sf = np.sqrt(f * (1 - f) / N)
#     return f, sf


# def number_of_sigma_to_prob(Z):
#     return sp_norm.sf(Z)


# def prob_to_number_of_sigma(P):
#     if P > 1:
#         P /= 100
#     return np.abs(sp_norm.isf(P))


# def inverse_percentile(array, x, as_percent=True, return_uncertainty=False):
#     x = (array < x).sum()
#     N = len(array)
#     f, sf = compute_fraction_and_uncertainty(x, N)
#     if as_percent:
#         f *= 100
#         sf *= 100
#     if return_uncertainty:
#         return f, sf
#     else:
#         return f


# def complementary_inverse_percentile(array, x, as_percent=True, return_uncertainty=False):

#     if return_uncertainty:
#         f, sf = inverse_percentile(
#             array,
#             x,
#             as_percent=as_percent,
#             return_uncertainty=True,
#         )
#         if as_percent:
#             return 100 - f, sf
#         else:
#             return 1 - f, sf

#     else:
#         f = inverse_percentile(
#             array,
#             x,
#             as_percent=as_percent,
#             return_uncertainty=True,
#         )

#         if as_percent:
#             return 100 - f
#         else:
#             return 1 - f


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

    return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), translate[magnitude])


# def group_contains_nans(group):
#     if group["f_C2T"].isna().sum() > 0 or group["f_G2A"].isna().sum() > 0:
#         return True
#     else:
#         return False


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

from psutil import cpu_count


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


def set_max_fpr(cut, thresholds, fpr, tpr):
    index = np.argmax(fpr > cut) - 1
    out = {"threshold": thresholds[index], "FPR": fpr[index], "TPR": tpr[index]}
    return out


def set_minimum_tpr(cut, thresholds, fpr, tpr):
    index = np.argmax(tpr > cut)
    out = {"threshold": thresholds[index], "FPR": fpr[index], "TPR": tpr[index]}
    return out


#%%


class ProgressParallel(Parallel):
    # https://stackoverflow.com/a/61900501

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

# Scientific Library
import numpy as np
import pandas as pd

# Standard Library
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from importlib import reload
import os.path
from pprint import pformat, pprint
import sys
import time
from typing import Iterable
from urllib.request import urlopen

# Third Party
import numpyro
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
    TransferSpeedColumn,
)
from tqdm.auto import tqdm

# First Party
from metadamage import fileloader, fit, plot, utils


numpyro.enable_x64()

#%%


def main(filenames, cfg):

    console = utils.console
    progress = utils.progress

    if cfg.verbose:
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

    all_fit_results = {}
    N_files = len(filenames)

    with progress:

        task_id_overall = progress.add_task(
            f"Overall progress", total=N_files, progress_type="overall"
        )

        for filename in filenames:
            name = utils.extract_name(filename)
            progress.add_task("task_name", progress_type="name", name=name)

            if not utils.file_is_valid(filename):
                console.print(f"Got error here: {name}")
                continue

            cfg.filename = filename
            cfg.name = name

            df = fileloader.load_dataframe(cfg)

            if not utils.is_df_accepted(df, cfg):
                continue

            if cfg.do_make_fits:
                cfg.set_number_of_fits(df)
                d_fits, df_results = fit.get_fits(df, cfg)
                all_fit_results[cfg.name] = df_results

                if cfg.do_make_plots:
                    plot.set_style()
                    plot.plot_error_rates(cfg, df, d_fits, df_results)

            progress.advance(task_id_overall)
        progress.advance(task_id_overall)

    if len(all_fit_results) >= 1:
        plot.set_style()
        N_alignments_mins = [0, 10, 100, 1000, 10_000, 100_000]
        plot.plot_fit_results(all_fit_results, cfg, N_alignments_mins=N_alignments_mins)


# if utils.is_ipython():

#     print("Doing iPython plot")

#     filenames = [
#         "./data/input/mikkel_data/LB-Ext-64-Lib-64-Index1.col.sorted.sam.gz.family.bdamage.gz.taxid.counts.txt"
#         # "./data/input/data_ancient.txt",
#         # "./data/input/data_control.txt",
#     ]

#     reload(utils)

#     cfg = utils.Config(
#         max_fits=100,
#         max_plots=10,
#         max_cores=-1,
#         max_position=15,
#         min_damage=None,
#         min_sigma=None,
#         min_alignments=None,
#         sort_by=utils.SortBy.alignments,
#         verbose=True,
#         force_reload_files=False,
#         force_plots=False,
#         force_fits=False,
#         version="0.0.0",
#     )

#     # Standard Library
#     import os
#     from pathlib import Path

#     path = Path().cwd().parent
#     os.chdir(path)

#     filenames = list(Path("./data/input/").rglob("mikkel_data/*taxid.counts.txt"))

#     if False:
#         # if True:
#         main(filenames, cfg)

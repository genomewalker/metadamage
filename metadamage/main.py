# Scientific Library
import numpy as np
import pandas as pd

# Standard Library
from importlib import reload
from pprint import pformat, pprint

# Third Party
import numpyro
from rich.console import Console
from tqdm.auto import tqdm

# First Party
from metadamage import fileloader, fit, plot, utils


numpyro.enable_x64()

#%%


def main(filenames, cfg):

    console = Console()

    if cfg.verbose:
        console.print(
            f"\nRunning [bold green]metadamage[/bold green]",
            f"on {len(filenames)} file(s)",
            f"using the following configuration: \n",
        )
        console.print(cfg)

    console.rule("[bold red]Chapter 2")

    all_fit_results = {}

    for filename in filenames:

        if not utils.file_is_valid(filename):
            print(f"Got error here: {filename}")
            continue

        cfg.filename = filename
        cfg.name = utils.extract_name(filename)

        df = fileloader.load_dataframe(cfg)
        cfg.set_number_of_fits(df)

        if cfg.do_make_fits:
            d_fits, df_results = fit.get_fits(df, cfg)
            all_fit_results[cfg.name] = df_results

            if cfg.do_make_plots:
                plot.set_style()
                plot.plot_error_rates(cfg, df, d_fits, df_results)

    if len(all_fit_results) >= 1:
        plot.set_style()
        N_alignments_mins = [0, 10, 100, 1000, 10_000, 100_000]
        plot.plot_fit_results(all_fit_results, cfg, N_alignments_mins=N_alignments_mins)


if utils.is_ipython():

    print("Doing iPython plot")

    filenames = [
        "./data/input/Lok-75-Sample-2a-Ext-A17-Lib17A-Index1.sorted.sam.gz.family.bdamage.gz.taxid.counts.txt"
        # "./data/input/data_ancient.txt",
        # "./data/input/data_control.txt",
    ]

    reload(utils)

    cfg = utils.Config(
        max_fits=None,
        max_plots=5,
        max_cores=2,
        max_position=15,
        min_damage=None,
        min_sigma=None,
        min_alignments=None,
        sort_by=utils.SortBy.damage,
        verbose=True,
        force_reload_files=False,
        force_plots=False,
        force_fits=False,
        version="0.0.0",
    )

    import os
    from pathlib import Path

    path = Path().cwd().parent
    os.chdir(path)

    if False:
        # if True:
        main(filenames, cfg)

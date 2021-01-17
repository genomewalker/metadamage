import numpy as np
import pandas as pd
from importlib import reload

from tqdm.auto import tqdm
from pprint import pprint, pformat

from rich.console import Console

from MADpy import fileloader
from MADpy import fit
from MADpy import plot
from MADpy import utils

import numpyro

numpyro.enable_x64()

#%%


def main(filenames, cfg):

    console = Console()

    if cfg.verbose:
        console.print(
            f"\nRunning [bold green]MADpy[/bold green]",
            f"on {len(filenames)} file(s)",
            f"using the following configuration: \n",
        )
        console.print(cfg)

    console.rule("[bold red]Chapter 2")

    all_fit_results = {}

    N_inner_loop = 1 + cfg.make_fits + cfg.make_plots
    bar_format = "{desc}"  # |{bar}| [{elapsed}]
    tqdm_kwargs = dict(bar_format=bar_format, dynamic_ncols=True, total=N_inner_loop, leave=False)
    pad = utils.string_pad_left_and_right

    with tqdm(filenames, desc="Overall progress", dynamic_ncols=True) as it:

        for filename in it:

            if not utils.file_is_valid(filename):
                print(f"Got error here: {filename}")
                continue

            cfg.filename = filename
            cfg.name = utils.extract_name(filename)
            it.set_postfix(name=cfg.name)

            with tqdm(**tqdm_kwargs) as pbar:
                pbar.set_description(pad("Loading", left=4))
                df = fileloader.load_dataframe(cfg)
                cfg.set_number_of_fits(df)
                pbar.update()

                d_fits = None
                if cfg.make_fits:
                    pbar.set_description(pad("Fitting", left=4))
                    d_fits, df_results = fit.get_fits(df, cfg)
                    all_fit_results[cfg.name] = df_results
                    pbar.update()

                if cfg.make_plots:
                    pbar.set_description(pad("Plotting", left=4))
                    plot.set_style()
                    plot.plot_error_rates(cfg, df, d_fits=d_fits)
                    pbar.update()

    if len(all_fit_results) >= 1:
        plot.set_style()
        N_alignments_mins = [0, 10, 100, 1000, 10_000, 100_000]
        plot.plot_fit_results(all_fit_results, cfg, N_alignments_mins=N_alignments_mins)


if utils.is_ipython():

    print("Doing iPython plot")

    filenames = [
        "./data/input/data_ancient.txt",
        "./data/input/data_control.txt",
    ]

    reload(utils)

    cfg = utils.Config(
        max_fits=None,
        max_plots=None,
        max_cores=2,
        verbose=True,
        force_reload_files=False,
        force_plots=False,
        force_fits=False,
        version="0.1.0",
    )

    # if False:
    if True:
        main(filenames, cfg)


#%%


# %%

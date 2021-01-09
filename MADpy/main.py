import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.groupby import groupby
from tqdm import tqdm
from p_tqdm import p_umap
from functools import partial
from dotmap import DotMap as DotDict

import datetime
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from importlib import reload

import os
import tempfile

os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib
import seaborn as sns

# import tracemalloc

import MADpy
import MADpy
import MADpy
import MADpy

import numpyro

numpyro.enable_x64()


plot.set_rc_params(fig_dpi=50)
# paths = fileloader.load_paths()

plot_latex = True

if utils.is_ipython() and not plot_latex:
    matplotlib.style.use("default")
    sns.set_style("white")
    sns.set_context("talk", font_scale=1, rc={"lines.linewidth": 2})


cfg = DotDict(
    {
        "name": "KapK",
        # "N_taxids": 1000,
        "N_taxids": None,
        # "N_aligmnets_minimum": 0,
        # "N_reads_minimum": 0,
        "max_pos": None,
        "verbose": True,
        #
        "make_fits": True,
        "make_plots": True,
        "max_plots": 100,
        #
        "force_reload": False,
        "force_plots": False,
        "force_fits": False,
        #
        # "parallel_plots": True,
        "parallel_plots": False,
        #
        "num_cores": 6,
    }
)

#%%


# names = ["KapK"]
# names = ["control"]
names = ["KapK", "control"]

all_fit_results = {}

# if __name__ == "__main__":
for name in names:

    print(f"\n\n{name}:", flush=True)
    cfg.name = name
    cfg.filename = utils.get_filename(cfg)

    df = fileloader.load_dataframe(cfg)
    df_top_N = fileloader.get_top_N_taxids(df, cfg.N_taxids)
    group = next((group for name, group in df_top_N.groupby("taxid", sort=False)))
    taxid = group["taxid"].iloc[0]

    if False:
        taxid = 1773
        taxid = 9606
        taxid = 203104
        taxid = 168807
        group = df.query(f"taxid == @taxid")

        reload(plot)
        plot.plot_single_group(group, cfg)

    d_fits = None
    if cfg.make_fits:
        d_fits, df_results = fit.get_fits(df_top_N, cfg)
        all_fit_results[name] = df_results

    if False:
        plot.plot_single_group(group, cfg, d_fits)

    # reload(fit)
    if cfg.make_plots:
        plot.plot_individual_error_rates(cfg, df, d_fits=d_fits)
        plt.close("all")


# x = x


#%%


if len(all_fit_results) != 0:

    fig, ax = plt.subplots(figsize=(10, 10))
    plot.plot_fit_results(
        all_fit_results,
        ax,
        xlim=(-3, 18),
        ylim=(0, 0.7),
        alpha_plot=0.1,
        alpha_hist=0.0,
    )
    if cfg.make_plots:
        fig.savefig(f"./figures/all_fit_results__N_taxids__{cfg.N_taxids}.pdf")
else:
    print("No fits to plot")

# %%

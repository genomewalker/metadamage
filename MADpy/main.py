import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.groupby import groupby
from tqdm import tqdm

# from p_tqdm import p_umap
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

from MADpy import fileloader
from MADpy import fit
from MADpy import plot
from MADpy import utils

import numpyro

numpyro.enable_x64()


plot.set_rc_params(fig_dpi=50)
# paths = fileloader.load_paths()

if utils.is_ipython():
    plot.set_style(style_path="style.mplstyle")

cfg = DotDict(
    {
        "name": "KapK",
        # "N_taxids": 1000,
        "N_taxids": 10,
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
        "num_cores": 5,
    }
)

#%%


filenames = {
    "KapK": "../data/input/KapK-12-1-35-Ext-12-Lib-12-Index2.col.sorted.sam.gz.family.bdamage.gz.counts.txt",
    "control": "../data/input/EC-Ext-14-Lib-14-Index1.col.sorted.sam.gz.family.bdamage.gz.counts.txt",
}

names = ["control"]
names = ["KapK"]

filenames = {k: v for k, v in filenames.items() if k in names}

all_fit_results = {}

# if __name__ == "__main__":
for name, filename in filenames.items():

    print(f"\n\n{name}:", flush=True)
    cfg.name = name
    cfg.filename = filename

    df = fileloader.load_dataframe(cfg)
    df_top_N = fileloader.get_top_N_taxids(df, cfg.N_taxids)
    group = next((group for name, group in df_top_N.groupby("taxid", sort=False)))
    taxid = group["taxid"].iloc[0]

    if False:
        taxid = 9606
        taxid = 9615
        taxid = 1491
        group = df.query(f"taxid == @taxid")

        reload(plot)
        plot.plot_single_group(group, cfg)

    d_fits = None
    if cfg.make_fits:
        d_fits, df_results = fit.get_fits(df, cfg)
        all_fit_results[name] = df_results

    if False:
        reload(plot)
        plot.plot_single_group(group, cfg, d_fits)

    # reload(fit)
    if cfg.make_plots:
        plot.plot_individual_error_rates(cfg, df, d_fits=d_fits)
        plt.close("all")


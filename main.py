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

#%%


def main(filenames, cfg):

    all_fit_results = {}

    for filename in filenames:
        cfg["filename"] = filename
        cfg["name"] = utils.extract_name(filename)

        df = fileloader.load_dataframe(cfg)
        # df_top_N = fileloader.get_top_N_taxids(df, cfg.N_taxids)

        d_fits = None
        if cfg.make_fits:
            d_fits, df_results = fit.get_fits(df, cfg)
            all_fit_results[cfg.name] = df_results

        if cfg.make_plots:
            plot.set_style()
            plot.plot_error_rates(cfg, df, d_fits=d_fits)

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

    # filenames = [
    #     "./data/input/KapK-12-1-35-Ext-12-Lib-12-Index2.col.sorted.sam.gz.family.bdamage.gz.counts.txt",
    #     "./data/input/EC-Ext-14-Lib-14-Index1.col.sorted.sam.gz.family.bdamage.gz.counts.txt",
    # ]

    cfg = DotDict(
        {
            "N_taxids": -1,
            "max_pos": None,
            "verbose": True,
            "make_fits": True,
            "make_plots": True,
            "max_plots": 100,
            "force_reload": False,
            "force_plots": False,
            "force_fits": False,
            "parallel_plots": True,
            "num_cores": 5,
        }
    )

    main(filenames, cfg)


# filenames = {
#     "KapK-12-1-35-Ext-12-Lib-12-Index2": "./data/input/KapK-12-1-35-Ext-12-Lib-12-Index2.col.sorted.sam.gz.family.bdamage.gz.counts.txt",
#     "EC-Ext-14-Lib-14-Index1": "./data/input/EC-Ext-14-Lib-14-Index1.col.sorted.sam.gz.family.bdamage.gz.counts.txt",
# }

# # names = ["EC-Ext-14-Lib-14-Index1"]  # control
# # names = ["KapK-12-1-35-Ext-12-Lib-12-Index2"]
# # filenames = {k: v for k, v in filenames.items() if k in names}

# all_fit_results = {}

# # if __name__ == "__main__":
# for name, filename in filenames.items():

#     print(f"\n\n{name}:", flush=True)
#     cfg.name = name
#     cfg.filename = filename

#     df = fileloader.load_dataframe(cfg)
#     df_top_N = fileloader.get_top_N_taxids(df, cfg.N_taxids)
#     group = next((group for name, group in df_top_N.groupby("taxid", sort=False)))
#     taxid = group["taxid"].iloc[0]

#     if False:
#         taxid = 9606
#         group = df.query(f"taxid == @taxid")
#         plot.plot_single_group(group, cfg)

#     d_fits = None
#     if cfg.make_fits:
#         d_fits, df_results = fit.get_fits(df, cfg)
#         all_fit_results[name] = df_results

#     if False:
#         reload(plot)
#         plot.plot_single_group(group, cfg, d_fits)

#     # reload(fit)
#     if cfg.make_plots:
#         plot.set_style()
#         plot.plot_fit_results(all_fit_results, cfg, N_alignments_min=10_000)

# Scientific Library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Standard Library
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from importlib import reload
import logging

# Third Party
import numpyro

# First Party
from metadamage import fileloader, fit, plot, utils
from metadamage.progressbar import console, progress


numpyro.enable_x64()
logger = logging.getLogger(__name__)

#%%


def main(filenames, cfg):

    utils.initial_print(filenames, cfg)

    # all_fit_results = {}
    N_files = len(filenames)
    bad_files = 0

    with progress:

        task_id_overall = progress.add_task(
            f"Overall progress",
            total=N_files,
            progress_type="overall",
        )

        for filename in filenames:

            name = utils.extract_name(filename)

            if not utils.file_is_valid(filename):
                logger.error(f"{name} is not a valid file. Skipping for now.")
                bad_files += 1
                continue

            progress.add_task("task_name", progress_type="name", name=name)
            cfg.filename = filename
            cfg.name = name

            df = fileloader.load_dataframe(cfg)
            # print(len(pd.unique(df.taxid)))
            # continue
            # group = utils.get_specific_taxid(df, taxid=-1)  # get very first group

            if not utils.is_df_accepted(df, cfg):
                continue

            if cfg.do_make_fits:
                cfg.set_number_of_fits(df)
                d_fits, df_results = fit.get_fits(df, cfg)
                # all_fit_results[cfg.name] = df_results

                # if cfg.do_make_plots:
                #     # plot.set_style()
                #     plot.plot_error_rates(cfg, df, d_fits, df_results)

            progress.refresh()
            progress.advance(task_id_overall)
            logger.debug("End of loop\n")

    # if all files were bad, raise error
    if bad_files == N_files:
        raise Exception("All files were bad!")

    # if len(all_fit_results) >= 1 and cfg.do_make_plots:
    #     # plot.set_style()
    #     N_alignments_mins = [0, 10, 100, 1000, 10_000, 100_000]
    #     plot.plot_fit_results(all_fit_results, cfg, N_alignments_mins=N_alignments_mins)


if utils.is_ipython():

    print("Doing iPython plot")

    filenames = [
        # "./data/input/ugly/KapK_small.UglyPrint.txt"
    ]

    reload(utils)

    cfg = utils.Config(
        max_fits=10,
        # max_plots=0,
        max_cores=-1,
        max_position=15,
        # min_damage=None,
        # min_sigma=None,
        min_alignments=10,
        min_y_sum=10,
        # sort_by=utils.SortBy.alignments.value,
        substitution_bases_forward=utils.SubstitutionBases.CT.value,
        substitution_bases_reverse=utils.SubstitutionBases.GA.value,
        # force_reload_files=False,
        # force_plots=False,
        force_fits=False,
        # force_no_plots=False,
        version="0.0.0",
    )

    # Standard Library
    import os
    from pathlib import Path

    path = Path().cwd().parent
    os.chdir(path)

    filenames = sorted(Path("./data/input/").rglob("ugly/*.txt"))
    filename = filenames[2]
    filename = filenames[3]

    if False:
        # if True:
        main(filenames, cfg)

        # filename_parquet = cfg.filename_parquet
        filename_parquet = "./data/parquet/n_sigma_test.parquet"

        df = pd.read_parquet(filename_parquet)
        taxid = 115547
        taxid = 3745
        taxid = 395312
        group = utils.get_specific_taxid(df, taxid)  # get very first group


#%%

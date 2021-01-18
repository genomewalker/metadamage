import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

from MADpy_pkg import utils


set_ACTG = set(["A", "C", "G", "T"])


columns_name_mapping = {
    "#taxid": "taxid",
    "Nalignments": "N_alignments",
    "Direction": "direction",
    "Pos": "pos",
}


def get_base_columns(df):
    base_columns = []
    for column in df.columns:
        if len(column) == 2 and column[0] in set_ACTG and column[1] in set_ACTG:
            base_columns.append(column)
    return base_columns


def add_N_counts(df):
    """ Adds the toal number of bases """
    base_columns = get_base_columns(df)
    df["N_counts"] = df[base_columns].sum(axis=1)
    return df


def get_reference_columns(df, ref):
    ref_columns = []
    for column in get_base_columns(df):
        if column[0] == ref:
            ref_columns.append(column)
    return ref_columns


def add_reference_counts(df, ref):
    reference_columns = get_reference_columns(df, ref)
    df[ref] = df[reference_columns].sum(axis=1)
    return df


def compute_fraction_and_uncertainty(x, N, set_zero_to_nan=False):
    f = x / N
    if set_zero_to_nan:
        f = f.mask(x == 0, np.nan)
    sf = np.sqrt(f * (1 - f) / N)
    return f, sf


def compute_error_rates(df, ref, obs):
    s_ref_obs = ref + obs
    x = df[s_ref_obs]
    N_ref = df[ref]
    # N_ref = df[ref_columns].sum(axis=1)
    f, sf = compute_fraction_and_uncertainty(x, N_ref)
    return f, sf


def add_error_rates(df, ref, obs, include_uncertainties=False):
    f, sf = compute_error_rates(df, ref, obs)
    df[f"f_{ref}2{obs}"] = f
    if include_uncertainties:
        df[f"sf_{ref}2{obs}"] = sf
    return df


def add_error_rates_other(df, include_uncertainties=False):
    others = ["AC", "AG", "AT", "CA", "CG", "GC", "GT", "TA", "TC", "TG"]
    N_A = df[get_reference_columns(df, ref="A")].sum(axis=1)
    N_C = df[get_reference_columns(df, ref="C")].sum(axis=1)
    N_G = df[get_reference_columns(df, ref="G")].sum(axis=1)
    N_T = df[get_reference_columns(df, ref="T")].sum(axis=1)
    numerator = df[others].sum(axis=1)
    denominator = 3 * N_A + 2 * N_C + 2 * N_G + 3 * N_T
    f, sf = compute_fraction_and_uncertainty(numerator, denominator)
    df[f"f_other"] = f
    if include_uncertainties:
        df[f"sf_other"] = sf
    return df


def make_position_1_indexed(df):
    "Make the position, z, one-indexed (opposed to zero-indexed)"
    df["pos"] += 1
    return df


def make_reverse_position_negative(df):
    pos = df["pos"]
    is_reverse = ~utils.is_forward(df)
    pos_reverse = pos[~utils.is_forward(df)]
    # pos_reverse *= -1
    df["pos"] = df["pos"].mask(is_reverse, -pos_reverse)
    return df


def delayed_list(lst, length):
    @dask.delayed(nout=length)
    def delayed_list_tmp(lst):
        out = []
        for l in lst:
            out.append(l)
        return out

    return delayed_list_tmp(lst)


def delayed_list_unknown_length(lst):
    @dask.delayed()
    def delayed_list_tmp(lst):
        out = []
        for l in lst:
            out.append(l)
        return out

    return delayed_list_tmp(lst)


def extract_top_max_fits_dask(df, max_fits):
    top_max_fits = df.groupby("taxid", observed=True)["N_alignments"].sum().nlargest(max_fits).index
    df_top_N = df[df["taxid"].isin(delayed_list(top_max_fits, max_fits))]
    return df_top_N


def extract_top_max_fits(df, max_fits):
    top_max_fits = df.groupby("taxid", observed=True)["N_alignments"].sum().nlargest(max_fits).index
    df_top_N = df[df["taxid"].isin(top_max_fits)]
    return df_top_N


def remove_base_columns(df):
    columns_to_keep = []
    columns = df.columns
    base_columns = get_base_columns(df)
    for col in columns:
        if col not in base_columns:
            columns_to_keep.append(col)
    return df[columns_to_keep]


def keep_only_base_columns(df, base_cols_to_keep):
    columns_to_keep = []
    columns = df.columns
    base_columns = get_base_columns(df)
    base_cols_to_discard = [base for base in base_columns if base not in base_cols_to_keep]
    for col in columns:
        if col not in base_cols_to_discard:
            columns_to_keep.append(col)
    return df[columns_to_keep]


def sort_by_alignments(df_top_N):
    pos = df_top_N["pos"]
    df_top_N["order"] = pos.mask(pos > 0, 1 / pos)
    return df_top_N.sort_values(by=["N_alignments", "order"], ascending=False).drop(
        columns=["order"]
    )


def cut_NANs_away(df):
    # we throw away rows with no C references or G references
    nan_mask = (df["C"] == 0) | (df["G"] == 0)
    df_nans = df.loc[nan_mask]
    bad_taxids = df_nans.taxid.unique()
    mask_bad = df.taxid.isin(bad_taxids)
    return df.loc[~mask_bad]


def get_top_max_fits(df, number_of_fits):
    if number_of_fits is not None and number_of_fits > 0:
        return df.pipe(extract_top_max_fits, number_of_fits)
    else:
        return df


def _load_dataframe_dask(filename):

    with Client(processes=False) as client:

        # REFERNCE_OBSERVERET: "AC" means reference = "A", observed = "C"
        # In my terminology: A2C or A->C

        # client = Client(processes=False)

        df = (
            dd.read_csv(filename, sep="\t")
            .rename(columns=columns_name_mapping)
            # compute error rates
            .pipe(add_reference_counts, ref="C")
            .pipe(add_reference_counts, ref="G")
            .pipe(add_error_rates, ref="C", obs="T")
            # .pipe(add_error_rates, ref="C", obs="C")
            .pipe(add_error_rates, ref="G", obs="A")
            # .pipe(add_error_rates, ref="G", obs="G")
            .pipe(add_error_rates_other)
            # add other information
            .pipe(make_position_1_indexed)
            .pipe(make_reverse_position_negative)
            .pipe(keep_only_base_columns, ["C", "CT", "G", "GA"])
            # .pipe(keep_only_base_columns, [])
            # turns dask dataframe into pandas dataframe
            .compute()
            .pipe(cut_NANs_away)  # remove any taxids containing nans
            .reset_index(drop=True)
            .pipe(sort_by_alignments)
        )
    # client.shutdown()
    utils.clean_up_after_dask()
    return df


def load_dataframe(cfg):

    filename_parquet = f"./data/parquet/{cfg.name}.parquet"

    if utils.file_exists(filename_parquet, cfg.force_reload_files):
        if cfg.verbose:
            tqdm.write("Loading DataFrame from parquet-file.")
        df = pd.read_parquet(filename_parquet)
        return df

    if cfg.verbose:
        tqdm.write("Creating DataFrame, please wait.")

    try:
        df = _load_dataframe_dask(cfg.filename)
    except FileNotFoundError as e:
        raise AssertionError(f"\n\nFile: {cfg.filename} not found. \n{e}")

    categorical_cols = ["taxid", "direction"]  # not max_fits as it is numerical
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    if cfg.verbose:
        tqdm.write("Saving DataFrame to file (in data/parquet/) for faster loading. \n")
    utils.init_parent_folder(filename_parquet)
    df.to_parquet(filename_parquet, engine="pyarrow")

    return df


#%%

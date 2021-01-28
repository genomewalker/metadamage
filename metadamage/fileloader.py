# Scientific Library
import numpy as np
import pandas as pd

# Standard Library
import warnings

# Third Party
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from tqdm.auto import tqdm

# First Party
from metadamage import utils
from metadamage.progressbar import console, progress


ACTG = ["A", "C", "G", "T"]

ref_obs_bases = []
for ref in ACTG:
    for obs in ACTG:
        ref_obs_bases.append(f"{ref}{obs}")


# fmt: off
# fmt: on

columns = [
    "taxid",
    "name",
    "rank",
    "N_alignments",
    "strand",
    "position",
    *ref_obs_bases,
]


def set_column_names(df, method="dask"):

    df_names = (
        df.iloc[:, 0]
        # regex which finds : (colon) within quotes
        .str.replace(r"(?<![\"]):(?![\"])", "", regex=True)
        .str.replace('"', "")
        .str.split(":", expand=True, n=3)
    )
    df_names = df_names.rename(columns=dict(zip(df_names.columns, columns[:4])))
    df_bases = df.iloc[:, 1:]
    df_bases = df_bases.rename(columns=dict(zip(df_bases.columns, columns[4:])))
    if method == "dask":
        df = dd.concat(
            [df_names, df_bases],
            axis="columns",
            ignore_unknown_divisions=True,
        )
    else:
        df = pd.concat([df_names, df_bases], axis="columns")
    return df


def convert_dtypes(df):
    df = df.astype({"taxid": "int", "N_alignments": "int"})
    return df


# def strip_colons_from_file(filename):
#     filename_out = filename.replace(".txt", ".tmp.txt")
#     command = f"sed $'s/:/ \t/g' {filename} > {filename_out}"
#     import os

#     os.popen(command)
#     return filename_out


#%%


def clean_up_after_dask():
    utils.delete_folder("./dask-worker-space")


# def is_reverse_strand(strand):
#     matches = ["reverse", "3", "-"]
#     if any(match in strand.lower() for match in matches):
#         return True
#     else:
#         return False


# def is_forward_strand(strand):
#     return not is_reverse_strand(strand)


def get_subsitution_bases_to_keep(cfg):
    # ["C", "CT", "G", "GA"]
    forward = cfg.substitution_bases_forward
    reverse = cfg.substitution_bases_reverse
    bases_to_keep = [forward[0], forward, reverse[0], reverse]
    return bases_to_keep


#%%


def get_base_columns(df):
    base_columns = []
    for column in df.columns:
        if len(column) == 2 and column[0] in ACTG and column[1] in ACTG:
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
    df[f"f_{ref}{obs}"] = f
    if include_uncertainties:
        df[f"sf_{ref}{obs}"] = sf
    return df


# def add_error_rates_other(df, include_uncertainties=False):
#     others = ["AC", "AG", "AT", "CA", "CG", "GC", "GT", "TA", "TC", "TG"]

#     N_A = df[get_reference_columns(df, ref="A")].sum(axis=1)
#     N_C = df[get_reference_columns(df, ref="C")].sum(axis=1)
#     N_G = df[get_reference_columns(df, ref="G")].sum(axis=1)
#     N_T = df[get_reference_columns(df, ref="T")].sum(axis=1)
#     numerator = df[others].sum(axis=1)
#     denominator = 3 * N_A + 2 * N_C + 2 * N_G + 3 * N_T
#     f, sf = compute_fraction_and_uncertainty(numerator, denominator)
#     df[f"f_other"] = f
#     if include_uncertainties:
#         df[f"sf_other"] = sf
#     return df


def make_position_1_indexed(df):
    "Make the position, z, one-indexed (opposed to zero-indexed)"
    df["position"] += 1
    return df


def make_reverse_position_negative(df):
    pos = df["position"]
    is_reverse = ~utils.is_forward(df)
    pos_reverse = pos[~utils.is_forward(df)]
    # pos_reverse *= -1
    df["position"] = df["position"].mask(is_reverse, -pos_reverse)
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
    top_max_fits = (
        df.groupby("taxid", observed=True)["N_alignments"]
        .sum()
        .nlargest(max_fits)
        .index
    )
    df_top_N = df[df["taxid"].isin(delayed_list(top_max_fits, max_fits))]
    return df_top_N


def extract_top_max_fits(df, max_fits):
    top_max_fits = (
        df.groupby("taxid", observed=True)["N_alignments"]
        .sum()
        .nlargest(max_fits)
        .index
    )
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


def keep_only_base_columns(df, cfg):
    base_columns_to_keep = get_subsitution_bases_to_keep(cfg)
    columns_to_keep = []
    columns = df.columns
    base_columns = get_base_columns(df)
    base_cols_to_discard = [
        base for base in base_columns if base not in base_columns_to_keep
    ]
    for col in columns:
        if col not in base_cols_to_discard:
            columns_to_keep.append(col)
    return df[columns_to_keep]


def sort_by_alignments(df_top_N):
    pos = df_top_N["position"]
    df_top_N["order"] = pos.mask(pos > 0, 1 / pos)
    return df_top_N.sort_values(
        by=["N_alignments", "taxid", "order"], ascending=False
    ).drop(columns=["order"])


# def cut_NANs_away(df):
#     # we throw away rows with no C references or G references
#     bad_taxids = df.query("C == 0 | G == 0").taxid.unique()
#     df_nans_removed = df.query("taxid not in @bad_taxids")
#     return df_nans_removed


def replace_nans_with_zeroes(df):
    return df.fillna(0)


def get_top_max_fits(df, number_of_fits):
    if number_of_fits is not None and number_of_fits > 0:
        return df.pipe(extract_top_max_fits, number_of_fits)
    else:
        return df


def remove_taxids_with_too_few_alignments(df, cfg):
    return df.query(f"N_alignments >= {cfg.min_alignments}")


# def remove_quotes_from_string_columns(df):
#     for col in df.select_dtypes(include=["object"]).columns:
#         df[col] = df[col].str.replace('"', "")
#     return df


def compute_dataframe_with_dask(cfg, use_processes=True):

    # Standard Library
    import logging

    filename = cfg.filename

    if cfg.max_cores == 1:
        use_processes = False

    # http://localhost:8787/status
    with Client(
        n_workers=cfg.max_cores,
        processes=use_processes,
        silence_logs=logging.ERROR,
    ) as client:

        df = (
            # dd.read_csv(filename, sep="\t")
            dd.read_csv(
                filename,
                sep="\t",
                # sep=":|\t",
                header=None,
                # blocksize=50e6,  # chunksize 50mb
            )
            # .rename(columns=columns_name_mapping)
            .pipe(set_column_names)
            .pipe(convert_dtypes)
            .pipe(remove_taxids_with_too_few_alignments, cfg)
            # compute error rates
            .pipe(add_reference_counts, ref=cfg.substitution_bases_forward[0])
            .pipe(add_reference_counts, ref=cfg.substitution_bases_reverse[0])
            .pipe(
                add_error_rates,
                ref=cfg.substitution_bases_forward[0],
                obs=cfg.substitution_bases_forward[1],
            )
            .pipe(
                add_error_rates,
                ref=cfg.substitution_bases_reverse[0],
                obs=cfg.substitution_bases_reverse[1],
            )
            # .pipe(add_error_rates_other)
            # add other information
            .pipe(make_position_1_indexed)
            .pipe(make_reverse_position_negative)
            .pipe(keep_only_base_columns, cfg)
            .pipe(replace_nans_with_zeroes)
            # turns dask dataframe into pandas dataframe
            .compute()
            # .pipe(cut_NANs_away)  # remove any taxids containing nans
            .reset_index(drop=True)
            .pipe(sort_by_alignments)
        )

    # client.shutdown()
    # cluster.close()
    clean_up_after_dask()

    df2 = df.astype(
        {
            "taxid": "category",
            "name": "category",
            "rank": "category",
            "strand": "category",
        }
    )

    for col in df2.select_dtypes(include=["integer"]).columns:
        df2.loc[:, col] = pd.to_numeric(df2[col], downcast="integer")

    for col in df2.select_dtypes(include=["float"]).columns:
        df2.loc[:, col] = pd.to_numeric(df2[col], downcast="float")

    return df


def load_dataframe(cfg):

    filename_parquet = cfg.filename_parquet

    if utils.file_exists(filename_parquet, cfg.force_reload_files):
        # if cfg.verbose:
        #     console.print("  Loading DataFrame from parquet-file.")
        df = pd.read_parquet(filename_parquet)
        return df

    # if cfg.verbose:
    #     console.print("  Creating DataFrame, please wait.")
    df = compute_dataframe_with_dask(cfg, use_processes=True)

    # if cfg.verbose:
    #     console.print("Saving DataFrame to file (in data/parquet/) for faster loading. \n")
    utils.init_parent_folder(filename_parquet)
    df.to_parquet(filename_parquet, engine="pyarrow")

    return df

# Scientific Library
import numpy as np
import pandas as pd

# Standard Library
import logging
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


logger = logging.getLogger(__name__)

#%%

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


#%%


def clean_up_after_dask():
    utils.delete_folder("./dask-worker-space")


def get_subsitution_bases_to_keep(cfg):
    # ["C", "CT", "G", "GA"]
    forward = cfg.substitution_bases_forward
    reverse = cfg.substitution_bases_reverse
    bases_to_keep = [forward[0], forward, reverse[0], reverse]
    return bases_to_keep


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


def compute_y_sum_total(group, cfg):
    y_sum_total = 0
    forward_bases = (
        cfg.substitution_bases_forward[0] + cfg.substitution_bases_forward[1]
    )
    y_sum_total += group[group.position > 0][forward_bases].sum()
    reverse_bases = (
        cfg.substitution_bases_reverse[0] + cfg.substitution_bases_reverse[1]
    )
    y_sum_total += group[group.position < 0][reverse_bases].sum()
    return y_sum_total


def add_y_sum_counts(df, cfg):

    meta = pd.Series(
        [],
        name="y_sum_total",
        index=pd.Index([], name="taxid", dtype=int),
        dtype=int,
    )

    ds = df.groupby("taxid").apply(compute_y_sum_total, cfg, meta=meta)
    # ds = df.groupby("taxid").apply(compute_y_sum_total, cfg)
    ds = ds.reset_index()  # .rename(columns={0: "y_sum_total"})
    # ds = ds.reset_index().rename(columns={0: "y_sum_total"})
    df = dd.merge(df, ds, on=["taxid"])
    return df


# def filter_taxids_passing_y_sum_cut(df, cfg):

#     ds = df.groupby("taxid").apply(compute_y_sum_total, cfg)
#     accepted_taxids = ds[ds > cfg.min_y_sum].index

#     if isinstance(df, dask.dataframe.DataFrame):
#         length = len(df) // (cfg.max_position * 2)
#         df_accepted = df[df["taxid"].isin(delayed_list(accepted_taxids, length=length))]

#     else:
#         df_accepted = df[df["taxid"].isin(accepted_taxids)]
#     return df_accepted


def get_top_max_fits(df, N_fits):
    if N_fits is not None and N_fits > 0:
        return df.pipe(extract_top_max_fits, N_fits)
    else:
        return df


def remove_taxids_with_too_few_alignments(df, cfg):
    return df.query(f"N_alignments >= {cfg.min_alignments}")


def remove_taxids_with_too_few_y_sum_total(df, cfg):
    return df.query(f"y_sum_total >= {cfg.min_y_sum}")


def compute_dataframe_with_dask(cfg, use_processes):

    # Standard Library

    filename = cfg.filename

    # if cfg.num_cores == 1:
    # use_processes = False

    # http://localhost:8787/status
    with Client(
        n_workers=cfg.num_cores,
        processes=use_processes,
        # processes=True,
        # silence_logs=logging.ERROR,
        silence_logs=logging.CRITICAL,
        # silence_logs=True,
        # silence_logs=False,
        local_directory="./dask-worker-space",
        # asynchronous=False,
        # silence_logs=False,
        # processes=False,
    ):

        df = (
            # dd.read_csv(filename, sep="\t")
            dd.read_csv(
                filename,
                sep="\t",
                # sep=":|\t",
                header=None,
                names=columns,
                # blocksize=50e6,  # chunksize 50mb
            )
            # .pipe(remove_taxids_with_too_few_alignments, cfg)
            # compute error rates
            .pipe(add_reference_counts, ref=cfg.substitution_bases_forward[0])
            .pipe(add_reference_counts, ref=cfg.substitution_bases_reverse[0])
            # error rates forward
            .pipe(
                add_error_rates,
                ref=cfg.substitution_bases_forward[0],
                obs=cfg.substitution_bases_forward[1],
            )
            # error rates reverse
            .pipe(
                add_error_rates,
                ref=cfg.substitution_bases_reverse[0],
                obs=cfg.substitution_bases_reverse[1],
            )
            # add other information
            .pipe(make_position_1_indexed)
            .pipe(make_reverse_position_negative)
            .pipe(replace_nans_with_zeroes)
            .pipe(add_y_sum_counts, cfg=cfg)
            # turns dask dataframe into pandas dataframe
            .compute()
            # .pipe(remove_taxids_with_too_few_y_sum_total, cfg)
            # .pipe(cut_NANs_away)  # remove any taxids containing nans
            .reset_index(drop=True)
            .pipe(sort_by_alignments)
            .reset_index(drop=True)
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

    query = f"(N_alignments >= {cfg.min_alignments}) & (y_sum_total >= {cfg.min_y_sum})"
    return df2.query(query)


def load_dataframe(cfg):

    key = "counts"

    if utils.file_exists(cfg.filename_out):
        logger.info(f"Loading DataFrame from hdf5-file.")

        # exclude = ["max_cores", "force_fits", "num_cores", "N_fits"]
        include = [
            "min_alignments",
            "min_y_sum",
            "substitution_bases_forward",
            "substitution_bases_reverse",
        ]
        if utils.metadata_is_similar(cfg, key, include=include):
            df = utils.load_from_hdf5(filename=cfg.filename_out, key=key)
            cfg.set_number_of_fits(df)
            return df
        else:
            raise AssertionError(f"Different metadata is not yet implemented")

    logger.info(f"Creating DataFrame, please wait.")
    # use_processes = True if utils.is_macbook() else False
    # use_processes = cfg.processes
    df = compute_dataframe_with_dask(cfg, use_processes=True)
    cfg.set_number_of_fits(df)

    logger.info(f"Saving DataFrame to hdf5-file (in data/out/) for faster loading..")
    utils.init_parent_folder(cfg.filename_out)

    utils.save_to_hdf5(filename=cfg.filename_out, key=key, value=df)
    utils.save_metadata_to_hdf5(
        filename=cfg.filename_out,
        key=key,
        value=df,
        metadata=cfg.to_dict(),
    )

    return df

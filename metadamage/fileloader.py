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
from metadamage import utils, io
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
    "tax_id",
    "tax_name",
    "tax_rank",
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


# def delayed_list(lst, length):
#     @dask.delayed(nout=length)
#     def delayed_list_tmp(lst):
#         out = []
#         for l in lst:
#             out.append(l)
#         return out
#     return delayed_list_tmp(lst)


# def extract_top_max_fits_dask(df, max_fits):
#     top_max_fits = (
#         df.groupby("tax_id", observed=True)["N_alignments"]
#         .sum()
#         .nlargest(max_fits)
#         .index
#     )
#     df_top_N = df[df["tax_id"].isin(delayed_list(top_max_fits, max_fits))]
#     return df_top_N


def delayed_list_unknown_length(lst):
    @dask.delayed()
    def delayed_list_tmp(lst):
        out = []
        for l in lst:
            out.append(l)
        return out

    return delayed_list_tmp(lst)


def extract_top_max_fits(df, max_fits):
    top_max_fits = (
        df.groupby("tax_id", observed=True)["N_alignments"]
        .sum()
        .nlargest(max_fits)
        .index
    )
    df_top_N = df[df["tax_id"].isin(top_max_fits)]
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
        by=["N_alignments", "tax_id", "order"], ascending=False
    ).drop(columns=["order"])


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
        index=pd.Index([], name="tax_id", dtype=int),
        dtype=int,
    )

    ds = df.groupby("tax_id").apply(compute_y_sum_total, cfg, meta=meta)
    # ds = df.groupby("tax_id").apply(compute_y_sum_total, cfg)
    ds = ds.reset_index()  # .rename(columns={0: "y_sum_total"})
    # ds = ds.reset_index().rename(columns={0: "y_sum_total"})
    df = dd.merge(df, ds, on=["tax_id"])
    return df


def get_top_max_fits(df, N_fits):
    if N_fits is not None and N_fits > 0:
        return df.pipe(extract_top_max_fits, N_fits)
    else:
        return df


# def remove_tax_ids_with_too_few_alignments(df, cfg):
#     return df.query(f"N_alignments >= {cfg.min_alignments}")


# def remove_tax_ids_with_too_few_y_sum_total(df, cfg):
#     return df.query(f"y_sum_total >= {cfg.min_y_sum}")

from psutil import cpu_count


def filter_cut_based_on_cfg(df, cfg):
    query = f"(N_alignments >= {cfg.min_alignments}) & (y_sum_total >= {cfg.min_y_sum})"
    return df.query(query)


def compute_dataframe_with_dask(cfg, use_processes=True):

    # Standard Library
    filename = cfg.filename

    # if cfg.num_cores == 1:
    # use_processes = False

    # http://localhost:8787/status
    # with Client(
    #     n_workers=cfg.num_cores,
    #     processes=use_processes,
    #     silence_logs=logging.CRITICAL,
    #     local_directory="./dask-worker-space",
    # ):

    # do not allow dask to use all the cores.
    # important to remove bugs on HEP
    n_workers = int(min(cfg.num_cores, cpu_count() * 0.6))
    logger.info(f"Dask: number of workers = {n_workers}.")

    with LocalCluster(
        n_workers=n_workers,
        processes=use_processes,
        silence_logs=logging.CRITICAL,
        # threads_per_worker=1,
    ) as cluster, Client(cluster) as client:

        df = (
            dd.read_csv(
                filename,
                sep="\t",
                header=None,
                names=columns,
            )
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
            .pipe(filter_cut_based_on_cfg, cfg)
            # turns dask dataframe into pandas dataframe
            .compute()
            # .pipe(remove_tax_ids_with_too_few_y_sum_total, cfg)
            # .pipe(cut_NANs_away)  # remove any tax_ids containing nans
            .reset_index(drop=True)
            .pipe(sort_by_alignments)
            .reset_index(drop=True)
        )

    # client.shutdown()
    # cluster.close()
    clean_up_after_dask()

    # df["shortname"] = cfg.shortname

    # np.iinfo("uint32")

    # # fix to make parquet see it as a category
    # df.loc[:, 'tax_id'] = df["tax_id"].astype("uint32")

    # df["df_type"] = "counts"
    categories = ["tax_id", "tax_name", "tax_rank", "strand", "shortname"]
    df2 = utils.downcast_dataframe(df, categories, fully_automatic=False)
    return df2


def load_dataframe(cfg):

    # key = "counts"

    # if utils.file_exists(cfg.filename_out):
    #     logger.info(f"Loading DataFrame from hdf5-file.")

    #     if cfg.ignore_metadata:
    #         logger.info(f"Ignoring metadata and simply loads the file.")
    #         df = utils.load_from_hdf5(filename=cfg.filename_out, key=key)
    #         cfg.set_number_of_fits(df)
    #         return df

    #     # exclude = ["max_cores", "force_fits", "num_cores", "N_fits"]
    #     include = [
    #         "min_alignments",
    #         "min_y_sum",
    #         "substitution_bases_forward",
    #         "substitution_bases_reverse",
    #     ]
    #     if utils.metadata_is_similar(cfg, key, include=include):
    #         df = utils.load_from_hdf5(filename=cfg.filename_out, key=key)
    #         cfg.set_number_of_fits(df)
    #         return df

    #     else:
    #         filename = cfg.filename_out
    #         metadata_file = utils.load_metadata_from_hdf5(filename=filename, key=key)
    #         metadata_cfg = cfg.to_dict()
    #         print("metadata file: ", metadata_file)
    #         print("metadata cfg:  ", metadata_cfg)
    #         raise AssertionError(f"Different metadata is not yet implemented")

    logger.info(f"Creating DataFrame, please wait.")
    df = compute_dataframe_with_dask(cfg, use_processes=True)

    parquet = io.Parquet()
    parquet.save(
        filename=f"./data/out/counts/{cfg.shortname}.parquet",
        df=df,
        metadata=cfg.to_dict(),
    )

    # cfg.set_number_of_fits(df)

    # logger.info(f"Saving DataFrame to hdf5-file (in data/out/) for faster loading..")
    # utils.init_parent_folder(cfg.filename_out)

    # utils.save_to_hdf5(filename=cfg.filename_out, key=key, value=df)
    # utils.save_metadata_to_hdf5(
    #     filename=cfg.filename_out,
    #     key=key,
    #     value=df,
    #     metadata=cfg.to_dict(),
    # )

    return df


#%%


# def save_hdf5_test(df1, df2, df3, df4, cfg):
#     filename = "storage.hdf5"
#     for df in [df1, df2]:
#         shortname = df["shortname"].iloc[0]
#         key = f"counts/{shortname}"
#         IO_HDF5().save(
#             df=df,
#             filename=filename,
#             key=key,
#             metadata=cfg.to_dict(),
#         )
#     for df in [df3, df4]:
#         key = f"fit_results"
#         IO_HDF5().save(
#             df=df,
#             filename=filename,
#             key=key,
#             metadata=cfg.to_dict(),
#         )


# def load_hdf5_test():
#     filename = "storage.hdf5"
#     df, metadata = IO_HDF5().load(filename=filename, key="counts/XXX")
#     df2, metadata2 = IO_HDF5().load(filename=filename, key="fit_results")
#     return df, metadata, df2, metadata2

# shortnames = ["EC-Ext-14-Lib-14-Index1", "KapK-12-1-24-Ext-1-Lib-1-Index2"]
# for shortname in shortnames:
#     df_shortname = df_counts_hdf5.query(f"shortname=='{shortname}'")
#     table = pa.Table.from_pandas(df_shortname)
#     table2 = IO_Parquet()._update_table_metadata(table, metadata=cfg.to_dict())
#     pq.write_table(table2, f"./data/out/counts/{shortname}.parquet", version="2.0")

# pq.read_table("./data/out/counts").to_pandas()
# pq.read_table("./data/out/counts").to_pandas().dtypes
# pq.read_metadata("./data/out/counts/EC-Ext-14-Lib-14-Index1.parquet")



# def save_parquet_test(df1, df2, df3, df4, cfg):
#     metadata = cfg.to_dict()
#     for df in [df1, df2]:
#         filename = "parquet"
#         IO_Parquet().save(
#             filename=filename,
#             df=df,
#             metadata=metadata,
#             partition_cols="shortname",
#         )
#     for df in [df3, df4]:
#         filename = "parquet_34"
#         IO_Parquet().save(filename=filename, df=df, metadata=metadata)


# def load_parquet_test():
#     df, metadata = IO_Parquet().load(filename="parquet", shortname="XXX")
#     df2, metadata2 = IO_Parquet().load(filename="parquet_34")
#     return df, metadata, df2, metadata2


#%%

if False:

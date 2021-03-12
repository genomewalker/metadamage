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

    df["shortname"] = cfg.shortname
    df["df_type"] = "counts"
    categories = ["tax_id", "tax_name", "tax_rank", "strand", "shortname", "df_type"]
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

    IO_HDF5().save(
        df=df,
        filename="./data/out/hdf5_test.hdf5",
        key=f"counts/{cfg.shortname}",
        metadata=cfg.to_dict(),
    )

    IO_Parquet().save(
        filename=f"./data/out/parquet_test/{cfg.shortname}.parquet",
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


import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import json
import warnings
from tqdm.auto import tqdm

from pandas import HDFStore


class IO_HDF5:
    def load(self, filename, key):
        with HDFStore(filename, mode="r") as hdf:
            df = hdf.select(key)
            metadata = hdf.get_storer(key).attrs.metadata
        return df, metadata

    def load_multiple_keys(self, filename, keys):
        all_dfs = []
        with HDFStore(filename, mode="r") as hdf:
            for key in tqdm(keys):
                df_tmp = hdf.select(key)
                all_dfs.append(df_tmp)
            # metadata = hdf.get_storer(key).attrs.metadata
        return all_dfs

    def save(self, df, filename, key, metadata=None):
        utils.init_parent_folder(filename)
        if metadata is None:
            metadata = {}
        with warnings.catch_warnings():
            message = "object name is not a valid Python identifier"
            warnings.filterwarnings("ignore", message=message)
            with HDFStore(filename, mode="a") as hdf:
                hdf.append(key, df, format="table", data_columns=True)
                hdf.get_storer(key).attrs.metadata = metadata

    def get_keys(self, filename):
        with HDFStore(filename, mode="r") as hdf:
            keys = list(set(hdf.keys()))

        # remove meta keys
        keys = sorted([key for key in keys if not "/meta/" in key])
        return keys


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


from pathlib import Path


class IO_Parquet:
    def __init__(self):
        self.custom_meta_key = "metadamage"

    def _read_metadata(self, filename):
        schema = pq.read_schema(filename)
        metadata_json = schema.metadata[self.custom_meta_key.encode()]
        metadata = json.loads(metadata_json)
        return metadata

    def _load_table(self, filename, shortname=None):
        if shortname is None:
            table = pq.read_table(filename)
        else:
            raise AssertionError("shortname not implemted yet for reading")
            # table = pq.read_table(filter=ds.field("shortname") == shortname).to_pandas()
        return table

    def load_single_file(self, filename, shortname=None):
        metadata = self._read_metadata(filename)
        table = self._load_table(filename=filename, shortname=shortname)
        df = table.to_pandas()
        return df, metadata

    def load_dir(self, filename, shortname=None):
        table = self._load_table(filename=filename, shortname=shortname)
        df = table.to_pandas()
        return df

    def load(self, filename, shortname=None):
        if Path(filename).is_dir():
            return self.load_dir(filename=filename, shortname=shortname)
        else:
            return self.load_single_file(filename=filename, shortname=shortname)

    def _add_metadata_to_table(self, table, metadata):
        if metadata is None:
            metadata = {}
        custom_meta_json = json.dumps(metadata)
        updated_metadata = {
            self.custom_meta_key.encode(): custom_meta_json.encode(),
            **table.schema.metadata,
        }
        return table.replace_schema_metadata(updated_metadata)

    def save(self, filename, df, metadata=None):
        utils.init_parent_folder(filename)
        table = pa.Table.from_pandas(df)
        table = self._add_metadata_to_table(table, metadata)
        # pq.write_to_dataset(table, filename, partition_cols=partition_cols)
        pq.write_table(table, filename, version="2.0")


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

    from tqdm.auto import tqdm

    filename_hdf5 = "./data/out/hdf5_test.hdf5"
    keys_hdf5 = IO_HDF5().get_keys(filename_hdf5)

    # all_df = []
    # for key in tqdm(keys_hdf5):
    #     df_tmp, metadata = IO_HDF5().load(
    #         filename=filename_hdf5, key="counts/KapK-12-1-24-Ext-1-Lib-1-Index2"
    #     )
    #     all_df.append(df_tmp)

    all_df = IO_HDF5().load_multiple_keys(filename=filename_hdf5, keys=keys_hdf5)
    df_counts1 = pd.concat(all_df, axis="index", ignore_index=True)

    def concatenate(dfs, **kwargs):
        """Concatenate while preserving categorical columns.

        NB: We change the categories in-place for the input dataframes"""
        from pandas.api.types import union_categoricals
        import pandas as pd

        # Iterate on categorical columns common to all dfs
        for col in set.intersection(
            *[set(df.select_dtypes(include="category").columns) for df in dfs]
        ):
            # Generate the union category across dfs for this column
            uc = union_categoricals([df[col] for df in dfs])
            # Change to union category for all dataframes
            for df in dfs:
                df[col] = pd.Categorical(df[col].values, categories=uc.categories)
        return pd.concat(dfs, **kwargs)

    df_counts2 = concatenate(all_df, axis="index", ignore_index=True)

    df_counts1.memory_usage(deep=True) / 1e6
    df_counts1.memory_usage(deep=True).sum() / 1e6
    df_counts2.memory_usage(deep=True) / 1e6
    df_counts2.memory_usage(deep=True).sum() / 1e6

    IO_HDF5().save(
        df=df_counts2,
        filename=filename_hdf5,
        key="counts_combined",
    )

    df_counts_hdf5, metadata_hdf5 = IO_HDF5().load(
        filename=filename_hdf5, key="counts_combined"
    )

    shortnames = ["EC-Ext-14-Lib-14-Index1", "KapK-12-1-24-Ext-1-Lib-1-Index2"]
    for shortname in shortnames:
        df_shortname = df_counts_hdf5.query(f"shortname=='{shortname}'")
        filename = f"./data/out/counts/{shortname}.parquet"
        IO_Parquet().save(filename=filename, df=df_shortname, metadata=cfg.to_dict())

    IO_Parquet().load("./data/out/counts").dtypes

    df_parquet, meta_parquet = IO_Parquet().load(
        "./data/out/counts/EC-Ext-14-Lib-14-Index1.parquet"
    )
    df_parquet.dtypes
    meta_parquet

    # %timeit IO_HDF5().load(filename=filename_hdf5, key="counts_combined")

    # df3 = pd.read_parquet(
    #     path="./data/out/counts",
    #     engine="pyarrow",
    #     # columns=['shortname'],
    #     filters=[("shortname", "=", "KapK-12-1-24-Ext-1-Lib-1-Index2"), ("tax_id", "=", "1")],
    # )
    # df3

    pd.read_parquet(
        path="./data/out/counts/KapK-12-1-24-Ext-1-Lib-1-Index2.parquet"
    ).dtypes

    filename_parquet = "./data/out/parquet_test.parquet"
    df_counts, metadata = IO_Parquet().load(filename_parquet, shortname=None)

    df3 = pd.read_parquet(
        path="./data/out/parquet_test.parquet/",
        engine="pyarrow",
        # columns=['shortname'],
        # filters=[("shortname", "=", "XXX"), ("tax_id", "=", "1")],
    )

    pd.read_parquet(
        path="./data/out/parquet_test.parquet/shortname=KapK-12-1-24-Ext-1-Lib-1-Index2/207fe0f890e94c61b2602bd673b37d55.parquet"
    ).dtypes

    IO_Parquet().save(
        filename="./data/out/test",
        df=df_counts_hdf5,
        metadata=cfg.to_dict(),
        # partition_cols="shortname",
    )

    def f_test(x):
        print(x)
        return "-".join(x) + "1.parquet"

    table = pa.Table.from_pandas(df_counts_hdf5)
    table2 = IO_Parquet()._update_table_metadata(table, metadata=cfg.to_dict())
    pq.write_to_dataset(
        table2,
        root_path="./data/out/test",
        partition_cols=["shortname"],
        partition_filename_cb=f_test,
    )

    pq.write_table(table2, "example.parquet", version="2.0")

    # https://issues.apache.org/jira/browse/ARROW-6114

    pd.read_parquet(path="./data/out/test").dtypes

    pd.read_parquet(path="example.parquet").dtypes


# if False:
#     #     pass

#     # else:

#     df1 = df  # .iloc[:10]

#     df2 = df1.copy(deep=True)
#     df2["shortname"] = "XXX"
#     categories = ["tax_id", "tax_name", "tax_rank", "strand", "shortname", "df_type"]
#     df2 = utils.downcast_dataframe(df2, categories)

#     df3 = pd.DataFrame.from_dict(
#         {
#             "a": range(2),
#             "b": np.random.randn(2),
#             "c": ["a", "b"],
#             "shortname": ["KapK-198A-Ext-55-Lib-55-Index1", "XXX"],
#             "df_type": ["fit_results", "fit_results"],
#         }
#     )

#     df4 = pd.DataFrame.from_dict(
#         {
#             "a": [2, 3],
#             "b": np.random.randn(2),
#             "c": ["c", "d"],
#             "shortname": ["KapK-198A-Ext-55-Lib-55-Index1", "XXX"],
#             "df_type": ["fit_results", "fit_results"],
#         }
#     )

#     # %time save_hdf5_test(df1, df2, df3, df4, cfg)
#     # %timeit df_hdf5, metadata_hdf5, df2_hdf5, metadata2_hdf5 = load_hdf5_test()
#     save_hdf5_test(df1, df2, df3, df4, cfg)
#     df_hdf5, metadata_hdf5, df2_hdf5, metadata2_hdf5 = load_hdf5_test()

#     # %time save_parquet_test(df1, df2, df3, df4, cfg)
#     # %timeit df_parquet, metadata_parquet, df2_parquet, metadata2_parquet = load_parquet_test()
#     save_parquet_test(df1, df2, df3, df4, cfg)
#     df_parquet, metadata_parquet, df2_parquet, metadata2_parquet = load_parquet_test()

#     # df1.to_parquet(
#     #     path="analytics",
#     #     engine="pyarrow",
#     #     compression="snappy",
#     #     partition_cols=["shortname"],
#     # )

#     # df2.to_parquet(
#     #     path="analytics",
#     #     engine="pyarrow",
#     #     compression="snappy",
#     #     partition_cols=["shortname"],
#     # )

#     # # df3 = pd.read_parquet(
#     # #     path="analytics",
#     # #     engine="pyarrow",
#     # #     # columns=['shortname'],
#     # #     filters=[("shortname", "=", "XXX"), ("tax_id", "=", "1")],
#     # # )

#     # # pd.read_parquet(
#     # #     path="analytics",
#     # #     engine="pyarrow",
#     # #     # columns=['shortname'],
#     # #     # filters=[('shortname', '=', 'XXX'), ('tax_id', '=', '1')]
#     # # )

#     # # index not important for counts
#     # table1 = pa.Table.from_pandas(df, preserve_index=False)
#     # table2 = pa.Table.from_pandas(df2, preserve_index=False)

#     # # pq.write_table(table, "example.parquet", version="2.0")

#     # # # Local dataset write
#     # # pq.write_to_dataset(table, root_path="dataset_name", partition_cols=["shortname"])

#     # # table3 = pq.read_table("dataset_name")
#     # # table3.to_pandas()

#     # #%%

#     # # table = table3

#     # # Path("parquet_dataset").mkdir(exist_ok=True)
#     # # pq.write_table(table, "parquet_dataset/data1.parquet")
#     # # pq.write_table(table2, "parquet_dataset/data2.parquet")

#     # # dataset = ds.dataset("parquet_dataset", format="parquet")
#     # # dataset.files

#     # # print(dataset.schema.to_string(show_field_metadata=False))

#     # # dataset.to_table().to_pandas()

#     # # dataset.to_table(columns=["tax_name", "shortname"]).to_pandas()

#     # # dataset.to_table(filter=ds.field("tax_id") == 1).to_pandas()
#     # # ds.field("a") != 3
#     # # ds.field("a").isin([1, 2, 3])

#     # for table in [table1, table2]:

#     #     pq.write_to_dataset(
#     #         table,
#     #         "parquet_dataset_partitioned",
#     #         partition_cols=["df_type", "shortname"],
#     #     )

#     # dataset = ds.dataset(
#     #     "parquet_dataset_partitioned",
#     #     format="parquet",
#     #     partitioning="hive",  # important to retreave the shortname column
#     # )
#     # dataset.files
#     # dataset.to_table().to_pandas()
#     # dataset.to_table(filter=ds.field("shortname") == "XXX").to_pandas()

#     # table_different = pa.table(
#     #     {
#     #         "a": range(2),
#     #         "b": np.random.randn(2),
#     #         "c": ["a", "b"],
#     #         "shortname": ["KapK-198A-Ext-55-Lib-55-Index1", "XXX"],
#     #         "df_type": ["fit_results", "fit_results"],
#     #     }
#     # )
#     # table_different.to_pandas()

#     # pq.write_to_dataset(
#     #     table_different,
#     #     "parquet_dataset_partitioned",
#     #     partition_cols=["df_type", "shortname"],
#     # )

#     # dataset = ds.dataset(
#     #     "parquet_dataset_partitioned",
#     #     format="parquet",
#     #     partitioning="hive",  # important to retreave the shortname column
#     # )
#     # dataset.files
#     # dataset.to_table(filter=ds.field("df_type") == "counts").to_pandas()
#     # dataset.to_table(filter=ds.field("df_type") == "fit_results").to_pandas()

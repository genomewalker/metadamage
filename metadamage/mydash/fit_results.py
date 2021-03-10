# Scientific Library
import numpy as np
import pandas as pd

# Standard Library
from functools import partial
from pathlib import Path

# Third Party
import dill
from joblib import Memory
import plotly.express as px

# First Party
from metadamage import mydash, utils


cachedir = "memoization"
memory = Memory(cachedir, verbose=0)

# @memory.cache

#%%


@memory.cache
def load_dataframes_memoization(files):
    df_counts = _load_df_counts(files)
    df_fit_results = _load_df_fit_results(files)
    df_fit_predictions = _load_df_fit_predictions(files)
    return df_counts, df_fit_results, df_fit_predictions


def _load_single_dataframe(files, key, ignore_index):
    df_list = []
    for filename in files:
        cfg = utils.load_metadata_from_hdf5(filename, key="counts")
        name = cfg["name"]
        df = utils.load_from_hdf5(filename, key)
        df["filename"] = name
        df_list.append(df)

    df = pd.concat(df_list, axis=0, ignore_index=ignore_index)
    categories = ["taxid", "name", "rank", "strand", "filename"]
    df = utils.downcast_dataframe(df, categories)
    return df


def _load_df_counts(files):
    df_counts = _load_single_dataframe(
        files=files,
        key="counts",
        ignore_index=True,
    )
    return df_counts


def _load_df_fit_results(files):
    df = _load_single_dataframe(
        files=files,
        key="fit_results",
        ignore_index=False,
    )
    df["taxid"] = df.index
    df = df.reset_index(drop=True)

    df["N_alignments_log10"] = np.log10(df["N_alignments"])
    df["N_alignments_sqrt"] = np.sqrt(df["N_alignments"])
    with np.errstate(divide="ignore", invalid="ignore"):
        df["N_sum_total_log10"] = np.log10(df["N_sum_total"])

    df_fit_results = df
    return df_fit_results


def _load_df_fit_predictions(files):
    df_fit_predictions = _load_single_dataframe(
        files=files,
        key="fit_predictions",
        ignore_index=True,
    )
    return df_fit_predictions


def find_good_files(folder):
    filenames = list(Path(folder).rglob("*.hdf5"))

    good_files = []
    for filename in filenames:
        keys = utils.get_hdf5_keys(filename, ignore_subgroups=True)
        if "counts" in keys and "fit_results" in keys and "fit_predictions" in keys:
            good_files.append(filename)
        else:
            print(filename, keys)
    return good_files


#%%


class FitResults:
    def __init__(self, folder, verbose=True):
        self.verbose = verbose
        self._load_dataframes(folder)
        # self._load_df_counts(verbose)
        # self._load_df_fit_results(verbose)
        # self._load_df_fit_predictions(verbose)
        self._compute_ranges()
        self._set_cmap()
        self._set_hover_info()
        self._set_dimensions()
        self._set_labels()
        self._set_dimensions_forward_reverse()

    #%%

    def _load_dataframes(self, folder):
        self.folder = folder
        self.filenames_hdf5 = find_good_files(folder)

        if len(self.filenames_hdf5) == 0:
            raise AssertionError(f"No output files found in {self.folder}.")

        # unique_id = "".join([str(file) for file in self.filenames_hdf5])
        filenames_hdf5 = tuple(str(file) for file in self.filenames_hdf5)

        dataframes = load_dataframes_memoization(filenames_hdf5)
        df_counts, df_fit_results, df_fit_predictions = dataframes

        self.df_counts = df_counts
        self.df_fit_results = df_fit_results
        self.df_fit_predictions = df_fit_predictions

        self.filenames = list(self.df_counts.filename.unique())
        self.columns = list(self.df_fit_results.columns)
        self.set_marker_size(marker_transformation="sqrt")

    #%%

    def _get_range_of_column(self, column, spacing):
        array = self.df_fit_results[column]
        array = array[np.isfinite(array) & array.notnull()]
        range_min = array.min()
        range_max = array.max()
        delta = range_max - range_min
        ranges = [range_min - delta / spacing, range_max + delta / spacing]
        return ranges

    def _compute_ranges(self, spacing=20):
        ranges = {}
        for column in self.columns:
            try:
                ranges[column] = self._get_range_of_column(column, spacing=spacing)
            except TypeError:
                pass
        self.ranges = ranges

    def set_marker_size(self, marker_transformation="sqrt", marker_size_max=30):

        df = self.df_fit_results

        if isinstance(marker_transformation, list) and isinstance(
            marker_size_max, list
        ):
            if len(marker_transformation) == 0 and len(marker_size_max) == 0:
                return None
            marker_transformation = marker_transformation[0]
            marker_size_max = marker_size_max[0]

        if marker_transformation == "identity":
            df.loc[:, "size"] = df["N_alignments"]

        elif marker_transformation == "sqrt":
            df.loc[:, "size"] = np.sqrt(df["N_alignments"])

        elif marker_transformation == "log10":
            df.loc[:, "size"] = np.log10(df["N_alignments"])

        elif marker_transformation == "constant":
            df.loc[:, "size"] = np.ones_like(df["N_alignments"])

        else:
            raise AssertionError(
                f"Did not recieve proper marker_transformation: {marker_transformation}"
            )

        self.max_of_size = np.max(df["size"])
        self.marker_size_max = marker_size_max
        return None

    def filter(self, filters, df="df_fit_results"):
        query = ""
        for dimension, filter in filters.items():

            if filter is None:
                continue

            elif dimension == "filenames":
                query += f"(filename in {filter}) & "

            elif dimension == "filename":
                query += f"(filename == '{filter}') & "

            elif dimension == "taxid":
                query += f"(taxid == {filter}) & "

            elif dimension == "taxids":
                query += f"(taxid in {filter}) & "

            else:
                low, high = filter
                if dimension in ["N_alignments", "y_sum_total", "N_sum_total"]:
                    low = mydash.utils.log_transform_slider(low)
                    high = mydash.utils.log_transform_slider(high)
                query += f"({low} <= {dimension} <= {high}) & "

        query = query[:-2]
        # print(query)

        if "fit_results" in df:
            return self.df_fit_results.query(query)
        elif "counts" in df:
            return self.df_counts.query(query)
        else:
            raise AssertionError(
                f"df = {df} not implemented yet, "
                "only 'df_fit_results' and 'df_counts"
            )

    # def get_mismatch_group(self, filename, taxid):
    def get_single_count_group(self, filename, taxid):
        return self.df_counts.query(f"filename == '{filename}' & taxid == {taxid}")

    def get_single_fit_prediction(self, filename, taxid):
        return self.df_fit_predictions.query(
            f"filename == '{filename}' & taxid == {taxid}"
        )

    def _set_cmap(self):
        # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
        cmap = px.colors.qualitative.D3 * 10

        d_cmap = {}
        for i, (name, _) in enumerate(
            self.df_fit_results.groupby("filename", sort=False)
        ):
            d_cmap[name] = cmap[i]

        self.cmap = cmap
        self.d_cmap = d_cmap

    def _set_hover_info(self):

        self.custom_data_columns = [
            "filename",
            "tax_name",
            "tax_rank",
            "taxid",
            "n_sigma",
            "D_max",
            "q_mean",
            "concentration_mean",
            "asymmetry",
            "normalized_noise",
            "N_alignments",
            "N_sum_total",
            "y_sum_total",
        ]

        self.hovertemplate = (
            "<b>%{customdata[0]}</b><br><br>"
            "<b>Tax</b>: <br>"
            "    Name: %{customdata[1]} <br>"
            "    Rank: %{customdata[2]} <br>"
            "    ID:   %{customdata[3]} <br><br>"
            "<b>Fit Results</b>: <br>"
            "    n sigma:  %{customdata[4]:9.2f} <br>"
            "    D max:    %{customdata[5]:9.2f} <br>"
            "    q:        %{customdata[6]:9.2f} <br>"
            "    phi:      %{customdata[7]:9.3s} <br>"
            "    asymmetry:%{customdata[8]:9.2f} <br>"
            "    noise:    %{customdata[9]:9.2f} <br><br>"
            "<b>Counts</b>: <br>"
            "    N alignments:%{customdata[10]:6.3s} <br>"
            "    N sum total: %{customdata[11]:6.3s} <br>"
            "    y sum total: %{customdata[12]:6.3s} <br>"
            "<extra></extra>"
        )

        # if "y_sum_total" in self.columns:
        #     self.has_y_sum_total = True
        # else:
        #     self.custom_data_columns.remove("y_sum_total")
        #     self.hovertemplate = self.hovertemplate.replace(
        #         "    y sum total: %{customdata[12]:6.3s} <br>", ""
        #     )
        #     self.has_y_sum_total = False

        self.customdata = self.df_fit_results[self.custom_data_columns]

    def _set_dimensions(self):
        self.dimensions = [
            "n_sigma",
            "D_max",
            "N_alignments_log10",
            "N_sum_total_log10",
            "q_mean",
            "concentration_mean",
            "asymmetry",
            "normalized_noise",
        ]

    def _set_labels(self):

        labels_list = [
            r"$\large n_\sigma$",
            r"$\large D_\mathrm{max}$",
            r"$\large \log_{10} N_\mathrm{alignments}$",
            r"$\large \log_{10} N_\mathrm{sum}$",
            r"$\large \bar{q}$",
            r"$\large \bar{\phi}$",
            r"$\large \alpha$",
            r"$\large \mathrm{noise}$",
        ]

        iterator = zip(self.dimensions, labels_list)
        self.labels = {dimension: label for dimension, label in iterator}

    def iterate_over_dimensions(self):
        row = 1
        column = 1
        for dimension in self.dimensions:
            yield dimension, row, column
            if column >= 4:
                row += 1
                column = 1
            else:
                column += 1

    def _set_dimensions_forward_reverse(self):
        self.dimensions_forward_reverse = {
            "n_sigma": r"$\large n_\sigma$",
            "D_max": r"$\large D_\mathrm{max}$",
            "N_z1": r"$\large N_{z=1}$",
            "N_sum": r"$\large N_\mathrm{sum}$",
            "y_sum": r"$\large y_\mathrm{sum}$",
            "normalized_noise": r"$\large \mathrm{noise}$",
        }
        # if not self.has_y_sum_total:
        #     self.dimensions_forward_reverse.pop("y_sum")

    def iterate_over_dimensions_forward_reverse(self, N_cols):
        showlegend = True
        for i, dimension in enumerate(self.dimensions_forward_reverse.keys()):
            column = i % N_cols
            row = (i - column) // N_cols
            column += 1
            row += 1

            forward = f"{dimension}_forward"
            reverse = f"{dimension}_reverse"

            yield dimension, row, column, showlegend, forward, reverse
            showlegend = False

    def parse_click_data(self, click_data, variable):
        try:
            index = self.custom_data_columns.index(variable)
            value = click_data["points"][0]["customdata"][index]
            return value

        except Exception as e:
            raise e

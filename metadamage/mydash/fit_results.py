# Scientific Library
import numpy as np
import pandas as pd

# Standard Library
from pathlib import Path

# Third Party
import plotly.express as px
import dill

# First Party
from metadamage import mydash, utils


class FitResults:
    def __init__(self):
        self._load_df_fit_results()
        self._load_fit_predictions()
        self._load_df_mismatch_counts()
        self._compute_ranges()
        self._set_cmap()
        self._set_hover_info()
        self._set_dimensions()
        self._set_labels()
        self._set_names()
        self._set_dimensions_forward_reverse()

    def _load_df_fit_results(self):

        input_folder = "./data/input/"
        fit_results_folder = "./data/fits/"
        input_files = list(Path(input_folder).rglob("*.txt"))

        if len(input_files) == 0:
            raise AssertionError(f"No csv files (fit results) found in {input_folder}.")

        dfs = []
        names = []
        for file in input_files:

            name = utils.extract_name(file)
            fit_file = list(Path(fit_results_folder).rglob(f"{name}*.csv"))

            if len(fit_file) == 0:
                continue

            elif len(fit_file) > 1:
                print("Got more than 1 fit file for df_fit_results.", fit_file)
                print(f"Choosing {fit_file[0]}")
                fit_file = fit_file[0]

            elif len(fit_file) == 1:
                fit_file = fit_file[0]

            df = pd.read_csv(fit_file)
            cols = list(df.columns)
            cols[0] = "taxid"
            df.columns = cols
            # name = utils.extract_name(file)
            df["name"] = name
            dfs.append(df)
            names.append(name)

        df = pd.concat(dfs, axis=0, ignore_index=True)
        df["N_alignments_log10"] = np.log10(df["N_alignments"])
        df["N_alignments_sqrt"] = np.sqrt(df["N_alignments"])
        df["N_sum_total_log10"] = np.log10(df["N_sum_total"])
        self.df_fit_results = df
        self.columns = list(self.df_fit_results.columns)
        self.set_marker_size(marker_transformation="sqrt")
        self.names = names

    def _load_fit_predictions(self):

        input_folder = "./data/input/"
        fit_results_folder = "./data/fits/"
        input_files = list(Path(input_folder).rglob("*.txt"))

        if len(input_files) == 0:
            raise AssertionError(
                f"No Parquet files (fit results) found in {input_folder}."
            )

        d_fits_median = {}
        d_fits_hpdi = {}

        for file in input_files:

            name = utils.extract_name(file)
            fit_file = list(Path(fit_results_folder).rglob(f"{name}*.dill"))

            if len(fit_file) == 0:
                continue

            elif len(fit_file) > 1:
                print("Got more than 1 fit file for fit prediction.", fit_file)
                print(f"Choosing {fit_file[0]}")
                fit_file = fit_file[0]

            elif len(fit_file) == 1:
                fit_file = fit_file[0]

            d_fits_taxid = utils.load_dill(fit_file)[0]
            d_fits_median_taxid = {
                taxid: d_fits_taxid[taxid]["median"] for taxid in d_fits_taxid.keys()
            }
            d_fits_hpdi_taxid = {
                taxid: d_fits_taxid[taxid]["hpdi"] for taxid in d_fits_taxid.keys()
            }

            d_fits_median[name] = d_fits_median_taxid
            d_fits_hpdi[name] = d_fits_hpdi_taxid

        self.d_fits_median = d_fits_median
        self.d_fits_hpdi = d_fits_hpdi

    def _load_df_mismatch_counts(self):

        # input_folder = "./data/parquet"
        # input_files = list(Path("").rglob(f"{input_folder}/*.parquet"))

        input_folder = "./data/input/"
        parquet_folder = "./data/parquet/"
        input_files = list(Path(input_folder).rglob("*.txt"))

        if len(input_files) == 0:
            raise AssertionError(
                f"No Parquet files (fit results) found in {input_folder}."
            )

        dfs = []
        for file in input_files:

            name = utils.extract_name(file)
            fit_file = list(Path(parquet_folder).rglob(f"{name}*.parquet"))

            if len(fit_file) == 0:
                continue

            elif len(fit_file) > 1:
                print("Got more than 1 fit file for parquet.", fit_file)
                print(f"Choosing {fit_file[0]}")
                fit_file = fit_file[0]

            elif len(fit_file) == 1:
                fit_file = fit_file[0]

            df = pd.read_parquet(fit_file)
            df["name"] = name
            dfs.append(df)

        df = pd.concat(dfs, axis=0, ignore_index=True)
        self.df_mismatch = df

    def _get_range_of_column(self, column, spacing):
        range_min = self.df_fit_results[column].min()
        range_max = self.df_fit_results[column].max()
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

    def _set_names(self):
        self.names = list(self.df_fit_results.name.unique())

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

            elif dimension == "names":
                query += f"(name in {filter}) & "

            elif dimension == "name":
                query += f"(name == '{filter}') & "

            elif dimension == "taxid":
                query += f"(taxid == {filter}) & "

            else:
                low, high = filter
                if dimension == "N_alignments":
                    low = mydash.utils.transform_slider(low)
                    high = mydash.utils.transform_slider(high)
                query += f"({low} <= {dimension} <= {high}) & "

        query = query[:-2]
        # print(query)

        if df == "df_fit_results":
            return self.df_fit_results.query(query)
        elif df == "df_mismatch":
            return self.df_mismatch.query(query)
        else:
            raise AssertionError(
                f"df = {df} not implemented yet, "
                "only 'df_fit_results' and 'df_mismatch"
            )

    def get_mismatch_group(self, name, taxid):
        return self.df_mismatch.query(f"name == '{name}' & taxid == {taxid}")

    def get_fit_predictions(self, name, taxid):
        return {
            "median": self.d_fits_median[name][taxid],
            "hdpi": self.d_fits_hpdi[name][taxid],
        }

    def _set_cmap(self):
        # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
        cmap = px.colors.qualitative.D3

        d_cmap = {}
        for i, (name, _) in enumerate(self.df_fit_results.groupby("name", sort=False)):
            d_cmap[name] = cmap[i]

        self.cmap = cmap
        self.d_cmap = d_cmap

    def _set_hover_info(self):

        self.custom_data_columns = [
            "name",
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

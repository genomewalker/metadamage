# Scientific Library
import numpy as np
import pandas as pd
from pathlib import Path

import plotly.express as px

from metadamage import utils, dash_utils


class FitResults:
    def __init__(self):
        self._load_df_fit_results()
        self._compute_ranges()
        self._set_cmap()
        self._set_hover_info()
        self._set_dimensions()
        self._set_labels()
        self._set_names()
        self._set_dimensions_forward_reverse()

    def _load_df_fit_results(self):

        input_folder = "./data/fits"
        input_files = list(Path("").rglob(f"{input_folder}/*.csv"))

        if len(input_files) == 0:
            raise AssertionError(f"No csv files (fit results) found in {input_folder}.")

        dfs = []
        for file in input_files:
            df = pd.read_csv(file)
            cols = list(df.columns)
            cols[0] = "taxid"
            df.columns = cols
            name = utils.extract_name(file, max_length=20)
            df["name"] = name
            dfs.append(df)

        df = pd.concat(dfs, axis=0, ignore_index=True)
        df["N_alignments_log10"] = np.log10(df["N_alignments"])
        df["N_alignments_sqrt"] = np.sqrt(df["N_alignments"])
        # df["N_alignments_str"] = df.apply(
        # lambda row: utils.human_format(row["N_alignments"]), axis=1
        # )
        df["N_sum_total_log10"] = np.log10(df["N_sum_total"])
        # df["N_sum_total_str"] = df.apply(
        # lambda row: utils.human_format(row["N_sum_total"]), axis=1
        # )
        self.df = df
        self.columns = list(self.df.columns)

    def _get_range_of_column(self, column, spacing):
        range_min = self.df[column].min()
        range_max = self.df[column].max()
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
        self.names = list(self.df.name.unique())

    def filter(self, filters):
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
                    low = dash_utils.transform_slider(low)
                    high = dash_utils.transform_slider(high)
                query += f"({low} <= {dimension} <= {high}) & "

        query = query[:-2]
        # print(query)
        return self.df.query(query)

    def _set_cmap(self):
        # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
        cmap = px.colors.qualitative.D3

        d_cmap = {}
        for i, (name, _) in enumerate(self.df.groupby("name", sort=False)):
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
            "<extra></extra>"
        )

        self.customdata = self.df[self.custom_data_columns]

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

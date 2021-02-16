import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Scientific Library
import numpy as np
import pandas as pd
from pathlib import Path

# import plotly as py
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from metadamage import utils

#%%

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


#%%


class FitResults:
    def __init__(self):
        self.df = self.load_df_fit_results()
        self.columns = list(self.df.columns)
        self.ranges = self.compute_ranges()
        self._set_cmap()
        self._set_hover_info()
        self._set_dimensions()
        self._set_labels()

    def load_df_fit_results(self):

        input_files = list(Path("").rglob("./data/fits/*.csv"))

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
        df["N_alignments_str"] = df.apply(
            lambda row: utils.human_format(row["N_alignments"]), axis=1
        )
        df["N_sum_total_log10"] = np.log10(df["N_sum_total"])
        df["N_sum_total_str"] = df.apply(
            lambda row: utils.human_format(row["N_sum_total"]), axis=1
        )

        return df

    def _get_range_of_column(self, column, spacing=20):
        range_min = self.df[column].min()
        range_max = self.df[column].max()
        delta = range_max - range_min
        ranges = [range_min - delta / spacing, range_max + delta / spacing]
        return ranges

    def compute_ranges(self, spacing=40):
        ranges = {}
        for column in self.columns:
            try:
                ranges[column] = self._get_range_of_column(column, spacing=spacing)
            except TypeError:
                pass
        return ranges

    def cut_N_alignments(self, slider_N_alignments):
        low, high = slider_N_alignments
        low = transform_slider(low)
        high = transform_slider(high)
        return self.df.query(f"{low} <= N_alignments <= {high}")

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
            "taxid",
            "n_sigma",
            "D_max",
            "N_alignments_str",
            "N_sum_total_str",
        ]

        self.hovertemplate = (
            "<b>%{customdata[0]}</b><br><br>"
            "taxid: %{customdata[1]}<br>"
            "<br>n sigma: %{customdata[2]:5.2f}<br>"
            "D max:    %{customdata[3]:.2f}<br>"
            "<br>N alignments: %{customdata[4]}<br>"
            "N sum total:   %{customdata[5]}<br>"
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

    def iterate_over_rows_and_columns(self):
        row = 1
        column = 1
        for dimension in self.dimensions:
            yield dimension, row, column
            if column >= 4:
                row += 1
                column = 1
            else:
                column += 1


fit_results = FitResults()


#%%


def create_fit_results_figure(df_cutted):

    fig = px.scatter(
        df_cutted,
        x="n_sigma",
        y="D_max",
        size="N_alignments_sqrt",
        color="name",
        hover_name="name",
        size_max=30,
        opacity=0.2,
        color_discrete_sequence=fit_results.cmap,
        custom_data=fit_results.custom_data_columns,
        range_x=fit_results.ranges["n_sigma"],
        range_y=[0, 1],
    )

    fig.update_traces(hovertemplate=fit_results.hovertemplate, marker_line_width=0)

    fig.update_layout(
        title=f"Fit Results",
        xaxis_title=r"$\Large n_\sigma$",
        yaxis_title=r"$\Large D_\mathrm{max}$",
        font_size=16,
        legend=dict(title="Files", title_font_size=20, font_size=16),
        width=1600,
        height=800,
    )

    return fig


#%%


def create_scatter_matrix_figure(df_cutted):

    fig = px.scatter_matrix(
        df_cutted,
        dimensions=fit_results.dimensions,
        color="name",
        hover_name="name",
        size_max=10,
        color_discrete_sequence=fit_results.cmap,
        labels=fit_results.labels,
        opacity=0.1,
        custom_data=fit_results.custom_data_columns,
    )

    fig.update_traces(
        diagonal_visible=False,
        showupperhalf=False,
        hovertemplate=fit_results.hovertemplate,
    )

    # manually set ranges for scatter matrix
    iterator = enumerate(fit_results.dimensions)
    ranges = {i + 1: fit_results.ranges[col] for i, col in iterator}
    for axis in ["xaxis", "yaxis"]:
        fig.update_layout({axis + str(k): {"range": v} for k, v in ranges.items()})

    fig.update_layout(
        title="Scatter Matrix",
        font_size=12,
        title_font_size=40,
        legend=dict(title="Files", title_font_size=20, font_size=16),
        width=1000 * 1.1,
        height=800 * 1.1,
    )

    return fig


#%%


def plotly_histogram(
    data,
    name,
    dimension,
    bins=50,
    density=True,
    range=None,
    showlegend=True,
):
    data = data[~np.isnan(data)]
    binned = np.histogram(data, bins=bins, density=density, range=range)
    trace = go.Scatter(
        x=binned[1],
        y=binned[0],
        mode="lines",
        name=name,
        legendgroup=name,
        line_shape="hv",  # similar to 'mid' in matplotlib,
        showlegend=showlegend,
        marker_color=fit_results.d_cmap[name],
        hovertemplate=(
            "<b>" + f"{name}" + "</b><br><br>"
            f"{dimension}" + ": %{x:5.2f}<br>"
            "Counts: %{y}<br>"
            "<extra></extra>"
        ),
    )

    return trace, np.max(binned[0])


def plot_histograms(df_cutted):

    fig = make_subplots(rows=2, cols=4)

    showlegend = True

    for dimension, row, column in fit_results.iterate_over_rows_and_columns():
        highest_y_max = 0
        for name, group in df_cutted.groupby("name", sort=False):
            trace, y_max = plotly_histogram(
                group[dimension],
                name,
                dimension,
                bins=50,
                density=False,
                range=fit_results.ranges[dimension],
                showlegend=showlegend,
            )
            fig.add_trace(trace, col=column, row=row)
            if y_max > highest_y_max:
                highest_y_max = y_max

        showlegend = False
        fig.update_xaxes(title_text=fit_results.labels[dimension], row=row, col=column)
        fig.update_yaxes(range=(0, highest_y_max * 1.1), row=row, col=column)
        if column == 1:
            fig.update_yaxes(title_text="Counts", row=row, col=column)

    # Update title and width, height
    fig.update_layout(
        height=800,
        width=1600,
        title=dict(text="1D Histograms", font_size=30),
        legend=dict(
            title_text="Files", title_font_size=20, font_size=16, tracegroupgap=2
        ),
    )

    return fig


#%%


def get_range_of_dataframe_column_slider(df, column="N_alignments"):

    N_alignments_min = float(df[column].min())
    N_alignments_max = float(df[column].max())
    N_steps = 1000
    step = (N_alignments_max - N_alignments_min) / N_steps

    slider = dcc.RangeSlider(
        id="range-slider",
        min=N_alignments_min,
        max=N_alignments_max,
        step=step,
        marks={
            N_alignments_min: str(N_alignments_min),
            N_alignments_max: str(N_alignments_max),
        },
        value=[N_alignments_min, N_alignments_max],
        allowCross=False,
        updatemode="mouseup",
        included=True,
    )
    return slider


def transform_slider(x):
    return 10 ** x


def get_log_range_slider(df, column="N_alignments"):

    N_alignments_log = np.log10(df[column])

    N_alignments_min = np.floor(N_alignments_log.min())
    N_alignments_max = np.ceil(N_alignments_log.max())
    N_steps = 100
    step = (N_alignments_max - N_alignments_min) / N_steps

    marks_steps = np.arange(N_alignments_min, N_alignments_max + 1)
    marks = {int(i): f"{utils.human_format(transform_slider(i))}" for i in marks_steps}
    marks[marks_steps[0]] = "No Minimum"
    marks[marks_steps[-1]] = "No Maximum"

    slider = dcc.RangeSlider(
        id="log-range-slider",
        min=N_alignments_min,
        max=N_alignments_max,
        step=step,
        marks=marks,
        value=[N_alignments_min, N_alignments_max],
        allowCross=False,
        updatemode="mouseup",
        included=True,
    )
    return slider


def is_ipython():
    return hasattr(__builtins__, "__IPYTHON__")


#%%

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/"
        "MathJax.js?config=TeX-MML-AM_CHTML",
    ],
)

app.layout = html.Div(
    [
        html.H1("Fit Results"),
        dcc.Graph(id="scatter-plot"),
        html.Div(id="log-range-slider-output"),
        html.Br(),
        get_log_range_slider(fit_results.df, column="N_alignments"),
    ],
    style={
        "width": "80%",
        "marginLeft": 20,
        "marginRight": 20,
    },
)


@app.callback(
    Output("log-range-slider-output", "children"),
    [Input("log-range-slider", "value")],
)
def update_output(slider_range):
    low, high = slider_range
    low = transform_slider(low)
    high = transform_slider(high)
    return f"Number of alignments interval: [{utils.human_format(low)}, {utils.human_format(high)}]"


#%%


@app.callback(
    Output("scatter-plot", "figure"),
    [Input("log-range-slider", "value")],
)
def update_figure(slider_N_alignments):

    df_cutted = fit_results.cut_N_alignments(slider_N_alignments)

    fig = create_fit_results_figure(df_cutted)
    # fig = create_scatter_matrix_figure(df_cutted)
    # fig = plot_histograms(df_cutted)

    return fig


if __name__ == "__main__" and not is_ipython():
    app.run_server(debug=True)

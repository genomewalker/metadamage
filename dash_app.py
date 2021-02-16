import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
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

    def filter_N_alignments(self, slider_N_alignments):
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


def create_fit_results_figure(df_filtered, width=1200, height=700):

    fig = px.scatter(
        df_filtered,
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
        font_size=16,
        xaxis_title=r"$\Large n_\sigma$",
        yaxis_title=r"$\Large D_\mathrm{max}$",
        title=dict(text="Fit Results", font_size=30),
        legend=dict(title="Files", title_font_size=20, font_size=16),
        width=width,
        height=height,
        uirevision=True,  # important for not reshowing legend after change in slider
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


def create_histograms_figure(df_filtered, width=1200, height=700):

    fig = make_subplots(rows=2, cols=4)

    showlegend = True

    for dimension, row, column in fit_results.iterate_over_rows_and_columns():
        highest_y_max = 0
        for name, group in df_filtered.groupby("name", sort=False):
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
        font_size=12,
        title=dict(text="1D Histograms", font_size=30),
        legend=dict(title="Files", title_font_size=20, font_size=16),
        width=width,
        height=height,
        uirevision=True,  # important for not reshowing legend after change in slider
    )

    return fig


#%%


def create_scatter_matrix_figure(df_filtered, width=1200, height=700):

    fig = px.scatter_matrix(
        df_filtered,
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
        font_size=12,
        title=dict(text="Scatter Matrix", font_size=30),
        legend=dict(title="Files", title_font_size=20, font_size=16),
        width=width,
        height=height,
        uirevision=True,  # important for not reshowing legend after change in slider
    )

    return fig


#%%


def transform_slider(x):
    return 10 ** x


def get_range_slider_keywords(df, column="N_alignments", do_log=True, N_steps=100):

    if do_log:

        N_alignments_log = np.log10(df[column])
        N_alignments_min = np.floor(N_alignments_log.min())
        N_alignments_max = np.ceil(N_alignments_log.max())

        marks_steps = np.arange(N_alignments_min, N_alignments_max + 1)
        f = lambda x: utils.human_format(transform_slider(x))
        marks = {int(i): f"{f(i)}" for i in marks_steps}

    else:

        N_alignments = np.log10(df[column])
        N_alignments_min = float(N_alignments.min())
        N_alignments_max = float(N_alignments.max())
        marks = (
            {
                N_alignments_min: str(N_alignments_min),
                N_alignments_max: str(N_alignments_max),
            },
        )

    step = (N_alignments_max - N_alignments_min) / N_steps

    marks[marks_steps[0]] = "No Minimum"
    marks[marks_steps[-1]] = "No Maximum"

    return dict(
        min=N_alignments_min,
        max=N_alignments_max,
        step=step,
        marks=marks,
        value=[N_alignments_min, N_alignments_max],
        allowCross=False,
        updatemode="mouseup",
        included=True,
    )


def is_ipython():
    return hasattr(__builtins__, "__IPYTHON__")


#%%


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/"
        "MathJax.js?config=TeX-MML-AM_CHTML",
    ],
)


app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.Br(),
        html.H1("Fit Results"),
        html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(label="Overview", tab_id="fig_fit_results"),
                dbc.Tab(label="Histograms", tab_id="fig_histograms"),
                dbc.Tab(label="Scatter Matrix", tab_id="fig_scatter_matrix"),
            ],
            id="tabs",
            active_tab="fig_fit_results",
        ),
        html.Div(id="tab-content", className="p-4"),
        html.Hr(),
        html.Div(id="range-slider-N-alignments-output"),
        dcc.RangeSlider(
            id="range-slider-N-alignments",
            **get_range_slider_keywords(
                fit_results.df,
                column="N_alignments",
                do_log=True,
                N_steps=100,
            ),
        ),
    ]
)


@app.callback(
    Output("range-slider-N-alignments-output", "children"),
    [Input("range-slider-N-alignments", "value")],
)
def update_output(slider_range):
    low, high = slider_range
    if True:
        low = transform_slider(low)
        high = transform_slider(high)
    return f"Number of alignments interval: [{utils.human_format(low)}, {utils.human_format(high)}]"


@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)
def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab and data is not None:
        if active_tab == "fig_fit_results":
            return dcc.Graph(figure=data["fig_fit_results"])
        elif active_tab == "fig_histograms":
            return dcc.Graph(figure=data["fig_histograms"])
        elif active_tab == "fig_scatter_matrix":
            return dcc.Graph(figure=data["fig_scatter_matrix"])
    return "No tab selected"


@app.callback(
    Output("store", "data"),
    [Input("range-slider-N-alignments", "value")],
)
def generate_all_figures(slider_N_alignments):
    """
    This callback generates the three graphs (figures) based on the filter
    and stores in the DCC store for faster change of tabs.
    """
    df_filtered = fit_results.filter_N_alignments(slider_N_alignments)

    fig_fit_results = create_fit_results_figure(df_filtered, width=1200, height=700)
    fig_histograms = create_histograms_figure(df_filtered, width=1200, height=700)
    fig_scatter_matrix = create_scatter_matrix_figure(
        df_filtered, width=1200, height=700
    )

    # fig_fit_results['layout']['uirevision'] = True

    # save figures in a dictionary for sending to the dcc.Store
    return {
        "fig_fit_results": fig_fit_results,
        "fig_histograms": fig_histograms,
        "fig_scatter_matrix": fig_scatter_matrix,
    }


#%%


if __name__ == "__main__" and not is_ipython():
    app.run_server(debug=True)

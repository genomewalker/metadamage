# Scientific Library
import numpy as np
import pandas as pd

# Standard Library
from pathlib import Path
from threading import Timer
import webbrowser

# Third Party
import dash
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# First Party
from metadamage import utils


#%%

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


#%%


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
                    low = transform_slider(low)
                    high = transform_slider(high)
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
            # "<i>Tax</i>:<br>"
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


fit_results = FitResults()


#%%


def create_fit_results_figure(df_filtered):

    fig = px.scatter(
        df_filtered,
        x="n_sigma",
        y="D_max",
        size="N_alignments_sqrt",
        color="name",
        hover_name="name",
        size_max=30,
        opacity=0.2,
        color_discrete_map=fit_results.d_cmap,
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
        legend=dict(
            title="Files",
            title_font_size=20,
            font_size=16,
            itemsizing="constant",
            itemclick=False,
            itemdoubleclick=False,
        ),
        # width=width,
        # height=height,
        # uirevision=True,  # important for not reshowing legend after change in slider
        hoverlabel_font_family="Monaco, Lucida Console, Courier, monospace",
        # margin=dict(
        #     t=50,  # top margin: 30px
        #     b=20,  # bottom margin: 10px
        # ),
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


def create_histograms_figure(df_filtered):

    fig = make_subplots(rows=2, cols=4)

    showlegend = True

    for dimension, row, column in fit_results.iterate_over_dimensions():
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

    fig.update_layout(
        font_size=12,
        title=dict(text="1D Histograms", font_size=30),
        legend=dict(
            title="Files",
            title_font_size=20,
            font_size=16,
            tracegroupgap=2,
            itemsizing="constant",
            itemclick=False,
            itemdoubleclick=False,
        ),
        # width=width,
        # height=height,
        # uirevision=True,  # important for not reshowing legend after change in slider
        hoverlabel_font_family="Monaco, Lucida Console, Courier, monospace",
    )

    return fig


#%%


def create_scatter_matrix_figure(df_filtered):

    fig = px.scatter_matrix(
        df_filtered,
        dimensions=fit_results.dimensions,
        color="name",
        hover_name="name",
        size_max=10,
        color_discrete_map=fit_results.d_cmap,
        # color_discrete_sequence=fit_results.cmap,
        # category_orders={"name": list(fit_results.d_cmap.keys())},
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
        legend=dict(
            title="Files",
            title_font_size=20,
            font_size=16,
            itemsizing="constant",
            itemclick=False,
            itemdoubleclick=False,
        ),
        # width=width,
        # height=height,
        # uirevision=True,  # important for not reshowing legend after change in slider
        hoverlabel_font_family="Monaco, Lucida Console, Courier, monospace",
        dragmode="zoom",
    )

    return fig


#%%


def create_forward_reverse_figure(df_filtered):

    N_rows = 3
    N_cols = 2
    subtitles = list(fit_results.dimensions_forward_reverse.values())
    fig = make_subplots(rows=N_rows, cols=N_cols, subplot_titles=subtitles)

    for it in fit_results.iterate_over_dimensions_forward_reverse(N_cols):
        dimension, row, column, showlegend, forward, reverse = it

        for name, group in df_filtered.groupby("name", sort=False):
            kwargs = dict(
                name=name,
                mode="markers",
                legendgroup=name,
                marker=dict(color=fit_results.d_cmap[name], opacity=0.2),
                hovertemplate=fit_results.hovertemplate,
                customdata=np.array(group[fit_results.custom_data_columns].values),
            )

            fig.add_trace(
                go.Scatter(
                    x=group[forward],
                    y=group[reverse],
                    showlegend=showlegend,
                    **kwargs,
                ),
                row=row,
                col=column,
            )

        fig.update_xaxes(range=fit_results.ranges[forward], row=row, col=column)
        fig.update_yaxes(range=fit_results.ranges[reverse], row=row, col=column)

    # Update xaxis properties
    fig.update_xaxes(title_text="Forward")
    fig.update_yaxes(title_text="Reverse")

    fig.update_layout(
        font_size=12,
        title=dict(text="Forward / Reverse", font_size=30),
        legend=dict(
            title="Files",
            title_font_size=20,
            font_size=16,
            itemsizing="constant",
            itemclick=False,
            itemdoubleclick=False,
            tracegroupgap=2,
        ),  #
        # width=width,
        # height=height,
        # uirevision=True,  # important for not reshowing legend after change in slider
        hoverlabel_font_family="Monaco, Lucida Console, Courier, monospace",
    )

    return fig


#%%


def create_empty_figures(width, height):
    fig = go.Figure()

    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=0.5,
        text="Please select some files to plot",
        font_size=20,
        showarrow=False,
    )

    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        width=width,
        height=height,
    )
    return fig


#%%


def create_empty_dataframe_for_datatable(s=None):
    if s is None:
        s = "Please click any datapoint on the graph"
    return pd.DataFrame({"name": [s]})


def get_data_table_keywords():

    data_table_columns_dtypes = {
        # File name
        "name": {"name": "File Name"},
        # Tax Info
        "tax_name": {"name": "Tax Name"},
        "tax_rank": {"name": "Tax Rank"},
        "taxid": {"name": "Tax ID"},
        # Fit Results
        "n_sigma": {
            "name": "n sigma",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "D_max": {
            "name": "D max",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "q_mean": {
            "name": "q",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "concentration_mean": {
            "name": "Concentration",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "asymmetry": {
            "name": "Assymmetry",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "normalized_noise": {
            "name": "Noise",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        # Counts
        "N_alignments": {
            "name": "N alignments",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "N_sum_total": {
            "name": "N sum total",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        # Forward & Reverse
        "n_sigma_forward": {
            "name": "n sigma, forward",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "n_sigma_reverse": {
            "name": "n sigma, reverse",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "D_max_forward": {
            "name": "D max, forward",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "D_max_reverse": {
            "name": "D max, reverse",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "N_z1_forward": {
            "name": "N z=1, forward",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "N_z1_reverse": {
            "name": "N z=1, reverse",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "N_sum_forward": {
            "name": "N sum, forward",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "N_sum_reverse": {
            "name": "N sum, reverse",
            "type": "numeric",
            "format": {"specifier": ".3s"},
        },
        "normalized_noise_forward": {
            "name": "Noise, forward",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
        "normalized_noise_reverse": {
            "name": "Noise, reverse",
            "type": "numeric",
            "format": {"specifier": ".3f"},
        },
    }

    columns = [
        {"id": col, **dtypes} for col, dtypes in data_table_columns_dtypes.items()
    ]

    kwargs = dict(
        id="data_table",
        columns=columns,
        data=create_empty_dataframe_for_datatable().to_dict("records"),
        style_table={
            "overflowX": "auto",
        },
        style_data={"border": "0px"},
        style_cell={"fontFamily": "sans-serif", "fontSize": "12px"},
        # inspired by https://github.com/plotly/dash-table/issues/231
        style_header={
            "backgroundColor": "white",
            "fontWeight": "bold",
            "border": "0px",
        },
        style_data_conditional=[
            {
                "if": {"row_index": 0},
                "backgroundColor": "#F9F9F9",
                "borderTop": "1px solid black",
            },
            {
                "if": {"state": "selected"},
                "backgroundColor": "#F9F9F9",
                "borderTop": "1px solid black",
                "border": "0px",
            },
        ],
    )

    return kwargs


#%%


def transform_slider(x):
    return 10 ** x


def get_range_slider_keywords(df, column="N_alignments", N_steps=100):

    if column == "N_alignments":

        range_log = np.log10(df[column])
        range_min = np.floor(range_log.min())
        range_max = np.ceil(range_log.max())

        marks_steps = np.arange(range_min, range_max + 1)
        f = lambda x: utils.human_format(transform_slider(x))
        marks = {int(i): f"{f(i)}" for i in marks_steps}

        marks[marks_steps[0]] = {"label": "No Minimum", "style": {"color": "#a3ada9"}}
        marks[marks_steps[-1]] = {"label": "No Maximum", "style": {"color": "#a3ada9"}}

    elif column == "D_max":
        range_min = 0.0
        range_max = 1.0
        marks = {
            # 0: "0.0",
            0.25: "0.25",
            0.5: "0.5",
            0.75: "0.75",
            # 1: "1.0",
        }
        marks[0] = {"label": "No Minimum", "style": {"color": "#a3ada9"}}
        marks[1] = {"label": "No Maximum", "style": {"color": "#a3ada9"}}

    step = (range_max - range_min) / N_steps

    return dict(
        min=range_min,
        max=range_max,
        step=step,
        marks=marks,
        value=[range_min, range_max],
        allowCross=False,
        updatemode="mouseup",
        included=True,
    )


def is_ipython():
    return hasattr(__builtins__, "__IPYTHON__")


def open_browser():
    # webbrowser.open_new("http://localhost:8050")
    webbrowser.open("http://localhost:8050")


#%%

card_D_max_slider = dbc.Card(
    html.Div(
        [
            dbc.Row(
                dcc.Markdown(id="range-slider-D-max-output"),
                justify="center",
            ),
            dbc.Row(
                dbc.Col(
                    dcc.RangeSlider(
                        id="range-slider-D-max",
                        **get_range_slider_keywords(
                            fit_results.df,
                            column="D_max",
                            N_steps=100,
                        ),
                    ),
                ),
                justify="center",
            ),
        ],
        style={
            "marginBottom": "1em",
            "marginTop": "0.5em",
            "marginLeft": "0.7em",
            "marginRight": "0.7em",
        },
    ),
    outline=True,
    color="white",
    # className="w-100",
)

card_N_alignments_slider = dbc.Card(
    html.Div(
        [
            dbc.Row(
                dcc.Markdown(id="range-slider-N-alignments-output"),
                justify="center",
            ),
            dbc.Row(
                dbc.Col(
                    dcc.RangeSlider(
                        id="range-slider-N-alignments",
                        **get_range_slider_keywords(
                            fit_results.df,
                            column="N_alignments",
                            N_steps=100,
                        ),
                    ),
                ),
                justify="center",
            ),
        ],
        style={
            "marginBottom": "1em",
            "marginTop": "0.5em",
            "marginLeft": "0.7em",
            "marginRight": "0.7em",
        },
    ),
    outline=True,
    color="white",
)

dropdown = dcc.Dropdown(
    id="dropdown",
    options=[{"label": name, "value": name} for name in fit_results.names],
    value=fit_results.names[:10],
    multi=True,
    placeholder="Select files to plot",
)


card_dropdown = dbc.Card(
    [
        html.H3("File Selection", className="card-title"),
        dropdown,
    ],
    body=True,  # spacing before border
)

card_filters = dbc.Card(
    [
        html.H3("Filters", className="card-title"),
        card_N_alignments_slider,
        card_D_max_slider,
    ],
    body=True,  # spacing before border
)


card_datatable = dbc.Card(
    [
        dbc.Row(
            [
                # dbc.Col(html.H3("Table", className="card-title"), md=1),
                dbc.Col(
                    DataTable(**get_data_table_keywords()),
                    # md=10,
                ),
            ],
        )
    ],
    body=True,  # spacing before border
)


card_filter_and_dropdown = dbc.Card(
    [
        html.Br(),
        card_dropdown,
        html.Br(),
        card_filters,
    ],
    body=True,  # spacing before border
    outline=True,  # together with color, makes a transparent/white border
    color="white",
)


card_main_plot = dbc.Card(
    [
        html.Br(),
        html.Div(
            id="tab_figure"
        ),  # this has to be a div for "reset axis" to work properly
    ],
    body=True,  # spacing before border
    outline=True,  # together with color, makes a transparent/white border
    color="white",
)


#%%

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/"
        "MathJax.js?config=TeX-MML-AM_CHTML",
    ],
    suppress_callback_exceptions=True,
)

app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(html.H1("Fit Results"), md=2),
                dbc.Col(
                    dbc.Tabs(
                        [
                            dbc.Tab(label="Overview", tab_id="fig_fit_results"),
                            dbc.Tab(label="Histograms", tab_id="fig_histograms"),
                            dbc.Tab(
                                label="Scatter Matrix", tab_id="fig_scatter_matrix"
                            ),
                            dbc.Tab(
                                label="Forward / Reverse", tab_id="fig_forward_reverse"
                            ),
                        ],
                        id="tabs",
                        active_tab="fig_fit_results",
                    ),
                ),
            ],
        ),
        # html.Hr(),
        # html.Br(),
        dbc.Row(
            [
                dbc.Col(card_filter_and_dropdown, md=2),
                dbc.Col(card_main_plot, md=8),
            ],
        ),
        card_datatable,
    ],
    fluid=True,
)


@app.callback(
    Output("range-slider-N-alignments-output", "children"),
    Input("range-slider-N-alignments", "value"),
)
def update_markdown_N_alignments(slider_range):
    low, high = slider_range
    low = transform_slider(low)
    high = transform_slider(high)

    # https://tex-image-link-generator.herokuapp.com/
    latex = (
        r"![\large N_\mathrm{alignments}](https://render.githubusercontent.com/render/"
        + r"math?math=%5Cdisplaystyle+%5Clarge+N_%5Cmathrm%7Balignments%7D)"
    )
    return latex + fr": \[{utils.human_format(low)}, {utils.human_format(high)}\]"


@app.callback(
    Output("range-slider-D-max-output", "children"),
    Input("range-slider-D-max", "value"),
)
def update_markdown_D_max(slider_range):
    low, high = slider_range

    # https://tex-image-link-generator.herokuapp.com/
    latex = (
        r"![\large D_\mathrm{max}](https://render.githubusercontent.com/render/"
        + r"math?math=%5Cdisplaystyle+%5Clarge+D_%5Cmathrm%7Bmax%7D)"
    )
    return latex + f": \[{low:.2f}, {high:.2f}\]"


@app.callback(
    Output("data_table", "data"),
    Input("graph", "clickData"),
)
def make_clickData_table(clickData):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if clickData is not None:
        index_taxid = fit_results.custom_data_columns.index("taxid")
        index_name = fit_results.custom_data_columns.index("name")
        try:
            name = clickData["points"][0]["customdata"][index_name]
            taxid = clickData["points"][0]["customdata"][index_taxid]
            # df_filtered = fit_results.filter({"name": name, "taxid": taxid})
            df_filtered = fit_results.df.query(f"taxid == {taxid} & name == '{name}'")
            return df_filtered.to_dict("records")

        # when selecting histogram without customdata
        except KeyError:
            s = "Does not work for binned data (histograms)"
            return create_empty_dataframe_for_datatable(s).to_dict("records")

    else:
        return create_empty_dataframe_for_datatable().to_dict("records")


@app.callback(
    Output("tab_figure", "children"),
    Input("tabs", "active_tab"),
    Input("store", "data"),
)
def render_tab_figure(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """

    kwargs = dict(
        id="graph",
        config={
            "displaylogo": False,
            "doubleClick": "reset",
            "showTips": True,
            "modeBarButtonsToRemove": [
                "select2d",
                "lasso2d",
                "autoScale2d",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
                "toggleSpikelines",
            ],
        },
        # https://css-tricks.com/fun-viewport-units/
        style={"width": "90%", "height": "78vh"},
    )

    if active_tab and data is not None:
        if active_tab == "fig_fit_results":
            return dcc.Graph(figure=data["fig_fit_results"], **kwargs)
        elif active_tab == "fig_histograms":
            return dcc.Graph(figure=data["fig_histograms"], **kwargs)
        elif active_tab == "fig_scatter_matrix":
            return dcc.Graph(figure=data["fig_scatter_matrix"], **kwargs)
        elif active_tab == "fig_forward_reverse":
            return dcc.Graph(figure=data["fig_forward_reverse"], **kwargs)

    print("Got here", active_tab, data)
    return "No tab selected"


@app.callback(
    Output("store", "data"),
    Input("range-slider-N-alignments", "value"),
    Input("range-slider-D-max", "value"),
    Input("dropdown", "value"),
    # Input("tab_figure", "restyleData"),
)
def generate_all_figures(
    slider_N_alignments,
    slider_D_max,
    dropdown_names,
    # restyleData,
):
    """
    This callback generates the three graphs (figures) based on the filter
    and stores in the DCC store for faster change of tabs.
    """

    # print(restyleData)

    height = 800
    width = 1100

    if dropdown_names is None or len(dropdown_names) == 0:
        fig_empty = create_empty_figures(width=width, height=height)
        return {
            "fig_fit_results": fig_empty,
            "fig_histograms": fig_empty,
            "fig_scatter_matrix": fig_empty,
            "fig_forward_reverse": fig_empty,
        }

    df_filtered = fit_results.filter(
        {
            "N_alignments": slider_N_alignments,
            "D_max": slider_D_max,
            "names": dropdown_names,
        }
    )

    fig_fit_results = create_fit_results_figure(
        df_filtered,
    )
    fig_histograms = create_histograms_figure(
        df_filtered,
    )
    fig_scatter_matrix = create_scatter_matrix_figure(
        df_filtered,
    )

    fig_forward_reverse = create_forward_reverse_figure(
        df_filtered,
    )

    # save figures in a dictionary for sending to the dcc.Store
    return {
        "fig_fit_results": fig_fit_results,
        "fig_histograms": fig_histograms,
        "fig_scatter_matrix": fig_scatter_matrix,
        "fig_forward_reverse": fig_forward_reverse,
    }


#%%

if __name__ == "__main__" and not is_ipython():
    # Timer(1, open_browser).start()
    app.run_server(debug=True)

#%%

# df = fit_results.df

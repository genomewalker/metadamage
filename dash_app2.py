# Scientific Library
import numpy as np
import pandas as pd

# Standard Library
from pathlib import Path
from threading import Timer
import webbrowser

# Third Party
import dash
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from importlib import reload

# First Party
from metadamage import mydash, utils


from about_time import about_time

#%%

mydash.utils.set_custom_theme()
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

#%%

fit_results = mydash.fit_results.FitResults()

#%%

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.COSMO],
    external_scripts=[
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/"
        "MathJax.js?config=TeX-MML-AM_CHTML",
    ],
    suppress_callback_exceptions=True,
    title="Metadamage",
    update_title="Updating...",
)

# to allow custom css
app.scripts.config.serve_locally = True


graph_kwargs = dict(
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
    # style={"width": "90%", "height": "70vh"},
)


#%%

tabs_kwargs = [
    dict(label="Overview", tab_id="fig_fit_results"),
    dict(label="Histograms", tab_id="fig_histograms"),
    dict(label="Scatter Matrix", tab_id="fig_scatter_matrix"),
    dict(label="Forward / Reverse", tab_id="fig_forward_reverse"),
]

tabs = dbc.Tabs(
    [dbc.Tab(**tab) for tab in tabs_kwargs],
    id="tabs",
    active_tab="fig_fit_results",
)

card_tabs = dbc.Card(
    [tabs],
    body=True,
)

#%%

card_form_mismatch = dbc.Card(
    [
        html.H3("Dropdowns", className="card-title"),
        html.Div(
            [
                dcc.Dropdown(
                    id="dropdown_mismatch_filename",
                    options=[
                        {"label": name, "value": name} for name in fit_results.names
                    ],
                    clearable=False,
                ),
            ],
        ),
        html.Div([dcc.Dropdown(id="dropdown_mismatch_taxid", clearable=False)]),
        html.Hr(),
        html.Div(id="display-selected-values"),
    ],
    body=True,
)


card_mismatch = dbc.Card(
    [
        card_form_mismatch,
        dcc.Graph(
            figure=mydash.figures.create_empty_figure(),
            id="graph_mismatch",
            **graph_kwargs,
        ),
    ],
    body=True,
)


#%%

card_datatable = dbc.Card(
    [
        html.H3("Datatable", className="card-title"),
        DataTable(**mydash.datatable.get_data_table_keywords(id="data_table")),
    ],
    body=True,
)


#%%

form_dropdown = dbc.FormGroup(
    [
        dbc.Label("Dropdown"),
        mydash.elements.get_dropdown_file_selection(
            id="dropdown_file_selection",
            fit_results=fit_results,
        ),
    ]
)

form_range_slider_N_alignments = dbc.FormGroup(
    [
        dbc.Label("Range Slider N_alignments"),
        dcc.RangeSlider(
            id="range_slider_N_alignments",
            **mydash.elements.get_range_slider_keywords(
                fit_results,
                column="N_alignments",
                N_steps=100,
            ),
        ),
    ]
)


form_range_slider_D_max = dbc.FormGroup(
    [
        dbc.Label("Range Slider D max"),
        dcc.RangeSlider(
            id="range_slider_D_max",
            **mydash.elements.get_range_slider_keywords(
                fit_results,
                column="D_max",
                N_steps=100,
            ),
        ),
    ]
)

card_form = dbc.Card(
    [
        html.H3("Filters", className="card-title"),
        dbc.Form(
            [form_dropdown, form_range_slider_N_alignments, form_range_slider_D_max]
        ),
    ],
    body=True,  # spacing before border
)

#%%


form_overview_marker_size = dbc.FormGroup(
    [
        dbc.Label("Marker Size", className="mr-2"),
        # dbc.Col(
        html.Div(
            dcc.Slider(
                # id="slider_overview_marker_size",
                # possible fix for ReferenceError
                id={"type": "slider_overview_marker_size", "index": 0},
                **mydash.elements.get_slider_keywords(),
            ),
            style={"min-width": "300px"},
        )
        # width=5,
        # ),
    ],
    className="mr-6",
)


form_overview_marker_transformation = dbc.FormGroup(
    [
        dbc.Label("Marker Transformation", className="mr-2"),
        # dbc.Col(
        dcc.Dropdown(
            # id="dropdown_overview_marker_transformation",
            # possible fix for ReferenceError
            id={"type": "dropdown_overview_marker_transformation", "index": 0},
            options=[
                {"label": "Identity", "value": "identity"},
                {"label": "Sqrt", "value": "sqrt"},
                {"label": "Log", "value": "log10"},
                {"label": "Constant", "value": "constant"},
            ],
            value="sqrt",
            searchable=False,
            clearable=False,
            style={"min-width": "100px"},
            # ),
            # width=5,
        ),
    ],
    className="mr-6",
)

button_reset = dbc.FormGroup(
    [
        dbc.Label("", className="mr-5"),
        dbc.Button(
            "Reset",
            id={"type": "button_reset", "index": 0},
            color="primary",
        ),
    ],
    className="mr-6",
)


card_overview_marker = dbc.Card(
    [
        # html.H3("Markers", className="card-title"),
        dbc.Form(
            [
                form_overview_marker_size,
                form_overview_marker_transformation,
                button_reset,
            ],
            inline=True,
        ),
    ],
    body=True,  # spacing before border
    # className="mr-6",
)


#%%


def get_figure_from_df_and_tab(df_fit_results_filtered, active_tab):
    if active_tab == "fig_fit_results":
        fig = mydash.figures.plot_fit_results(fit_results, df_fit_results_filtered)
    elif active_tab == "fig_histograms":
        fig = mydash.figures.plot_histograms(fit_results, df_fit_results_filtered)
    elif active_tab == "fig_scatter_matrix":
        fig = mydash.figures.plot_scatter_matrix(fit_results, df_fit_results_filtered)
    elif active_tab == "fig_forward_reverse":
        fig = mydash.figures.plot_forward_reverse(fit_results, df_fit_results_filtered)
    else:
        print("got here: get_figure_from_df_and_tab")
    return fig


def get_main_figure(data_or_df, active_tab="fig_fit_results"):
    if data_or_df is None:
        return mydash.figures.create_empty_figure(s="")

    if isinstance(data_or_df, list):
        df = pd.DataFrame.from_records(data_or_df)
    elif isinstance(data_or_df, pd.DataFrame):
        df = data_or_df
    else:
        raise AssertionError(f"Got wrong type for data_or_df: {type(data_or_df)}")
    return get_figure_from_df_and_tab(df, active_tab)


def make_tab_from_data(data_or_df=None, active_tab="fig_fit_results"):

    figure = get_main_figure(data_or_df, active_tab)
    main_graph = dcc.Graph(figure=figure, id="main_graph", **graph_kwargs)

    if active_tab == "fig_fit_results" or active_tab == "overview":

        return (
            dbc.Container(
                [
                    dbc.Row(
                        dbc.Col(main_graph, width=12),
                        justify="center",
                    ),
                    dbc.Row(
                        dbc.Col(card_overview_marker, width=10),
                        justify="center",
                    ),
                ],
            ),
        )

        # return [
        #     main_graph,
        #     card_overview_marker,
        # ]

    elif active_tab == "fig_histograms":
        return (
            dbc.Container(
                [
                    dbc.Row(
                        dbc.Col(main_graph, width=12),
                        justify="center",
                    ),
                ],
            ),
        )

    elif active_tab == "fig_scatter_matrix":
        return (
            dbc.Container(
                [
                    dbc.Row(
                        dbc.Col(main_graph, width=12),
                        justify="center",
                    ),
                ],
            ),
        )

    elif active_tab == "fig_forward_reverse":
        return (
            dbc.Container(
                [
                    dbc.Row(
                        dbc.Col(main_graph, width=12),
                        justify="center",
                    ),
                ],
            ),
        )

    else:
        print("got here: make_tab_from_data")


#%%

card_graph = dbc.Card(
    [
        html.Div(
            make_tab_from_data(active_tab="overview"),  # main_graph
            id="main_graph_div",
            className="loader-fade",
        ),
    ],
    body=True,  # spacing before border
    # style={"height": "50vh"},
)


#%%

import time

app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(card_form, width=2),
                dbc.Col([card_tabs, card_graph], width=6),
                dbc.Col(card_mismatch, width=4),
            ],
            justify="center",
            # className="h-75",
        ),
        html.Hr(),
        dbc.Row(
            [dbc.Col(card_datatable, width=12)],
            justify="center",
            # className="h-25",
        ),
    ],
    fluid=True,  # fill available horizontal space and resize fluidly
    # style={"height": "90vh"},
)

#%%


from dash.exceptions import PreventUpdate


@app.callback(
    Output("store", "data"),
    Input("dropdown_file_selection", "value"),
    Input("range_slider_N_alignments", "value"),
    Input("range_slider_D_max", "value"),
    Input({"type": "slider_overview_marker_size", "index": ALL}, "value"),
    Input({"type": "dropdown_overview_marker_transformation", "index": ALL}, "value"),
)
def filter_fit_results(
    dropdown_file_selection,
    range_slider_N_alignments,
    range_slider_D_max,
    marker_size_max,
    marker_transformation,
):

    # if no files selected
    if not dropdown_file_selection:
        raise PreventUpdate

    fit_results.set_marker_size(marker_transformation, marker_size_max)
    df_fit_results_filtered = fit_results.filter(
        {
            "N_alignments": range_slider_N_alignments,
            "D_max": range_slider_D_max,
            "names": dropdown_file_selection,
        }
    )

    return df_fit_results_filtered.to_dict("records")


@app.callback(
    Output("main_graph", "figure"),
    Input("store", "data"),
    Input("tabs", "active_tab"),
)
def update_main_graph(data, active_tab):
    if active_tab is None:
        print("update_main_graph got active_tab == None")
    if data is None:
        print("update_main_graph got data == None")

    figure = get_main_figure(data, active_tab)

    # allows to size of marker to change without loosing zoom level
    if active_tab == "fig_fit_results":
        figure["layout"]["uirevision"] = True

    return figure


@app.callback(
    Output("main_graph_div", "children"),
    Input("tabs", "active_tab"),
    Input({"type": "button_reset", "index": ALL}, "n_clicks"),
    State("store", "data"),
)
def update_tab_layout(active_tab, button_n, data):
    if active_tab is None:
        print("update_tab_layout got active_tab == None")
    if data is None:
        # print("update_tab_layout got data == None")
        raise PreventUpdate

    return make_tab_from_data(data, active_tab)


#%%


@app.callback(
    Output("graph_mismatch", "figure"),
    Input("tabs", "active_tab"),
    Input("dropdown_mismatch_taxid", "value"),
    Input("dropdown_mismatch_filename", "value"),
)
def update_mismatch_plot(active_tab, dropdown_mismatch_taxid, dropdown_name):

    if dropdown_mismatch_taxid is None:
        if active_tab == "fig_histograms":
            s = "Does not work for binned data"
            return mydash.figures.create_empty_figure(s=s)
        else:
            return mydash.figures.create_empty_figure()

    try:
        group = fit_results.get_mismatch_group(
            name=dropdown_name,
            taxid=dropdown_mismatch_taxid,
        )
        fit = fit_results.get_fit_predictions(
            name=dropdown_name,
            taxid=dropdown_mismatch_taxid,
        )
        chosen_mismatch_columns = ["C→T", "G→A"]
        fig = mydash.figures.plot_mismatch_fractions(
            group,
            chosen_mismatch_columns,
            fit,
        )
        return fig

    # when selecting histogram without customdata
    except KeyError:
        raise PreventUpdate


#%%


@app.callback(
    Output("data_table", "data"),
    Input("main_graph", "clickData"),
    Input("tabs", "active_tab"),
)
def update_data_table(click_data, active_tab):
    if click_data is None:
        if active_tab == "fig_histograms":
            s = "Does not work for binned data (histograms)"
            ds = mydash.datatable.create_empty_dataframe_for_datatable(s)
        else:
            ds = mydash.datatable.create_empty_dataframe_for_datatable()
        return ds.to_dict("records")

    try:
        taxid = fit_results.parse_click_data(click_data, variable="taxid")
        name = fit_results.parse_click_data(click_data, variable="name")
        df_fit_results_filtered = fit_results.filter({"name": name, "taxid": taxid})
        return df_fit_results_filtered.to_dict("records")

    # when selecting histogram without customdata
    except KeyError:
        raise PreventUpdate


#%%


def get_taxid_options_based_on_filename(name, df_string="df_fit_results"):
    """df_string is a string, eg. df_fit_results or  df_mismatch.
    The 'df_' part is optional
    """
    taxids = sorted(fit_results.filter({"name": name}, df="mismatch")["taxid"].unique())
    options = [{"label": i, "value": i} for i in taxids]
    return options


@app.callback(
    Output("dropdown_mismatch_taxid", "options"),
    Input("dropdown_mismatch_filename", "value"),
)
def update_dropdown_taxid_options(name):
    return get_taxid_options_based_on_filename(name, df_string="mismatch")


@app.callback(
    Output("display-selected-values", "children"),
    Input("dropdown_mismatch_taxid", "value"),
    Input("main_graph", "clickData"),
)
def update_click_data_text(taxid, click_data):
    if click_data is not None:
        try:
            taxid = fit_results.parse_click_data(click_data, variable="taxid")
        except KeyError:
            raise PreventUpdate
    return "you have selected Tax ID {}".format(taxid)


@app.callback(
    Output("dropdown_mismatch_filename", "value"),
    Output("dropdown_mismatch_taxid", "value"),
    Input("main_graph", "clickData"),
)
def update_dropdowns_based_on_click_data(click_data):
    if click_data is not None:
        try:
            taxid = fit_results.parse_click_data(click_data, variable="taxid")
            name = fit_results.parse_click_data(click_data, variable="name")
            return name, taxid
        except KeyError:
            raise PreventUpdate
    else:
        raise PreventUpdate


#%%


def is_ipython():
    return hasattr(__builtins__, "__IPYTHON__")


if __name__ == "__main__" and not is_ipython():
    app.run_server(debug=True)


else:

    name = "KapK-198A-Ext-55-Lib-55-Index1"
    name = "SJArg-1-Nit"
    taxid = 33969

    df_fit_results = fit_results.filter(
        {"taxid": taxid, "name": name}, df="df_fit_results"
    )

    group = fit_results.get_mismatch_group(name=name, taxid=taxid)
    fit = fit_results.get_fit_predictions(name=name, taxid=taxid)

    chosen_mismatch_columns = ["C→T", "G→A"]

    # #%%

    fig = mydash.figures.plot_mismatch_fractions(
        group, chosen_mismatch_columns, fit=fit
    )
    fig

    group[["position", "CT", "C"]]

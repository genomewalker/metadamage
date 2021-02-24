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

card_mismatch = dbc.Card(
    [
        dcc.Graph(
            figure=mydash.figures.create_empty_figure(s="Please select a point"),
            id="graph_mismatch",
            **graph_kwargs,
        ),
    ],
    body=True,
)

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
        dbc.Label("Marker Size"),
        dcc.Slider(
            id="slider_overview_marker_size",
            **mydash.elements.get_slider_keywords(),
        ),
    ]
)


form_overview_marker_transformation = dbc.FormGroup(
    [
        dbc.Label("Marker Transformation"),
        dcc.Dropdown(
            id="dropdown_overview_marker_transformation",
            options=[
                {"label": "Identity", "value": "identity"},
                {"label": "Sqrt", "value": "sqrt"},
                {"label": "Log", "value": "log10"},
                {"label": "Constant", "value": "constant"},
            ],
            value="sqrt",
            searchable=False,
            clearable=False,
        ),
    ]
)

card_overview_marker = dbc.Card(
    [
        html.H3("Markers", className="card-title"),
        dbc.Form(
            [form_overview_marker_size, form_overview_marker_transformation],
        ),
    ],
    body=True,  # spacing before border
)


#%%

card_graph = dbc.Card(
    [
        html.Div(
            dcc.Graph(
                figure=mydash.figures.create_empty_figure(s=""),
                id="main_graph",
                **graph_kwargs,
            ),
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
        # dcc.Store(id="store"),
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
    Output("main_graph_div", "children"),
    [
        Input("dropdown_file_selection", "value"),
        Input("range_slider_N_alignments", "value"),
        Input("range_slider_D_max", "value"),
        Input("tabs", "active_tab"),
    ],
)
def main_plot(
    dropdown_file_selection,
    range_slider_N_alignments,
    range_slider_D_max,
    active_tab,
):

    # if no files selected
    if not dropdown_file_selection:
        raise PreventUpdate

    df_fit_results_filtered = fit_results.filter(
        {
            "N_alignments": range_slider_N_alignments,
            "D_max": range_slider_D_max,
            "names": dropdown_file_selection,
        }
    )

    if active_tab == "fig_fit_results":
        fig = mydash.figures.plot_fit_results(fit_results, df_fit_results_filtered)
        # graph = dcc.Graph(figure=fig, id="main_graph", **graph_kwargs)
        # card_graph = dbc.Card([graph], body=True)  # spacing before border
        # container = dbc.Container(
        #     [
        #         dbc.Row([card_graph]),
        #         dbc.Row([card_overview_marker]),
        #     ]
        # )
        # return container

    elif active_tab == "fig_histograms":
        fig = mydash.figures.plot_histograms(fit_results, df_fit_results_filtered)

    elif active_tab == "fig_scatter_matrix":
        fig = mydash.figures.plot_scatter_matrix(fit_results, df_fit_results_filtered)

    elif active_tab == "fig_forward_reverse":
        fig = mydash.figures.plot_forward_reverse(fit_results, df_fit_results_filtered)

    else:
        print(f"{active_tab} not implemented yet")
        raise PreventUpdate

    return dcc.Graph(figure=fig, id="main_graph", **graph_kwargs)


#%%


@app.callback(
    Output("graph_mismatch", "figure"),
    Input("main_graph", "clickData"),
)
def mismatch_plot(click_data):
    if click_data is None:
        raise PreventUpdate
    taxid = fit_results.parse_click_data(click_data, variable="taxid")
    name = fit_results.parse_click_data(click_data, variable="name")
    group = fit_results.get_mismatch_group(name=name, taxid=taxid)
    fit = fit_results.get_fit_predictions(name=name, taxid=taxid)
    chosen_mismatch_columns = ["C→T", "G→A"]
    fig = mydash.figures.plot_mismatch_fractions(group, chosen_mismatch_columns, fit)
    return fig


#%%


@app.callback(
    Output("data_table", "data"),
    Input("main_graph", "clickData"),
)
def mismatch_plot(click_data):
    if click_data is None:
        ds = mydash.datatable.create_empty_dataframe_for_datatable()
        return ds.to_dict("records")

    try:
        taxid = fit_results.parse_click_data(click_data, variable="taxid")
        name = fit_results.parse_click_data(click_data, variable="name")
        df_fit_results_filtered = fit_results.filter({"name": name, "taxid": taxid})
        return df_fit_results_filtered.to_dict("records")

    # when selecting histogram without customdata
    except KeyError:
        s = "Does not work for binned data (histograms)"
        ds = mydash.datatable.create_empty_dataframe_for_datatable(s)
        return ds.to_dict("records")


#%%


def is_ipython():
    return hasattr(__builtins__, "__IPYTHON__")


if __name__ == "__main__" and not is_ipython():
    app.run_server(debug=True)


else:

    name = "SJArg-1-Nit"
    taxid = 33969

    group = fit_results.filter({"taxid": taxid, "name": name}, df="df_mismatch")
    chosen_mismatch_columns = ["C→T", "G→A"]

    fig = mydash.figures.plot_mismatch_fractions(group, chosen_mismatch_columns)
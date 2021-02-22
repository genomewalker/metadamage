# Scientific Library
import numpy as np
import pandas as pd

# Third Party
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable

# First Party
from metadamage import mydash, utils


def get_range_slider_keywords(df, column="N_alignments", N_steps=100):

    if column == "N_alignments":

        range_log = np.log10(df[column])
        range_min = np.floor(range_log.min())
        range_max = np.ceil(range_log.max())

        marks_steps = np.arange(range_min, range_max + 1)
        f = lambda x: utils.human_format(mydash.utils.transform_slider(x))
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


#%%


def get_card_D_max_slider(fit_results):

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

    return card_D_max_slider


def get_card_N_alignments_slider(fit_results):

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

    return card_N_alignments_slider


def get_dropdown_file_selection(fit_results):

    dropdown_file_selection = dcc.Dropdown(
        id="dropdown_file_selection",
        options=[{"label": name, "value": name} for name in fit_results.names],
        value=fit_results.names[:10],
        multi=True,
        placeholder="Select files to plot",
    )

    return dropdown_file_selection


def get_card_dropdown_file_selection(fit_results):

    card_dropdown_file_selection = dbc.Card(
        [
            html.H3("File Selection", className="card-title"),
            get_dropdown_file_selection(fit_results),
        ],
        body=True,  # spacing before border
    )
    return card_dropdown_file_selection


def get_card_filters(fit_results):

    card_filters = dbc.Card(
        [
            html.H3("Filters", className="card-title"),
            get_card_N_alignments_slider(fit_results),
            get_card_D_max_slider(fit_results),
        ],
        body=True,  # spacing before border
    )

    return card_filters


def get_card_datatable(fit_results):

    card_datatable = dbc.Card(
        [
            dbc.Row(
                [
                    # dbc.Col(html.H3("Table", className="card-title"), md=1),
                    dbc.Col(
                        DataTable(**mydash.datatable.get_data_table_keywords()),
                        # md=10,
                    ),
                ],
            )
        ],
        body=True,  # spacing before border
    )

    return card_datatable


def get_dropdown_marker_transformation(fit_results):

    dropdown_marker_transformation = dcc.Dropdown(
        id="dropdown_marker_transformation",
        options=[
            {"label": "Identity", "value": "identity"},
            {"label": "Sqrt", "value": "sqrt"},
            {"label": "Log", "value": "log10"},
            {"label": "Constant", "value": "constant"},
        ],
        value="sqrt",
        searchable=False,
        clearable=False,
    )

    return dropdown_marker_transformation


def get_slider_keywords():
    marks = [1, 10, 20, 30, 40, 50, 60]
    return dict(
        min=1,
        max=60,
        step=1,
        value=30,
        marks={mark: str(mark) for mark in marks},
    )


def get_card_marker_size_slider(fit_results):

    card_marker_size_slider = dbc.Card(
        html.Div(
            [
                dbc.Row(
                    dcc.Markdown(id="slider-marker-size-output"),
                    justify="center",
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.Slider(
                            id="slider-marker-size",
                            **get_slider_keywords(),
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

    return card_marker_size_slider


def get_card_dropdown_marker_transformation(fit_results):

    card_dropdown_marker_transformation = dbc.Card(
        [
            html.H3("Marker Size", className="card-title"),
            get_card_marker_size_slider(fit_results),
            get_dropdown_marker_transformation(fit_results),
        ],
        body=True,  # spacing before border
    )
    return card_dropdown_marker_transformation


def get_card_filter_and_dropdown_file_selection(fit_results):

    card_filter_and_dropdown_file_selection = dbc.Card(
        [
            html.Br(),
            get_card_dropdown_file_selection(fit_results),
            html.Br(),
            get_card_filters(fit_results),
            html.Br(),
            get_card_dropdown_marker_transformation(fit_results),
        ],
        body=True,  # spacing before border
        outline=True,  # together with color, makes a transparent/white border
        color="white",
    )
    return card_filter_and_dropdown_file_selection


def get_card_main_plot(fit_results):

    card_main_plot = dbc.Card(
        [
            html.Br(),
            # this has to be a div for "reset axis" to work properly
            html.Div(id="tab_figure"),
        ],
        body=True,  # spacing before border
        outline=True,  # together with color, makes a transparent/white border
        color="white",
    )

    return card_main_plot

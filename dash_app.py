# Scientific Library
import numpy as np
import pandas as pd

# Standard Library
from importlib import reload
from pathlib import Path
from threading import Timer
import time
import webbrowser

# Third Party
from about_time import about_time
import dash
from dash.dependencies import ALL, Input, MATCH, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# First Party
from metadamage import mydash, taxonomy, utils, io

#%%

mydash.utils.set_custom_theme()
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

#%%

folder = Path("./data/out")
verbose = True

with about_time() as at1:
    fit_results = mydash.fit_results.FitResults(
        folder, verbose=verbose, very_verbose=False
    )
# print(f"{at1.duration_human}")


#%%BOOTSTRAP

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
    style={"width": "100%", "height": "55vh"},
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

mismatch_dropdown_shortnames = dbc.FormGroup(
    [
        dbc.Label("Filenames:", className="mr-2"),
        html.Div(
            dcc.Dropdown(
                id="dropdown_mismatch_shortname",
                options=[
                    {"label": shortname, "value": shortname}
                    for shortname in fit_results.shortnames
                ],
                clearable=False,
            ),
            style={"min-width": "300px"},
        ),
    ],
    className="mr-5",
)

mismatch_dropdown_tax_ids = dbc.FormGroup(
    [
        dbc.Label("Tax IDs:", className="mr-2"),
        html.Div(
            dcc.Dropdown(id="dropdown_mismatch_tax_id", clearable=True),
            style={"min-width": "150px"},
        ),
    ],
    className="mr-3",
)


mismatch_dropdowns = dbc.Card(
    [
        dbc.Form(
            [
                mismatch_dropdown_shortnames,
                mismatch_dropdown_tax_ids,
            ],
            inline=True,
        ),
    ],
    body=True,
)


card_mismatch_dropdowns_and_graph = dbc.Card(
    [
        dbc.Row(
            dbc.Col(mismatch_dropdowns, width=10),
            justify="center",
        ),
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

import itertools

filter_tax_id = dbc.Row(
    [
        dbc.Col(
            html.Br(),
            width=12,
        ),
        dbc.Col(
            html.H3("Filter specific taxa"),
            width=12,
        ),
        dbc.Col(
            dbc.FormGroup(
                [
                    dcc.Dropdown(
                        id="tax_id_filter_input",
                        options=[
                            {"label": tax, "value": tax}
                            for tax in itertools.chain.from_iterable(
                                [
                                    fit_results.all_tax_ranks,
                                    fit_results.all_tax_names,
                                    fit_results.all_tax_ids,
                                ]
                            )
                        ],
                        clearable=True,
                        multi=True,
                        placeholder="Select taxas...",
                    ),
                ],
            ),
            width=12,
        ),
        dbc.Col(
            html.Br(),
            width=12,
        ),
        dbc.Col(
            html.H3("Filter taxanomic descendants"),
            width=12,
        ),
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Input(
                        id="tax_id_filter_input_descendants",
                        placeholder="Input goes here...",
                        type="text",
                        autoComplete="off",
                    ),
                ]
            ),
            width=12,
        ),
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Checklist(
                        options=[
                            {"label": "Include subspecies", "value": True},
                        ],
                        value=[True],
                        id="tax_id_filter_subspecies",
                    ),
                ]
            ),
            width=12,
        ),
        dbc.Col(
            html.P(id="tax_id_filter_counts_output"),
            # xl=9,
            width=12,
        ),
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Button(
                        "Plot",
                        id="tax_id_plot_button",
                        color="primary",
                        block=True,
                    ),
                ]
            ),
            # xl=3,
            # offset=3,
            # width=6,
            xs={"size": 11, "offset": 1},
            sm={"size": 11, "offset": 1},
            md={"size": 10, "offset": 2},
            lg={"size": 9, "offset": 3},
            width={"size": 6, "offset": 6},
        ),
    ],
    justify="between",
    form=True,
)


filters_collapse_tax_id = html.Div(
    [
        dbc.Button(
            "Filter Tax IDs",
            id="filters_toggle_tax_ids_button",
            color="secondary",
            block=True,
            outline=True,
            size="lg",
        ),
        dbc.Collapse(
            filter_tax_id,
            id="filters_dropdown_tax_ids",
            is_open=False,
        ),
    ]
)


#%%

filter_dropdown_file = dbc.FormGroup(
    [
        # dbc.Label("Dropdown"),
        html.Br(),
        dbc.Col(
            html.H3("Filter input samples"),
            width=12,
        ),
        mydash.elements.get_dropdown_file_selection(
            id="dropdown_file_selection",
            fit_results=fit_results,
            shortnames_to_show="each",  # one for each first letter in shortname
        ),
    ]
)


filters_collapse_files = html.Div(
    [
        dbc.Button(
            "Filter Files",
            id="filters_toggle_files_button",
            color="secondary",
            block=True,
            outline=True,
            size="lg",
        ),
        dbc.Collapse(
            filter_dropdown_file,
            id="filters_dropdown_files",
            is_open=False,
        ),
    ]
)


def make_range_slider_filter(column, N_steps=100):
    return dbc.FormGroup(
        [
            dbc.Label(f"Range Slider {column}"),
            dcc.RangeSlider(
                id=f"range_slider_{column}",
                **mydash.elements.get_range_slider_keywords(
                    fit_results,
                    column=column,
                    N_steps=N_steps,
                ),
            ),
        ]
    )


slider_names = [
    "n_sigma",
    "D_max",
    "q_mean",
    "concentration_mean",
    "N_alignments",
    "y_sum_total",
    "N_sum_total",
]

filters_collapse_ranges = html.Div(
    [
        dbc.Button(
            "Filter Fit Results",
            id="filters_toggle_ranges_button",
            color="secondary",
            block=True,
            outline=True,
            size="lg",
        ),
        dbc.Collapse(
            [
                html.Br(),
                dbc.Col(
                    html.H3("Filter fit results"),
                    width=12,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="dropdown_slider",
                        options=[
                            {"label": shortname, "value": shortname}
                            for shortname in slider_names
                        ],
                        value=[],
                        multi=True,
                        placeholder="Select a variable to filter on...",
                        optionHeight=30,
                        # style={"overflow-y": "scroll", "height": "100px"},
                    ),
                    width=12,
                    # style={"overflow-y": "scroll", "height": "500px"},
                ),
                dbc.Col(
                    id="dynamic_slider-container",
                    children=[],
                    width=12,
                ),
            ],
            id="filters_dropdown_ranges_button",
            is_open=False,
        ),
    ]
)


filter_card = dbc.Card(
    [
        html.H1("Select filters:", className="card-title"),
        html.Hr(),
        dbc.Form(
            [
                filters_collapse_files,
                html.Hr(),
                filters_collapse_tax_id,
                html.Hr(),
                filters_collapse_ranges,
            ]
        ),
    ],
    # style={"maxHeight": "1000px", "overflow": "auto"},
    body=True,  # spacing before border
)

#%%


form_overview_marker_size = dbc.FormGroup(
    [
        dbc.Label("Marker scaling", className="mr-2"),
        # dbc.Col(
        html.Div(
            dcc.Slider(  # possible fix for ReferenceError
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
        dbc.Label("Marker transformation", className="mr-2"),
        # dbc.Col(
        dcc.Dropdown(  # possible fix for ReferenceError
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


app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(filter_card, width=3),
                dbc.Col([card_tabs, card_graph], width=6),
                dbc.Col(card_mismatch_dropdowns_and_graph, width=3),
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
        dbc.Modal(
            [
                dbc.ModalHeader("Filtering Error"),
                dbc.ModalBody(
                    "Too restrictive filtering, no points left to plot. "
                    "Please choose a less restrictive filtering."
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="modal_close_button", className="ml-auto")
                ),
            ],
            centered=True,
            id="modal",
        ),
    ],
    fluid=True,  # fill available horizontal space and resize fluidly
    # style={"height": "90vh"},
)


#%%


def key_is_in_list_case_insensitive(lst, key):
    return any([key.lower() in s.lower() for s in lst])


@app.callback(
    Output("dropdown_file_selection", "value"),
    Input("dropdown_file_selection", "value"),
)
def update_dropdown_when_Select_All(dropdown_file_selection):
    if key_is_in_list_case_insensitive(dropdown_file_selection, "Select all"):
        dropdown_file_selection = fit_results.shortnames
    elif key_is_in_list_case_insensitive(dropdown_file_selection, "Deselect"):
        dropdown_file_selection = mydash.elements.get_shortnames_each(
            fit_results.shortnames
        )
    return dropdown_file_selection


def append_to_list_if_exists(d, key, value):
    if key in d:
        d[key].append(value)
    else:
        d[key] = [value]


def apply_tax_id_filter(d_filter, tax_id_filter_input):
    if tax_id_filter_input is None or len(tax_id_filter_input) == 0:
        return None

    for tax in tax_id_filter_input:
        if tax in fit_results.all_tax_ids:
            append_to_list_if_exists(d_filter, "tax_ids", tax)
        elif tax in fit_results.all_tax_names:
            append_to_list_if_exists(d_filter, "tax_names", tax)
        elif tax in fit_results.all_tax_ranks:
            append_to_list_if_exists(d_filter, "tax_ranks", tax)
        else:
            raise AssertionError(f"Tax {tax} could not be found. ")


def apply_tax_id_descendants_filter(d_filter, tax_name, tax_id_filter_subspecies):
    if tax_name is None:
        return None

    tax_ids = taxonomy.extract_descendant_tax_ids(
        tax_name,
        include_subspecies=include_subspecies(tax_id_filter_subspecies),
    )
    N_tax_ids = len(tax_ids)
    if N_tax_ids != 0:
        if "tax_id" in d_filter:
            d_filter["tax_ids"].extend(tax_ids)
        else:
            d_filter["tax_ids"] = tax_ids


@app.callback(
    Output("store", "data"),
    Output("modal", "is_open"),
    Input("dropdown_file_selection", "value"),
    Input("tax_id_filter_input", "value"),
    Input("tax_id_plot_button", "n_clicks"),
    Input({"type": "dynamic_slider", "index": ALL}, "value"),
    Input({"type": "slider_overview_marker_size", "index": ALL}, "value"),
    Input({"type": "dropdown_overview_marker_transformation", "index": ALL}, "value"),
    Input("modal_close_button", "n_clicks"),
    State("tax_id_filter_input_descendants", "value"),
    State("tax_id_filter_subspecies", "value"),
    State({"type": "dynamic_slider", "index": ALL}, "id"),
    State("modal", "is_open"),
    # State("tabs", "active_tab"),
)
def filter_fit_results(
    dropdown_file_selection,
    tax_id_filter_input,
    tax_id_button,
    slider_values,
    marker_size_max,
    marker_transformation,
    n_clicks_modal,
    tax_id_filter_input_descendants,
    tax_id_filter_subspecies,
    slider_ids,
    modal_is_open,
    # active_tab,
    prevent_initial_call=True,
):

    # if modal is open and the "close" button is clicked, close down modal
    if n_clicks_modal and modal_is_open:
        return dash.no_update, False

    # if no files selected
    if not dropdown_file_selection:
        raise PreventUpdate

    with about_time() as at1:

        fit_results.set_marker_size(marker_transformation, marker_size_max)

        d_filter = {"shortnames": dropdown_file_selection}
        slider_names = [id["index"] for id in slider_ids]
        for shortname, values in zip(slider_names, slider_values):
            d_filter[shortname] = values

        apply_tax_id_filter(
            d_filter,
            tax_id_filter_input,
        )

        apply_tax_id_descendants_filter(
            d_filter,
            tax_id_filter_input_descendants,
            tax_id_filter_subspecies,
        )

        df_fit_results_filtered = fit_results.filter(d_filter)

    # print(f"Time taken to filter : {at1.duration_human}")

    # raise modal warning if no results due to too restrictive filtering
    if len(df_fit_results_filtered) == 0:
        return dash.no_update, True

    return df_fit_results_filtered.to_dict("records"), dash.no_update


#%%


def list_is_none_or_empty(l):
    return l is None or len(l) == 0


def get_id_dict(child):
    return child["props"]["id"]


def find_index_in_children(children, id_type, search_index):
    for i, child in enumerate(children):
        d_id = get_id_dict(child)
        if d_id["type"] == id_type and d_id["index"] == search_index:
            return i


def get_current_names(current_ids):
    return [x["index"] for x in current_ids if x]


def slider_is_added(current_names, dropdown_names):
    "Returns True if a new slider is added, False otherwise"
    return set(current_names).issubset(dropdown_names)


def get_name_of_added_slider(current_names, dropdown_names):
    return list(set(dropdown_names).difference(current_names))[0]


def get_name_of_removed_slider(current_names, dropdown_names):
    return list(set(current_names).difference(dropdown_names))[0]


def remove_name_from_children(column, children, id_type):
    " Given a column, remove the corresponding child element from children"
    index = find_index_in_children(children, id_type=id_type, search_index=column)
    children.pop(index)


def get_slider_name(column, low_high):
    if isinstance(low_high, dict):
        low = low_high["min"]
        high = low_high["max"]
    elif isinstance(low_high, (tuple, list)):
        low = low_high[0]
        high = low_high[1]

    if column in mydash.utils.log_transform_columns:
        low = mydash.utils.log_transform_slider(low)
        high = mydash.utils.log_transform_slider(high)

    low = utils.human_format(low)
    high = utils.human_format(high)

    return f"{column}: [{low}, {high}]"


def make_new_slider(column, id_type, N_steps=100):

    d_range_slider = mydash.elements.get_range_slider_keywords(
        fit_results,
        column=column,
        N_steps=N_steps,
    )

    return dbc.Container(
        [
            dbc.Row(html.Br()),
            dbc.Row(
                html.P(
                    get_slider_name(column, d_range_slider),
                    id={"type": "dynamic_slider_name", "index": column},
                ),
                justify="center",
            ),
            dbc.Row(
                dbc.Col(
                    dcc.RangeSlider(
                        id={"type": "dynamic_slider", "index": column},
                        **d_range_slider,
                    ),
                    width=12,
                ),
            ),
        ],
        id={"type": id_type, "index": column},
    )


@app.callback(
    Output("dynamic_slider-container", "children"),
    Input("dropdown_slider", "value"),
    State("dynamic_slider-container", "children"),
    State({"type": "dynamic_slider", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def add_or_remove_slider(
    dropdown_names,
    children,
    current_ids,
):

    id_type = "dbc"

    current_names = get_current_names(current_ids)

    # add new slider
    if slider_is_added(current_names, dropdown_names):
        column = get_name_of_added_slider(current_names, dropdown_names)
        new_element = make_new_slider(column, id_type=id_type)
        children.append(new_element)

    # remove selected slider
    else:
        column = get_name_of_removed_slider(current_names, dropdown_names)
        remove_name_from_children(column, children, id_type=id_type)

    return children


@app.callback(
    Output({"type": "dynamic_slider_name", "index": MATCH}, "children"),
    Input({"type": "dynamic_slider", "index": MATCH}, "value"),
    State({"type": "dynamic_slider", "index": MATCH}, "id"),
    prevent_initial_call=True,
)
def update_slider_name(dynamic_slider_values, dynamic_slider_name):
    column = dynamic_slider_name["index"]
    name = get_slider_name(column, dynamic_slider_values)
    return name


#%%


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
    Input("dropdown_mismatch_tax_id", "value"),
    Input("dropdown_mismatch_shortname", "value"),
)
def update_mismatch_plot(active_tab, dropdown_mismatch_tax_id, dropdown_name):

    if dropdown_mismatch_tax_id is None:
        if active_tab == "fig_histograms":
            s = "Does not work for binned data"
            return mydash.figures.create_empty_figure(s=s)
        else:
            return mydash.figures.create_empty_figure()

    try:
        group = fit_results.get_single_count_group(
            shortname=dropdown_name,
            tax_id=dropdown_mismatch_tax_id,
        )
        fit = fit_results.get_single_fit_prediction(
            shortname=dropdown_name,
            tax_id=dropdown_mismatch_tax_id,
        )
        chosen_mismatch_columns = ["C→T", "G→A"]
        fig = mydash.figures.plot_count_fraction(
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
        tax_id = fit_results.parse_click_data(click_data, column="tax_id")
        shortname = fit_results.parse_click_data(click_data, column="shortname")
        df_fit_results_filtered = fit_results.filter(
            {"shortname": shortname, "tax_id": tax_id}
        )
        return df_fit_results_filtered.to_dict("records")

    # when selecting histogram without customdata
    except KeyError:
        raise PreventUpdate


#%%


def get_tax_id_options_based_on_shortname(shortname, df_string="df_fit_results"):
    """df_string is a string, eg. df_fit_results or  df_counts.
    The 'df_' part is optional
    """
    tax_ids = sorted(
        fit_results.load_df_counts_shortname(shortname, columns="tax_id")[
            "tax_id"
        ].unique()
    )
    options = [{"label": i, "value": i} for i in tax_ids]
    return options


@app.callback(
    Output("dropdown_mismatch_tax_id", "options"),
    Input("dropdown_mismatch_shortname", "value"),
)
def update_dropdown_tax_id_options(shortname):
    # if shortname is None:
    # print("update_dropdown_tax_id_options got shortname==None")
    return get_tax_id_options_based_on_shortname(shortname, df_string="counts")


@app.callback(
    Output("dropdown_mismatch_shortname", "value"),
    Output("dropdown_mismatch_tax_id", "value"),
    Input("main_graph", "clickData"),
    State("tabs", "active_tab"),
)
def update_dropdowns_based_on_click_data(click_data, active_tab):
    if click_data is not None:
        if active_tab == "fig_histograms":
            # print("update_dropdowns_based_on_click_data got here")
            raise PreventUpdate
        try:
            tax_id = fit_results.parse_click_data(click_data, column="tax_id")
            shortname = fit_results.parse_click_data(click_data, column="shortname")
            return shortname, tax_id
        except KeyError:
            # print("update_dropdowns_based_on_click_data got KeyError")
            raise PreventUpdate
            # return None, None
    else:
        # print("update_dropdowns_based_on_click_data got click_data == None")
        raise PreventUpdate


#%%


def include_subspecies(subspecies):
    if len(subspecies) == 1:
        return True
    return False


@app.callback(
    Output("tax_id_filter_counts_output", "children"),
    # Input("tax_id_filter_button", "n_clicks"),
    Input("tax_id_filter_input_descendants", "value"),
    Input("tax_id_filter_subspecies", "value"),
)
def update_tax_id_filter_counts(tax_name, subspecies):

    if tax_name is None or tax_name == "":
        return f"No specific Tax IDs selected, defaults to ALL."
        # raise PreventUpdate

    tax_ids = taxonomy.extract_descendant_tax_ids(
        tax_name,
        include_subspecies=include_subspecies(subspecies),
    )
    N_tax_ids = len(tax_ids)
    if N_tax_ids == 0:
        return f"Couldn't find any Tax IDs for {tax_name} in NCBI"
    return f"Found {utils.human_format(N_tax_ids)} Tax IDs for {tax_name} in NCBI"


#%%


@app.callback(
    Output("filters_dropdown_files", "is_open"),
    Output("filters_toggle_files_button", "outline"),
    Input("filters_toggle_files_button", "n_clicks"),
    State("filters_dropdown_files", "is_open"),
)
def toggle_collapse_files(n, is_open):
    # after click
    if n:
        return not is_open, is_open
    # initial setup
    return is_open, True


@app.callback(
    Output("filters_dropdown_tax_ids", "is_open"),
    Output("filters_toggle_tax_ids_button", "outline"),
    Input("filters_toggle_tax_ids_button", "n_clicks"),
    State("filters_dropdown_tax_ids", "is_open"),
)
def toggle_collapse_tax_ids(n, is_open):
    if n:
        return not is_open, is_open
    return is_open, True


@app.callback(
    Output("filters_dropdown_ranges_button", "is_open"),
    Output("filters_toggle_ranges_button", "outline"),
    Input("filters_toggle_ranges_button", "n_clicks"),
    State("filters_dropdown_ranges_button", "is_open"),
)
def toggle_collapse_ranges(n, is_open):
    if n:
        return not is_open, is_open
    return is_open, True


#%%


def is_ipython():
    return hasattr(__builtins__, "__IPYTHON__")


if __name__ == "__main__" and not is_ipython():
    app.run_server(debug=True)
    # app.run_server(debug=False)


else:

    mydash.elements.get_range_slider_keywords(
        fit_results,
        column="D_max",
        N_steps=100,
    )
    mydash.elements.get_range_slider_keywords(
        fit_results,
        column="n_sigma",
        N_steps=100,
    )

    shortname = "KapK-12-1-24-Ext-1-Lib-1-Index2"
    # shortname = "SJArg-1-Nit"
    tax_id = 131567

    df_fit_results_all = fit_results.df_fit_results

    df_fit_results = fit_results.filter({"tax_id": tax_id, "shortname": shortname})

    group = fit_results.get_single_count_group(shortname=shortname, tax_id=tax_id)
    fit = fit_results.get_single_fit_prediction(shortname=shortname, tax_id=tax_id)

    chosen_mismatch_columns = ["C→T", "G→A"]

    # # #%%

    fig = mydash.figures.plot_count_fraction(group, chosen_mismatch_columns, fit=fit)
    fig

    group[["position", "CT", "C"]]

    reload(taxonomy)
    tax_name = "Ursus"
    tax_name = "Mammalia"
    # tax_name = "Chordata"
    # tax_name = "Salmon"
    tax_ids = taxonomy.extract_descendant_tax_ids(tax_name)

    df_fit_results = fit_results.filter({"tax_ids": tax_ids})

    df = df_fit_results_all.query("y_sum_total > 1_000_000")
    N_shortnames = df.shortname.nunique()
    symbol_sequence = [str(i) for i in range(N_shortnames)]

    fig2 = mydash.figures.plot_fit_results(fit_results, df)
    # mydash.figures.plot_histograms(fit_results, df)
    # mydash.figures.plot_scatter_matrix(fit_results, df)
    # mydash.figures.plot_forward_reverse(fit_results, df)

    mydash.elements.get_dropdown_file_selection(
        id="dropdown_file_selection",
        fit_results=fit_results,
        shortnames_to_show="each",
    )

    fit_results.all_tax_ranks
    fit_results.all_tax_names
    fit_results.all_tax_ids

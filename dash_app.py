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
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# First Party
from metadamage import utils, mydash

#%%

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
                dbc.Col(
                    mydash.elements.get_card_filter_and_dropdown(fit_results), md=2
                ),
                dbc.Col(mydash.elements.get_card_main_plot(fit_results), md=8),
            ],
        ),
        mydash.elements.get_card_datatable(fit_results),
    ],
    fluid=True,
)


@app.callback(
    Output("range-slider-N-alignments-output", "children"),
    Input("range-slider-N-alignments", "value"),
)
def update_markdown_N_alignments(slider_range):
    low, high = slider_range
    low = mydash.utils.transform_slider(low)
    high = mydash.utils.transform_slider(high)

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
            return dash_datatable.create_empty_dataframe_for_datatable(s).to_dict(
                "records"
            )

    else:
        return mydash.datatable.create_empty_dataframe_for_datatable().to_dict(
            "records"
        )


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
        fig_empty = mydash.figures.create_empty_figures(width=width, height=height)
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

    fig_fit_results = mydash.figures.create_fit_results_figure(
        fit_results,
        df_filtered,
    )
    fig_histograms = mydash.figures.create_histograms_figure(
        fit_results,
        df_filtered,
    )
    fig_scatter_matrix = mydash.figures.create_scatter_matrix_figure(
        fit_results,
        df_filtered,
    )
    fig_forward_reverse = mydash.figures.create_forward_reverse_figure(
        fit_results,
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


def is_ipython():
    return hasattr(__builtins__, "__IPYTHON__")


if __name__ == "__main__" and not is_ipython():
    # Timer(1, open_browser).start()
    app.run_server(debug=True)

#%%
else:
    df = fit_results.df
    fig_fit_results = mydash.figures.create_fit_results_figure(fit_results, df)
    fig_fit_results
# %%

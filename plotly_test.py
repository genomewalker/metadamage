import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

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


#%%

# https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
cmap = px.colors.qualitative.D3

d_cmap = {}
for i, (name, _) in enumerate(df.groupby("name", sort=False)):
    d_cmap[name] = cmap[i]

# cmap = px.colors.qualitative.Dark24

#%%

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

df

#%%

custom_data_columns = ["name", "taxid", "N_alignments_str", "N_sum_total_str"]
hovertemplate = (
    "<b>%{customdata[0]}</b><br><br>"
    "taxid: %{customdata[1]}<br>"
    "<br>n sigma: %{x:5.2f}<br>"
    "D max:    %{y:.2f}<br>"
    "<br>N alignments: %{customdata[2]}<br>"
    "N sum total:   %{customdata[3]}<br>"
    "<extra></extra>"
)


#%%

fig_fit_results = px.scatter(
    df,
    x="n_sigma",
    y="D_max",
    size="N_alignments_sqrt",
    color="name",
    hover_name="name",
    size_max=30,
    opacity=0.2,
    color_discrete_sequence=cmap,
    custom_data=custom_data_columns,
)


fig_fit_results.update_yaxes(range=[0, 1])
fig_fit_results.update_traces(hovertemplate=hovertemplate, marker_line_width=0)


fig_fit_results.update_layout(
    title="Fit Results",
    xaxis_title=r"$\Large n_\sigma$",
    yaxis_title=r"$\Large D_\mathrm{max}$",
    font_size=16,
    legend=dict(title="Files", title_font_size=20, font_size=16),
    width=1400,
    height=800,
)


fig_fit_results.write_html(
    "./figures/fig_fit_results.html",
    include_mathjax="cdn",
    auto_open=False,
)


#%%

dimensions = ["D_max", "n_sigma", "q_mean"]
labels_list = [r"$D max$", r"$n_{\sigma}$", r"$q$"]
labels = {dimension: label for dimension, label in zip(dimensions, labels_list)}

fig_scatter_matrix = px.scatter_matrix(
    df,
    dimensions=dimensions,
    color="name",
    hover_name="name",
    size_max=10,
    color_discrete_sequence=cmap,
    labels=labels,
    opacity=0.1,
    custom_data=custom_data_columns,
)

fig_scatter_matrix.update_traces(
    diagonal_visible=False,
    showupperhalf=False,
    hovertemplate=hovertemplate,
)
fig_scatter_matrix.update_layout(legend_title="Files", font_size=16)


fig_scatter_matrix.update_layout(
    title="Scatter Matrix",
    font_size=12,
    title_font_size=40,
    legend=dict(title="Files", title_font_size=20, font_size=16),
    width=1400,
    height=800,
)


fig_scatter_matrix.write_html(
    "./figures/fig_scatter_matrix.html",
    include_mathjax="cdn",
    auto_open=False,
)
# %%



# fig["layout"].pop("updatemenus") # optional, drop animation buttons

fig.write_html(
    "./figures/fig_test.html",
    include_mathjax="cdn",
    auto_open=False,
)
# %%

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

app = dash.Dash(__name__)

N_alignments_min = df.N_alignments.min()
N_alignments_max = df.N_alignments.max()
N_steps = 100
step = (N_alignments_max - N_alignments_min) // N_steps

app.layout = html.Div(
    [
        dcc.Graph(id="scatter-plot"),
        html.P("N alignments:"),
        dcc.RangeSlider(
            id="range-slider",
            min=N_alignments_min,
            max=N_alignments_max,
            step=step,
            marks={
                N_alignments_min: str(N_alignments_min),
                N_alignments_max: str(N_alignments_max),
            },
            value=[N_alignments_min, N_alignments_max],
        ),
    ]
)


@app.callback(Output("scatter-plot", "figure"), [Input("range-slider", "value")])
def update_bar_chart(slider_range):
    low, high = slider_range

    range_x = [
        df.n_sigma.min() - np.abs(df.n_sigma.min() / 10),
        df.n_sigma.max() + np.abs(df.n_sigma.max() / 10),
    ]

    fig = px.scatter(
        df.query(f"{low} <= N_alignments <= {high}"),
        x="n_sigma",
        y="D_max",
        size="N_alignments_sqrt",
        color="name",
        hover_name="name",
        size_max=30,
        range_x=range_x,
        range_y=[0, 1],
        opacity=0.2,
        color_discrete_sequence=cmap,
        custom_data=custom_data_columns,
    )

    return fig


app.run_server(debug=True)

# %%

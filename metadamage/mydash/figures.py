# Scientific Library
import numpy as np
import pandas as pd

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#%%


def create_fit_results_figure(fit_results, df_filtered):

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
        xaxis_title=r"$\Large n_\sigma$",
        yaxis_title=r"$\Large D_\mathrm{max}$",
        title_text="Fit Results",
        legend_title="Files",
    )

    return fig


#%%


def plotly_histogram(
    fit_results,
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


def create_histograms_figure(fit_results, df_filtered):

    fig = make_subplots(rows=2, cols=4)

    showlegend = True

    for dimension, row, column in fit_results.iterate_over_dimensions():
        highest_y_max = 0
        for name, group in df_filtered.groupby("name", sort=False):
            trace, y_max = plotly_histogram(
                fit_results=fit_results,
                data=group[dimension],
                name=name,
                dimension=dimension,
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
        title_text="1D Histograms",
        legend_title="Files",
    )

    return fig


#%%


def create_scatter_matrix_figure(fit_results, df_filtered):

    fig = px.scatter_matrix(
        df_filtered,
        dimensions=fit_results.dimensions,
        color="name",
        hover_name="name",
        size_max=10,
        color_discrete_map=fit_results.d_cmap,
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
        title_text="Scatter Matrix",
        legend_title="Files",
    )

    return fig


#%%


def create_forward_reverse_figure(fit_results, df_filtered):

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
        title_text="Forward / Reverse",
        legend_title="Files",
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

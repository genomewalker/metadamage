# Scientific Library
import numpy as np

# Standard Library
import webbrowser

# Third Party
from PIL import ImageColor
import plotly.graph_objects as go
import plotly.io as pio


def set_custom_theme():

    pio.templates["custom_template"] = go.layout.Template(
        layout=go.Layout(
            font_size=16,
            title_font_size=30,
            legend=dict(
                # title="",
                title_font_size=20,
                font_size=16,
                itemsizing="constant",
                itemclick=False,
                itemdoubleclick=False,
            ),
            hoverlabel_font_family="Monaco, Lucida Console, Courier, monospace",
            dragmode="zoom",
            # width=width,
            # height=height,
            # uirevision=True,
            # margin=dict(
            #     t=50,  # top margin: 30px
            #     b=20,  # bottom margin: 10px
            # ),
        )
    )

    # pio.templates.default = "plotly_white"
    pio.templates.default = "simple_white+custom_template"

    return None


#%%


def log_transform_slider(x):
    return np.where(x < 0, 0, 10 ** np.clip(x, 0, a_max=None))


def open_browser():
    # webbrowser.open_new("http://localhost:8050")
    webbrowser.open("http://localhost:8050")


#%%


def hex_to_rgb(hex_string, opacity=1):
    rgb = ImageColor.getcolor(hex_string, "RGB")
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"

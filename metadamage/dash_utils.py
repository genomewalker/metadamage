import plotly.io as pio
import plotly.graph_objects as go


def set_custom_theme():

    pio.templates["custom_template"] = go.layout.Template(
        layout=go.Layout(
            font_size=16,
            title_font_size=30,
            legend=dict(
                title="Files",
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


def transform_slider(x):
    return 10 ** x


import webbrowser
def open_browser():
    # webbrowser.open_new("http://localhost:8050")
    webbrowser.open("http://localhost:8050")


from dash_html_components import Div
from dash_core_components import Checklist


new_list = [
    Div(
        children=[
            Checklist(
                id={"index": 0, "type": "done"},
                options=[{"label": "", "value": "done"}],
                value=[],
                style={"display": "inline"},
                labelStyle={"display": "inline"},
            ),
            Div(
                children="1",
                id={"index": 0},
                style={"display": "inline", "margin": "10px"},
            ),
        ],
        style={"clear": "both"},
    ),
    Div(
        children=[
            Checklist(
                id={"index": 1, "type": "done"},
                options=[{"label": "", "value": "done"}],
                value=[],
                style={"display": "inline"},
                labelStyle={"display": "inline"},
            ),
            Div(
                children="2",
                id={"index": 1},
                style={"display": "inline", "margin": "10px"},
            ),
        ],
        style={"clear": "both"},
    ),
    Div(
        children=[
            Checklist(
                id={"index": 2, "type": "done"},
                options=[{"label": "", "value": "done"}],
                value=[],
                style={"display": "inline"},
                labelStyle={"display": "inline"},
            ),
            Div(
                children="3",
                id={"index": 2},
                style={"display": "inline", "margin": "10px"},
            ),
        ],
        style={"clear": "both"},
    ),
]
list_out = [
    [
        Div(
            children=[
                Checklist(
                    id={"index": 0, "type": "done"},
                    options=[{"label": "", "value": "done"}],
                    value=[],
                    style={"display": "inline"},
                    labelStyle={"display": "inline"},
                ),
                Div(
                    children="1",
                    id={"index": 0},
                    style={"display": "inline", "margin": "10px"},
                ),
            ],
            style={"clear": "both"},
        ),
        Div(
            children=[
                Checklist(
                    id={"index": 1, "type": "done"},
                    options=[{"label": "", "value": "done"}],
                    value=[],
                    style={"display": "inline"},
                    labelStyle={"display": "inline"},
                ),
                Div(
                    children="2",
                    id={"index": 1},
                    style={"display": "inline", "margin": "10px"},
                ),
            ],
            style={"clear": "both"},
        ),
        Div(
            children=[
                Checklist(
                    id={"index": 2, "type": "done"},
                    options=[{"label": "", "value": "done"}],
                    value=[],
                    style={"display": "inline"},
                    labelStyle={"display": "inline"},
                ),
                Div(
                    children="3",
                    id={"index": 2},
                    style={"display": "inline", "margin": "10px"},
                ),
            ],
            style={"clear": "both"},
        ),
    ],
    "",
]

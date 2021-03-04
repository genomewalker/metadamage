import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import json

#%%

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
    # prevent_initial_callbacks=True,
)

N_steps = 10

slider_kw_args = {
    "A": {
        "min": 0.0,
        "max": 1.0,
        "step": 0.1,
        "marks": {
            0.25: "0.25",
            0.5: "0.5",
            0.75: "0.75",
            0: {"label": "No Min.", "style": {"color": "#a3ada9"}},
            1: {"label": "No Max.", "style": {"color": "#a3ada9"}},
        },
        "value": [0.0, 1.0],
        "allowCross": False,
        "updatemode": "mouseup",
        "included": True,
    },
    "B": {
        "min": 0.0,
        "max": 100.0,
        "step": 10.0,
        "marks": {
            25: "25",
            50: "50",
            75: "75",
            0: {"label": "No Min.", "style": {"color": "#a3ada9"}},
            100: {"label": "No Max.", "style": {"color": "#a3ada9"}},
        },
        "value": [0.0, 100.0],
        "allowCross": False,
        "updatemode": "mouseup",
        "included": True,
    },
}

#%%

app.layout = dbc.Container(
    [
        dbc.Row("Dash Test"),
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="dropdown",
                    options=[
                        {"label": name, "value": name} for name in slider_kw_args.keys()
                    ],
                    value=[],
                    multi=True,
                    placeholder="Select a variable to filter on...",
                ),
                width=12,
            ),
        ),
        dbc.Row(dbc.Col(id="dynamic-slider-container", children=[], width=12)),
        dbc.Row(dbc.Col(id="dynamic-slider-summary", children=[], width=12)),
    ]
)


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


def remove_name_from_children(name, children, id_type):
    " Given a name, remove the corresponding child element from children"
    index = find_index_in_children(children, id_type=id_type, search_index=name)
    children.pop(index)


#%%


def make_new_slider(name, id_type):
    return dbc.Row(
        [
            dbc.Col(
                html.Div(
                    id={
                        "type": "dynamic-output",
                        "index": name,
                    }
                ),
            ),
            dbc.Col(
                dcc.RangeSlider(
                    id={
                        "type": "dynamic-slider",
                        "index": name,
                    },
                    **slider_kw_args[name],
                ),
                width=12,
            ),
        ],
        id={"type": id_type, "index": name},
    )


@app.callback(
    Output("dynamic-slider-container", "children"),
    Input("dropdown", "value"),
    State("dynamic-slider-container", "children"),
    State({"type": "dynamic-slider", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def add_or_remove_slider(
    dropdown_names,
    children,
    current_ids,
):

    # id_type = "div"
    id_type = "dbc"

    current_names = get_current_names(current_ids)

    # add new slider
    if slider_is_added(current_names, dropdown_names):
        name = get_name_of_added_slider(current_names, dropdown_names)
        new_element = make_new_slider(name, id_type=id_type)
        children.append(new_element)

    # remove selected slider
    else:
        name = get_name_of_removed_slider(current_names, dropdown_names)
        remove_name_from_children(name, children, id_type=id_type)

    return children


@app.callback(
    Output({"type": "dynamic-output", "index": MATCH}, "children"),
    Input({"type": "dynamic-slider", "index": MATCH}, "value"),
    State({"type": "dynamic-slider", "index": MATCH}, "id"),
)
def update_slider_title(value, id):
    return html.Div(f"Dropdown {id['index']} = {value}")


@app.callback(
    Output("dynamic-slider-summary", "children"),
    Input({"type": "dynamic-slider", "index": ALL}, "value"),
    State({"type": "dynamic-slider", "index": ALL}, "id"),
)
def update_slider_summary(slider_values, ids):
    if list_is_none_or_empty(slider_values):
        return "No sliders yet"
    names = [id["index"] for id in ids]
    s = ""
    for name, slider in zip(names, slider_values):
        s += f"Name {name} = {slider} \n"
    return s


#%%

if __name__ == "__main__":
    app.run_server(debug=True)

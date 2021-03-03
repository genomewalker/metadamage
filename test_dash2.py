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


app.layout = html.Div(
    [
        html.Div("Dash Test"),
        # dcc.Input(id="input", autoComplete="off"),
        dcc.Dropdown(
            id="dropdown",
            # id={"type": "dropdown", "index": 0},
            options=[{"label": name, "value": name} for name in slider_kw_args.keys()],
            # value=["A"],
            value=[],
            multi=True,
        ),
        # dbc.Container(
        #     id="sliders",
        # ),
        html.Div(id="dynamic-slider-container", children=[]),
    ]
)


#%%


# def get_title(name, minval, maxval):
#     return name + f": [{minval}, {maxval}]"


# def iteratively_search_for_index(d):

#     if d is None:
#         return d

#     if "index" in d:
#         return d["index"]

#     if "id" in d:
#         return iteratively_search_for_index(d["id"])

#     if "props" in d:
#         return iteratively_search_for_index(d["props"])

#     if "children" in d:
#         return iteratively_search_for_index(d["children"])

#     return None


# def find_all_sliders(children):
#     if children is None:
#         return {}
#     out = {iteratively_search_for_index(child) for child in children}
#     out.remove(None)
#     return out


def get_recursively(search_dict, field):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    """
    fields_found = []

    for key, value in search_dict.items():

        if key == field:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_recursively(value, field)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_recursively(item, field)
                    for another_result in more_results:
                        fields_found.append(another_result)

    return fields_found


def find_index_in_children(children, search_index):
    for i, child in enumerate(children):
        indices = get_recursively(child, "index")
        if len(set(indices)) != 1:
            raise AssertionError("Something fishy here")
        if search_index in set(indices):
            return i


def get_current_names(current_ids):
    return [x["index"] for x in current_ids if x]


@app.callback(
    Output("dynamic-slider-container", "children"),
    Input("dropdown", "value"),
    State("dynamic-slider-container", "children"),
    State({"type": "dynamic-slider", "index": ALL}, "id"),
)
def add_slider(
    dropdown_names,
    children,
    current_ids,
):

    if dropdown_names is None or len(dropdown_names) == 0:
        raise PreventUpdate

    # print("")
    # print("")
    # print("")
    # print("")
    current_names = get_current_names(current_ids)
    # print(current_names)
    # print("")
    # print(dropdown_names)
    # print("")
    # print(children)
    # print("")

    if set(current_names).issubset(dropdown_names):

        # name to add
        name = list(set(dropdown_names).difference(current_names))[0]

        new_element = html.Div(
            [
                html.Div(
                    id={
                        "type": "dynamic-output",
                        "index": name,
                    }
                ),
                dcc.RangeSlider(
                    id={
                        "type": "dynamic-slider",
                        "index": name,
                    },
                    **slider_kw_args[name],
                ),
            ]
        )
        children.append(new_element)
        # print(children)
        return children
    else:

        # name to remove
        name = list(set(current_names).difference(dropdown_names))[0]
        # print(name)
        index = find_index_in_children(children, search_index=name)
        children.pop(index)
        return children


@app.callback(
    Output({"type": "dynamic-output", "index": MATCH}, "children"),
    Input({"type": "dynamic-slider", "index": MATCH}, "value"),
    State({"type": "dynamic-slider", "index": MATCH}, "id"),
)
def display_output(value, id):
    return html.Div("Dropdown {} = {}".format(id["index"], value))


#%%

if __name__ == "__main__":
    app.run_server(debug=True)


#%%


# %%

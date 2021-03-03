import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
)

app.layout = html.Div(
    [
        html.Div("Dash To-Do list"),
        dcc.Input(id="input", autoComplete="off"),
        html.Button("Add", id="add_button"),
        html.Button("Clear Done", id="clear_button"),
        html.Div(id="item_list_out"),
        html.Div(id="summary_out"),
    ]
)

style_todo = {"display": "inline", "margin": "10px"}
style_done = {"textDecoration": "line-through", "color": "#888"}
style_done.update(style_todo)


@app.callback(
    [
        Output("item_list_out", "children"),
        Output("input", "value"),
    ],
    [
        Input("add_button", "n_clicks"),
        Input("input", "n_submit"),
        Input("clear_button", "n_clicks"),
    ],
    [
        State("input", "value"),
        State({"index": ALL}, "children"),
        State({"index": ALL, "type": "done"}, "value"),
    ],
)
def edit_list(
    add,
    input_enter,
    clear,
    new_item,
    items,
    items_done,
):
    print("")

    # print(dash.callback_context.triggered)
    triggered = [t["prop_id"] for t in dash.callback_context.triggered]
    print(f"{triggered=}")

    adding = len(
        [1 for i in triggered if i in ("add_button.n_clicks", "input.n_submit")]
    )
    print(f"{adding=}")

    clearing = len([1 for i in triggered if i == "clear_button.n_clicks"])
    print(f"{clearing=}")

    new_spec = [
        (text, done) for text, done in zip(items, items_done) if not (clearing and done)
    ]
    print(f"{new_spec=}")

    if adding:
        new_spec.append((new_item, []))

    new_list = []
    for i, (text, done) in enumerate(new_spec):
        print(f"{i=}", f"{text=}", f"{done=}")
        div = html.Div(
            [
                dcc.Checklist(
                    id={"index": i, "type": "done"},
                    options=[{"label": "", "value": "done"}],
                    value=done,
                    style={"display": "inline"},
                    labelStyle={"display": "inline"},
                ),
                html.Div(
                    text, id={"index": i}, style=style_done if done else style_todo
                ),
            ],
            style={"clear": "both"},
        )
        new_list.append(div)

    print(f"{new_list=}")

    list_out = [new_list, "" if adding else new_item]
    print(f"{list_out=}")

    return list_out


@app.callback(
    Output({"index": MATCH}, "style"),
    Input({"index": MATCH, "type": "done"}, "value"),
)
def change_style_for_done(done):
    return style_done if done else style_todo


@app.callback(
    Output("summary_out", "children"),
    Input({"index": ALL, "type": "done"}, "value"),
)
def update_summary_out(done):
    count_all = len(done)
    count_done = len([d for d in done if d])
    result = f"{count_done} of {count_all} items completed"
    if count_all:
        result += f" - {count_done / count_all:.0%}"
    return result


if __name__ == "__main__":
    app.run_server(debug=True)

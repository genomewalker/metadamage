# Third Party
import dash
from dash.dependencies import ALL, Input, MATCH, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

#%%

style_todo = {"display": "inline", "margin": "10px"}
style_done = {"textDecoration": "line-through", "color": "#888"}
style_done.update(style_todo)


#%%BOOTSTRAP

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Card(
    dbc.Container(
        [
            dbc.Row(html.Div("Dash To-Do list")),
            dbc.Row(
                [
                    dbc.Col(dbc.Input(id="new-item"), width=1),
                    dbc.Col(dbc.Button("Add", id="add"), width=1),
                    dbc.Col(dbc.Button("Clear Done", id="clear-done"), width=1),
                ]
                # justify="center",
            ),
            dbc.Row(html.Div(id="list-container")),
            dbc.Row(html.Div(id="totals")),
        ],
        fluid=True,  # fill available horizontal space and resize fluidly
        # style={"height": "10vh", "widht": "10%"},
    )
)


#%%


@app.callback(
    [
        Output("list-container", "children"),
        Output("new-item", "value"),
    ],
    [
        Input("add", "n_clicks"),
        Input("new-item", "n_submit"),
        Input("clear-done", "n_clicks"),
    ],
    [
        State("new-item", "value"),
        State({"index": ALL}, "children"),
        State({"index": ALL, "type": "done"}, "value"),
    ],
)
def edit_list(add, add2, clear, new_item, items, items_done):
    print("edit_list", add, add2, clear, new_item, items, items_done)
    triggered = [t["prop_id"] for t in dash.callback_context.triggered]
    adding = len([1 for i in triggered if i in ("add.n_clicks", "new-item.n_submit")])
    clearing = len([1 for i in triggered if i == "clear-done.n_clicks"])
    new_spec = [
        (text, done) for text, done in zip(items, items_done) if not (clearing and done)
    ]

    if adding:
        new_spec.append((new_item, []))

    new_list = [
        html.Div(
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
        for i, (text, done) in enumerate(new_spec)
    ]
    return [new_list, "" if adding else new_item]


@app.callback(
    Output({"index": MATCH}, "style"),
    Input({"index": MATCH, "type": "done"}, "value"),
)
def mark_done(done):
    print("mark_done", done)
    return style_done if done else style_todo


@app.callback(
    Output("totals", "children"),
    Input({"index": ALL, "type": "done"}, "value"),
)
def show_totals(done):
    print("show_totals", done)
    count_all = len(done)
    count_done = len([d for d in done if d])
    result = "{} of {} items completed".format(count_done, count_all)
    if count_all:
        result += " - {}%".format(int(100 * count_done / count_all))
    return result


if __name__ == "__main__":
    app.run_server(debug=True)

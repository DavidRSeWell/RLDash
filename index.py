import dash_core_components as dcc
import dash_html_components as html
import dash_table

from dash.dependencies import Input, Output,State
from app import app
from apps import k_arm_bandit

app.layout = html.Div(children=[

        html.Div([

            html.Div([

                html.Div([

                    html.A(
                        html.Button('K-Arm Bandit', type="button", className="btn btn-primary")
                        , href="/k_arm"),

                ], className="btn-group-vertical")

            ],className="col-sm-2"),

            html.Div([

                dcc.Location(id='url', refresh=False),
                html.Div(id='page-content')

            ], className="col-sm-10")


        ],className="row"),

    ],style = {"margin-top":"50px","margin-left":"0px","margin-right":"0px"}, className="container")


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):

    if pathname == '/k_arm':

        return k_arm_bandit.layout

    else:
        return "404"



if __name__ == '__main__':

    app.run_server(debug=True)
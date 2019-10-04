# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly_express as px
import plotly.graph_objs as go

from Algorithms.BasicRL.k_arm_bandit import run_e_greedy_bandit
from app import app
from dash.dependencies import Input,Output,State
from scipy.stats import norm


tips = px.data.tips()
col_options = [dict(label=x, value=x) for x in tips.columns]
dimensions = ["x", "y", "color", "facet_col", "facet_row"]

layout_posterior = go.Layout(

    xaxis={'title': 'X'},
    yaxis={'title': 'Reward dist', 'range': [-1, 1]},

)

fig = dict(data=[],layout=layout_posterior)

layout = html.Div([

    html.Div([

        html.Div([

            html.Form([

                html.Div([

                    html.Label('Lever number'),
                    dcc.Slider(
                        id='lever-number',
                        min=1,
                        max=10,
                        step=1,
                        marks={i: str(i) for i in range(1, 11)},
                        value=5,
                    ),

                ],className="form-group"),


                html.Div([

                    html.Label('Init Q'),
                    dcc.Slider(
                        id='init-q-value',
                        min=0,
                        max=5,
                        step=1,
                        marks={i: str(i) for i in range(1, 6)},
                        value=0,
                    ),

                ],className="form-group"),

                html.Div([

                    html.Label('Epsilon'),
                    dcc.Slider(
                        id='epsilon-value',
                        min=0,
                        max=1,
                        step=0.1,
                        marks={i/10.0: str(i/10.0) for i in range(1, 11)},
                        value=0,
                    ),

                ],className="form-group"),

                html.Div([

                    dcc.Dropdown(
                        options=[
                            {'label': 'e-greedy', 'value': 'greedy'},
                            {'label': 'UCB', 'value': 'UCB'},
                            {'label': 'e-greedy dynamic', 'value': 'greedy-2'}
                        ],
                        value='greedy',
                        id='algorithm-select'
                    )
                ],style={"margin-top":"25px"})


            ],id='model-parameter-form',style={"margin-bottom":"25px"}),


            html.Div([
                html.Button(id='submit-button', n_clicks=0, type="button",children='Sample', className="btn btn-primary"),
                html.Button(id='run-model', n_clicks=0, type="button", children='Run', className="btn btn-primary"),
                html.Button(id='toggle-model', n_clicks=0, type="button", children='Toggle', className="btn btn-primary")

            ],style={"margin-top":"25px"})


        ],className="col-sm-12"),



    ],className="row"),

    html.Div([

        html.Div([

            html.H3(children="Reward distribution",className='text-center'),

            dcc.Graph(id='reward-distribution', figure=fig)

        ],className="col-sm-12")

    ],className="row",style={"margin-top":"10px"}),

    ###############
    # Reward Graph
    ###############
    html.Div([

        html.Div([

            html.H3(children="Reward Graph",className='text-center'),

            dcc.Graph(id='reward-graph',style={"width": "75%", "display": "inline-block"})

        ],className="col-sm-12")

    ],className="row",style={"margin-top":"10px"}),

    html.Div([

        html.Div([

            html.H3('Lever Counts')

        ],className="col-sm-12",id="datatable-lever-holder")

    ],className="row",style={"margin-top":"10px"}),

    html.Div(id='interval-container'),

    # Hidden div inside the app that stores the intermediate value
    html.Div(id='model-state',children='stop'),

    html.Div(id='run-model-hidden', style={'display': 'none'}),
    html.Div(id='place-holder-div', style={'display': 'none'}),


],style = {"margin-top":"50px","margin-left":"0px","margin-right":"0px"},className="container",id='main-container')


#################
# Lever Graph
#################
@app.callback(
    dash.dependencies.Output('reward-distribution','figure'),
    [dash.dependencies.Input('submit-button','n_clicks')],
    [dash.dependencies.State('lever-number','value')]
)
def update_current_sample(n_clicks,input1):

    traces = []
    for i in range(input1):

        # get random mean
        rand_mean = np.random.uniform(0,5)

        y_norm = norm.rvs(loc=rand_mean,size=1000)

        name = 'lever' + str(i) + ' mean: ' + str(rand_mean)

        tracei = {
            "type":"violin",
            "y":y_norm,
            "box": {
                "visible": True
            },
            "name":name,
            "mean":rand_mean,
            "std":1
        }

        traces.append(tracei)


    layout_posterior = go.Layout(

        xaxis={'title': 'lever number'},
        yaxis={'title': 'Reward dist', 'range': [-1, 1]},

    )

    fig = dict(data=traces)

    return fig



#################
# Run Model
#################
@app.callback(
    Output('reward-graph','figure'),
    [Input('run-model', 'n_clicks')],
    [State('reward-distribution','figure'),
     State('epsilon-value', 'value'),
     State('init-q-value', 'value'),
     State('algorithm-select', 'value')]
)
def run_model(n_click,reward_figure,epsilon_value,init_q_value,algo_type):

    print("Running run_model task")

    epochs = 1000
    if n_click <= 0:
        data = {}
        df = pd.DataFrame(data)

    else:
        #result = k_arm_bandit.delay(hdf_file_path, reward_figure, epochs, epsilon_value,init_q_value,algo_type)

        dset, avg_reward, q_values, count_values = run_e_greedy_bandit(reward_figure,epochs,epsilon_value,init_q_value,algo_type)

        x = [i for i in range(len(dset))]

        data = {
            'epoch':x,
            'avg_reward':dset
        }
        df = pd.DataFrame(data)

        fig = px.scatter(df,x='epoch',y='avg_reward')


    return px.scatter(df,x='epoch',y='avg_reward',height=700)

    print('')
    print("Done running model")
    #print("avg reward: " + str(avg_reward))

'''
#################
# Reward Graph
#################

@app.callback(
    Output('reward-graph', 'figure'),
    [Input('datatable-file', 'selected_row_indices')],
    [State('datatable-file', 'rows')])
def update_figure(selected_row_indices,rows):

    dff = pd.DataFrame(rows)

    traces = []

    for i in (selected_row_indices or []):

        file_name = str(rows[i]['files'])

        file_path = '/Users/befeltingu/RLResearch/Data/k_arm_bandit/' + file_name

        f = h5py.File(file_path, 'r', libver='latest',
                      swmr=True)

        dset = f["avg_reward"][:]  # fetch all the datas

        init_q = f["init_q"].value

        num_bandits = file_name.split('_')[1]

        epsilon = f['epsilon'].value

        label_name = "Levers={levers} epsilon={epsilon} q0={init_q}".format(levers=num_bandits,epsilon=epsilon,init_q=init_q)

        trace = go.Scatter(
                    x=[x for x in range(len(dset))],
                    y=dset,
                    name=label_name

                )

        traces.append(trace)

        f.close()

    figure = {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'iteration'},
            yaxis={'title': 'loss'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            #legend={'x': 0, 'y': 1},
        )
    }

    return figure


@app.callback(
    Output('datatable-lever-holder', 'children'),
    [Input('datatable-file', 'selected_row_indices')],
    [State('datatable-file', 'rows')])
def update_figure(selected_row_indices,rows):

    df = pd.DataFrame(rows)

    q_values = []

    for i in (selected_row_indices or []):

        file_name = str(rows[i]['files'])

        file_path = '/Users/befeltingu/RLResearch/Data/k_arm_bandit/' + file_name

        f = h5py.File(file_path, 'r', libver='latest',
                      swmr=True)

        count_vaues = f['lever_count'][:]

        q_values.append([file_name] + list(count_vaues))


        f.close()

    if len(q_values) == 0:

        table = dt.DataTable(
            rows=[],
            # optional - sets the order of columns
            columns=[],
            row_selectable=True,
            filterable=True,
            sortable=True,
            selected_row_indices=[],
            id='datatable-lever-counts'
        )

        return table


    columns = ["lever_name"] + ["lever_" + str(i) for i in range(len(q_values[0]) - 1)]

    q_df = pd.DataFrame(q_values,columns=columns)

    table = dt.DataTable(
        rows=q_df.to_dict('records'),
        # optional - sets the order of columns
        columns=columns,
        row_selectable=True,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
        id='datatable-lever-counts'
    )

    return table




#################
# File DataTable
#################
@app.callback(
    dash.dependencies.Output('datatable-file','rows'),
    [],
    [],
    [Event('fetch-files','click')]
)
def populate_file_container():


    file_list = []
    for file in os.listdir('/Users/befeltingu/RLResearch/Data/k_arm_bandit/'):
        file_list.append(file)

    file_df = pd.DataFrame({'files':file_list})


    return file_df.to_dict('records')

@app.callback(
    dash.dependencies.Output('place-holder-div','children'),
    [],
    [],
    [Event('delete-files','click')]
)
def populate_file_container():

    for file in os.listdir('/Users/befeltingu/RLResearch/Data/k_arm_bandit/'):

        os.remove('/Users/befeltingu/RLResearch/Data/k_arm_bandit/' + file)

    return ''
'''


'''
@app.callback(
    dash.dependencies.Output('interval-container','children'),
    [],
    [],
    [Event('run-model', 'click')]
)
def run_model():

    return dcc.Interval(
        id='interval-component',
        interval=2 * 1000,  # in milliseconds
        n_intervals=0
    )'''

'''
@app.callback(
    dash.dependencies.Output('model-state','children'),
    [],
    [dash.dependencies.State('model-state','children')],
    [Event('toggle-model', 'click')]
)
def toggle_model(model_state):

    if model_state == 'stop':
        return "start"

    else:
        return "stop"
'''


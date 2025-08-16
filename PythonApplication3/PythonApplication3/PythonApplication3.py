import dash
from dash import dcc, html, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import io
import base64

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "CSV Joiner"

app.layout = dbc.Container([
    html.H1("CSV Join Tool", className="mt-3"),

    dbc.Row([
        dbc.Col([
            html.H5("Upload CSV 1"),
            dcc.Upload(
                id='upload-data1',
                children=html.Div(['Drag & Drop or ', html.A('Select File')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                       'borderWidth': '1px', 'borderStyle': 'dashed',
                       'borderRadius': '5px', 'textAlign': 'center'},
                multiple=False
            ),
            html.Div(id='file1-name', style={'marginTop': 5}),
        ], width=6),

        dbc.Col([
            html.H5("Upload CSV 2"),
            dcc.Upload(
                id='upload-data2',
                children=html.Div(['Drag & Drop or ', html.A('Select File')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                       'borderWidth': '1px', 'borderStyle': 'dashed',
                       'borderRadius': '5px', 'textAlign': 'center'},
                multiple=False
            ),
            html.Div(id='file2-name', style={'marginTop': 5}),
        ], width=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Join Type"),
            dcc.Dropdown(
                id='join-type',
                options=[
                    {'label': 'Inner', 'value': 'inner'},
                    {'label': 'Left', 'value': 'left'},
                    {'label': 'Right', 'value': 'right'},
                    {'label': 'Outer', 'value': 'outer'}
                ],
                value='inner'
            )
        ], width=3),

        dbc.Col([
            html.Label("Join Columns - CSV 1"),
            dcc.Dropdown(id='join-col1', placeholder="Select columns from CSV 1", multi=True)
        ], width=4),

        dbc.Col([
            html.Label("Join Columns - CSV 2"),
            dcc.Dropdown(id='join-col2', placeholder="Select columns from CSV 2", multi=True)
        ], width=4),
    ], className="mb-4"),

    dbc.Button("Join Files", id="join-btn", color="primary", className="mb-3"),
    html.Hr(),

    html.Div(id='table-container'),

    dcc.Download(id="download-joined"),
    dbc.Button("Download Joined CSV", id="download-btn", color="success", className="mt-3")
])

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))

@app.callback(
    [Output('file1-name', 'children'),
     Output('join-col1', 'options'),
     Output('upload-data1', 'contents')],
    Input('upload-data1', 'contents'),
    State('upload-data1', 'filename')
)
def handle_upload1(contents, filename):
    if contents is None:
        return no_update, [], None
    df = parse_contents(contents)
    return f"Uploaded: {filename}", [{'label': c, 'value': c} for c in df.columns], contents

@app.callback(
    [Output('file2-name', 'children'),
     Output('join-col2', 'options'),
     Output('upload-data2', 'contents')],
    Input('upload-data2', 'contents'),
    State('upload-data2', 'filename')
)
def handle_upload2(contents, filename):
    if contents is None:
        return no_update, [], None
    df = parse_contents(contents)
    return f"Uploaded: {filename}", [{'label': c, 'value': c} for c in df.columns], contents

@app.callback(
    Output('table-container', 'children'),
    Input('join-btn', 'n_clicks'),
    State('upload-data1', 'contents'),
    State('upload-data2', 'contents'),
    State('join-col1', 'value'),
    State('join-col2', 'value'),
    State('join-type', 'value')
)
def join_csvs(n_clicks, file1, file2, cols1, cols2, join_type):
    if not n_clicks or not file1 or not file2 or not cols1 or not cols2:
        raise dash.exceptions.PreventUpdate

    if not isinstance(cols1, list) or not isinstance(cols2, list):
        return html.Div("Please select at least one column on each side for join.", style={'color': 'red'})

    df1 = parse_contents(file1)
    df2 = parse_contents(file2)

    try:
        df_joined = df1.merge(df2, left_on=cols1, right_on=cols2, how=join_type)
    except Exception as e:
        return html.Div(f"Error during join: {e}", style={'color': 'red'})

    return dash_table.DataTable(
        data=df_joined.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df_joined.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        filter_action='native',
        sort_action='native'
    )

@app.callback(
    Output("download-joined", "data"),
    Input("download-btn", "n_clicks"),
    State('upload-data1', 'contents'),
    State('upload-data2', 'contents'),
    State('join-col1', 'value'),
    State('join-col2', 'value'),
    State('join-type', 'value'),
    prevent_initial_call=True
)
def download_csv(n_clicks, file1, file2, cols1, cols2, join_type):
    if not file1 or not file2:
        raise dash.exceptions.PreventUpdate
    if not isinstance(cols1, list) or not isinstance(cols2, list):
        raise dash.exceptions.PreventUpdate
    df1 = parse_contents(file1)
    df2 = parse_contents(file2)
    df_joined = df1.merge(df2, left_on=cols1, right_on=cols2, how=join_type)
    return dcc.send_data_frame(df_joined.to_csv, "joined.csv", index=False)

if __name__ == '__main__':
    app.run_server(debug=True)

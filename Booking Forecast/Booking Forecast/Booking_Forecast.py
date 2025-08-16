# ota_forecast_dashboard.py
# Dash app: Upload OTA daily sales CSV and generate forecasts (Prophet, SARIMA, Holt-Winters, Naive-seasonal)
# Two independent blocks:
#  - Block 1: Data Exploration (upload, inspect actuals, descriptive stats, filters)
#  - Block 2: Forecasting (choose model, horizon, run forecast on full or subset, per-category charts, export CSV)

import base64
import io
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# Optional: Prophet
HAVE_PROPHET = False
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


# -------------------------------
# Helpers
# -------------------------------

def parse_csv_contents(contents: str) -> pd.DataFrame:
    if not contents:
        return pd.DataFrame()
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.BytesIO(decoded))
    except Exception as e:
        # try with utf-8 decode
        s = decoded.decode('utf-8', errors='replace')
        df = pd.read_csv(io.StringIO(s))

    # Normalize variety of possible column names
    rename_map = {
        'bk_category': 'BK_CATEGORY', 'category': 'BK_CATEGORY',
        'cns_channel': 'CNS_CHANNEL', 'channel': 'CNS_CHANNEL', 'CNS_CHANNEL_BUDGET': 'CNS_CHANNEL',
        'epm_geography': 'EPM_GEOGRAPHY', 'country': 'EPM_GEOGRAPHY',
        'bk_date': 'BK_DATE', 'date': 'BK_DATE',
        'count': 'COUNT', 'bookings': 'COUNT', 'qty': 'COUNT', 'BK_COUNT': 'COUNT'
    }
    df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)

    required = {'BK_CATEGORY', 'CNS_CHANNEL', 'EPM_GEOGRAPHY', 'BK_DATE'}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # If COUNT is not present, assume row-level booking and set COUNT=1
    if 'COUNT' not in df.columns:
        df['COUNT'] = 1

    df['BK_DATE'] = pd.to_datetime(df['BK_DATE'])
    df['COUNT'] = pd.to_numeric(df['COUNT'], errors='coerce').fillna(0).astype(float)
    
    # Proactively fill missing categorical data to prevent sorting errors later
    for col in ['BK_CATEGORY', 'CNS_CHANNEL', 'EPM_GEOGRAPHY']:
        if col in df.columns:
            df[col].fillna('Unknown', inplace=True)

    return df


def ensure_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    daily = df.groupby('BK_DATE', as_index=False)['COUNT'].sum().sort_values('BK_DATE')
    if daily.empty:
        return pd.DataFrame(columns=['BK_DATE', 'COUNT'])
    full = pd.date_range(daily['BK_DATE'].min(), daily['BK_DATE'].max(), freq='D')
    daily = daily.set_index('BK_DATE').reindex(full).fillna(0.0).rename_axis('BK_DATE').reset_index()
    return daily


# Forecasting engines (reuse previous implementations)

def fc_holt_winters(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    y = daily.set_index('BK_DATE')['COUNT']
    model = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=7, initialization_method='estimated')
    fit = model.fit(optimized=True)
    fitted = fit.fittedvalues
    fc_values = fit.forecast(horizon)
    future_idx = pd.date_range(y.index.max() + pd.Timedelta(days=1), periods=horizon, freq='D')
    fc = pd.DataFrame({'BK_DATE': list(y.index) + list(future_idx), 'FC': list(fitted) + list(fc_values)})
    return fc


def fc_sarima(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    y = daily.set_index('BK_DATE')['COUNT'].asfreq('D').fillna(0.0)
    orders = [(1,1,1), (2,1,2), (1,0,1)]
    sorders = [(1,1,1), (0,1,1), (1,0,1)]
    speriods = [7]
    best = (None, float('inf'))
    best_res = None
    for order in orders:
        for sorder in sorders:
            for sp in speriods:
                try:
                    model = SARIMAX(y, order=order, seasonal_order=(*sorder, sp), enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    if res.aic < best[1]:
                        best = (res, res.aic)
                        best_res = res
                except Exception:
                    continue
    if best_res is None:
        return fc_holt_winters(daily, horizon)
    fitted = best_res.fittedvalues.asfreq('D')
    forecast = best_res.get_forecast(steps=horizon).predicted_mean
    future_idx = pd.date_range(y.index.max() + pd.Timedelta(days=1), periods=horizon, freq='D')
    fc = pd.DataFrame({'BK_DATE': list(y.index) + list(future_idx), 'FC': list(fitted) + list(forecast)})
    return fc


def fc_prophet(daily: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not HAVE_PROPHET:
        raise RuntimeError('Prophet not installed')
    dfp = daily.rename(columns={'BK_DATE': 'ds', 'COUNT': 'y'})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=horizon, freq='D')
    fc = m.predict(future)
    out = fc[['ds', 'yhat']].rename(columns={'ds': 'BK_DATE', 'yhat': 'FC'}).sort_values('BK_DATE').reset_index(drop=True)
    return out


def fc_naive_seasonal(daily: pd.DataFrame, horizon: int, seasonal_period=7) -> pd.DataFrame:
    y = daily.set_index('BK_DATE')['COUNT'].asfreq('D').fillna(0.0)
    if y.empty:
         # Cannot forecast on empty data, return empty forecast df
        future_idx = pd.date_range(start=pd.Timestamp.now(tz='UTC').normalize(), periods=horizon, freq='D')
        return pd.DataFrame({'BK_DATE': future_idx, 'FC': [0]*horizon})
        
    if len(y) < seasonal_period:
        mean_val = y.mean() if not y.empty else 0
        future_idx = pd.date_range(y.index.max() + pd.Timedelta(days=1), periods=horizon, freq='D')
        fc_vals = [mean_val] * horizon
    else:
        last_season = y[-seasonal_period:]
        repeats = int(np.ceil(horizon / seasonal_period))
        fc_vals = (list(last_season) * repeats)[:horizon]
        future_idx = pd.date_range(y.index.max() + pd.Timedelta(days=1), periods=horizon, freq='D')
    fitted = y
    fc = pd.DataFrame({'BK_DATE': list(y.index) + list(future_idx), 'FC': list(fitted) + list(fc_vals)})
    return fc


def run_forecast_engine(daily: pd.DataFrame, model: str, horizon: int) -> pd.DataFrame:
    if daily.empty or len(daily) < 14:
        return fc_naive_seasonal(daily, horizon)
    if model == 'Holt-Winters':
        return fc_holt_winters(daily, horizon)
    if model == 'SARIMA':
        return fc_sarima(daily, horizon)
    if model == 'Prophet':
        return fc_prophet(daily, horizon)
    if model == 'Naive-Seasonal':
        return fc_naive_seasonal(daily, horizon)
    raise ValueError('Unknown model')


# -------------------------------
# Dash app layout
# -------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = 'OTA Forecast (Exploration + Models)'

app.layout = dbc.Container([
    html.H2('OTA Sales Explorer & Forecast'),
    dcc.Store(id='stored-data'),
    dcc.Store(id='stored-forecast-data'),
    html.H3('Block 1: Data Exploration'),
    dcc.Upload(id='upload-data', children=html.Div(['Drag & Drop or ', html.A('Select CSV')]),
               style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center'}),
    html.Div(id='upload-message', style={'marginTop': '8px'}),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='explore-category', placeholder='Category (multi)', multi=True), md=4),
        dbc.Col(dcc.Dropdown(id='explore-channel', placeholder='Channel (multi)', multi=True), md=4),
        dbc.Col(dcc.Dropdown(id='explore-geo', placeholder='Geography (multi)', multi=True), md=4),
    ], className='mb-2'),
    dcc.DatePickerRange(id='explore-dates', display_format='YYYY-MM-DD'),
    dcc.Graph(id='explore-actuals'),
    html.Div(id='explore-stats'),
    dash_table.DataTable(id='explore-table', page_size=10, style_table={'overflowX': 'auto'}),
    html.Hr(),
    html.H3('Block 2: Forecasting'),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='forecast-category', placeholder='Filter by Category', multi=True), md=4),
        dbc.Col(dcc.Dropdown(id='forecast-channel', placeholder='Filter by Channel', multi=True), md=4),
        dbc.Col(dcc.Dropdown(id='forecast-geo', placeholder='Filter by Geo', multi=True), md=4),
    ], className='mb-2'),
    dbc.Row([
        dbc.Col(dcc.DatePickerRange(id='forecast-dates', display_format='YYYY-MM-DD'), md=12)
    ], className='mb-2'),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='model-select', options=[
            {'label': 'Holt-Winters', 'value': 'Holt-Winters'},
            {'label': 'SARIMA', 'value': 'SARIMA'},
            {'label': 'Prophet', 'value': 'Prophet'},
            {'label': 'Naive-Seasonal', 'value': 'Naive-Seasonal'}
        ], value='Holt-Winters'), md=3),
        dbc.Col(dcc.Input(id='horizon', type='number', min=7, max=365, value=90, placeholder='Horizon (days)'), md=2),
        dbc.Col(dcc.Dropdown(id='training-window', options=[
            {'label': 'All History', 'value': 'ALL'},
            {'label': 'Trailing 12 months', 'value': 'TTM'},
            {'label': 'Trailing 24 months', 'value': 'TTM24'}
        ], value='TTM24'), md=3),
        # ADDED: Granularity dropdown
        dbc.Col(dcc.Dropdown(id='forecast-granularity', options=[
            {'label': 'Granularity: By Category', 'value': 'category'},
            {'label': 'Granularity: By Category & Channel', 'value': 'category_channel'},
            {'label': 'Granularity: Max Detail (All)', 'value': 'all'},
        ], value='category'), md=3),
        dbc.Col(dbc.Button('Run Forecast', id='run-forecast', color='primary', className='w-100'), md=1)
    ], className='mb-2'),
    dbc.Alert("Warning: Forecasting at max detail on a large, unfiltered dataset can be very slow.", color="warning", id='granularity-warning', is_open=False),
    
    dcc.Loading(children=[
        dcc.Graph(id='forecast-overall'),
        html.Div(id='forecast-category-charts'),
    ]),
    dash_table.DataTable(id='forecast-output-table', page_size=15, style_table={'overflowX': 'auto'}),
    html.Br(),
    dbc.Button('Download Forecast CSV', id='download-btn', color='success'),
    dcc.Download(id='download-forecast'),
    html.Hr(),
    html.P('Notes: For Prophet select it only if installed (pip install prophet).')
], fluid=True)


# -------------------------------
# Callbacks
# -------------------------------
@app.callback(
    Output('granularity-warning', 'is_open'),
    Input('forecast-granularity', 'value')
)
def show_granularity_warning(granularity):
    return granularity == 'all'


@app.callback(
    [Output('stored-data', 'data'),
     Output('upload-message', 'children'),
     Output('explore-category', 'options'),
     Output('explore-channel', 'options'),
     Output('explore-geo', 'options'),
     Output('explore-dates', 'start_date'),
     Output('explore-dates', 'end_date'),
     Output('forecast-category', 'options'),
     Output('forecast-channel', 'options'),
     Output('forecast-geo', 'options'),
     Output('forecast-dates', 'start_date'),
     Output('forecast-dates', 'end_date'),
    ],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    if not contents:
        return no_update
    try:
        df = parse_csv_contents(contents)
    except Exception as e:
        return None, html.Div(['Error parsing CSV: ', html.Code(str(e))]), [], [], [], None, None, [], [], [], None, None

    cats = [{'label': c, 'value': c} for c in sorted(df['BK_CATEGORY'].unique())]
    chs = [{'label': c, 'value': c} for c in sorted(df['CNS_CHANNEL'].unique())]
    geos = [{'label': c, 'value': c} for c in sorted(df['EPM_GEOGRAPHY'].unique())]
    min_date = df['BK_DATE'].min().date()
    max_date = df['BK_DATE'].max().date()

    return (df.to_json(date_format='iso', orient='records'), f'Loaded: {filename}',
            cats, chs, geos, min_date, max_date,
            cats, chs, geos, min_date, max_date)


@app.callback(
    [Output('explore-actuals', 'figure'),
     Output('explore-stats', 'children'),
     Output('explore-table', 'data'),
     Output('explore-table', 'columns')],
    [Input('stored-data', 'data'),
     Input('explore-category', 'value'),
     Input('explore-channel', 'value'),
     Input('explore-geo', 'value'),
     Input('explore-dates', 'start_date'),
     Input('explore-dates', 'end_date')]
)
def update_explore(json_data, cat, chan, geo, start_date, end_date):
    if not json_data:
        return go.Figure(), '', [], []
    df = pd.read_json(json_data, orient='records')
    df['BK_DATE'] = pd.to_datetime(df['BK_DATE'])

    dff = df.copy()
    if cat: dff = dff[dff['BK_CATEGORY'].isin(cat)]
    if chan: dff = dff[dff['CNS_CHANNEL'].isin(chan)]
    if geo: dff = dff[dff['EPM_GEOGRAPHY'].isin(geo)]
    if start_date and end_date:
        dff = dff[(dff['BK_DATE'].dt.date >= pd.to_datetime(start_date).date()) & (dff['BK_DATE'].dt.date <= pd.to_datetime(end_date).date())]

    daily = ensure_daily(dff)
    fig = go.Figure(layout={'template':'plotly_dark'})
    fig.update_layout(title='Daily Actual Sales')
    if not daily.empty:
        fig.add_trace(go.Scatter(x=daily['BK_DATE'], y=daily['COUNT'], mode='lines', name='Actual'))
    stats = {}
    if not daily.empty:
        s = daily['COUNT']
        stats = {'Total': float(s.sum()), 'Mean': float(s.mean()), 'Median': float(s.median()), 'Std': float(s.std()), 'Min': float(s.min()), 'Max': float(s.max()), 'Days': int(s.count())}
        stats_html = html.Ul([html.Li(f"{k}: {v:,.2f}" if isinstance(v, float) else f"{k}: {v}") for k, v in stats.items()])
    else:
        stats_html = ''

    dff_display = dff.copy()
    dff_display['BK_DATE'] = dff_display['BK_DATE'].dt.strftime('%Y-%m-%d')
    columns = [{'name': c, 'id': c} for c in dff_display.columns]
    data = dff_display.to_dict('records')
    return fig, stats_html, data, columns


@app.callback(
    [Output('forecast-overall', 'figure'),
     Output('forecast-category-charts', 'children'),
     Output('forecast-output-table', 'data'),
     Output('forecast-output-table', 'columns'),
     Output('stored-forecast-data', 'data')],
    Input('run-forecast', 'n_clicks'),
    [State('stored-data', 'data'),
     State('forecast-category', 'value'),
     State('forecast-channel', 'value'),
     State('forecast-geo', 'value'),
     State('forecast-dates', 'start_date'),
     State('forecast-dates', 'end_date'),
     State('model-select', 'value'),
     State('horizon', 'value'),
     State('training-window', 'value'),
     State('forecast-granularity', 'value')], # MODIFIED: Get granularity value
    prevent_initial_call=True
)
def run_forecast_callback(n_clicks, json_data, cat, chan, geo, start_date, end_date, model, horizon, training_window, granularity):
    try:
        if not json_data: return no_update

        df = pd.read_json(json_data, orient='records')
        df['BK_DATE'] = pd.to_datetime(df['BK_DATE'])

        dff = df.copy()
        if cat: dff = dff[dff['BK_CATEGORY'].isin(cat)]
        if chan: dff = dff[dff['CNS_CHANNEL'].isin(chan)]
        if geo: dff = dff[dff['EPM_GEOGRAPHY'].isin(geo)]
        if start_date and end_date:
            dff = dff[(dff['BK_DATE'].dt.date >= pd.to_datetime(start_date).date()) & (dff['BK_DATE'].dt.date <= pd.to_datetime(end_date).date())]

        daily_all = ensure_daily(dff)
        train = daily_all
        if not daily_all.empty:
            if training_window == 'TTM':
                last = daily_all['BK_DATE'].max()
                train = daily_all[daily_all['BK_DATE'] >= (last - pd.Timedelta(days=365))]
            elif training_window == 'TTM24':
                last = daily_all['BK_DATE'].max()
                train = daily_all[daily_all['BK_DATE'] >= (last - pd.Timedelta(days=730))]

        horizon = 90 if horizon is None else int(horizon)
        fc_overall = run_forecast_engine(train, model, horizon)
        
        overall_fig = go.Figure(layout={'template':'plotly_dark'})
        overall_fig.update_layout(title='Overall Forecast vs. Actuals (Filtered Scope)')
        if not daily_all.empty:
            overall_fig.add_trace(go.Scatter(x=daily_all['BK_DATE'], y=daily_all['COUNT'], mode='lines', name='Actual'))
        if not fc_overall.empty:
            overall_fig.add_trace(go.Scatter(x=fc_overall['BK_DATE'], y=fc_overall['FC'], mode='lines', name='Forecast', line=dict(dash='dash')))

        cat_children = []
        rows = []

        # MODIFIED: Determine grouping based on selected granularity
        if granularity == 'category':
            group_cols = ['BK_CATEGORY']
        elif granularity == 'category_channel':
            group_cols = ['BK_CATEGORY', 'CNS_CHANNEL']
        else: # 'all'
            group_cols = ['BK_CATEGORY', 'CNS_CHANNEL', 'EPM_GEOGRAPHY']
        
        if dff.empty:
            combinations = []
        else:
            combinations = dff[group_cols].drop_duplicates().sort_values(by=group_cols).itertuples(index=False)

        for combo in combinations:
            combo_dict = combo._asdict()
            query_parts = [f"`{col}` == '{val}'" for col, val in combo_dict.items()]
            query_string = " & ".join(query_parts)
            part = dff.query(query_string)

            daily_c = ensure_daily(part)
            
            train_c = daily_c
            if not daily_c.empty:
                if training_window == 'TTM':
                    last = daily_c['BK_DATE'].max()
                    train_c = daily_c[daily_c['BK_DATE'] >= (last - pd.Timedelta(days=365))]
                elif training_window == 'TTM24':
                    last = daily_c['BK_DATE'].max()
                    train_c = daily_c[daily_c['BK_DATE'] >= (last - pd.Timedelta(days=730))]
            
            fc_c = run_forecast_engine(train_c, model, horizon)
            
            fig = go.Figure(layout={'template':'plotly_dark'})
            if not daily_c.empty:
                fig.add_trace(go.Scatter(x=daily_c['BK_DATE'], y=daily_c['COUNT'], mode='lines', name='Actual'))
            if not fc_c.empty:
                fig.add_trace(go.Scatter(x=fc_c['BK_DATE'], y=fc_c['FC'], mode='lines', name='Forecast', line=dict(dash='dash')))

            title = ' | '.join(combo_dict.values())
            cat_children.append(html.Div([html.H5(title), dcc.Graph(figure=fig)]))

            if not fc_c.empty:
                actual_c = daily_c.rename(columns={'COUNT': 'Actual'})
                merged = actual_c.merge(fc_c.rename(columns={'FC': 'Forecast'}), on='BK_DATE', how='outer').sort_values('BK_DATE')
                for col, val in combo_dict.items():
                    merged[col] = val
                rows.append(merged)

        out_df = pd.DataFrame()
        if rows:
            out_df = pd.concat(rows).reset_index(drop=True)
            # Reorder columns for clarity
            cols_order = group_cols + [col for col in out_df.columns if col not in group_cols]
            out_df = out_df[cols_order]

        table_columns = [{'name': col, 'id': col} for col in out_df.columns]
        
        display_df = out_df.copy()
        if not display_df.empty:
            display_df['BK_DATE'] = display_df['BK_DATE'].dt.strftime('%Y-%m-%d')
            for col in ['Actual', 'Forecast']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map('{:,.2f}'.format)

        table_data = display_df.to_dict('records')
        forecast_json = out_df.to_json(date_format='iso', orient='records')

        return overall_fig, cat_children, table_data, table_columns, forecast_json

    except Exception as e:
        error_fig = go.Figure(layout={'template':'plotly_dark'})
        error_fig.add_annotation(text=f'An error occurred: {e}', showarrow=False)
        error_message = dbc.Alert(f"Failed to generate forecast. Error: {e}", color="danger")
        return error_fig, [error_message], [], [], None


@app.callback(
    Output('download-forecast', 'data'),
    Input('download-btn', 'n_clicks'),
    State('stored-forecast-data', 'data'),
    prevent_initial_call=True,
)
def download_csv(n_clicks, forecast_json):
    if not forecast_json:
        return no_update
    
    df_to_download = pd.read_json(forecast_json, orient='records')
    return dcc.send_data_frame(df_to_download.to_csv, "forecast_output.csv", index=False)


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
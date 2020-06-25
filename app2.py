from __future__ import print_function
import dash
import dash_core_components as dcc
import dash_html_components as html
import yfinance as yf
import datetime
import plotly.graph_objs as go
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from utils import dividends_splits, get_values, market_tail, \
    csv_saver, dividend_saver, tickers_from_market
from dash.dependencies import Output, Input
import dash_table as dt
from plotly.subplots import make_subplots
import numpy as np
from arch import arch_model
from openpyxl import load_workbook
import openpyxl
import xlrd
import plotly.express as px
import plotly.io as pio
from arch import arch_model
from statsmodels.tsa import stattools

pio.templates
pd.core.common.is_list_like = pd.api.types.is_list_like

tickers = ['^MXX',
           'LABB.MX',
           'BIMBOA.MX',
           'CUERVO.MX',
           'ASURB.MX',
           'ALPEKA.MX',
           'TLEVISACPO.MX',
           'GCARSOA1.MX',
           'MEGACPO.MX',
           'GENTERA.MX',
           'RA.MX',
           'BOLSAA.MX',
           'LIVEPOLC-1.MX',
           'GFNORTEO.MX',
           'GAPB.MX',
           'ALSEA.MX',
           'PINFRA.MX',
           'CEMEXCPO.MX',
           'KIMBERA.MX',
           'AC.MX',
           'IENOVA.MX',
           'GCC.MX',
           'BSMXB.MX',
           'BBAJIOO.MX',
           'GMEXICOB.MX',
           'GRUMAB.MX',
           'AMXL.MX',
           'PE&OLES.MX',
           'OMAB.MX']
tickers.sort()
#csv_saver(tickers)
#dividend_saver(tickers)

header_table_color = '#555555'

start = datetime.datetime.today() - relativedelta(years=5)
end = datetime.datetime.today()

app = dash.Dash()
app.layout = html.Div(style={'backgroundColor': '#111111', "border-color": "#111111"}, children=[

    html.Div([
        html.H1(children="BMV - Stock Analyzer",
                style={'display': 'center-block', 'font-family': 'verdana', 'padding-left': '800px',
                       'padding-bottom': '80px', 'color': 'white', 'text-decoration': 'underline'}),

        html.Div(id='market_table',
                 style={'font-family': 'verdana',
                        'display': 'center-block',
                        'padding-left': '155px', 'padding-bottom': '80px', 'width': '1600px'})

    ]),

    dcc.Dropdown(id='drop-down-tickers', options=[{'label': i, 'value': i} for i in tickers], value='AC.MX',
                 style={'font-family': 'verdana', 'width': '210px', 'padding-left': '80px'}),

    html.Div(dcc.Graph(id="graph_close")),

    html.Div(id='today_table',
             style={'font-family': 'verdana',
                    'width': '1200px', 'display': 'inline-block', 'margin-left': 'auto', 'margin-right': 'auto',
                    'padding-left': '80px', 'vertical-align': 'top', 'padding-top': '0px', 'padding-bottom': '0px',
                    'position': 'relative', 'bottom': 0, 'right': 0}),

    html.Div(id='dividend_table',
             style={'font-family': 'verdana',
                    'width': '400px', 'display': 'inline-block', 'margin-left': 'auto', 'margin-right': 'auto',
                    'padding-left': '80px', 'vertical-align': 'top', 'padding-top': '0px', 'padding-bottom': '0px',
                    'position': 'relative', 'bottom': 0, 'right': 0}),

    html.Div([

        dcc.RadioItems(id="window-checker", style={"padding-left": "80px", "padding-bottom": "20px",
                                                   'display': 'block',
                                                   'font-family': 'verdana', 'color': 'white'},
                       options=[
                           {'label': '5D', 'value': 5},
                           {'label': '15D', 'value': 15},
                           {'label': '1M', 'value': 20},
                           {'label': 'Q', 'value': 60},
                           {'label': 'Y', 'value': 252},
                           {'label': '2Y', 'value': 252 * 2},
                           {'label': '5Y', 'value': 252 * 5}
                       ], value=252 * 5),

        html.Div(id='stat_table',
                 style={'font-family': 'verdana',
                        'width': '800px', 'display': 'inline-block',
                        'padding-left': '80px', 'padding-top': '20px', 'padding-bottom': '50px'}),

        html.Div(id='info_table',
                 style={'font-family': 'verdana',
                        'width': '860px', 'display': 'inline-block',
                        'padding-left': '20px', 'padding-top': '20px', 'padding-bottom': '50px'}),

        html.Div([
            html.H2(children="Analysis of Returns",
                    style={'display': 'center-block', 'font-family': 'verdana', 'padding-left': '800px',
                           'padding-bottom': '20px', 'color': 'white', 'text-decoration': 'underline'}),

            html.H3(children="Returns of Asset over Time",
                    style={'display': 'center-block', 'font-family': 'verdana', 'padding-left': '120px',
                           'padding-bottom': '00px', 'color': 'white', 'display': 'inline-block'}),

            html.H3(children="Distribution of Returns",
                    style={'display': 'center-block', 'font-family': 'verdana', 'padding-left': '850px',
                           'padding-bottom': '0px', 'color': 'white', 'display': 'inline-block',
                           'position': 'relative'}),
        ]),

        html.Div(dcc.Graph(id="return-figure", style={'font-family': 'verdana', 'display': 'center-block',
                                                      'padding-left': '30px', 'padding-top': '1px',
                                                      'padding-bottom': '0px'})),

        html.Div([
            html.H2(children="Analysis of Volatility",
                    style={'display': 'center-block', 'font-family': 'verdana', 'padding-left': '800px',
                           'padding-bottom': '20px', 'color': 'white', 'text-decoration': 'underline'}),

            html.H3(children="Volatility over Time",
                    style={'display': 'center-block', 'font-family': 'verdana', 'padding-left': '120px',
                           'padding-bottom': '00px', 'color': 'white', 'display': 'inline-block'}),

        ]),

        html.Div(dcc.Graph(id="volatility-figure", style={'font-family': 'verdana', 'display': 'center-block',
                                                          'padding-left': '30px', 'padding-top': '0px',
                                                          'padding-bottom': '0px'})),

        html.Div([
            html.H2(children="Analysis of Correlation with Index",
                    style={'display': 'center-block', 'font-family': 'verdana', 'padding-left': '800px',
                           'padding-bottom': '20px', 'color': 'white', 'text-decoration': 'underline'}),

            html.H3(children="Correlation over Time",
                    style={'display': 'center-block', 'font-family': 'verdana', 'padding-left': '120px',
                           'padding-bottom': '00px', 'color': 'white', 'display': 'inline-block'}),

        ]),

        html.Div(dcc.Graph(id="correlation-figure", style={'font-family': 'verdana', 'display': 'center-block',
                                                           'padding-left': '30px', 'padding-top': '0px',
                                                           'padding-bottom': '30px'}))
    ])
])


@app.callback(dash.dependencies.Output('market_table', 'children'),
              [dash.dependencies.Input('drop-down-tickers', 'value')])
def market_table(input_value):
    columns = ["Open", "High", "Low", "Close", "Volume"]
    prices = pd.DataFrame(columns=columns)
    wb_tickers = xlrd.open_workbook("Tickers.xlsx")
    for ticker in tickers:
        sheet = wb_tickers.sheet_by_name(ticker)
        x = pd.DataFrame(sheet.row_values(-1)).T
        x_2 = pd.DataFrame(sheet.row_values(-2)).T
        last_two = x_2.append(x, ignore_index=True)
        last_two_change = last_two.pct_change()
        prices.loc[ticker, "Open"] = round(last_two.iloc[1][1], 4)
        prices.loc[ticker, "High"] = round(last_two.iloc[1][2], 4)
        prices.loc[ticker, "Low"] = round(last_two.iloc[1][3], 4)
        prices.loc[ticker, "Close"] = round(last_two.iloc[1][4], 4)
        prices.loc[ticker, "Volume"] = round(last_two.iloc[1][5], 4)
        prices.loc[ticker, "Chg. Close"] = str(round(last_two_change.loc[1, 4] * 100, 2)) + str('%')
        prices.loc[ticker, "Chg. Volume"] = str(round(last_two_change.loc[1, 5] * 100, 2)) + str('%')
        prices.append(prices)
        prices = prices.rename_axis("Company")
    prices.reset_index(level=0, inplace=True)
    data = prices.to_dict("rows")
    columns = [{"name": i, "id": i, } for i in prices.columns]
    return dt.DataTable(data=data, columns=columns, style_cell={'textAlign': 'center', 'font-family': 'verdana',
                                                                'backgroundColor': '#111111', 'color': 'white'},
                        style_as_list_view=True, style_header={'fontWeight': 'bold',
                                                               'backgroundColor': header_table_color},
                        style_table={'height': '400px', 'overflowY': 'auto'})


@app.callback(dash.dependencies.Output('graph_close', 'figure'),
              [dash.dependencies.Input('drop-down-tickers', 'value')])
def update_fig(input_value):
    wb_prices = pd.ExcelFile('Tickers.xlsx')
    price_data = wb_prices.parse(input_value)
    # MA Graphs
    df = pd.DataFrame(price_data)
    MA_1 = 20
    MA_2 = 100
    df['MA1'] = df.Close.rolling(MA_1).mean()
    df['MA2'] = df.Close.rolling(MA_2).mean()

    trace_candlestick = go.Figure(data=[go.Candlestick(x=price_data.Date,
                                                       open=price_data['Open'],
                                                       high=price_data['High'],
                                                       low=price_data['Low'],
                                                       close=price_data['Close']),
                                        go.Scatter(x=df['Date'], y=df.MA1, line=dict(color='orange', width=2),
                                                   name="MA %i" % MA_1),
                                        go.Scatter(x=df['Date'], y=df.MA2, line=dict(color='green', width=2),
                                                   name="MA %i" % MA_2)
                                        ])

    trace_candlestick.layout = dict(title=str(input_value), autosize=True, height=700,
                                    xaxis=dict(
                                        rangeselector=dict(bgcolor='#000000',
                                                           buttons=list([
                                                               dict(count=7,
                                                                    label="1W",
                                                                    step="day",
                                                                    stepmode="backward"),
                                                               dict(count=1,
                                                                    label="1M",
                                                                    step="month",
                                                                    stepmode="backward"),
                                                               dict(count=3,
                                                                    label="3M",
                                                                    step="month",
                                                                    stepmode="backward"),
                                                               dict(count=6,
                                                                    label="6M",
                                                                    step="month",
                                                                    stepmode="backward"),
                                                               dict(count=1,
                                                                    label="YTD",
                                                                    step="year",
                                                                    stepmode="todate"),
                                                               dict(count=1,
                                                                    label="1Y",
                                                                    step="year",
                                                                    stepmode="backward"),
                                                               dict(count=2,
                                                                    label="2Y",
                                                                    step="year",
                                                                    stepmode="backward"),
                                                               dict(count=5,
                                                                    label="5Y",
                                                                    step="year",
                                                                    stepmode="backward"),
                                                           ])
                                                           ), rangeslider=dict(visible=False), type="date"
                                    )
                                    )
    trace_candlestick.update_layout(showlegend=False, yaxis_title="Price")
    trace_candlestick.update_yaxes(tickprefix="$")

    # Add Volume Figure

    fig_volume = go.Figure(data=[go.Bar(x=price_data["Date"], y=price_data["Volume"], name="Volume",
                                        marker_color='white')])

    # Adding Traces
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.2], vertical_spacing=0.05)
    fig.add_traces([trace_candlestick.data[0]]) #trace_candlestick.data[1], trace_candlestick.data[2]]
    fig.add_traces([fig_volume.data[0]], [2], [1])
    fig.layout.update(trace_candlestick.layout)
    fig.layout.update(fig_volume.layout)
    # fig.update_yaxes(type="log")
    fig.layout.update(height=800)
    fig.layout.template = 'plotly_dark'
    return fig


@app.callback(dash.dependencies.Output('dividend_table', 'children'),
              [dash.dependencies.Input('drop-down-tickers', 'value')])
def update_dividend(input_value):
    wb_dividend = pd.ExcelFile('Dividend.xlsx')
    dividends = wb_dividend.parse(input_value)
    dividends.Date = pd.DatetimeIndex(dividends.Date).strftime("%Y-%m-%d")
    data = dividends.to_dict("rows")
    columns = [{"name": i, "id": i, } for i in dividends.columns]
    return dt.DataTable(data=data, columns=columns, style_cell={'textAlign': 'center', 'font-family': 'verdana',
                                                                'backgroundColor': '#111111', 'color': 'white'},
                        style_as_list_view=True, style_header={'fontWeight': 'bold',
                                                               'backgroundColor': header_table_color},
                        style_table={'height': '200px', 'overflowY': 'auto'})


@app.callback(dash.dependencies.Output('today_table', 'children'),
              [dash.dependencies.Input('drop-down-tickers', 'value')])
def update_today_data(input_value):
    wb_tickers = xlrd.open_workbook("Tickers.xlsx")
    sheet = wb_tickers.sheet_by_name(input_value)
    x = pd.DataFrame(sheet.row_values(-1)).T
    x_2 = pd.DataFrame(sheet.row_values(-2)).T
    last_two = x_2.append(x, ignore_index=True)
    last_two_change = last_two.pct_change()
    last_two_change.reset_index(level=0, inplace=True)
    today_table = pd.DataFrame(index=[''])
    today_table["Open"] = round(x.iloc[0][1], 2)
    today_table["High"] = round(x.iloc[0][2], 2)
    today_table["Low"] = round(x.iloc[0][3], 2)
    today_table["Close"] = round(x.iloc[0][4], 2)
    today_table["Volume"] = x.iloc[0][5]
    today_table["Chg. Close"] = str(round(last_two_change.iloc[1][4] * 100, 2)) + str('%')
    today_table["Chg. Volume"] = str(round(last_two_change.iloc[1][5] * 100, 2)) + str('%')
    data = today_table.to_dict("rows")
    columns = [{"name": i, "id": i, } for i in today_table.columns]
    return dt.DataTable(data=data, columns=columns, style_cell={'textAlign': 'center', 'font-family': 'verdana',
                                                                'backgroundColor': '#111111', 'color': 'white'},
                        style_as_list_view=True, style_header={'fontWeight': 'bold',
                                                               'backgroundColor': header_table_color},
                        style_table={'height': '100px', 'overflowY': 'auto'})


@app.callback(dash.dependencies.Output('return-figure', 'figure'),
              [dash.dependencies.Input('drop-down-tickers', 'value'),
               dash.dependencies.Input('window-checker', 'value')])
def update_returns_figure(input_value, window_value):
    wb_prices = pd.ExcelFile('Tickers.xlsx')

    price_data = wb_prices.parse(input_value).tail(window_value)
    price_data.reset_index(level=0, inplace=True)
    normalized = pd.DataFrame(columns=['Date', "Normalized Returns"])
    normalized['Date'] = price_data['Date']
    normalized['Normalized Returns'] = (price_data['Close'] / price_data['Close'][:1].values) - 1

    market_prices = wb_prices.parse('^MXX').tail(window_value)
    market_prices.reset_index(level=0, inplace=True)
    market = pd.DataFrame(columns=['Date', "Normalized Returns"])
    market['Date'] = market_prices['Date']
    market['Normalized Returns'] = (market_prices['Close'] / market_prices['Close'][:1].values) - 1

    trace_returns_figure = go.Figure()
    trace_returns_figure.add_trace(go.Scatter(x=normalized['Date'], y=(normalized['Normalized Returns']),
                                              name=(str(input_value) + str(" Returns")), marker_line_width=2,
                                              marker_line_color="blue"))

    trace_returns_figure.add_trace(go.Scatter(x=market['Date'], y=(market['Normalized Returns']),
                                              name="Market Returns", marker_line_width=2,
                                              marker_line_color="red"))
    trace_returns_figure.update_layout(showlegend=False, yaxis_title="Return", title="Historical Returns")

    log_returns = pd.DataFrame(columns=['Date', "Close"])
    log_returns['Date'] = price_data['Date']
    log_returns['Close'] = price_data['Close']
    log_returns['Log Returns'] = np.log(log_returns['Close']).diff()
    return_histo = go.Figure(data=[go.Histogram(x=log_returns['Log Returns'], histnorm='probability',
                                                name=(str(input_value) + str(" Frequency of Returns")),
                                                marker_color="blue", marker_line_width=1, marker_line_color="white")])
    return_histo.update_layout(showlegend=False, yaxis_title="Returns", title="Frequency")

    fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=0.025)
    fig2.add_traces([trace_returns_figure.data[0]], [1], [1])
    fig2.add_traces([trace_returns_figure.data[1]], [1], [1])
    fig2.add_traces([return_histo.data[0]], [1], [2])
    fig2.layout.update(height=500, title=str(input_value) + " Visual Look on Returns", yaxis_tickformat='.2%')
    fig2.layout.template = 'plotly_dark'
    return fig2


@app.callback(dash.dependencies.Output('stat_table', 'children'),
              [dash.dependencies.Input('drop-down-tickers', 'value'),
               dash.dependencies.Input('window-checker', 'value')])
def update_stat_table(input_value, window_value):
    wb_prices = pd.ExcelFile('Tickers.xlsx')
    price_data = wb_prices.parse(input_value)
    price_data.reset_index(level=0, inplace=True)
    ### RETURNS IMPLEMENTATION
    log_returns = pd.DataFrame(columns=['Close'])
    log_returns['Close'] = price_data['Close']
    log_returns['Log Returns'] = np.log(log_returns['Close']).diff().tail(window_value)

    ### GARCH MODEL IMPLEMENTATION
    data = pd.DataFrame(index=price_data.index, columns=['Returns'])
    data['Returns'] = price_data['Close'].pct_change() * 100
    data = data.dropna()
    aic_value = [arch_model(data, vol='Garch', p=lag, o=0, q=lag, dist='Normal').fit(disp='off').aic
                 for lag in range(1, 11)]
    lag = int(np.array(aic_value).argmin() + 1)
    model = arch_model(data, vol='Garch', p=lag, o=0, q=lag, dist='Normal')
    results = model.fit(update_freq=5)
    yhat = pd.DataFrame(results.forecast(horizon=20).variance.iloc[-1] ** 0.5)
    estimacion_vol = pd.DataFrame([yhat.iloc[i].values[0] for i in (0, 4, 9, 19)],
                                  index=['Tomorrow', 'W', '15D', 'Month'], columns=[input_value]).round(2)
    stat_measures = pd.DataFrame(index=["Return Mean Annualized"], columns=[""])
    ## Mean of Returns in Window Value
    stat_measures.loc["Return Mean Annualized"] = str(round(np.mean(log_returns['Log Returns'] * 252 * 100), 2)) \
                                                  + str('%')
    ## Expected Volatility
    stat_measures.loc["Expected Daily Volatility for Tomorrow"] = str(estimacion_vol.iat[0, 0]) + str('%')
    stat_measures.loc["Expected Daily Volatility at the Week"] = str(estimacion_vol.iat[1, 0]) + str('%')
    stat_measures.loc["Expected Daily Volatility at 15 days"] = str(estimacion_vol.iat[2, 0]) + str('%')
    stat_measures.loc["Expected Daily Volatility at Month"] = str(estimacion_vol.iat[3, 0]) + str('%')

    stat_measures.index.names = ['Stats Measures']
    stat_measures.reset_index(level=0, inplace=True)
    data = stat_measures.to_dict("rows")
    columns = [{"name": i, "id": i, } for i in stat_measures.columns]
    return dt.DataTable(data=data, columns=columns, style_cell={'textAlign': 'center', 'font-family': 'verdana',
                                                                'backgroundColor': '#111111', 'color': 'white'},
                        style_as_list_view=True, style_header={'fontWeight': 'bold',
                                                               'backgroundColor': header_table_color},
                        style_table={'overflowY': 'auto'})


@app.callback(dash.dependencies.Output('volatility-figure', 'figure'),
              [dash.dependencies.Input('drop-down-tickers', 'value'),
               dash.dependencies.Input('window-checker', 'value')])
def update_volatility_figure(input_value, window_value):
    wb_prices = pd.ExcelFile('Tickers.xlsx')
    price_data = wb_prices.parse(input_value)

    price_data.set_index('Date', inplace=True, drop=True)
    dailyret = pd.DataFrame(index=price_data.index, columns=['Returns'])
    dailyret['Returns'] = price_data['Close'].pct_change().dropna(axis=0)
    # %% Ventanas de Volatilidad
    vol_mov_5 = pd.DataFrame(dailyret.rolling(5).std().dropna()).rename(columns={"Returns": "Volatility Week"})
    vol_mov_15 = pd.DataFrame(dailyret.rolling(15).std().dropna()).rename(columns={"Returns": "Volatility 15 days"})
    vol_mov_20 = pd.DataFrame(dailyret.rolling(20).std().dropna()).rename(columns={"Returns": "Volatility Month"})
    vol_mov_90 = pd.DataFrame(dailyret.rolling(60).std().dropna()).rename(columns={"Returns": "Volatility Quarter"})
    vol_mov_365 = pd.DataFrame(dailyret.rolling(252).std().dropna()).rename(columns={"Returns": "Volatility Year"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol_mov_5.index, y=vol_mov_5['Volatility Week'],
                             mode='lines',
                             name='Vol. Week'))
    fig.add_trace(go.Scatter(x=vol_mov_15.index, y=vol_mov_15['Volatility 15 days'],
                             mode='lines',
                             name='Vol. 15 days'))
    fig.add_trace(go.Scatter(x=vol_mov_20.index, y=vol_mov_20['Volatility Month'],
                             mode='lines',
                             name='Vol. Month'))
    fig.add_trace(go.Scatter(x=vol_mov_90.index, y=vol_mov_90['Volatility Quarter'],
                             mode='lines',
                             name='Vol. Quarter'))
    fig.add_trace(go.Scatter(x=vol_mov_365.index, y=vol_mov_365['Volatility Year'],
                             mode='lines',
                             name='Vol. Year'))

    fig.layout.update(height=600, title=str(input_value) + " Volatility Windows", yaxis_title="Volatility",
                      showlegend=True, yaxis_tickformat='.2%')
    fig.layout.template = 'plotly_dark'
    return fig


@app.callback(dash.dependencies.Output('correlation-figure', 'figure'),
              [dash.dependencies.Input('drop-down-tickers', 'value'),
               dash.dependencies.Input('window-checker', 'value')])
def update_correlation_figure(input_value, window_value):
    wb_prices = pd.ExcelFile('Tickers.xlsx')
    price_data = wb_prices.parse(input_value)
    price_data.set_index('Date', inplace=True, drop=True)
    closes = price_data['Close']
    market = wb_prices.parse('^MXX')
    market.set_index('Date', inplace=True, drop=True)
    market_closes = market['Close']
    both_closes = pd.DataFrame(index=price_data.index, columns=[str(input_value), '^MXX'])
    both_closes[str(input_value)] = closes
    both_closes["^MXX"] = market_closes

    # %% Ventanas de Correlacion
    corr_mov_5 = pd.DataFrame(both_closes[str(input_value)].rolling(5).corr(both_closes['^MXX']).dropna())
    corr_mov_15 = pd.DataFrame(both_closes[str(input_value)].rolling(10).corr(both_closes['^MXX']).dropna())
    corr_mov_20 = pd.DataFrame(both_closes[str(input_value)].rolling(20).corr(both_closes['^MXX']).dropna())
    corr_mov_90 = pd.DataFrame(both_closes[str(input_value)].rolling(60).corr(both_closes['^MXX']).dropna())
    corr_mov_365 = pd.DataFrame(both_closes[str(input_value)].rolling(252).corr(both_closes['^MXX']).dropna())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=corr_mov_5.index, y=corr_mov_5[0],
                             mode='lines',
                             name='Corr. Week'))
    fig.add_trace(go.Scatter(x=corr_mov_15.index, y=corr_mov_15[0],
                             mode='lines',
                             name='Corr. 15 days'))
    fig.add_trace(go.Scatter(x=corr_mov_20.index, y=corr_mov_20[0],
                             mode='lines',
                             name='Corr. Month'))
    fig.add_trace(go.Scatter(x=corr_mov_90.index, y=corr_mov_90[0],
                             mode='lines',
                             name='Corr. Quarter'))
    fig.add_trace(go.Scatter(x=corr_mov_365.index, y=corr_mov_365[0],
                             mode='lines',
                             name='Corr. Year'))

    fig.layout.update(height=600, title=str(input_value) + " Correlation Windows with Market Index",
                      yaxis_title="Correlation", showlegend=True, yaxis_tickformat='.2%')
    fig.layout.template = 'plotly_dark'
    return fig


@app.callback(dash.dependencies.Output('info_table', 'children'),
              [dash.dependencies.Input('drop-down-tickers', 'value')])
def update_info(input_value):
    ticker = yf.Ticker(input_value)
    x = ticker.info
    y = pd.DataFrame.from_dict(x, orient='index')
    y.loc["marketCap"] = '$' + (y.loc["marketCap"].astype(float) / 1000000).round(2).astype(str) + 'MM'
    info_df = y.loc[["shortName", "sector", "industry", "country", "marketCap",
                    "exchange", "exchangeTimezoneShortName", "market",  "currency", "beta", "fiftyTwoWeekHigh",
                     "fiftyTwoWeekLow"]]
    info_df.index.names = [input_value]
    info_df = info_df.rename({0: 'Info'}, axis=1)
    info_df.reset_index(level=0, inplace=True)
    data = info_df.to_dict("rows")
    columns = [{"name": str(i), "id": str(i), } for i in info_df.columns]
    return dt.DataTable(data=data, columns=columns, style_cell={'textAlign': 'center', 'font-family': 'verdana',
                                                                'backgroundColor': '#111111', 'color': 'white'},
                        style_as_list_view=True, style_header={'fontWeight': 'bold',
                                                               'backgroundColor': header_table_color},
                        style_table={'height': '200px', 'overflowY': 'auto'})


if __name__ == '__main__':
    app.run_server(debug=True)

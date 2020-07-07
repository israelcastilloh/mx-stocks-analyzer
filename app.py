# -*- coding: utf-8 -*-
from __future__ import print_function
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import dash_table as dt
from plotly.subplots import make_subplots
import numpy as np
from arch import arch_model
import plotly.io as pio
from Tickers_and_prices import prices_from_index, update_prices, dividend_download
from html_style import tab_style, selected_tab_style, stock_analyzer_titles, suggestion_text, tables_styler, fontsize, \
    multi_table_styler

## test test
def initializer_stock_analysis(input_value):
    global historicos, closes, mercado, dividendos, tickers, historicos, indx, index_df
    indx = input_value
    ventana = 365 * 5.1  # User input, maybe
    tickers, historicos, closes = prices_from_index(indx, ventana)
    tickers, historicos, closes = update_prices(indx, ventana)
    if indx == 'MXX':
        mercado = closes['^' + indx]
        dividendos = dividend_download(tickers)
        index_name = yf.Ticker('^' + indx)
    else:
        mercado = closes[indx]
        dividendos = {}
        index_name = yf.Ticker(indx)

    j = pd.DataFrame.from_dict(index_name.info, orient='index')
    index_df = str(j.loc["shortName"].values).strip("[]")
    for ticker in tickers:
        try:
            y = pd.DataFrame(historicos[ticker]['Dividends']).dropna()
            y.reset_index(level=0, inplace=True)
            y = y.sort_values(by='Date', ascending=False)
            dividendos[ticker] = y
        except:
            pass
    initializer_stock_analysis.historicos = historicos
    initializer_stock_analysis.tickers = tickers
    initializer_stock_analysis.tickers.sort()
    initializer_stock_analysis.indx = indx
initializer_stock_analysis("SPY")

app = dash.Dash()
server = app.server

app.title = 'PAP Stock Analyzer'
app.layout = html.Div(style={'backgroundColor': '#111111', "border-width": "1px",
                             "border-color": "#111111", 'margin': 0}, children=[
    dcc.Tabs([

        dcc.Tab(label="Stocks Analysis", style=tab_style(), selected_style=selected_tab_style(), children=[
            # html.Div([
            #     dcc.Input(id='etf-input', value="MXX", type="text"),
            #     html.Button(id="submit-button", n_clicks=0, children="submit")
            #     ]),
            html.Div(id='stock-analyzer', children=[
                html.H1(children=globals()["index_df"] + " Market Components",
                        style=stock_analyzer_titles()),

                html.P(closes.index[-1].strftime('%d-%b-%Y'),
                       style={'font-family': 'verdana', 'color': 'white', 'width': '320px', 'left': '4%',
                            'position': 'relative', 'align': 'left',
                              'vertical-align': 'middle', 'font-size': fontsize, 'fontWeight': 'bold'}),

                html.Div(id='market_table',
                         style=tables_styler('90%')),

            html.Div([
                dcc.Dropdown(id='drop-down-tickers',
                             options=[{'label': i, 'value': i} for i in tickers],
                             value=tickers[3],
                             style={'font-family': 'verdana', 'width': '320px', 'left': '20%',
                                    'vertical-align': 'middle', 'font-size': fontsize}),
            ]),
            html.Div(dcc.Graph(id="graph_close")),

            html.Div(id='today_table',
                     style=tables_styler('60%')),

            html.Div(id='info_table',
                     style=multi_table_styler('2.5%', '60%')),

            html.Div(id='dividend_table',
                     style=multi_table_styler('2.5%', '30%')),

            html.Div([
                html.H2(children="Daily Return Analysis", style=stock_analyzer_titles()),
            ]),

            html.Div([
                dcc.RadioItems(id="window-checker", style={"padding-left": "80px", "padding-bottom": "20px",
                                                           'display': 'block',
                                                           'font-family': 'verdana', 'color': 'white'},
                               options=[
                                   {'label': 'W', 'value': 5},
                                   {'label': '2W', 'value': 10},
                                   {'label': '1M', 'value': 20},
                                   {'label': 'Q', 'value': 63},
                                   {'label': '6M', 'value': 126},
                                   {'label': 'Y', 'value': 252},
                                   {'label': '2Y', 'value': 252 * 2},
                                   {'label': '5Y', 'value': 252 * 5}
                               ], value=252)
            ]),

            html.Div(
                dcc.Graph(id="return-figure", style={'font-family': 'verdana', 'display': 'center-block',
                                                     'padding-left': '30px'})),

            html.Div(id='stat_table', style=tables_styler('60%')),

            html.Div([
                html.H2(children="Volatility Analysis", style=stock_analyzer_titles()),
                html.H3(children="Click on graph legend to hide/show line", style=suggestion_text()),
            ]),

            html.Div(dcc.Graph(id="volatility-figure",
                               style={'font-family': 'verdana', 'display': 'center-block',
                                      'padding-left': '30px'})),

            html.Div([
                html.H2(children="Index Correlation Analysis", style=stock_analyzer_titles()),
                html.H3(children="Click on graph legend to hide/show line", style=suggestion_text()),
            ]),

            html.Div(dcc.Graph(id="correlation-figure",
                               style={'font-family': 'verdana', 'display': 'center-block',
                                      'padding-left': '30px',
                                      'padding-bottom': '30px'}))
        ])
        ]),
        dcc.Tab(label='Tab Test', style=tab_style(), selected_style=selected_tab_style(), children=[])
    ])
])

header_table_color = '#555555'
fontsize_titles = '35px'
hovertext_size = 24

# @app.callback(
#     Output('stock-analyzer', 'children'),
#     [Input('etf-input', 'value')],
#     [State('submit-button', 'n_clicks')])
# def initializer_stock_analysis(input_value):
#     global historicos, closes, mercado, dividendos, tickers, historicos, indx, index_df
#     indx = input_value
#     ventana = 365 * 5  # User input, maybe
#     tickers, historicos, closes = prices_from_index(indx, ventana)
#     tickers, historicos, closes = update_prices(indx, ventana)
#     if indx == 'MXX':
#         mercado = closes['^' + indx]
#         dividendos = dividend_download(tickers)
#         index_name = yf.Ticker('^' + indx)
#     else:
#         mercado = closes[indx]
#         dividendos = {}
#         index_name = yf.Ticker(indx)
#
#     j = pd.DataFrame.from_dict(index_name.info, orient='index')
#     index_df = str(j.loc["shortName"].values).strip("[]")
#     for ticker in tickers:
#         try:
#             y = pd.DataFrame(historicos[ticker]['Dividends']).dropna()
#             y.reset_index(level=0, inplace=True)
#             y = y.sort_values(by='Date', ascending=False)
#             dividendos[ticker] = y
#         except:
#             pass
#     tickers.sort()

#tickers = initializer_stock_analysis.tickers
#historicos = initializer_stock_analysis.historicos
#indx = initializer_stock_analysis.indx

@app.callback(dash.dependencies.Output('market_table', 'children'),
              [dash.dependencies.Input('drop-down-tickers', 'value')])
def market_table(input_value):
    columns = ["Open", "High", "Low", "Close", "Volume"]
    prices = pd.DataFrame(columns=columns)
    for ticker in tickers:
        x = pd.DataFrame(historicos[ticker].iloc[-1]).T
        x_2 = pd.DataFrame(historicos[ticker].iloc[-2]).T
        last_two = x_2.append(x)
        last_two_change = last_two.pct_change()
        for col in columns:
            prices.loc[ticker, col] = round(x[col][0], 4)
        prices.loc[ticker, "Chg. Close"] = str(round(last_two_change.Close[-1] * 100, 2)) + str('%')
        if last_two_change.Close[-1] > 0:
            prices.loc[ticker, ""] = '🟢️️'
        else:
            prices.loc[ticker, ""] = '🔴'
        prices.loc[ticker, "Chg. Volume"] = str(round(last_two_change.Volume[-1] * 100, 1)) + str('%')
        prices.loc[ticker, "Volume"] = (prices.loc[ticker, "Volume"].astype(float) / 1000).round(2).\
                                           astype(str) + 'M'
        prices.append(prices)
        prices = prices.round(2)
        prices = prices.rename_axis("Company")
    prices.reset_index(level=0, inplace=True)
    data = prices.to_dict("rows")
    columns = [{"name": i, "id": i, } for i in prices.columns]
    return dt.DataTable(data=data, columns=columns, style_cell={'textAlign': 'center', 'font-family': 'verdana',
                                                                'backgroundColor': '#111111', 'color': 'white'},
                        style_as_list_view=True, sort_action='native', fixed_rows={'headers': True},
                        style_header={'fontWeight': 'bold',
                                      'backgroundColor': header_table_color},
                        style_table={'height': '300px', 'overflowY': 'auto'})


@app.callback(dash.dependencies.Output('graph_close', 'figure'),
              [dash.dependencies.Input('drop-down-tickers', 'value')])
def update_fig(input_value):
    price_data = historicos[input_value]
    # MA Graphs
    df = price_data.copy()
    MA_1 = 50
    MA_2 = 100
    df['MA1'] = df.Close.rolling(MA_1).mean()
    df['MA2'] = df.Close.rolling(MA_2).mean()

    trace_candlestick = go.Figure(data=[go.Candlestick(x=price_data.index,
                                                       open=price_data['Open'],
                                                       high=price_data['High'],
                                                       low=price_data['Low'],
                                                       close=price_data['Close'], name=(str(input_value))),
                                        go.Scatter(x=df.index, y=df.MA1, line=dict(color='orange', width=2),
                                                   name="MA %i" % MA_1),
                                        go.Scatter(x=df.index, y=df.MA2, line=dict(color='green', width=2),
                                                   name="MA %i" % MA_2)
                                        ])

    trace_candlestick.layout = dict(title=str(input_value), autosize=True, height=700,
                                    hoverlabel=dict(font=dict(size=hovertext_size)),
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

    fig_volume = go.Figure(data=[go.Bar(x=price_data.index, y=price_data["Volume"], name="Volume",
                                        marker_color='white')])

    # Adding Traces
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.2], vertical_spacing=0.05)
    fig.add_traces([trace_candlestick.data[0]])  # trace_candlestick.data[1], trace_candlestick.data[2]]
    fig.add_traces([fig_volume.data[0]], [2], [1])
    fig.layout.update(trace_candlestick.layout)
    fig.layout.update(fig_volume.layout)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), ])  # hide weekends
    fig.layout.update(height=800)
    fig.layout.template = 'plotly_dark'
    return fig


@app.callback(dash.dependencies.Output('dividend_table', 'children'),
              [dash.dependencies.Input('drop-down-tickers', 'value')])
def update_dividend(input_value):
    dividends = dividendos[input_value]
    dividends.Date = pd.DatetimeIndex(dividends.Date).strftime("%Y-%m-%d")
    data = dividends.to_dict("rows")
    columns = [{"name": i, "id": i, } for i in dividends.columns]
    return dt.DataTable(data=data, columns=columns, style_cell={'textAlign': 'center', 'font-family': 'verdana',
                                                                'backgroundColor': '#111111', 'color': 'white'},
                        style_as_list_view=True, style_header={'fontWeight': 'bold',
                                                               'backgroundColor': header_table_color}, fixed_rows={'headers': True},
                        style_table={'height': '200px', 'overflowY': 'auto'})


@app.callback(dash.dependencies.Output('today_table', 'children'),
              [dash.dependencies.Input('drop-down-tickers', 'value')])
def update_today_data(input_value):
    x = pd.DataFrame(historicos[input_value].iloc[-1]).T
    x_2 = pd.DataFrame(historicos[input_value].iloc[-2]).T
    last_two = x_2.append(x)
    last_two_change = last_two.pct_change()
    last_two_change.reset_index(level=0, inplace=True)
    today_table = pd.DataFrame(index=[''])
    for col in x.columns[:-1]:
        today_table[col] = round(x[col][0], 2)
    today_table["Chg. Close"] = str(round(last_two_change.iloc[-1][6] * 100, 2)) + str('%')
    if last_two_change.iloc[-1][6] > 0:
        today_table[""] = '🟢️'
    else:
        today_table[""] = '🔴'
    today_table["Chg. Volume"] = str(round(last_two_change.iloc[-1][5] * 100, 1)) + str('%')
    today_table["Volume"] = (today_table["Volume"].astype(float) / 1000).round(2).astype(str) + 'M'
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
    wb_prices = historicos[input_value].copy()
    price_data = wb_prices.tail(window_value)
    price_data.reset_index(level=0, inplace=True)
    normalized = pd.DataFrame(columns=['Date', "Normalized Returns"])
    normalized['Date'] = price_data['Date']
    normalized['Normalized Returns'] = (price_data['Adj Close'] / price_data['Adj Close'][:1].values) - 1
    if indx == 'MXX':
        market_prices = historicos['^' + indx].copy().tail(window_value)
    else:
        market_prices = historicos[indx].copy().tail(window_value)
    market_prices.reset_index(level=0, inplace=True)
    market = pd.DataFrame(columns=['Date', "Normalized Returns"])
    market['Date'] = market_prices['Date']
    market['Normalized Returns'] = (market_prices['Adj Close'] / market_prices['Adj Close'][:1].values) - 1

    trace_returns_figure = go.Figure()
    trace_returns_figure.add_trace(go.Scatter(x=normalized['Date'], y=(normalized['Normalized Returns']),
                                              name=(str(input_value)), marker_line_width=2,
                                              marker_line_color="blue"))

    trace_returns_figure.add_trace(go.Scatter(x=market['Date'], y=(market['Normalized Returns']),
                                              name=str(indx) + " Returns", marker_line_width=2,
                                              marker_line_color="red"))
    trace_returns_figure.update_layout(showlegend=False, yaxis_title="Return", title="Historical Returns")

    log_returns = pd.DataFrame(columns=['Date', "Close"])
    log_returns['Date'] = price_data['Date']
    log_returns['Close'] = price_data['Adj Close']
    log_returns['Log Returns'] = np.log(log_returns['Close']).diff()
    return_histo = go.Figure(data=[go.Histogram(x=log_returns['Log Returns'], histnorm='probability',
                                                name=(str("Frequency")),
                                                marker_color="blue", marker_line_width=1, marker_line_color="white")])
    return_histo.update_layout(showlegend=False, yaxis_title="Frequency", title="Returns")

    fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=0.03,
                         subplot_titles=['Cumulative Returns over Time', 'Daily Returns Distribution Histogram'],
                         column_widths=(5, 3))
    fig2.add_traces([trace_returns_figure.data[0]], [1], [1])
    fig2.add_traces([trace_returns_figure.data[1]], [1], [1])
    fig2.add_traces([return_histo.data[0]], [1], [2])
    fig2.layout.update(height=500, title=str(input_value), yaxis_tickformat='.2%',
                       hoverlabel=dict(font=dict(size=hovertext_size)))
    fig2.layout.template = 'plotly_dark'
    return fig2


@app.callback(dash.dependencies.Output('stat_table', 'children'),
              [dash.dependencies.Input('drop-down-tickers', 'value'),
               dash.dependencies.Input('window-checker', 'value')])
def update_stat_table(input_value, window_value):
    price_data = historicos[input_value].copy()
    price_data.reset_index(level=0, inplace=True)
    ### RETURNS IMPLEMENTATION
    log_returns = pd.DataFrame(columns=['Close'])
    log_returns['Close'] = price_data['Adj Close']
    log_returns['Log Returns'] = np.log(log_returns['Close']).diff().tail(window_value)

    ### GARCH MODEL IMPLEMENTATION
    data = pd.DataFrame(index=price_data.index, columns=['Returns'])
    data['Returns'] = price_data['Adj Close'].pct_change() * 100
    data = data.dropna()
    aic_value = [arch_model(data, vol='Garch', p=lag, o=0, q=lag, dist='Normal').fit(disp='off').aic
                 for lag in range(1, 11)]
    lag = int(np.array(aic_value).argmin() + 1)
    model = arch_model(data, vol='Garch', p=lag, o=0, q=lag, dist='Normal')
    results = model.fit(disp='off')
    yhat = pd.DataFrame(results.forecast(horizon=20).variance.iloc[-1] ** 0.5)
    estimacion_vol = pd.DataFrame([yhat.iloc[i].values[0] for i in (0, 4, 9, 19)],
                                  index=['Tomorrow', 'W', '15D', 'Month'], columns=[input_value]).round(2)
    stat_measures = pd.DataFrame(index=["Return Mean Annualized"], columns=[""])
    ## Mean of Returns in Window Value
    # stat_measures.loc["Return Mean Annualized"] = str(round(np.mean(log_returns['Log Returns'] * 252 * 100), 2)) \
    #                                               + str('%')
    stat_measures.loc["Return Mean Annualized"] = str(round(np.mean(data['Returns'].tail(window_value) * 252), 2))+str('%')

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
    price_data = historicos[input_value].copy()
    dailyret = pd.DataFrame(index=price_data.index, columns=['Returns'])
    dailyret['Returns'] = price_data['Close'].pct_change()
    dailyret = dailyret.dropna(axis=0)
    # Ventanas de Volatilidad
    vol_mov_5 = pd.DataFrame(dailyret.rolling(5).std().dropna()).rename(columns={"Returns": "Volatility Week"})
    vol_mov_15 = pd.DataFrame(dailyret.rolling(10).std().dropna()).rename(columns={"Returns": "Volatility 15 days"})
    vol_mov_20 = pd.DataFrame(dailyret.rolling(20).std().dropna()).rename(columns={"Returns": "Volatility Month"})
    vol_mov_90 = pd.DataFrame(dailyret.rolling(63).std().dropna()).rename(columns={"Returns": "Volatility Quarter"})
    vol_mov_180 = pd.DataFrame(dailyret.rolling(126).std().dropna()).rename(columns={"Returns": "Volatility 6M"})
    vol_mov_365 = pd.DataFrame(dailyret.rolling(252).std().dropna()).rename(columns={"Returns": "Volatility Year"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol_mov_5.index, y=vol_mov_5['Volatility Week'],
                             mode='lines',
                             name='Week'))
    fig.add_trace(go.Scatter(x=vol_mov_15.index, y=vol_mov_15['Volatility 15 days'],
                             mode='lines',
                             name='2 Weeks'))
    fig.add_trace(go.Scatter(x=vol_mov_20.index, y=vol_mov_20['Volatility Month'],
                             mode='lines',
                             name='Month'))
    fig.add_trace(go.Scatter(x=vol_mov_90.index, y=vol_mov_90['Volatility Quarter'],
                             mode='lines',
                             name='Quarter'))
    fig.add_trace(go.Scatter(x=vol_mov_180.index, y=vol_mov_180['Volatility 6M'],
                             mode='lines',
                             name='Semester'))
    fig.add_trace(go.Scatter(x=vol_mov_365.index, y=vol_mov_365['Volatility Year'],
                             mode='lines',
                             name='Year'))

    fig.layout.update(height=600, title=str(input_value) + " Volatility Windows", yaxis_title="Volatility",
                      showlegend=True, yaxis_tickformat='.2%', hoverlabel=dict(font=dict(size=hovertext_size)),
                      xaxis=dict(rangeselector=dict(bgcolor='#000000',
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
                                                    ), rangeslider=dict(visible=True), type='date'))
    fig.layout.template = 'plotly_dark'
    return fig


@app.callback(dash.dependencies.Output('correlation-figure', 'figure'),
              [dash.dependencies.Input('drop-down-tickers', 'value'),
               dash.dependencies.Input('window-checker', 'value')])
def update_correlation_figure(input_value, window_value):
    price_data = historicos[input_value].copy()
    close = price_data['Adj Close']
    market_closes = mercado[-len(close):]
    both_closes = pd.DataFrame(index=price_data.index, columns=[str(input_value), indx])
    both_closes[str(input_value)] = close
    both_closes[indx] = market_closes
    both_closes = both_closes.pct_change().dropna(axis=0)
    # Ventanas de Correlacion
    corr_mov_5 = pd.DataFrame(both_closes[str(input_value)].rolling(5).corr(both_closes[indx]).dropna())
    corr_mov_15 = pd.DataFrame(both_closes[str(input_value)].rolling(10).corr(both_closes[indx]).dropna())
    corr_mov_20 = pd.DataFrame(both_closes[str(input_value)].rolling(20).corr(both_closes[indx]).dropna())
    corr_mov_90 = pd.DataFrame(both_closes[str(input_value)].rolling(60).corr(both_closes[indx]).dropna())
    corr_mov_180 = pd.DataFrame(both_closes[str(input_value)].rolling(126).corr(both_closes[indx]).dropna())
    corr_mov_365 = pd.DataFrame(both_closes[str(input_value)].rolling(252).corr(both_closes[indx]).dropna())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=corr_mov_5.index, y=round(corr_mov_5[0], 4),
                             mode='lines',
                             name='Week'))
    fig.add_trace(go.Scatter(x=corr_mov_15.index, y=round(corr_mov_15[0], 4),
                             mode='lines',
                             name='2 Weeks'))
    fig.add_trace(go.Scatter(x=corr_mov_20.index, y=round(corr_mov_20[0], 4),
                             mode='lines',
                             name='Month'))
    fig.add_trace(go.Scatter(x=corr_mov_90.index, y=round(corr_mov_90[0], 4),
                             mode='lines',
                             name='Quarter'))
    fig.add_trace(go.Scatter(x=corr_mov_180.index, y=round(corr_mov_180[0], 4),
                             mode='lines',
                             name='Semester'))
    fig.add_trace(go.Scatter(x=corr_mov_365.index, y=round(corr_mov_365[0], 4),
                             mode='lines',
                             name='Year'))

    fig.layout.update(height=600, title=(str(input_value) + ' vs. ' + str(indx) + ' Correlation Windows'),
                      yaxis_title="Correlation", showlegend=True, yaxis_tickformat='.2',
                      yaxis=dict(range=[-1, 1]), hoverlabel=dict(font=dict(size=hovertext_size)),
                      xaxis=dict(rangeselector=dict(bgcolor='#000000',
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
                                                    ), rangeslider=dict(visible=True), type='date'))
    fig.layout.template = 'plotly_dark'
    return fig


@app.callback(dash.dependencies.Output('info_table', 'children'),
              [dash.dependencies.Input('drop-down-tickers', 'value')])
def update_info(input_value):
    ticker = yf.Ticker(input_value)
    x = ticker.info
    y = pd.DataFrame.from_dict(x, orient='index')
    y.loc["marketCap"] = '$' + (y.loc["marketCap"].astype(float) / 1000000).round(2).astype(str) + 'MM'
    y.loc["floatShares"] = (y.loc["floatShares"].astype(float) / 1000000).round(2).astype(str) + 'MM'
    y.loc["sharesOutstanding"] = (y.loc["sharesOutstanding"].astype(float) / 1000000).round(2).astype(str) + 'MM'
    info_df = y.loc[["shortName", "sector", "industry", "country", "marketCap",
                     "exchange", "exchangeTimezoneShortName", "market", "currency", "beta", "fiftyTwoWeekHigh",
                     "fiftyTwoWeekLow", "52WeekChange", "dividendYield", "trailingAnnualDividendYield", "trailingEps",
                     "forwardEps",
                     "trailingPE", "forwardPE", "priceToBook", "bookValue", "twoHundredDayAverage",
                     "payoutRatio", "profitMargins",  "floatShares",
                     "sharesOutstanding", "heldPercentInsiders"]]
    info_df = info_df.rename({'shortName': 'Name', 'sector': 'Sector', 'industry': 'Industry',
                              'country': 'Country', 'marketCap': 'Market Cap', 'exchange': 'Exchange',
                              'exchangeTimezoneShortName': 'Timezone', 'market': 'Market',
                              'currency': 'Currency', 'beta': 'Beta',
                              'dividendYield': 'Dividend Yield', 'fiftyTwoWeekHigh': '52-week High',
                              "fiftyTwoWeekLow":"52-week Low",
                              'trailingAnnualDividendYield': 'Trailing Annual Div. Yield',
                              'trailingEps': 'Trailing EPS', 'forwardEps': 'Forward EPS',
                              'trailingPE': 'Trailing PE', 'forwardPE': 'Forward PE',
                              'priceToBook': 'Price to Book', 'bookValue': 'Book Value',
                              'twoHundredDayAverage': '200-day MA', 'payoutRatio': 'Payout Ratio',
                              'profitMargins': 'Profit Margins', '52WeekChange': '52-week chg.',
                              'floatShares': 'Float Shares', 'sharesOutstanding': 'Shares Outs.',
                              "heldPercentInsiders": "Held % Insiders"}, axis=0)

    info_df.index.names = [input_value]
    info_df = info_df.rename({0: 'Info'}, axis=1)
    info_df.reset_index(level=0, inplace=True)
    data = info_df.to_dict("rows")
    columns = [{"name": str(i), "id": str(i), } for i in info_df.columns]
    return dt.DataTable(data=data, columns=columns, style_cell={'textAlign': 'center', 'font-family': 'verdana',
                                                                'backgroundColor': '#111111', 'color': 'white'},
                        style_as_list_view=True, style_header={'fontWeight': 'bold',
                                                               'backgroundColor': header_table_color}, fixed_rows={'headers': True},
                        style_table={'height': '200px', 'overflowY': 'auto'})


if __name__ == '__main__':
    app.run_server()

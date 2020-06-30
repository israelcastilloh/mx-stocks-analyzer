# Utility functions
import yfinance as yf
import pandas as pd
import dash_table as dt

#import requests
#import json


def dividends_splits(input_value):
    quotes = yf.Ticker(input_value)
    quotes.history(period='5y')
    dividends = pd.DataFrame(quotes.dividends)
    dividends.reset_index(level=0, inplace=True)
    return dividends


def get_values(input_value, interval):
    price_data = yf.download(input_value, period='5y', interval=interval, group_by='ticker', auto_adjust=True,
                             prepost=True, threads=True, proxy=None)
    return price_data


def update_dividend(input_value):
    dividends = dividends_splits(input_value)
    dividends.Date = pd.DatetimeIndex(dividends.Date).strftime("%Y-%m-%d")
    data = dividends.to_dict("rows")
    columns = [{"name": i, "id": i, } for i in dividends.columns]
    return dt.DataTable(data=data, columns=columns, style_cell={'textAlign': 'center', 'font-family': 'verdana'},
                        style_as_list_view=True, style_header={'fontWeight': 'bold'},
                        style_table={'height': '200px', 'overflowY': 'auto'})


def market_tail(interval):
    data_market_table = yf.download(tickers, period='1d', interval=interval, threads=True, group_by="ticker")
    return data_market_table


def csv_saver(tickers):
    import pickle
    # crear y guardar en un diccionario
    historicos = {}
    writer = pd.ExcelWriter("Tickers.xlsx", engine='openpyxl')
    for ticker in tickers:
        try:
            y = get_values(ticker, '1d')
            y.reset_index(level=0, inplace=True)
            y.to_excel(writer, ticker, index=False)
            writer.save()
            historicos[ticker] = y
        except:
            pass
        #print(y)
    # mandar historicos a un archivo pickle
    pickle.dump(historicos, open('save.p', 'wb'))
    return historicos


def dividend_saver(tickers):
    writer = pd.ExcelWriter("Dividend.xlsx", engine='openpyxl')
    for ticker in tickers:
        y = dividends_splits(ticker)
        y.to_excel(writer, ticker, index=False)
        writer.save()



def tickers_from_market(indx):
    import requests
    # Definir el indice con el que se quiere trabajar, en este caso el mercado mexicano
    #indx = 'MXX'
    url = 'https://finance.yahoo.com/quote/%5E' + indx + '/components'
    html = requests.get(url).content
    df_list = pd.read_html(html)
    df = df_list[-1]
    tickers = df.Symbol.tolist()
    tickers = ['LIVEPOLC-1.MX' if x == 'LIVEPOLC1.MX' else x for x in tickers]
    tickers = ['PE&OLES.MX' if x == 'PEOLES.MX' else x for x in tickers]
    tickers = ['KOFUBL.MX' if x == 'KOFL.MX' else x for x in tickers]
    tickers.insert(0, '^' + indx)
    return tickers

def download_prices(tickers, inicio):
    import pandas_datareader.data as web
    from datetime import date
    import pickle
    import datetime
    today = date.today()
    while (today.weekday()>4): today = today - datetime.timedelta(1)
    closes = pd.DataFrame(columns = tickers) #, index=web.YahooDailyReader(symbols=tickers[0], start=inicio, end=today, interval='d').read().index)
    historicos = {}
    for ticker in tickers:
        try:
            #df = yf.download(ticker, start=inicio, end = today)
            df= web.YahooDailyReader(symbols=ticker, start=inicio, end=today, interval='d').read()
            historicos[ticker] = df
            closes[ticker]=df['Adj Close']
        except:
            pass
    closes.index_name = 'Date'
    closes = closes.sort_index()
    closes = closes.dropna(axis=1)
    pickle.dump(historicos,open('save.p','wb'))
    return historicos, closes

def update_prices():
    import pickle
    import datetime
    import pandas_datareader.data as web
    from datetime import date
    today = date.today()
    while (today.weekday()>4): today = today - datetime.timedelta(1)
    # Cargar datos
    historicos = pickle.load(open('save.p', 'rb'))
    tickers = list(historicos.keys())
    closes = pd.DataFrame(columns = tickers)
    indx = 'MXX'
    inicio = historicos['^' + indx].index[-1] + datetime.timedelta(1)
    if today>inicio:
        closes = pd.DataFrame(columns = tickers)
        for ticker in tickers:
            try:
                df = web.YahooDailyReader(symbols=ticker, start=inicio, end=today, interval='d').read()
                historicos[ticker] = historicos[ticker].append(df)
                closes[ticker] = historicos[ticker]['Adj Close']
            except:
                pass
        # Guardar de nuevo
        closes.index_name = 'Date'
        closes = closes.sort_index()
        closes = closes.dropna(axis=1)
        pickle.dump(historicos, open('save.p', 'wb'))
    else:
        for ticker in tickers:
            closes[ticker] = historicos[ticker]['Adj Close']
    return historicos,closes

def dividends(tickers):
    #import pandas_datareader.data as web
    import yfinance as yf
    dividendos = {}
    for ticker in tickers:
        try:
            #Using web
            #y = web.YahooDivReader(symbols=ticker).read()
            #dividendos[ticker] = pd.DataFrame(y.value, index=y.index)
            #using yfinance
            y = pd.DataFrame(yf.Ticker(ticker).history(period='5y').Dividends)
            y = y.loc[~(y==0).all(axis=1)]
            y.reset_index(level=0, inplace=True)
            y = y.sort_values(by='Date',ascending=False)
            dividendos[ticker] = y
        except:
            pass
    return dividendos
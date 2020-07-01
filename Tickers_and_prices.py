#%% Download tickers
def download_tickers(indx):
    import pandas as pd
    import requests
    # Definir el indice con el que se quiere trabajar
    # para mexico se usa el indice
    # para extrankeras se usa un ETF que represente al sector/indice deseado
    # Se descargaran los componentes mas importantes
    # indx = 'SPY'
    if indx=='MXX':
        url = 'https://finance.yahoo.com/quote/%5E' + indx + '/components'
        html = requests.get(url).content
        df_list = pd.read_html(html)
        df = df_list[-1]
        tickers = df.Symbol.tolist()
        tickers = ['LIVEPOLC-1.MX' if x == 'LIVEPOLC1.MX' else x for x in tickers]
        tickers = ['PE&OLES.MX' if x == 'PEOLES.MX' else x for x in tickers]
        tickers = ['KOFUBL.MX' if x == 'KOFL.MX' else x for x in tickers]
        tickers.insert(0, 'WALMEX.MX')
        tickers.insert(0, 'BACHOCO.MX')
        tickers.insert(0, 'AEROMEX.MX')
        tickers.insert(0, 'VOLARA.MX')
        tickers.insert(0, 'AZTECACPO.MX')
        tickers.insert(0, 'LALAB.MX')
        tickers.insert(0, 'CHDRAUIB.MX')
        tickers.insert(0, '^' + indx)
    else:
        url = 'https://www.marketwatch.com/investing/fund/'+indx+'/holdings'
        html = requests.get(url).content
        df_list = pd.read_html(html)
        df = df_list[-1]
        tickers = df.Symbol.dropna().tolist()
        return tickers

#%% Download prices
def get_prices(tickers,ventana):
    import pandas as pd
    import pandas_datareader.data as web
    import pickle
    import datetime
    today = datetime.date.today()
    #ventana = 365*1
    inicio = today - datetime.timedelta(ventana)
    try: # Load file from pickle
        infile = open(indx+'.pkl','rb')
        historicos = pickle.load(infile)
        infile.close()
        tickers = list(historicos.keys())
        closes = pd.DataFrame(columns = tickers)
        for ticker in tickers:
                closes[ticker] = historicos[ticker]['Adj Close']
        print('Load succesful')
    except: # download from yahoo
        while (today.weekday()>4): today = today - datetime.timedelta(1)
        closes = pd.DataFrame(columns = tickers) #, index=web.YahooDailyReader(symbols=tickers[0], start=inicio, end=today, interval='d').read().index)
        historicos = {}
        i=0
        for ticker in tickers:
            try:
                df= web.YahooDailyReader(symbols=ticker, start=inicio, end=today, interval='d').read()
                historicos[ticker] = df
                closes[ticker]=df['Adj Close']
                i+=1
                print(ticker,' downloaded')
            except:
                pass
        print("Downloaded {} of {} succesfully".format(i, len(tickers)))
    closes.index_name = 'Date'
    closes = closes.sort_index()
    closes = closes.dropna(axis=1)
    outfile = open(indx+'.pkl','wb')
    pickle.dump(historicos,outfile)
    outfile.close()
    tickers = list(historicos.keys())
    return historicos, closes, tickers

#%% Download stock prices given an ETF
def prices_from_index(indx,ventana):
    import pandas as pd
    import requests
    import pandas_datareader.data as web
    import pickle
    import datetime
    today = datetime.date.today()
     # Definir el indice con el que se quiere trabajar
    # para mexico se usa el indice
    # para extrankeras se usa un ETF que represente al sector/indice deseado
    # Se descargaran los componentes mas importantes
    # indx = 'SPY'
    # GET TICKERS
    if indx=='MXX':
        url = 'https://finance.yahoo.com/quote/%5E' + indx + '/components'
        html = requests.get(url).content
        df_list = pd.read_html(html)
        df = df_list[-1]
        tickers = df.Symbol.tolist()
        tickers = ['LIVEPOLC-1.MX' if x == 'LIVEPOLC1.MX' else x for x in tickers]
        tickers = ['PE&OLES.MX' if x == 'PEOLES.MX' else x for x in tickers]
        tickers = ['KOFUBL.MX' if x == 'KOFL.MX' else x for x in tickers]
        tickers.insert(0, 'WALMEX.MX')
        tickers.insert(0, 'BACHOCO.MX')
        tickers.insert(0, 'AEROMEX.MX')
        tickers.insert(0, 'VOLARA.MX')
        tickers.insert(0, 'AZTECACPO.MX')
        tickers.insert(0, 'LALAB.MX')
        tickers.insert(0, 'CHDRAUIB.MX')
        tickers.insert(0, '^' + indx)
    else:
        url = 'https://www.marketwatch.com/investing/fund/'+indx+'/holdings'
        html = requests.get(url).content
        df_list = pd.read_html(html)
        df = df_list[-1]
        tickers = df.Symbol.dropna().tolist()
        tickers.insert(0,indx)
    # GET PRICES
    inicio = today - datetime.timedelta(ventana)
    try: # Load file from pickle
        infile = open(indx+'.pkl','rb')
        historicos = pickle.load(infile)
        infile.close()
        tickers = list(historicos.keys())
        closes = pd.DataFrame(columns = tickers)
        for ticker in tickers:
                closes[ticker] = historicos[ticker]['Adj Close']
        print('Load succesful')
    except: # download from yahoo
        while (today.weekday()>4): today = today - datetime.timedelta(1)
        closes = pd.DataFrame(columns = tickers) #, index=web.YahooDailyReader(symbols=tickers[0], start=inicio, end=today, interval='d').read().index)
        historicos = {}
        i=0
        for ticker in tickers:
            try:
                df= web.YahooDailyReader(symbols=ticker, start=inicio, end=today, get_actions=True,
                                         interval='d').read()
                historicos[ticker] = df
                closes[ticker]=df['Adj Close']
                i+=1
                print(ticker,' downloaded')
            except:
                print(ticker,' not available')
                pass
        print("Downloaded {} of {} succesfully".format(i, len(tickers)))
    closes.index_name = 'Date'
    closes = closes.sort_index()
    closes = closes.dropna(axis=1)
    outfile = open(indx+'.pkl','wb')
    pickle.dump(historicos,outfile)
    outfile.close()
    tickers = list(historicos.keys())
    
    return tickers, historicos, closes

#%% Update prices
def update_prices(indx, ventana):
    import pickle
    import datetime
    import pandas as pd
    import pandas_datareader.data as web
    today = datetime.date.today()
    while (today.weekday()>4): today = today - datetime.timedelta(1)
    # Cargar datos
    infile = open(indx+'.pkl','rb')
    historicos = pickle.load(infile)
    tickers = list(historicos.keys())
    closes = pd.DataFrame(columns = tickers)
    #closes = pd.DataFrame(columns = tickers)
    #indx = 'MXX'
    first_date = min([historicos[ticker].index[0] for ticker in tickers])
    inicio = today - datetime.timedelta(ventana)
    
    if inicio < first_date:
        closes = pd.DataFrame(columns = tickers)
        historicos = {}
        i=0
        for ticker in tickers:
            try:
                df= web.YahooDailyReader(symbols=ticker, start=inicio, end=today, get_actions=True,
                                         interval='d').read()
                historicos[ticker] = df
                closes[ticker]=df['Adj Close']
                i+=1
                print(ticker,' downloaded')
            except:
                print(ticker,' not available')
                pass
        print("Downloaded {} of {} succesfully".format(i, len(tickers)))
        closes.index_name = 'Date'
        closes = closes.sort_index()
        closes = closes.dropna(axis=1)
        outfile = open(indx+'.pkl','wb')
        pickle.dump(historicos,outfile)
        outfile.close()
        tickers = list(historicos.keys())
    
    last_date = max([historicos[ticker].index[-1] for ticker in tickers])
    if today>last_date:
        print('Updating data')
        closes = pd.DataFrame(columns = tickers)
        i=0
        for ticker in tickers:
            try:
                df = web.YahooDailyReader(symbols=ticker, start=last_date + datetime.timedelta(1), 
                                          end=today, get_actions=True, interval='d').read()
                historicos[ticker] = historicos[ticker].append(df)
                closes[ticker] = historicos[ticker]['Adj Close']
                i+=1
                print(ticker,' downloaded')
            except:
                print(ticker,' not available')
                pass
        # Guardar de nuevo
        closes.index_name = 'Date'
        closes = closes.sort_index()
        closes = closes.dropna(axis=1)
        outfile = open(indx+'.pkl','wb')
        pickle.dump(historicos,outfile)
        outfile.close()
        tickers = list(historicos.keys())
    else:
        for ticker in tickers:
            closes[ticker] = historicos[ticker]['Adj Close']
    return tickers, historicos, closes

#%% Dividend information
def dividend_download(tickers):
    #import pandas_datareader.data as web
    import yfinance as yf
    import pandas as pd
    dividendos = {}
    print('Downloading Dividend Information')
    for ticker in tickers:
        #try:
            #Using web
            #y = web.YahooDivReader(symbols=ticker).read()
            #dividendos[ticker] = pd.DataFrame(y.value, index=y.index)
            #using yfinance
        y = pd.DataFrame(yf.Ticker(ticker).history(period='5y').Dividends)
        y = y.loc[~(y==0).all(axis=1)]
        y.reset_index(level=0, inplace=True)
        y = y.sort_values(by='Date',ascending=False)
        dividendos[ticker] = y
        #except:
            #pass
    return dividendos
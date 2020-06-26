# Utility functions
import yfinance as yf
import pandas as pd
import requests


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


tickers = ["^MXX", "AMXL.MX", "WALMEX.MX", "FEMSAUBD.MX", "ALFAA.MX", "CEMEXCPO.MX", "BIMBOA.MX",
           "GFNORTEO.MX",
           "SORIANAB.MX",
           "ALPEKA.MX",
           "GCARSOA1.MX",
           "LIVEPOL1.MX",
           "TLEVISACPO.MX",
           "BSMX",
           "ORBIA.MX",
           "ELEKTRA.MX",
           "CHDRAUIB.MX",
           "AC.MX",
           "PE&OLES.MX",
           "GFINBURO.MX",
           "GRUMAB.MX",
           "LACOMERUBC.MX",
           "LALAB.MX",
           "AHMSA.MX",
           "AEROMEX.MX",
           "BACHOCOB.MX",
           "CULTIBAB.MX",
           "GNP.MX",
           "FRAGUAB.MX",
           "ICHB.MX",
           "KIMBERA.MX",
           "SIMECB.MX",
           "ALSEA.MX",
           "GPH1.MX",
           "VITROA.MX",
           "GIGANTE.MX",
           "KUOB.MX",
           "Q.MX",
           "GENTERA.MX",
           "GRUMAB.MX",
           "GFAMSAA.MX",
           "IDEALB-1.MX",
           "HERDEZ.MX",
           "VOLARA.MX",
           "AZTECACPO.MX",
           "MFRISCOA-1.MX",
           "PAPPEL.MX",
           "RASSINIA.MX",
           "LABB.MX",
           "MEGACPO.MX",
           "BEVIDESA.MX",
           "IENOVA.MX",
           "AXTELCPO.MX",
           "GCC.MX",
           "GISSAA.MX",
           "CMOCTEZ.MX",
           "BAFARB.MX",
           "GPROFUT.MX",
           "LAMOSA.MX",
           "RA.MX",
           "CABLECPO.MX",
           "CERAMICB.MX",
           "PINFRA.MX",
           "AGUA.MX",
           "CIEB.MX",
           "GCARSOA1.MX",
           "ARA.MX",
           "POCHTECB.MX",
           "ASURB.MX",
           "FINDEP.MX",
           "POSADASA.MX",
           "MINSAB.MX",
           "GAPB.MX",
           "INVEXA.MX",
           "CYDSASAA.MX",
           "MONEXB.MX",
           "COLLADO.MX",
           "UNIFINA.MX",
           "GFMULTIO.MX",
           "AUTLANB.MX",
           "PASAB.MX",
           "OMAB.MX",
           "GBMO.MX",
           "PV.MX",
           "CREAL.MX",
           "TMMA.MX",
           "VASCONI.MX",
           "FIBRAMQ12.MX",
           "GMD.MX",
           "CMRB.MX",
           "BOLSAA.MX",
           "VALUEGFO.MX",
           "MEDICAB.MX",
           "TERRA13.MX",
           "FINAMEXO.MX",
           "DANHOS13.MX",
           "GENSEG.MX",
           "FIHO12.MX",
           "CIDMEGA.MX",
           "HCITY.MX",
           "ARISTOSA.MX",
           "SPORTS.MX",
           "DINEB.MX",
           "CONVERA.MX",
           "VESTA.MX",
           "RCENTROA.MX",
           "FINN13.MX",
           "HOGARB.MX",
           "HOTEL.MX",
           "FSHOP13.MX",
           "TEAKCPO.MX",
           "LASEG.MX",
           "SAREB.MX",
           "FMTY14.MX",
           "INGEALB.MX",
           "EDOARDOB.MX",
           "FHIPO14.MX",
           "GEOB.MX",
           "GOMO.MX",
           "HOMEX.MX",
           "URBI.MX", "ACTINVRB.MX"]
tickers.sort()


def market_tail(interval):
    data_market_table = yf.download(tickers, period='1d', interval=interval, threads=True, group_by="ticker")
    return data_market_table


def csv_saver(tickers):
    writer = pd.ExcelWriter("Tickers.xlsx", engine='openpyxl')
    for ticker in tickers:
        y = get_values(ticker, '1d')
        y.reset_index(level=0, inplace=True)
        y.to_excel(writer, ticker, index=False)
        writer.save()
        print(y)


def dividend_saver(tickers):
    writer = pd.ExcelWriter("Dividend.xlsx", engine='openpyxl')
    for ticker in tickers:
        y = dividends_splits(ticker)
        y.to_excel(writer, ticker, index=False)
        writer.save()



def tickers_from_market():
    # Definir el indice con el que se quiere trabajar, en este caso el mercado mexicano
    indx = 'MXX'
    url = 'https://finance.yahoo.com/quote/%5E' + indx + '/components'
    html = requests.get(url).content
    df_list = pd.read_html(html)
    df = df_list[-1]
    tickers = df.Symbol.tolist()
    tickers = ['LIVEPOLC-1.MX' if x == 'LIVEPOLC1.MX' else x for x in tickers]
    tickers = ['PE&OLES.MX' if x == 'PEOLES.MX' else x for x in tickers]
    tickers.insert(0, '^' + indx)
    return tickers




import os
import yfinance as yf
import datetime as dt
from dateutil.relativedelta import relativedelta


class Extraction:
    """A class for extracting data from Yahoo finance"""

    def __init__(self, tickers, col, dir):
        self.tickers = tickers
        self.col = col
        self.dir = dir

    def get_data(self):
        start_date = dt.datetime.now() - relativedelta(years=15)
        end_date = dt.datetime.now()
        data = yf.download(self.tickers, start_date, end_date)[self.col]
        data = data[data != 0]
        data.dropna(axis=0, how='any', inplace=True)
        data.to_csv(os.path.join(self.dir, 'data.csv'))

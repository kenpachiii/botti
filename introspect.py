import glob
import math
from random import randint
import traceback
import numpy as np
import logging
import lzma
import os
import time
import re
import calendar
import mmap
import shutil
import tempfile
import pandas as pd
import multiprocessing as mp
from datetime import datetime, timezone, timedelta
from scipy.stats import jarque_bera, skew, norm, cauchy, gaussian_kde
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from botti.cache import Cache
from botti.enums import PositionStatus, PositionState

pd.set_option('display.precision', 10)
# pd.set_option('display.float_format', lambda x: '%.9f' % x)

logger = logging.getLogger(__name__)

GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[96m'
MAGENTA = '\033[95m'
PURPLE = '\033[94m'
END = '\033[0m'

def parse8601(timestamp=None):
    if timestamp is None:
        return timestamp
    yyyy = '([0-9]{4})-?'
    mm = '([0-9]{2})-?'
    dd = '([0-9]{2})(?:T|[\\s])?'
    h = '([0-9]{2}):?'
    m = '([0-9]{2}):?'
    s = '([0-9]{2})'
    ms = '(\\.[0-9]{1,3})?'
    tz = '(?:(\\+|\\-)([0-9]{2})\\:?([0-9]{2})|Z)?'
    regex = r'' + yyyy + mm + dd + h + m + s + ms + tz
    try:
        match = re.search(regex, timestamp, re.IGNORECASE)
        if match is None:
            return None
        yyyy, mm, dd, h, m, s, ms, sign, hours, minutes = match.groups()
        ms = ms or '.000'
        ms = (ms + '00')[0:4]
        msint = int(ms[1:])
        sign = sign or ''
        sign = int(sign + '1') * -1
        hours = int(hours or 0) * sign
        minutes = int(minutes or 0) * sign
        offset = timedelta(hours=hours, minutes=minutes)
        string = yyyy + mm + dd + h + m + s + ms + 'Z'
        dt = datetime.strptime(string, "%Y%m%d%H%M%S.%fZ")
        dt = dt + offset
        return calendar.timegm(dt.utctimetuple()) * 1000 + msint
    except (TypeError, OverflowError, OSError, ValueError):
        return None

class OrderBook:
    def __init__(self, *args):

        if isinstance(*args, bytes):
            self.parse(*args)

            return

        self.bids: list = args[0].get('bids').copy()
        self.asks: list = args[0].get('asks').copy()
        self.timestamp: int = int(args[0].get('timestamp'))

    def format(self):

        for i in range(0, len(self.bids)):
            self.bids[i] = '{} {}'.format(self.bids[i][0], self.bids[i][1]) 

        for i in range(0, len(self.asks)):
            self.asks[i] = '{} {}'.format(self.asks[i][0], self.asks[i][1]) 

        return 'timestamp:{};bids:{};asks:{}\n'.format(self.timestamp, ','.join(self.bids), ','.join(self.asks)).encode()

    def parse(self, book: bytes):

        for item in book.decode().split(';'):
            key, value = item.split(':')

            if key == 'bids' or key == 'asks':
                value = np.asarray([item.split(' ') for item in value.split(',')]).astype(float)

            if key == 'timestamp':
                value = int(value)

            setattr(self, key, value)

        return self

class Trade:
    def __init__(self, *args):

        if isinstance(*args, bytes):
            self.parse(*args)

            return

        self.id: str = args[0].get('id')
        self.price: float = args[0].get('price')
        self.amount: float = args[0].get('amount')
        self.side: str = args[0].get('side')
        self.timestamp: int = int(args[0].get('timestamp'))

    def format(self):
        return 'id:{};price:{};amount:{};side:{};timestamp:{}\n'.format(self.id, self.price, self.amount, self.side, self.timestamp).encode()

    def parse(self, trade: bytes):
        for item in trade.decode().split(';'):
            key, value = item.split(':')
            if key == 'timestamp':
                value = int(value)
            if key == 'amount' or key == 'price':
                value = float(value)
            setattr(self, key, value)

        return self

class FileReader:

    def __init__(self, path):
        self.files = glob.glob(path)
        self.files.sort()

        self.current_file = None

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __del__(self) -> None:

        if self.current_file:
            self.current_file.close()

    def load_temporary(self, file):
        with tempfile.TemporaryFile() as f_temp:  
            with lzma.open(file, mode = 'rb') as f:
                shutil.copyfileobj(f, f_temp)  
            f_temp.flush()  

            return f_temp

    def chunkify(self, fname, size = 1024*1024):

        with tempfile.NamedTemporaryFile(delete = False) as f_temp:
            with lzma.open(fname, mode = 'rb') as f:

                shutil.copyfileobj(f, f_temp)

                f_temp.flush()
                f_temp.seek(0)

                fend = os.path.getsize(f_temp.name)
                end = f_temp.tell()

                while True:
                    start = end
                    f_temp.seek(size, 1)
                    f_temp.readline()
                    end = f_temp.tell()
                    yield f_temp.name, start, end - start
                    if end > fend:
                        break

            f.close()
        f_temp.close()

    def next(self):
        
        try:

            if not self.current_file:
                self.current_file = lzma.open(self.files.pop(0), 'rb')

            if self.current_file.peek(1) == b'' and len(self.files) > 0:
                self.current_file = lzma.open(self.files.pop(0), 'rb')

            return self.current_file.readline().rstrip()
        except Exception as e:
            print(e)
            
        raise StopIteration()

    def search(self, f):
        pass
  
    def length(self):

        total = 0
        for file in self.files:
            with tempfile.TemporaryFile() as f_temp:  
                with lzma.open(file, mode = 'rb') as f:
                    shutil.copyfileobj(f, f_temp)  
                f_temp.flush()  
                
                total += os.path.getsize(f_temp.name)
                f_temp.close()

        return total

# example: self.df.apply(lookback_window, values = df2['column'], method = 'mean', axis = 1)
def lookback_window(row, values, method = 'sum', *args, **kwargs):
    loc = values.index.get_loc(row.name)
    return getattr(values.iloc[0: loc + 1], method)(*args, **kwargs)

def gaussian_kernel(row, values):
    loc = values.index.get_loc(row.name)

    if loc == 0:
        return np.nan

    return gaussian_kde(values.iloc[0: loc + 1]).resample(1)[0][0]

class Introspect:

    def __init__(self, **kwargs) -> None:

        path = os.path.join('data', kwargs.get('exchange', 'okx'))

        self.trades = glob.glob(os.path.join(path, 'trades', 'BTC-USDT-SWAP', '*'))
        self.trades.sort()

        self.cache = Cache('botti.db.test')

        self.df = pd.DataFrame(columns=['id', 'timestamp', 'price', 'amount', 'side'])

    def datetime(self, timestamp: float):

        if type(timestamp) == 'str':
            timestamp = float(timestamp)

        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    def safe_devisor(value):
        return np.max(value, 1)

    @staticmethod
    def kde_bandwith_est(kde):
        return 0
        
    def process_trades(self):

        # self.trades = self.trades[0:30]

        with tqdm(total=len(self.trades)) as pbar:

            for file in self.trades:

                self.df = pd.concat([self.df, pd.read_csv(file, header = 0, names = ['id', 'side', 'amount', 'price', 'timestamp'])], ignore_index = True)
                
                self.df['timestamp'] = self.df['timestamp'].astype(int)
                self.df['price'] = self.df['price'].astype(float)
                self.df['amount'] = self.df['amount'].astype(float)
     
                pbar.update(1)

        self.df.sort_values(by=['id'])
        self.df.fillna(0)

        df2: pd.DataFrame = self.df.groupby(['timestamp', 'side']).agg(amount=('amount', np.sum), price=('price', np.mean)).reset_index()

        df2.set_index('timestamp', inplace = True)
        df2.index = pd.to_datetime(df2.index, utc = True, unit = 'ms')

        seconds = 1800

        df2 = df2.resample(f'{seconds}S', kind = 'period', convention = 'start').agg(amount=('amount', np.sum), price=('price', np.mean)).dropna()

        df2.loc[:,('r')] = (np.log(df2['price'] / df2['price'].shift(1))).fillna(0)

        df2.loc[:,('mu')] = df2.apply(lookback_window, values = df2['r'], method = 'mean', axis = 1).fillna(1)
        df2.loc[:,('sigma')] = df2.apply(lookback_window, values = df2['r'], method = 'std', axis = 1).fillna(0)
        df2.loc[:,('sigma')] = df2.loc[:,('sigma')] * np.sqrt(seconds / 86400)

        df2.loc[:,('p')] = df2.apply(gaussian_kernel, values = df2['r'], axis = 1)

        print(df2.corr())

        print('\nsamples %i\n' % df2.index.size)

        # jb, p = jarque_bera(df2['r'])
        # print('jarque bera %.9f' % jb)
        # print('p-value %.9f' % p)
        print('skew %.9f' % skew(df2['r']))

        mse = ((df2['r'] - df2['p'])**2).mean(axis = None).real
        print('\nmse {}'.format(mse))

        df2['accuracy'] = np.equal(np.sign(df2['p']), np.sign(df2['r']))
        print('accuracy {}\n'.format(df2['accuracy'].sum() / df2['accuracy'].count()))

        df2.drop(['accuracy'], axis = 1, inplace = True)

        print(df2.tail(10).to_string())

        # df2['r'].plot.hist(bins = 'fd', density = True, alpha = 0.5)
        # df2['r'].plot.kde()

        # df2['p'].plot.hist(bins = 'fd', density = True)

        # mu, std = norm.fit(df2['r']) 
        # xmin, xmax = plt.xlim()
        # x = np.linspace(xmin, xmax, df2['r'].index.size)
        # pdf = norm.pdf(x, mu, std)
        # plt.plot(x, pdf, color = 'red')

        # mu, std = cauchy.fit(df2['r']) 
        # xmin, xmax = plt.xlim()
        # x = np.linspace(xmin, xmax, df2['r'].index.size)
        # pdf = cauchy.pdf(x, mu, std)
        # plt.plot(x, pdf, color = 'blue')

        # plt.xlabel('Returns')
        # plt.legend(['historical', 'p'])
        # plt.show()

    def dump_orders(self):

        import json

        orders = []

        values = self.cache.cur.execute('''SELECT * FROM orders;''').fetchall()
        for value in values:
            print({k: value[k] for k in value.keys()})

        # for order in orders:
        #     print(order.get('datetime'), order.get('id'), order.get('side'), order.get('average'), order.get('amount'), order.get('filled'), order.get('status'))

    def dump_position(self):
        values = self.cache.cur.execute('''SELECT * FROM position;''').fetchall()
        for value in values:
            print({k: value[k] for k in value.keys()})

def main():
    introspect = Introspect(exchange = 'okx')

    try:

        order = { 'id': 0, 'side': 'buy', 'filled': 1, 'remaining': 1, 'symbol': 'BTC/USD:BTC' }
        print(introspect.cache.insert_order(order))

        order = { 'id': 0, 'side': 'buy', 'filled': 2, 'remaining': 0, 'symbol': 'BTC/USD:BTC' }
        print(introspect.cache.insert_order(order))

        order = { 'id': 0, 'side': 'buy', 'filled': 2, 'remaining': 0, 'symbol': 'BTC/USD:BTC' }
        print(introspect.cache.insert_order(order))

        order = { 'id': 1, 'side': 'buy', 'filled': 2, 'remaining': 0, 'symbol': 'BTC/USD:BTC' }
        print(introspect.cache.insert_order(order))

        print(introspect.cache.fetch_order('BTC/USD:BTC'))
        
        # n_tic = time.time() * 1000
        # introspect.process_trades()
        # print('completed in {} ms'.format((time.time() * 1000) - n_tic))
    except Exception as e:
        print(e, traceback.print_tb(e.__traceback__))

if __name__ == "__main__":
    main() 

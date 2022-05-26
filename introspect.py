import glob
import math
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
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

from botti.cache import Cache
from botti.enums import PositionStatus, PositionState

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
        self.previous_line = None
        self.current_line = None
        self.collection = None

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __del__(self) -> None:

        if self.current_file:
            self.current_file.close()

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

        if self.current_line and f(self.current_line) == False:
            return self.previous_line

        if self.current_line and f(self.current_line):
            self.previous_line = self.current_line
            self.current_line = None
            return self.previous_line

        while line := self.next():
           
            if f(line) == False:
                self.current_line = line
                return self.previous_line

            self.previous_line = line
  
    def length(self):
        total = 0
        for file in self.files:
            with lzma.open(file, 'rb') as f:
                total += len(f.readlines())
        return total

def lookback_window(row, values, method = 'sum', *args, **kwargs):
    loc = values.index.get_loc(row.name)
    return getattr(values.iloc[0: loc + 1], method)(*args, **kwargs)

class Introspect:

    def __init__(self, **kwargs) -> None:

        path = os.path.join('data', kwargs.get('exchange', 'okx'))

        # self.order_book: dict = self.load_order_book(os.path.join(path, 'order_book', '2022-05-19.xz'))
        self.trades: FileReader = FileReader(os.path.join(path, 'trades', '*'))
        self.cache = Cache('botti.db')

    def load_order_book(self, path):

        order_book = {}

        with tempfile.TemporaryFile() as f_temp:  
            with lzma.open(path, mode = 'rb') as f:
                shutil.copyfileobj(f, f_temp)  
            f_temp.flush()  
            
            with mmap.mmap(f_temp.fileno(), length = 0, access = mmap.ACCESS_READ) as f_mmap:
                with tqdm(total=f_mmap.size()) as pbar:
                    while line := f_mmap.readline():

                        _, value = line.decode().split(';')[0].split(':')
                        value = int(value)

                        order_book[value] = line

                        pbar.update(len(line))
          
            f_temp.close()

        return order_book

    def datetime(self, timestamp: float):

        if type(timestamp) == 'str':
            timestamp = float(timestamp)

        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    def safe_devisor(value):
        return np.max(value, 1)

    def test_strategy(self):

        long_entries, short_entries = {}, {}
        long_success, short_success = [], []

        depth = 4
        tick_size = 0.1
        size = 25

        buy_amount, sell_amount = 1, 1
        timestamp = 0

        trades_count = self.trades.length()

        ask_spread_history = []
        bid_spread_history = []
        buy_amount_history = []
        sell_amount_history = []
        cum_ask_history = []
        cum_bid_history = []

        print('processing {} trade files'.format(trades_count))
        self.trades.next()
        while line := self.trades.next():

            trade = Trade(line)
    
            condition = lambda x: int(trade.timestamp) >= int(OrderBook(x).timestamp)
            book = OrderBook(self.order_book.search(condition))

            ask_price = float(getattr(book, 'asks')[0][0])
            bid_price = float(getattr(book, 'bids')[0][0])
            ask_spread = getattr(book, 'asks')[depth][0] - ask_price
            bid_spread = bid_price - getattr(book, 'bids')[depth][0]
            ask_volume = getattr(book, 'asks')[0][1]
            bid_volume = getattr(book, 'bids')[0][1]
            cum_ask_volume = np.sum(np.asarray(getattr(book, 'asks'))[:, 1]) 
            cum_bid_volume = np.sum(np.asarray(getattr(book, 'bids'))[:, 1]) 

            mid = ask_price + bid_price / 2

            fee_spread = ((mid * 1.0005**2) - mid)

            price = float(getattr(trade, 'price'))
            amount = float(getattr(trade, 'amount'))
            side = getattr(trade, 'side')

            long_keys = list(long_entries.keys())
            short_keys = list(short_entries.keys())

            if timestamp != getattr(trade, 'timestamp'):
                buy_amount, sell_amount = 1, 1

            timestamp = getattr(trade, 'timestamp')


            roc = 0
            if side == 'buy':

                if len(sell_amount_history) > 2:
                    a1, a2 = sell_amount_history[::-1][:2]
                    roc = amount - buy_amount_history[-1] / a2 - a1
                buy_amount += amount

            if side == 'sell':

                if len(buy_amount_history) > 2:
                    b1, b2 = buy_amount_history[::-1][:2]
                    roc = b2 - b1 / amount - sell_amount_history[-1]
                sell_amount += amount

            # 478052 81 0
            if ask_spread > fee_spread and ask_price not in long_keys:
                long_entries[ask_price] = (ask_spread, bid_spread, cum_ask_volume, cum_bid_volume, buy_amount, 0)

            if bid_spread > fee_spread and bid_price not in short_keys:
                short_entries[bid_price] = (ask_spread, bid_spread, cum_ask_volume, cum_bid_volume, sell_amount, 0)

            for i in range(0, len(long_keys)):

                key = long_keys[i]

                if key in long_success:
                    continue

                if price < key:

                    mdd = ((price - key) / key) * 100
                    if long_entries[key][-1] > mdd:
                        v = list(long_entries[key])

                        v[-1] = mdd

                        long_entries[key] = tuple(v)

                if price > key * 1.0005**2 and ask_volume > size:

                        long_success.append(key)

                        failed = 0
                        for key in long_keys:
                            if key not in long_success:
                                failed += 1

                        print('+ {entry}  {:06.2f} {:06.2f} {:09.2f} {:09.2f} {:08.2f} {:05.2f} {failed}'.format(entry=key, failed=failed, *long_entries[key]))

            for i in range(0, len(short_keys)):

                key = short_keys[i]

                if key in short_success:
                    continue

                if price > key:

                    mdd = -((price - key) / key) * 100
                    if short_entries[key][-1] > mdd:

                        v = list(short_entries[key])

                        v[-1] = mdd

                        short_entries[key] = tuple(v)

                if price < key * 0.9995**2 and bid_volume > size:
                    
                    short_success.append(key)

                    failed = 0
                    for key in short_keys:
                        if key not in short_success:
                            failed += 1

                    print('- {entry}  {:06.2f} {:06.2f} {:09.2f} {:09.2f} {:08.2f} {:05.2f} {failed}'.format(entry=key, failed=failed, *short_entries[key]))

        print(trades_count, len(long_success) + len(short_success), (len(long_entries) + len(short_entries)) - (len(long_success) + len(short_success)))

    def process_trades(self):

        depth = 4
        tick_size = 0.1

        previous_buy_amount, previous_sell_amount = 1, 1
        buy_change, sell_change = 1, 1

        entries = []
        total = self.trades.length()

        df = pd.DataFrame(columns=['timestamp', 'price', 'amount', 'side'])
        with tqdm(total=total) as pbar:

            # order_book_keys = np.asarray(list(self.order_book.keys()))

            for line in lzma.open(self.trades.files.pop(0), 'rb').readlines():
 
                trade = Trade(line)
        
                # index = np.argwhere(int(trade.timestamp) >= order_book_keys)
                # if index.size > 0:
                #     key = order_book_keys[index.size]
                #     book = OrderBook(self.order_book[key])

                timestamp = getattr(trade, 'timestamp')
                price = getattr(trade, 'price')
                amount = getattr(trade, 'amount')
                side = getattr(trade, 'side')

                # asks = getattr(book, 'asks')
                # bids = getattr(book, 'bids')

                df.loc[df.index.size] = [timestamp, price, amount, side]

                pbar.update(1)

            df2: pd.DataFrame = df.groupby(['timestamp', 'side']).agg(avg_tx_size=('amount', np.mean), amount=('amount', np.sum), price=('price', np.mean)).reset_index()
         
            df2['timestamp_diff'] = (df2['timestamp'] - df2.shift(1)['timestamp']) 
            df2['timestamp_diff_avg'] = df2.apply(lookback_window, values = df2['timestamp_diff'], method = 'mean', axis = 1).fillna(0)

            df2['log_returns'] = np.log(df2['price'] / df2.shift(1)['price'])
            df2['sd'] = df2.apply(lookback_window, values = df2['log_returns'], method = 'std', axis = 1).fillna(0)

            df2['volatility'] = df2.apply(lambda row: row.sd * np.sqrt((60 / 86400) / 365), axis = 1).fillna(0)

            df2['spread'] = df2.apply(lambda row: row.price * row.volatility * np.sqrt(row.avg_tx_size / row.amount), axis = 1).fillna(0)

            df2.fillna(0)
            df2.to_json('./introspect.json')

            # df['bid_ask_spread'] = df.apply(lambda row: row.asks[0][0] - row.bids[0][0], axis = 1)
            # df['ask_spread'] = df.apply(lambda row: row.asks[4][0] - row.asks[0][0], axis = 1)
            # df['bid_spread'] = df.apply(lambda row: row.bids[0][0] - row.bids[4][0], axis = 1)

            # df['ask_cum_volume'] = df.apply(lambda row: np.sum(np.asarray(row.asks)[:, 1]), axis = 1)
            # df['bid_cum_volume'] = df.apply(lambda row: np.sum(np.asarray(row.bids)[:, 1]), axis = 1)

            # df2: pd.DataFrame = df.groupby(['timestamp', 'side']).agg(avg_tx_size=('amount', np.mean), amount=('amount', np.sum), price=('price', np.mean), ask_cum_volume=('ask_cum_volume', np.mean), bid_cum_volume=('bid_cum_volume', np.mean)).reset_index()
         
            # df2['timestamp_diff'] = (df2['timestamp'] - df2.shift(1)['timestamp']) 

            # df2['timestamp_diff_avg'] = df2.apply(lookback_window, values = df2['timestamp_diff'], method = 'mean', axis = 1).fillna(0)
 
            # df2['sell_to_bid_cum'] = df2.apply(lambda row: row.amount / row.bid_cum_volume if row.side == 'sell' else row.amount / row.ask_cum_volume, axis = 1)
            # df2['buy_to_ask_cum'] = df2.apply(lambda row: row.amount / row.bid_cum_volume if row.side == 'sell' else row.amount / row.ask_cum_volume, axis = 1)
            
            # df.fillna(0)
            # df2.fillna(0)

            # print(df2)
            # print('dataframe size {} bytes'.format(df.memory_usage().sum()))

            # print('time between trades {} {} {}'.format(df2['timestamp_diff'].min(), df2['timestamp_diff'].max(), df2['timestamp_diff'].mean()))

            # print('volatility {} {} {}'.format(df2['volatility'].min(), df2['volatility'].max(), df2['volatility'].mean()))
            # print('spread {} {} {}'.format(df2['spread'].min(), df2['spread'].max(), df2['spread'].mean()))

            # print('bid ask spread {} {} {}'.format(df['bid_ask_spread'].min(), df['bid_ask_spread'].max(), df['bid_ask_spread'].mean()))

            # print('ask spread {} {} {}'.format(df['ask_spread'].min(), df['ask_spread'].max(), df['ask_spread'].mean()))
            # print('bid spread {} {} {}'.format(df['bid_spread'].min(), df['bid_spread'].max(), df['bid_spread'].mean()))

            # print('ask cum volume {} {} {}'.format(df['ask_cum_volume'].min(), df['ask_cum_volume'].max(), df['ask_cum_volume'].mean()))
            # print('bid cum volume {} {} {}'.format(df['bid_cum_volume'].min(), df['bid_cum_volume'].max(), df['bid_cum_volume'].mean()))

            # print('sell volume {} {} {}'.format(df2[df2['side'] == 'sell']['amount'].min(), df2[df2['side'] == 'sell']['amount'].max(), df2[df2['side'] == 'sell']['amount'].mean()))
            # print('buy volume {} {} {}'.format(df2[df2['side'] == 'buy']['amount'].min(), df2[df2['side'] == 'buy']['amount'].max(), df2[df2['side'] == 'buy']['amount'].mean()))
            
            # print('sell / bid cum {} {} {}'.format(df2['sell_to_bid_cum'].min(), df2['sell_to_bid_cum'].max(), df2['sell_to_bid_cum'].mean()))
            # print('buy / ask cum {} {} {}'.format(df2['buy_to_ask_cum'].min(), df2['buy_to_ask_cum'].max(), df2['buy_to_ask_cum'].mean()))

                # ask_price = getattr(book, 'asks')[0][0]
                # bid_price = getattr(book, 'bids')[0][0]
                # ask_spread = getattr(book, 'asks')[depth][0] - ask_price
                # bid_spread = bid_price - getattr(book, 'bids')[depth][0]
                # best_ask_volume = getattr(book, 'asks')[0][1]
                # best_bid_volume = getattr(book, 'bids')[0][1]
                # cum_ask_volume = np.sum(np.asarray(getattr(book, 'asks'))[:, 1]) 
                # cum_bid_volume = np.sum(np.asarray(getattr(book, 'bids'))[:, 1]) 

                # roc = 0
                # if side == 'buy':
                #     buy_change = amount - previous_buy_amount
                #     if buy_change == 0:
                #         buy_change = 1
                #     roc =  buy_change / sell_change

                # if side == 'sell':
                #     sell_change = amount - previous_sell_amount
                #     if sell_change == 0:
                #         sell_change = 1
                #     roc =  buy_change / sell_change

                # mid = ask_price + bid_price / 2

                # fee_spread = ((mid * 1.0005**2) - mid)

                # cum_ask_volume = '{:09.2f}'.format(cum_ask_volume)
                # cum_bid_volume = '{:09.2f}'.format(cum_bid_volume)

                # if price in entries:
                #     price = PURPLE + str(price) + END
        
                # if 'buy' in side and price not in entries:
                #     price = GREEN + str(price) + END

                # if 'sell' in side and price not in entries:
                #     price = RED + str(price) + END

                # if (ask_spread) > fee_spread and (bid_spread) < fee_spread * 0.2:
                #     ask_spread = BLUE + '{:06.2f}'.format(ask_spread) + END
                #     bid_spread = BLUE + '{:06.2f}'.format(bid_spread) + END
                # else:
                #     ask_spread = '{:06.2f}'.format(ask_spread)
                #     bid_spread = '{:06.2f}'.format(bid_spread)

                # if best_ask_volume > 1000:
                #     best_ask_volume = MAGENTA + '{:08.2f}'.format(best_ask_volume) + END
                # else:
                #     best_ask_volume = '{:08.2f}'.format(best_ask_volume)

                # if best_bid_volume > 1000:
                #     best_bid_volume = MAGENTA + '{:08.2f}'.format(best_bid_volume) + END
                # else:
                #     best_bid_volume = '{:08.2f}'.format(best_bid_volume)

                # if amount >= 1000:
                #     amount = MAGENTA + '{:07.2f}'.format(amount) + END
                # else:
                #     amount = '{:07.2f}'.format(amount)

                # if roc > 10:
                #     roc = PURPLE + '{:05.2f}'.format(roc) + END
                # else:
                #     roc = '{:06.2f}'.format(roc)

                # asks = list(np.asarray(getattr(book, 'asks'))[:5][:, 0])
                # bids = list(np.asarray(getattr(book, 'bids'))[:5][:, 0])
                # for i in range(0, len(asks)):

                #     if type(asks[i]) == 'str' or type(bids[i]) == 'str':
                #         continue

                #     if float(asks[i]) in entries:
                #         asks[i] = PURPLE + str(asks[i]) + END

                #     if float(bids[i]) in entries:
                #         bids[i] = PURPLE + str(bids[i]) + END

                # out = '{} - {} {} {} {} {} - {} {} {} {} {} - {} {} - {} {} - {} {} - {} {} - {}'.format(
                #     self.datetime(trade.timestamp / 1000), 
                #     asks[4], 
                #     asks[3], 
                #     asks[2], 
                #     asks[1], 
                #     asks[0], 
                #     bids[0], 
                #     bids[1], 
                #     bids[2], 
                #     bids[3], 
                #     bids[4], 
                #     cum_ask_volume,
                #     best_ask_volume, 
                #     price, 
                #     amount, 
                #     best_bid_volume, 
                #     cum_bid_volume,
                #     ask_spread, 
                #     bid_spread, 
                #     roc,
                # )

                # print(out)
        return total

    def dump_orders(self):

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

introspect = Introspect(exchange = 'okx')

# print(introspect.dump_position())

try:
    n_tic = time.process_time()
    total = introspect.process_trades()
    print('completed {} in {} ms'.format(total, 1000 * (time.process_time() - n_tic)))
except Exception as e:
    print(e.with_traceback())

# import pandas as pd
# import numpy as np

# df = pd.DataFrame(columns=['timestamp', 'amount', 'price', 'side', 'ask_cum_volume', 'bid_cum_volume'])
# df.loc[len(df.index)] = [1, 1, 40000, 'buy', 10, 20]
# df.loc[len(df.index)] = [1, 2, 40001, 'sell', 10, 20]
# df.loc[len(df.index)] = [2, 3, 45000, 'sell', 10, 20]
# df.loc[len(df.index)] = [3, 3, 45000, 'sell', 10, 20]
# df.loc[len(df.index)] = [4, 3, 45000, 'sell', 10, 20]

# df2: pd.DataFrame = df.groupby(['timestamp', 'side']).agg(avg_tx_size=('amount', np.mean), amount=('amount', np.sum), price=('price', 'first'), ask_cum_volume=('ask_cum_volume', 'first'), bid_cum_volume=('bid_cum_volume', 'first')).reset_index()

# df2['sd'] = df2.apply(lookback_window, values = df2['price'], method = 'std', axis = 1).fillna(0)

# df2['volatility'] = df2.apply(lambda row: row.price * row.sd * np.sqrt(1 / (df2['price'].index.get_loc(row.name) + 1)), axis = 1).fillna(0)

# print(df2.fillna(0))

# import struct, sys, time

# timestamp = int(time.time())

# bids = [[int(40000.45 * 1000), int(100.5 * 1000)]]
# asks = [[int(50000.45 * 1000), int(200.5 * 1000)]]

# bytes = b''
# bytes += struct.pack('i', timestamp)

# b = b''
# for arr in bids:
#     packed = struct.pack('i' * len(arr) , *arr)
#     b += packed

# a = b''
# for arr in asks:
#     packed = struct.pack('i' * len(arr) , *arr)
#     a += packed

# index = struct.pack('i', int(len(bids) + 2))

# bytes += index + b + a
# print(len(bytes))

# array = struct.unpack('i' * (len(bytes) // 4), bytes)
# import json

# df = pd.read_json('./introspect.json')

# print(df.to_string())
     



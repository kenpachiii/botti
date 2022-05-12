import glob
import json
import numpy as np
import logging
import zlib
import base64
import os
import time

from botti.cache import Cache
from botti.enums import PositionStatus, PositionState

logger = logging.getLogger(__name__)

class Introspect:

    def __init__(self, **kwargs) -> None:
        self.order_book: dict = self.read_order_book_files()
        self.trades: list = self.read_trade_files()
        self.cache = Cache('botti.db')

    def read_trade_files(self):

        trade_files = glob.glob('./dump/trades-*')
        trade_files.sort()

        trades = []

        for file in trade_files:
            data: list = json.loads(zlib.decompress(base64.b64decode(open(file).read())))

            if len(data) > 0:
                for trade in data:
                    trades.append(trade)

        return trades

    def read_order_book_files(self):

        order_book = {}

        order_book_files = glob.glob('./dump/order_book-*')
        order_book_files.sort()

        for file in order_book_files:
            data: dict = json.loads(zlib.decompress(base64.b64decode(open(file).read())))
            order_book[data.get('timestamp')] = data

        return order_book

    def process_trades(self):

        times = np.asarray(list(self.order_book.keys()))

        for trade in self.trades:

            index = np.argwhere(trade.get('timestamp') < times)
            # print(index)
            book = self.order_book[times[index[0][0]]]
            # print(trade.get('timestamp') >= times[index[-1][0]])

            print(book.get('datetime'), book.get('asks')[0][1], book.get('asks')[0][0], trade.get('price'), book.get('bids')[0][0], book.get('bids')[0][1])

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


introspect = Introspect()
print(introspect.process_trades())










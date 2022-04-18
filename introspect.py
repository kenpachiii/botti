import logging
import json
import os
import datetime

from botti.cache import Cache

logger = logging.getLogger(__name__)

class Introspect:

    def __init__(self, **kwargs):

        self.order_book = json.loads(open('order_book_dump').read())['BTC/USDT:USDT']
        self.cache = Cache('botti.db.dump')

    def dump_cache(self):
        self.cache.all()

    def dump_order_book(self):
        print(self.order_book)


introspect = Introspect()

# best bid > price > worst bid

def dump() -> None:

    path = os.path.join(os.getcwd(), 'dump')
    if not os.path.exists(path):
        os.mkdir(path)

    filename = 'order_book-' + datetime.datetime.now().isoformat()
    print(filename)

dump()
import logging
import json

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

print(not (introspect.order_book.get('bids')[0][0] > 39540.9 > introspect.order_book.get('bids')[1][0]))
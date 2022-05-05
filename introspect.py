import logging

from botti.cache import Cache

logger = logging.getLogger(__name__)

class Introspect:

    def __init__(self, **kwargs):
        # self.order_book = json.loads(open('./dump/order_book-2022-04-18T17:22:12.095802').read())['BTC/USDT:USDT']
        self.cache = Cache('botti.db')

    def dump_cache(self):
        self.cache.all()

    def dump_order_book(self):
        print(self.order_book)


introspect = Introspect()


print(introspect.dump_cache())

print(vars(introspect.cache.last))

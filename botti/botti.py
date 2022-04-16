from math import ceil
from xml.dom.minidom import parseString
import ccxtpro
import logging
import numpy as np
import asyncio
import json
import os

from botti.exchange import Exchange
from botti.cache import Cache
from botti.position import Position
from botti.retrier import retrier
from botti.sms import send_sms

logger = logging.getLogger(__name__)

class Botti:

    @classmethod
    def __init__(self, **kwargs: dict) -> None:

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.key: str = kwargs.get('key')
        self.secret: str = kwargs.get('secret')
        self.password: str = kwargs.get('password')
        self.test: bool = kwargs.get('test')

        self.symbol: str = 'BTC/USDT:USDT'
        self.fee: float = 0.0005
        self.leverage: int = 2

        self.p_t = 0
        self.cache: Cache = Cache()
        self.order_book: dict = {}

        self.okx: Exchange = ccxtpro.okx

        self.event = asyncio.Event()

    def __del__(self):
        """
        Destructor - clean up async stuff
        """
        self.close()

    def close(self):
        logger.info('{id} closed connection'.format(id=self.okx.id))
        self.loop.run_until_complete(self.okx.close())
        self.loop.close()

    def log_exception(self, origin, exception) -> None:
        logger.error('{id} {origin} - {error}'.format(id=self.okx.id, origin=origin, error=type(exception).__name__))
        send_sms('exception', 'origin: {id} {origin}\n\nmessage: {msg}'.format(id=self.okx.id, origin=origin, msg=type(exception).__name__))

    def market_depth(self, side, price, size) -> float:

        orders = np.asarray(self.order_book.get(side))

        # if worst bid price is greater than price, then return price 
        # implies order book window is out of range
        if 'bids' in side and orders[-1][0] > price:
            return price

        # if worst ask price is less than price, then return price 
        # implies order book window is out of range
        if 'asks' in side and orders[-1][0] < price:
            return price

        # find index where price < orders price
        index_arr = np.argwhere(price < orders[:,0])
        if index_arr.size > 0:
            # create orders slice and reverse
            orders = orders[:index_arr[-1][0]+1][::-1]

            # find index where position.open_amount <= cummulative contracts
            cum_index = np.argwhere(size <= np.cumsum(orders[:,1]))

            if cum_index.size > 0:
                return orders[cum_index[0][0]][0]
                   
        return price

    def break_even(self) -> tuple:

        position: Position = self.cache.position

        if not self.order_book or 'open' not in position.status:
            return (0, False) 

        break_even_price = position.open_avg * (1 + self.fee)**2

        # print(ceil(self.p_t), ceil(break_even_price), position.triggered)

        if self.p_t > break_even_price and position.triggered == 0:
            position.update({ 'triggered': 1 })
            self.cache.update(position)

        if self.p_t < break_even_price and position.triggered == 1:
            position.update({ 'triggered': 0 })
            self.cache.update(position)

        if position.triggered == 0:
            return (0, False)

        bid = self.market_depth('bids', break_even_price, self.cache.position.open_amount)
       
        # apply break even adjustment if needed
        if bid > break_even_price:
            logger.info('{exchange_id} break even adjusted - {_id} {_symbol} {break_even} -> {adj_price}'.format(**vars(position), exchange_id=self.okx.id, break_even=break_even_price, adj_price=bid))
            break_even_price = bid

        if ceil(self.p_t) == ceil(break_even_price):
            logger.info('{exchange_id} breaking even {_id} {_symbol} {p_t} < {break_even}'.format(exchange_id=self.okx.id, **vars(position), p_t=self.p_t, break_even=break_even_price))
            return (break_even_price, True)

        return (0, False)

    # FIXME: update to a moving window...?
    def trailing_entry(self) -> bool:

        if 'closed' not in self.cache.position.status:
            return False

        if not self.cache.last.id:
            logger.info('{id} trailing entry - no trades found'.format(id=self.okx.id))
            return True

        # upper limit
        if self.p_t > (self.cache.last.close_avg * 1.01):
            logger.info('{id} trailing entry - no trades found - upper limit hit {limit}'.format(id=self.okx.id, limit=self.cache.last.close_avg * 1.01))
            return True

        if self.p_t < (self.cache.last.close_avg * 0.98):
            logger.info('{id} trailing entry - no trades found - lower limit hit {limit}'.format(id=self.okx.id), limit=self.cache.last.close_avg * 0.98)
            return True

        return False

    def take_profits(self):
        return 'open' in self.cache.position.status and self.cache.position.open_amount > 0 and self.p_t > self.cache.position.open_avg * 1.05

    def add_position(self, order: dict) -> None:
        try:
            self.cache.insert(Position({ 
                'id': os.urandom(6).hex(), 
                'timestamp': order.get('timestamp'), 
                'symbol': order.get('symbol'), 
                'side': order.get('side'), 
                'open_amount': order.get('filled') if order.get('status') == 'closed' else 0.0, 
                'open_avg': order.get('average') if order.get('status') == 'closed' else 0.0, 
                'close_amount': 0.0, 
                'close_avg': 0.0, 
                'status': 'open' if order.get('status') == 'closed' else 'pending', 
                'triggered': 0 
            }))
            logger.info('{exchange_id} add position - {_id} {_symbol} {_timestamp} {_open_avg} {_open_amount} {_close_avg} {_close_amount} {_status}'.format(exchange_id = self.okx.id, **vars(self.cache.position)))
        except Exception as e:
            self.log_exception('add position', e)
        
    def update_position(self, order: dict) -> None:

        try: 

            position: Position = self.cache.position
            position.update({ 'timestamp': order.get('timestamp') })

            if order.get('status') == 'open':
                return
            
            type = 'add' if order.get('side') == position.side else 'reduce'
            if type == 'add':

                if order.get('status') == 'closed':
                    avg = position.position_avg('open', order)
                    position.update({ 'open_amount': position.open_amount + order.get('filled'), 'open_avg': avg })

                    if position.status == 'pending':
                        position.update({ 'status': 'open' })

            if type == 'reduce':
                 
                if order.get('status') == 'closed':
                    position.update({ 'open_amount': position.open_amount - order.get('filled') })
        
                    avg = position.position_avg('close', order)
                    position.update({ 'close_amount': position.close_amount + order.get('filled'), 'close_avg': avg })

                if position.open_amount == 0:
                    position.update({ 'status': 'closed' })

            logger.info('{exchange_id} update position - {_id} {_symbol} {_timestamp} {_open_avg} {_open_amount} {_close_avg} {_close_amount} {_status} {pnl}'.format(exchange_id = self.okx.id, pnl = position.pnl(self.leverage) if position.open_amount == 0 else '', **vars(position)))
            self.cache.update(position)

        except Exception as e:
            self.log_exception('update position', e)

    async def position_size(self) -> float:

        response: dict = None

        try:
            response = await self.okx.private_get_account_max_size({ 'instId': self.okx.market_id(self.symbol), 'tdMode': 'cross', 'ccy': 'BTC', 'leverage': 2 })
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception('position size', e)
        finally:
            return float(response.get('data')[0].get('maxBuy')) * (1 - self.fee)**2

    async def create_order(self, type: str, side: str, size: float, price: float = None, params: dict={}) -> None:
        try:
            await self.okx.create_order(self.symbol, type, side, size, price, params)
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception('create order', e)

        # reset event so consume waits on order
        # self.event.clear()

    async def check_open_position(self):

        logger.info('{id} checking for open positions'.format(id=self.okx.id))

        response: dict = {}
        try:
            response = await self.okx.fetch_position(self.symbol)

            if response:
                await self.orders_history()
            else:
                logger.info('{id} no position clearing cache'.format(id=self.okx.id))
                self.cache.clear()

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception('check open positions', e)

    # FIXME: combine orders_history and watch_orders
    async def orders_history(self):

        logger.info('{id} re-syncing orders'.format(id=self.okx.id))

        response: dict = {}
        try:
            # FIXME: params need to be 'smarter'
            # getting orders this way may not get cached by ccxtpro
            response = await self.okx.private_get_trade_orders_history({ 'instId': self.okx.market_id(self.symbol), 'instType': 'SWAP', 'limit': 100 })

            if response:
                orders = self.okx.parse_orders(response.get('data'), since = self.cache.position.timestamp + 1) # add 1 so since conditional ignores most recent entry
                
                for order in orders:

                    # TODO: does this need to be handled too...?
                    if order.get('status') in ['canceled', 'expired', 'rejected']:
                        logger.info('{exchange_id} recent order {status}'.format(exchange_id = self.okx.id, status = order.get('status')))
                        continue 

                    # adding position
                    if self.cache.position.symbol == None:
                        self.add_position(order)
                        continue

                    # updating position
                    if self.cache.position.symbol == order.get('symbol'):
                        self.update_position(order)

                # once orders are processed re-start consuming
                # self.event.set()

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception('orders history', e)

    async def watch_trades(self):

        try:
            while True:
                trades: list[dict] = await self.okx.watch_trades(self.symbol)

                for trade in trades:
                    self.p_t = trade.get('price')

                    # break even
                    price, ok = self.break_even()
                    if ok:
                        await self.create_order('fok', 'sell', self.cache.position.open_amount, price, params = { 'tdMode': 'cross', 'posSide': 'long' })

                    # trailing entry
                    if self.trailing_entry():
                        size = await self.position_size() 
                        price = self.cache.last.close_avg if self.cache.last.close_avg != 0 else self.p_t
                        await self.create_order('fok', 'buy', size, price, params = { 'tdMode': 'cross', 'posSide': 'long' })
                        
                    # take profits
                    if self.take_profits():
                        await self.create_order('fok', 'sell', self.cache.position.open_amount, self.p_t, params = { 'tdMode': 'cross', 'posSide': 'long' })
                        logger.info('{id} take profits - target hit'.format(id=self.okx.id))
                        send_sms('profits', 'target hit {}'.format(self.cache.last.pnl(self.leverage)))
  
                self.okx.trades[self.symbol].clear()

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception('watch trades', e)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e) 

    async def watch_order_book(self):

        try:
            while True:
                self.order_book = await self.okx.watch_order_book(self.symbol, limit = 100)
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception('watch order book', e)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e) 

    async def watch_orders(self):

        try:
            while True:

                # FIXME: ommiting symbol is the only way to get type ANY otherwise ccxt uses the market object
                # which seems to produce unfavorable results
                orders: list[dict] = await self.okx.watch_orders(self.symbol, limit = 1)

                with open('dump', 'w') as json_file:
                    json.dump(self.okx.orders, json_file, 
                                indent=4,  
                                separators=(',',': '))
                
                for order in orders:

                    # print(order['side'], order['status'], order['average'], order['filled'])

                    if order.get('status') in ['canceled', 'expired', 'rejected']:

                        # TODO: will this produce any unwanted side effects...?
                        if self.cache.position.get('status') == 'pending':
                            self.cache.clear()

                        logger.info('{exchange_id} recent order {status}'.format(exchange_id = self.okx.id, status=order.get('status')))
                        continue 

                    # adding position
                    if self.cache.position.symbol == None:
                        self.add_position(order)
                        continue

                    # updating position
                    if self.cache.position.symbol == order.get('symbol'):
                        self.update_position(order)

                # once orders are processed re-start consuming
                # self.event.set()

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception('watch orders', e)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e) 

    async def consume(self):

        await asyncio.sleep(5)

        try: 
            while True:

                # break even
                price, ok = self.break_even()
                if ok:
                    await self.create_order('fok', 'sell', self.cache.position.open_amount, price, params = { 'tdMode': 'cross', 'posSide': 'long' })

                # trailing entry
                if self.trailing_entry():
                    size = await self.position_size() * 0.5
                    price = self.cache.last.close_avg if self.cache.last.close_avg != 0 else self.p_t
                    await self.create_order('market', 'buy', size, None, params = { 'tdMode': 'cross', 'posSide': 'long' })
                    
                # take profits
                if self.take_profits():
                    await self.create_order('fok', 'sell', self.cache.position.open_amount, self.p_t, params = { 'tdMode': 'cross', 'posSide': 'long' })
                    logger.info('{id} take profits - target hit'.format(id=self.okx.id))
                    send_sms('profits', 'target hit {}'.format(self.cache.last.pnl(self.leverage)))

                # if an open or close order is created wait until watch orders receives it.
                # await self.event.wait()

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception('consume', e)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e) 

    async def system_status(self):
        try: 
            response: dict = await self.okx.public_get_system_status()
            for status in response.get('data'):
                print(self.okx.iso8601(int(status.get('begin'))), self.okx.iso8601(int(status.get('end'))), status.get('serviceType'), status.get('state'), status.get('system'), status.get('title'))
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception('system status', e)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e) 

    # FIXME: ccxt docs claim to handle all retry attempts
    # @retrier
    def run(self):

        logger.info('starting botti')

        try: 

            self.okx = Exchange({
                'asyncio_loop': self.loop,
                'newUpdates': True,
                'apiKey': self.key,
                'secret': self.secret,
                'password': self.password,
                'options': { 'watchOrderBook': { 'depth': 'books' }}
            })

            self.okx.set_sandbox_mode(self.test)
            self.loop.run_until_complete(self.okx.load_markets(reload=False))

            # make sure leverage is updated
            self.loop.run_until_complete(self.okx.set_leverage(self.leverage, self.symbol, params = { 'mgnMode': 'cross' }))

            # required to repopulate an already opened position
            self.loop.run_until_complete(self.check_open_position())
            loops = [
                    self.watch_orders(),
                    self.watch_order_book(),
                    self.watch_trades()
                    # self.consume()
                ]

            self.loop.run_until_complete(asyncio.gather(*loops))

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception('run', e)

            # raise NetworkError to be recieved by retrier
            if type(e).__name__ == 'NetworkError':
                self.close()
                raise ccxtpro.NetworkError(e) 

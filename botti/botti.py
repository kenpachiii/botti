import ccxtpro
import logging
import numpy as np
import asyncio
import time
import os
import json

from botti.cache import Cache
from botti.position import Position
from botti.retrier import retrier
from botti.sms import send

logger = logging.getLogger(__name__)

class Botti:

    @classmethod
    def __init__(self, **kwargs: dict) -> None:

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.key: str = kwargs['key']
        self.secret: str = kwargs['secret']
        self.password: str = kwargs['password']
        self.test: str = kwargs['test']

        self.symbol: str = 'BTC/USDT:USDT'
        self.fee: float = 0.0005
        self.leverage: int = 2

        self.p_t = 0
        self.cache: Cache = Cache()
        self.order_book: dict = {}

        self.okx: ccxtpro.okx = ccxtpro.okx

    def market_depth(self, side, price, size) -> float:

        orders = np.asarray(self.order_book[side])

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

    async def break_even(self) -> None:

        position: Position = self.cache.position

        if not self.order_book or 'open' not in position.status:
            return 

        break_even_price = position.open_avg * (1 + self.fee)**2

        if self.p_t > break_even_price and position.triggered == 0:
            position.update({ 'triggered': 1 })
            self.cache.update(position)

        if self.p_t < break_even_price and position.triggered == 1:
            position.update({ 'triggered': 0 })
            self.cache.update(position)

        if position.triggered == 0:
            return

        bid = self.market_depth('bids', break_even_price, self.cache.position.open_amount)

        # apply break even adjustment if needed
        if bid > break_even_price:
            logger.info('{exchange_id} break even adjusted - {_id} {_symbol} {break_even} -> {adj_price}'.format(**vars(position), exchange_id=self.okx.id, break_even=break_even_price, adj_price=bid))
            break_even_price = bid

        if self.p_t < break_even_price:
            logger.info('{exchange_id} breaking even {_id} {_symbol} {p_t} < {break_even}'.format(exchange_id=self.okx.id, **vars(position), p_t=self.p_t, break_even=break_even_price))
            await self.create_order('fok', 'sell', position.open_amount, break_even_price, params = { 'tdMode': 'cross', 'posSide': 'long' })


    async def trailing_entry(self) -> bool:

        price = self.cache.last.close_avg if self.cache.last.close_avg != 0 else self.p_t

        if 'closed' in self.cache.position.status:
            # logger.info('{id} trailing entry - no trades found'.format(id=self.okx.id))
            return True
        return False

        if 'closed' not in self.cache.position.status:
            return False

        # upper limit
        if self.p_t > (price * 1.01):
            # logger.info('{id} trailing entry - no trades found - upper limit hit {limit}'.format(id=self.okx.id, limit=price * 1.01))
            return True

        if self.p_t < (price * 0.98):
            # logger.info('{id} trailing entry - no trades found - lower limit hit {limit}'.format(id=self.okx.id), limit=price * 0.98)
            return True

        return False

    async def take_profits(self):
        return 'open' in self.cache.position.status and self.cache.position.open_amount > 0 and self.p_t > self.cache.position.open_avg * 1.05

    async def position_size(self) -> float:

        response: dict = None

        try:
            response = await self.okx.private_get_account_max_size({ 'instId': self.okx.market_id(self.symbol), 'tdMode': 'cross', 'ccy': 'BTC', 'leverage': 2 })
        except Exception as e:
            logger.error('{id} max size - {error}'.format(id=self.okx.id, error=str(e)))
        finally:
            return float(response['data'][0]['maxBuy']) * (1 - self.fee)**2

    async def create_order(self, type: str, side: str, size: float, price: float = None, params={}) -> None:
        
        try:
            await self.okx.create_order(self.symbol, type, side, size, price, params)
        except Exception as e:
            logger.error('{id} - create order - {error}'.format(id=self.okx.id, error=str(e)))

    async def add_position(self, order) -> None:

        try:
            self.cache.insert(Position({ 
                'id': os.urandom(6).hex(), 
                'timestamp': int(time.time()), 
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
            logger.error('{id} add position - {error}'.format(id=self.okx.id, error=str(e)))
        
    async def update_position(self, order: dict) -> None:

        try: 

            position: Position = self.cache.position

            # TODO: do pending unfilled orders need pre-processing?
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

            logger.info('{exchange_id} update position - {_id} {_symbol} {_timestamp} {_open_avg} {_open_amount} {_close_avg} {_close_amount} {_status} {pnl}'.format(exchange_id = self.okx.id, pnl = position.pnl() if position.open_amount == 0 else '', **vars(position)))
            self.cache.update(position)

        except Exception as e:
            logger.error('{id} update position - {error}'.format(id=self.okx.id, error=str(e)))

    @retrier
    async def watch_trades(self):
        try:
            while True:
                trades = await self.okx.watch_trades(self.symbol)

                for trade in trades:

                    self.p_t = trade['price']

                    # break even
                    await self.break_even()

                    # trailing entry
                    if await self.trailing_entry():
                        size = await self.position_size() * 0.5 # FIXME: for some reason max + market throws error
                        price = self.cache.last.close_avg if self.cache.last.close_avg != 0 else self.p_t
                        await self.create_order('market', 'buy', size, price, params = { 'tdMode': 'cross', 'posSide': 'long' })
                        
                    # take profits
                    if await self.take_profits():
                        logger.info('{id} take profits - target hit'.format(id=self.okx.id))
                        await self.create_order('market', 'sell', self.cache.position.open_amount, self.p_t, params = { 'tdMode': 'cross', 'posSide': 'long' })
               
                self.okx.trades[self.symbol].clear()

        except ccxtpro.NetworkError as e:
            raise ccxtpro.NetworkError(e)
        except Exception as e:
            logger.error('{id} watch trades - {error}'.format(id=self.okx.id, error=str(e)))

    @retrier
    async def watch_order_book(self):
        try:
            while True:
                self.order_book = await self.okx.watch_order_book(self.symbol, limit = 100)
        except ccxtpro.NetworkError as e:
            raise ccxtpro.NetworkError(e)
        except Exception as e:
            logger.error('{id} watch order book - {error}'.format(id=self.okx.id, error=str(e)))

    @retrier
    async def watch_orders(self):
        try:
            while True:
                orders = await self.okx.watch_orders(self.symbol, limit = 1)

                with open('dump', 'w') as json_file:
                    json.dump(self.okx.orders, json_file, 
                                indent=4,  
                                separators=(',',': '))
                
                for order in orders:

                    # print(order['side'], order['status'], order['average'], order['filled'])

                    if order.get('status') in ['canceled', 'expired', 'rejected']:
                        logger.info('{exchange_id} recent order {status}'.format(exchange_id = self.okx.id, status=order['status']))
                        continue 

                    # adding position
                    if self.cache.position.symbol == None:
                        await self.add_position(order)
                        continue

                    # updating position
                    if self.cache.position.symbol == order.get('symbol'):
                        await self.update_position(order)

        except ccxtpro.NetworkError as e:
            raise ccxtpro.NetworkError(e)
        except Exception as e:
            logger.error('{id} watch orders - {error}'.format(id=self.okx.id, error=str(e)))

    def __del__(self):
        """
        Destructor - clean up async stuff
        """
        self.close()

    def close(self):
        logger.info('{id} closed connection'.format(id=self.okx.id))
        self.loop.run_until_complete(self.okx.close())

    # TODO: execeptions arent always super descriptive
    # a notification should be sent on unrecoverable errors
    # data needs to be rechecked and updated after recoverable retries
    def run(self):

        logger.info('starting botti')

        try: 

            self.okx = ccxtpro.okx({
                'asyncio_loop': self.loop,
                'newUpdates': True,
                'apiKey': self.key,
                'secret': self.secret,
                'password': self.password,
                'options': { 'defaultType': 'swap' }
            })

            self.okx.set_sandbox_mode(self.test)
            self.loop.run_until_complete(self.okx.load_markets(reload=False))

            # await self.okx.set_leverage(self.leverage, self.symbol, params = { 'mgnMode': 'cross' })

            # required to repopulate an already opened position
            # self.loop.run_until_complete(self.check_open_position())
            loops = [
                    self.watch_orders(),
                    self.watch_order_book(),
                    self.watch_trades()
                ]

            self.loop.run_until_complete(asyncio.gather(*loops))

        # TODO: how should retrying work?
        except ccxtpro.NetworkError as e:
            raise ccxtpro.NetworkError(e)
        except Exception as e:
            logger.error('{id} run - {error}'.format(id=self.okx.id, error=str(e)))




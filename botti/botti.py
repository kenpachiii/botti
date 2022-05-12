import time
import ccxtpro
import logging
import numpy as np
import asyncio
import aiofiles
import json
import os
import datetime
import zlib
import base64

from botti.exceptions import log_exception
from botti.exchange import Exchange
from botti.cache import Cache
from botti.enums import SystemState, ServiceType, PositionStatus, PositionState
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

        self.symbol: str = kwargs.get('symbol')
        self.fee: float = kwargs.get('fee')
        self.leverage: int = kwargs.get('leverage')
        self.upper_limit: int = kwargs.get('upper_limit')
        self.lower_limit: int = kwargs.get('lower_limit')
        self.tp: int = kwargs.get('tp')

        self.collect = False

        self.cursor: float = -1 # -1 = needs to be set, 0 = upper limit hit, cursor > 0 = lower limit
        self.cache: Cache = Cache()

        self.queue = asyncio.Queue()

        self.okx: Exchange = ccxtpro.okx

    def __del__(self):
        """
        Destructor - clean up async stuff
        """
        self.close()

    def close(self):

        if len(asyncio.all_tasks(self.loop)) > 0:
            logger.info('{id} canceling tasks'.format(id=self.okx.id))
            for task in asyncio.all_tasks(self.loop):
                task.cancel()

        self.loop.run_until_complete(self.okx.close())
        logger.info('{id} closed connection'.format(id=self.okx.id))

        logger.info('{id} closed loop'.format(id=self.okx.id))
        self.loop.close()

    # FIXME: can this be parallel?
    async def dump(self):

        order_book_timestamp = 0
        trade_timestamp = 0

        try:

            while True:

                await asyncio.sleep(0)

                if len(self.okx.orderbooks) == 0:
                    continue

                path = os.path.join(os.getcwd(), 'dump')
                if not os.path.exists(path):
                    os.mkdir(path)

                timestamp = datetime.datetime.now().isoformat()

                book: dict = self.okx.orderbooks.get(self.symbol)
      
                if book.get('timestamp') > order_book_timestamp:

                    filename = 'order_book-' + timestamp
                    async with aiofiles.open(os.path.join(path, filename), mode='w') as f:
                        bytes = base64.b64encode(
                            zlib.compress(
                                json.dumps(book).encode('utf-8')
                            )
                        ).decode('ascii')
                        await f.write(bytes)

                    order_book_timestamp = book.get('timestamp')
                    
                if len(self.okx.trades[self.symbol]) > 0 and self.okx.trades[self.symbol][0].get('timestamp') > trade_timestamp:
                    filename = 'trades-' + timestamp
                    async with aiofiles.open(os.path.join(path, filename), mode='w') as f:
                        bytes = base64.b64encode(
                            zlib.compress(
                                json.dumps(self.okx.trades[self.symbol]).encode('utf-8')
                            )
                        ).decode('ascii')
                        await f.write(bytes)

                    trade_timestamp = self.okx.trades[self.symbol][0].get('timestamp')

                self.okx.trades[self.symbol].clear()

        except Exception as e:
            log_exception(e, self.okx.id)
 
    def market_depth(self, side: str, price: float, size: float, limit: float = 100) -> float:

        orders = np.asarray(self.okx.orderbooks[self.symbol].get(side))[:limit]

        # bid window = best bid > price > worst bid
        if 'bids' in side and not (orders[0][0] >= price >= orders[-1][0]):
            return price

        # ask window = best ask < price < worst ask
        if 'asks' in side and not (orders[0][0] <= price <= orders[-1][0]):
            return price

        index_arr: np.ndarray
        if 'bids' in side:
            # find index where price < orders price
            index_arr = np.argwhere(price <= orders[:, 0])

        if 'asks' in side:
            # find index where price > orders price
            index_arr = np.argwhere(price >= orders[:, 0])

        if index_arr.size > 0:
            orders = orders[:index_arr[-1][0]+1]

        return np.sum(orders[:, 1]) >= size

    def break_even(self) -> bool:

        break_even_price = self.cache.position.open_avg * (1 + self.fee)**2
        best_bid = self.okx.orderbooks[self.symbol].get('bids')[0][0]

        return best_bid > break_even_price

    def mid_point(self):
        return (self.okx.orderbooks[self.symbol].get('bids')[0][0] + self.okx.orderbooks[self.symbol].get('asks')[0][0]) / 2
 
    def trailing_entry(self) -> bool:

        if self.cache.fetch_order() == {} and self.cursor == -1:

            self.cursor = self.mid_point()

            logger.info(
                '{id} trailing entry - no previous entries - starting fresh at {entry}'.format(
                    id=self.okx.id,entry=self.cursor))

            return False

        self.cursor = self.cursor if self.cursor > 0 else self.mid_point()

        # upper limit # FIXME: make smarter
        if (self.okx.orderbooks[self.symbol].get('asks')[0][0] > (self.cursor * self.upper_limit)):

            logger.info('{id} trailing entry - no trades found - upper limit hit {limit}'.format(
                id=self.okx.id, limit=self.cursor * self.upper_limit))

            self.cursor = 0

            return True

        # lower limit
        if (self.okx.orderbooks[self.symbol].get('bids')[0][0] < (self.cursor * self.lower_limit)):

            logger.info('{id} trailing entry - no trades found - lower limit hit {limit}'.format(
                id=self.okx.id, limit=self.cursor * self.lower_limit))

            self.cursor = self.cursor * self.lower_limit

            return False

        return False

    def take_profits(self): 
        return self.okx.orderbooks[self.symbol].get('bids')[0][0] > self.cache.position.open_avg * self.tp and self.market_depth('bids', self.cache.position.open_avg * (1 + self.fee)**2, self.cache.position.open_amount)

    def process_orders(self, orders):

        for order in orders:

            self.cache.insert_order(order)

            if order.get('status') in ['canceled', 'expired', 'rejected']:

                logger.info('{exchange_id} recent order {status}'.format(
                    exchange_id=self.okx.id, status=order.get('status')))

                if self.cache.position.status is PositionStatus.PENDING:
                    self.cache.remove('position', self.cache.position.id)

                if self.cache.position.state is PositionState.EXIT and self.cache.position.side == order.get('side') and order.get('filled') > self.cache.position.pending_close_amount:
                    
                    args = self.symbol, 'limit', 'sell', order.get('filled') - self.cache.position.pending_close_amount, self.cache.position.open_avg * (1 + self.fee)**2, { 'tdMode': 'cross', 'posSide': 'long' }
                    self.queue.put_nowait(('create', PositionState.EXIT, args))

                    logger.info('{id} process orders - adjusting for order surplus'.format(id=self.okx.id))

                continue

            # adjust position
            self.adjust_position(order)

    def adjust_position(self, order: dict) -> None:

        try:

            if order.get('filled') == 0:
                return

            type = 'increase' if order.get('side') == self.cache.position.side else 'decrease'
            if type == 'increase':

                amount = order.get('filled')

                # fetch previous orders and get the non-cummulative amount
                orders = self.cache.fetch_orders(order.get('id'))
                if len(orders) > 1:
                    amount = amount - orders[len(orders) - 2].get('filled')
           
                self.cache.update(self.cache.position.id, 
                    {
                        'open_amount': self.cache.position.open_amount + amount, 
                        'open_avg': order.get('average')
                    }
                )

                if self.cache.position.status is PositionStatus.PENDING:
                    self.cache.update(self.cache.position.id, { 'status': PositionStatus.OPEN })

            if type == 'decrease':

                amount = order.get('filled')

                # fetch previous orders and get the non-cummulative amount
                orders = self.cache.fetch_orders(order.get('id'))
                if len(orders) > 1:
                    amount = amount - orders[len(orders) - 2].get('filled')

                self.cache.update(self.cache.position.id, 
                    {
                        'close_amount': self.cache.position.close_amount + amount, 
                        'close_avg': order.get('average')
                    }
                )

                if self.cache.position.open_amount == self.cache.position.close_amount:
                    self.cache.update(self.cache.position.id, { 'status': PositionStatus.CLOSED })

                    if self.cache.last.pnl(self.leverage) > 0:
                        send_sms('profits', 'position closed\n\n+{:.2f}%'.format(self.cache.last.pnl(self.leverage)))

            position = self.cache.position if self.cache.position.id else self.cache.last

            logger.info('{exchange_id} adjusted position - {id} {symbol} {open_avg} {open_amount} {close_avg} {close_amount} {status} {state} {pnl}'.format(
                exchange_id=self.okx.id, id=position.id,symbol=position.symbol, open_avg=position.open_avg, open_amount=position.open_amount,close_avg=position.close_avg, close_amount=position.close_amount,status=position.status,state=position.state,pnl=position.pnl(self.leverage) if position.open_amount == position.close_amount else ''))
            
        except Exception as e:
            log_exception(e, self.okx.id)
  
    async def position_size(self, side: str = 'long') -> float:

        response: dict = None

        try:
            params = { 'instId': self.okx.market_id(self.symbol), 'tdMode': 'cross', 'ccy': self.okx.markets.get(self.symbol).get('base'), 'leverage': self.leverage }
            response = await self.okx.private_get_account_max_size(params)
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)
        finally:

            data = response.get('data')[0]
            sz = data.get('maxBuy') if 'long' in side else data.get('maxSell')

            return float(sz) 

    async def consumer(self):

        while True:

            await asyncio.sleep(0)
            if not self.queue.empty():
                try:

                    (command, state, args) = self.queue.get_nowait()
                    if command == 'create':
                        await self.okx.create_order(*args)
                    
                    if command == 'cancel':
                        await self.okx.cancel_order(*args)

                except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
                    
                    # sometimes order completes before it can be canceled
                    if type(e).__name__ == 'OrderNotFound' and command == 'cancel':
                        logger.info('{id} consumer - order completed before canceling'.format(id=self.okx.id))

                        if self.cache.position.state is PositionState.EXIT and self.cache.position.pending_open_amount > self.cache.position.pending_close_amount:

                            args = self.symbol, 'limit', 'sell', self.cache.position.pending_open_amount - self.cache.position.pending_close_amount, self.cache.position.open_avg * (1 + self.fee)**2, { 'tdMode': 'cross', 'posSide': 'long' }
                            self.queue.put_nowait(('create', PositionState.EXIT, args))

                        continue

                    if type(e).__name__ == 'InsufficientFunds' and state is PositionState.ENTRY and command == 'create':
                        logger.info('{id} consumer - insufficient funds removing {position}'.format(id=self.okx.id,position=self.cache.position.id))
                        self.cache.remove('position', self.cache.position.id)
                        continue

                    if type(e).__name__ == 'InvalidOrder' and state is PositionState.ENTRY and command == 'create':
                        logger.info('{id} consumer - insufficient funds removing {position}'.format(id=self.okx.id,position=self.cache.position.id))
                        self.cache.remove('position', self.cache.position.id)
                        continue

                    log_exception(e, self.okx.id) 

                self.queue.task_done()

    # FIXME: doesnt handle missed entries
    async def strategy(self) -> None:

        while True:

            await asyncio.sleep(0)
            if len(self.okx.orderbooks) == 0:
                continue

            try:

                # exit 
                if self.cache.position.status is PositionStatus.OPEN and self.cache.position.state is PositionState.ENTRY:

                    if self.break_even():

                        self.cache.update(self.cache.position.id, { 'state': PositionState.EXIT })

                        order = self.cache.fetch_order()
                        if order.get('status') == 'open':
                            args = order.get('id'), self.symbol
                            self.queue.put_nowait(('cancel', PositionState.EXIT, args))

                        self.cache.update(self.cache.position.id, { 'pending_close_amount': self.cache.position.open_amount })

                        args = self.symbol, 'limit', 'sell', self.cache.position.open_amount, self.cache.position.open_avg * (1 + self.fee)**2, { 'tdMode': 'cross', 'posSide': 'long' }
                        self.queue.put_nowait(('create', PositionState.EXIT, args))

                # entry
                if not self.cache.position.status and self.trailing_entry():

                    self.cache.insert({ 'id': os.urandom(6).hex(), 'timestamp': int(time.time_ns() / 1000), 'symbol': self.symbol, 'side': 'buy', 'state': PositionState.ENTRY, 'status':  PositionStatus.PENDING })

                    size = await self.position_size('long') 
                    if size == 0:
                        logger.info('{id} trailing entry - size was zero'.format(id=self.okx.id))
                        continue
                    
                    best_ask = self.okx.orderbooks[self.symbol].get('asks')[0][0]

                    self.cache.update(self.cache.position.id, { 'pending_open_amount': size })

                    args = self.symbol, 'limit', 'buy', size, best_ask + 1, { 'tdMode': 'cross', 'posSide': 'long' }
                    self.queue.put_nowait(('create', PositionState.ENTRY, args))

            except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
                log_exception(e, self.okx.id)

    async def orders_history(self) -> None:

        order = self.cache.fetch_order()
        if order == {}:
            return

        logger.info('{id} re-syncing orders'.format(id=self.okx.id))

        response: dict = {}
        try:
            response = await self.okx.private_get_trade_orders_history({ 'instId': self.okx.market_id(self.symbol), 'instType': 'SWAP', 'state': 'filled', 'before': order.get('id') })

            if response:
                orders = self.okx.parse_orders(response.get('data'))
                self.process_orders(orders)

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)

    async def set_leverage(self):

        params={ 'mgnMode': 'cross' }

        try:
            response: dict = await self.okx.fetch_leverage(self.symbol, params)
            if int(response.get('data')[0].get('lever')) is not self.leverage:
                await self.okx.set_leverage(self.leverage, self.symbol, params)

                logger.info('{id} set leverage to {leverage}'.format(id=self.okx.id,leverage=self.leverage))

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e)

    async def watch_trades(self):

        try:
            while True:
                await self.okx.watch_trades(self.symbol)

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e)

    async def watch_order_book(self):

        try:
            while True:
                await self.okx.watch_order_book(self.symbol, limit=20)               
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e)

    async def watch_orders(self):
        try:
            while True:
                # limit = 1 seems to be only way that works to get only the newest updates from cache
                orders = await self.okx.watch_orders(self.symbol,limit=1)
                self.process_orders(orders)

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e)

    async def system_status(self):
        
        try:
            response: dict = await self.okx.public_get_system_status()
            for status in response.get('data'):

                title = status.get('title')
                start = self.okx.iso8601(int(status.get('begin')))
                end = self.okx.iso8601(int(status.get('end')))
                state = SystemState(status.get('state'))
                service = ServiceType(status.get('serviceType'))

                logger.warning(f'system status - {title} {start} {end} {state} {service}')

                send_sms('system-status', f'system status\n{title}\n{start}\n{end}\n{state}\n{service}')

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e)

    def run(self):

        logger.info('starting botti')

        try:

            self.okx = Exchange({
                'asyncio_loop': self.loop,
                'newUpdates': True,
                'apiKey': self.key,
                'secret': self.secret,
                'password': self.password,
                'options': { 'rateLimit': 10, 'watchOrderBook': { 'depth': 'books' }, 'newUpdates': True }
            })

            self.loop.run_until_complete(self.system_status())

            self.okx.set_sandbox_mode(self.test)

            self.loop.run_until_complete(self.okx.load_markets(reload=False))
            # make sure leverage is updated
            self.loop.run_until_complete(self.set_leverage())

            # required to repopulate an already opened position
            self.loop.run_until_complete(self.orders_history())
            loops = [
                self.watch_orders(),
                self.watch_order_book(),
                self.watch_trades(),
                self.consumer(),
                self.strategy(),
                self.dump() 
            ]

            self.loop.run_until_complete(asyncio.gather(*loops))

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)

            # raise so systemd restarts otherwise let daemon shutdown
            if type(e).__name__ == 'NetworkError':
    
                raise ccxtpro.NetworkError(e)

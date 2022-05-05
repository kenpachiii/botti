from math import ceil, floor
import ccxtpro
import logging
import numpy as np
import asyncio
import json
import os
import traceback
import datetime

from enum import Enum

from botti.exchange import Exchange
from botti.cache import Cache
from botti.position import Position, PositionStatus
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

        self.p_t: float = 0
        self.cursor: float = 0
        self.cache: Cache = Cache()

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

    def log_exception(self, e: Exception) -> None:

        frame = None

        stack = traceback.extract_tb(e.__traceback__)

        root = os.path.dirname(os.path.abspath(__file__))
        for s in stack:
            if root in s.filename:
                frame = s

        if type(e).__name__ == 'NetworkError':
            logger.warning('{id} - {file} - {f} - {error}'.format(id=self.okx.id, file=frame.filename, f=frame.name, error=e))
            send_sms('exception', 'network error')
            return

        # TODO: InvalidOrder typically gets thrown when multiple orders go through when only one is needed
        # figure out a way to prevent multiple orders from happening in the first place instead of this temp fix
        if type(e).__name__ == 'InvalidOrder':
            logger.warning('{id} - {file} - {f} - {t}'.format(id=self.okx.id, file=frame.filename, f=frame.name, t=type(e).__name__))
            return

        logger.error('{id} - {file} - {f} - {t}'.format(id=self.okx.id, file=frame.filename, f=frame.name, t=type(e).__name__))
        send_sms('exception', 'origin: {id} {origin}\n\ntype: {t}'.format(id=self.okx.id, origin=frame.filename + ' ' + frame.name, t=type(e).__name__))

    def dump(self) -> None:

        try:

            path = os.path.join(os.getcwd(), 'dump')
            if not os.path.exists(path):
                os.mkdir(path)

            while True:
                timestamp = datetime.datetime.now().isoformat()

                filename = 'order_book-' + timestamp
                with open(os.path.join(path, filename), 'w') as json_file:
                    json.dump(self.okx.orderbooks, json_file,
                            indent=4,
                            separators=(',', ': '))

                filename = 'trades-' + timestamp
                with open(os.path.join(path, filename), 'w') as json_file:
                    json.dump(self.okx.trades, json_file,
                            indent=4,
                            separators=(',', ': '))

        except Exception as e:
            self.log_exception(e)
 
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

            # create orders slice and reverse
            orders = orders[:index_arr[-1][0]+1][::-1]

            # find index where position.open_amount <= cummulative contracts
            cum_index = np.argwhere(size <= np.cumsum(orders[:, 1]))

            if cum_index.size > 0:
                return orders[cum_index[0][0]][0]

        return price

    def break_even(self) -> tuple:

        position: Position = self.cache.position

        if position.status is not PositionStatus.OPEN:
            return (0, False)

        break_even_price = position.open_avg * (1 + self.fee)**2

        if self.p_t > break_even_price and position.triggered == 0:
            position.update({ 'triggered': 1 })
            self.cache.update(position)

        if self.p_t < break_even_price and position.triggered == 1:
            position.update({ 'triggered': 0 })
            self.cache.update(position)
            logger.info('{exchange_id} failed to break even {_symbol} {p_t} < {break_even}'.format(
                exchange_id=self.okx.id, **vars(position), p_t=self.p_t, break_even=break_even_price))

        if position.triggered == 0:
            return (0, False)

        bid = self.market_depth('bids', break_even_price, self.cache.position.open_amount)

        # last > break even > entry

        if (self.p_t - bid) < 1:
            logger.info('{exchange_id} breaking even {_symbol} {p_t} - {bid} < 1'.format(
                exchange_id=self.okx.id, **vars(position), p_t=self.p_t, bid=bid))
            return (break_even_price, True)

        return (0, False)
 
    def trailing_entry(self) -> bool:

        if not self.cache.last.id:
            logger.info(
                '{id} trailing entry - no trades found'.format(id=self.okx.id))
            return True

        delta = self.cursor if self.cursor > 0 else self.cache.last.close_avg

        # upper limit
        if (self.p_t > (delta * self.upper_limit)):

            self.cursor = 0

            logger.info('{id} trailing entry - no trades found - upper limit hit {limit}'.format(
                id=self.okx.id, limit=delta * self.upper_limit))
            return True

        # lower limit
        if (self.p_t < (delta * self.lower_limit)):

            self.cursor = delta * self.lower_limit

            logger.info('{id} trailing entry - no trades found - lower limit hit {limit}'.format(
                id=self.okx.id, limit=self.cursor))
            return False

        return False

    def take_profits(self):
        return self.p_t > self.cache.position.open_avg * self.tp

    def handle_orders(self, orders: list, clear=False):

        for order in orders:
            if order.get('status') in ['canceled', 'expired', 'rejected']:

                if self.cache.position.status is PositionStatus.PENDING and clear:
                    self.cache.clear()

                logger.info('{exchange_id} recent order {status}'.format(
                    exchange_id=self.okx.id, status=order.get('status')))
                return

            # adding position
            if self.cache.position.symbol == None:
                self.add_position(order)
                return

            # updating position
            if self.cache.position.symbol == order.get('symbol'):
                self.update_position(order)

    def add_position(self, order: dict) -> None:
        try:

            position = Position({
                'id': os.urandom(6).hex(),
                'timestamp': order.get('timestamp'),
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'open_amount': order.get('filled') if order.get('status') == 'closed' else 0.0,
                'open_avg': order.get('average') if order.get('status') == 'closed' else 0.0,
                'close_amount': 0.0,
                'close_avg': 0.0,
                'status': PositionStatus.OPEN if order.get('status') == 'closed' else PositionStatus.PENDING,
                'triggered': 0
            })

            self.cache.insert(position)
            logger.info('{exchange_id} add position - {_id} {_symbol} {_timestamp} {_open_avg} {_open_amount} {_close_avg} {_close_amount} {_status}'.format(
                exchange_id=self.okx.id, **vars(position)))
        except Exception as e:
            self.log_exception(e)

    def update_position(self, order: dict) -> None:

        try:

            position: Position = self.cache.position
            position.update({'timestamp': order.get('timestamp')})

            if order.get('status') == 'open':
                return

            type = 'add' if order.get('side') == position.side else 'reduce'
            if type == 'add':

                if order.get('status') == 'closed':
                    avg = position.position_avg('open', order)
                    position.update(
                        {'open_amount': position.open_amount + order.get('filled'), 'open_avg': avg})

                    if position.status is PositionStatus.PENDING:
                        position.update({'status': PositionStatus.OPEN})

            if type == 'reduce':

                if order.get('status') == 'closed':
                    position.update(
                        {'open_amount': position.open_amount - order.get('filled')})

                    avg = position.position_avg('close', order)
                    position.update(
                        {'close_amount': position.close_amount + order.get('filled'), 'close_avg': avg})

                if position.open_amount == 0:
                    position.update({'status': PositionStatus.CLOSED})

                    if position.pnl(self.leverage) > 0:
                        send_sms('profits', 'position closed\n\n+{:.2f}%'.format(position.pnl(self.leverage)))

                    if position.pnl(self.leverage) < 0:
                        send_sms('earlyexit', 'position closed\n\n+{:.2f}%'.format(position.pnl(self.leverage)))

            logger.info('{exchange_id} update position - {_id} {_symbol} {_timestamp} {_open_avg} {_open_amount} {_close_avg} {_close_amount} {_status} {pnl}'.format(
                exchange_id=self.okx.id, pnl=position.pnl(self.leverage) if position.open_amount == 0 else '', **vars(position)))
            self.cache.update(position)

        except Exception as e:
            self.log_exception(e)

    async def portfolio_size(self) -> dict:

        response: dict = None

        try:
            response = await self.okx.fetch_balance(params={ 'currency': 'usdt' })
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception(e)
        finally:

            total = response.get('total')
            return total.get('USDT')
  
    async def position_size(self, side: str = 'long') -> float:

        response: dict = None

        try:
            params = { 'instId': self.okx.market_id(self.symbol), 'tdMode': 'cross', 'ccy': self.okx.markets.get(self.symbol).get('base'), 'leverage': self.leverage }
            response = await self.okx.private_get_account_max_size(params)
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception(e)
        finally:

            data = response.get('data')[0]
            sz = data.get('maxBuy') if 'long' in side else data.get('maxSell')

            return float(sz) 

    async def create_order(self, type: str, side: str, size: float, price: float = None, params: dict = {}) -> None:
        try:
            await self.okx.create_order(self.symbol, type, side, size, price, params)
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception(e)

    # TODO: something more fault tolerant
    async def check_open_position(self):

        logger.info('{id} checking for open positions'.format(id=self.okx.id))

        response: dict = {}
        try:
            response = await self.okx.fetch_position(self.symbol)

            if response:
                await self.orders_history()
            else:
                logger.info(
                    '{id} no position clearing cache'.format(id=self.okx.id))
                self.cache.clear()

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception(e)

    # TODO: something more fault tolerant
    async def orders_history(self):

        logger.info('{id} re-syncing orders'.format(id=self.okx.id))

        response: dict = {}
        try:
            # getting orders this way may not get cached by ccxtpro
            response = await self.okx.private_get_trade_orders_history({ 'instId': self.okx.market_id(self.symbol), 'instType': 'SWAP', 'limit': 100 })

            if response:
                # add 1 so since conditional ignores most recent entry
                orders = self.okx.parse_orders(response.get('data'), since=self.cache.position.timestamp + 1)
                self.handle_orders(orders)

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception(e)

    async def watch_trades(self):

        try:
            while True:
                trades: list[dict] = await self.okx.watch_trades(self.symbol)

                for trade in trades:
                    self.p_t = trade.get('price')

                    # break even
                    price, ok = self.break_even()
                    if self.cache.position.status is PositionStatus.OPEN and ok:
                        await self.create_order('fok', 'sell', self.cache.position.open_amount, price, params={'tdMode': 'cross', 'posSide': 'long'})

                    # trailing entry # TODO: add / cancel limit orders maybe?
                    if self.cache.position.status not in [PositionStatus.OPEN, PositionStatus.PENDING] and self.trailing_entry():

                        size = await self.position_size('long') 
                        if size == 0:
                            logger.info('{id} trailing entry - size was zero'.format(id=self.okx.id))
                            continue

                        await self.create_order('fok', 'buy', size, self.okx.orderbooks[self.symbol].get('asks')[0][0], params={'tdMode': 'cross', 'posSide': 'long'})

                    # take profits
                    if self.cache.position.status is PositionStatus.OPEN and self.take_profits():
                        await self.create_order('market', 'sell', self.cache.position.open_amount, None, params={'tdMode': 'cross', 'posSide': 'long'})
                        logger.info('{id} take profits - target hit'.format(id=self.okx.id))

                self.okx.trades[self.symbol].clear()

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception(e)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e)

    async def watch_order_book(self):

        try:
            while True:
                await self.okx.watch_order_book(self.symbol, limit=20)               
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception(e)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e)

    async def watch_orders(self):
        try:
            while True:

                orders: list[dict] = await self.okx.watch_orders(self.symbol, limit=1)
                self.handle_orders(orders, True)

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception(e)

            # make sure run recieves the error to retry
            if type(e).__name__ == 'NetworkError':
                raise ccxtpro.NetworkError(e)

    async def system_status(self):

        class State(Enum):
            SCHEDULED = 'scheduled'
            ONGOING = 'ongoing'
            COMPLETED = 'completed'
            CANCELED = 'canceled'

        class ServiceType(Enum):
            WEBSOCKET = '0'
            SPOTMARGIN = '1'
            FUTURES = '2'
            PERPETUAL = '3'
            OPTIONS = '4'
            TRADING = '5'
        
        try:
            response: dict = await self.okx.public_get_system_status()
            for status in response.get('data'):

                title = status.get('title')
                start = self.okx.iso8601(int(status.get('begin')))
                end = self.okx.iso8601(int(status.get('end')))
                state = State(status.get('state')).name
                service = ServiceType(status.get('serviceType')).name

                logger.warning(f'system status - {title} {start} {end} {state} {service}')

                send_sms('system-status', f'system status\n{title}\n{start}\n{end}\n{state}\n{service}')

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception(e)

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
                'options': { 'watchOrderBook': { 'depth': 'books' }}
            })

            self.loop.run_until_complete(self.system_status())

            self.okx.set_sandbox_mode(self.test)

            self.loop.run_until_complete(self.okx.load_markets(reload=False))
            # make sure leverage is updated
            self.loop.run_until_complete(self.okx.set_leverage(self.leverage, self.symbol, params={'mgnMode': 'cross'}))

            # required to repopulate an already opened position
            self.loop.run_until_complete(self.check_open_position())
            loops = [
                self.watch_orders(),
                self.watch_order_book(),
                self.watch_trades()
            ]

            self.loop.run_until_complete(asyncio.gather(*loops))

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            self.log_exception(e)

            # raise so systemd restarts otherwise let daemon shutdown
            if type(e).__name__ == 'NetworkError':
    
                raise ccxtpro.NetworkError(e)

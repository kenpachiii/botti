import time
import ccxtpro
import logging
import numpy as np
import pandas as pd
import asyncio
import os
import joblib
import multiprocessing as mp
import datetime

from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import SimpleExpSmoothing

from botti.exceptions import log_exception
from botti.exchange import Exchange
from botti.cache import Cache
from botti.enums import PositionStatus, PositionState, PositionSide
from botti.sms import send_sms

logger = logging.getLogger(__name__)

class Botti:

    def __init__(self, **kwargs: dict) -> None:

        self.symbol: str = kwargs.get('symbol')
        self.fee: float = 0.0005
        self.leverage: int = kwargs.get('leverage')

        self.cache: Cache = Cache()

        self.queue = asyncio.Queue()

        self.exchange: Exchange = ccxtpro.okx

        self.history: pd.DataFrame = kwargs.get('history')
        self.model = load_model('model')
        self.scalar: MinMaxScaler = joblib.load('model/assets/model.scalar')

    @staticmethod
    def seconds_until_30_minute():
        n = datetime.datetime.utcnow()
        return ((datetime.datetime.min - n) % datetime.timedelta(minutes = 30)).seconds 

    def break_even(self) -> bool:

        if self.cache.current_position(self.symbol).side is PositionSide.LONG:
            break_even_price = self.cache.current_position(self.symbol).open_avg * (1 + self.fee)**2
            best_bid = self.exchange.orderbooks[self.symbol].get('bids')[0][0]
            return best_bid > break_even_price

        if self.cache.current_position(self.symbol).side is PositionSide.SHORT:
            break_even_price = self.cache.current_position(self.symbol).open_avg * (1 - self.fee)**2
            best_ask = self.exchange.orderbooks[self.symbol].get('asks')[0][0]
            return best_ask < break_even_price

        return False

    @staticmethod
    def transform(df: pd.DataFrame) -> np.ndarray:
    
        df.loc[:,('r')] = (np.log(df['price'] / df['price'].shift(-1))).fillna(2.220446049250313e-16).to_numpy()

        es = SimpleExpSmoothing(df.loc[:,('r')], initialization_method = 'heuristic')
        es.fit(smoothing_level = 0.070, optimized = True, remove_bias = True, use_brute = True)

        df = df.reset_index()

        df.loc[:,('price')] = df['price'].rolling(48).mean()
        df.loc[:,('amount')] = df['amount'].rolling(48).mean()

        return np.stack((es.predict(es.params, start = 0, end = None)[47:], df['price'].dropna().to_numpy(), df['amount'].dropna().to_numpy()), axis = -1)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return self.scalar.transform(x.reshape(-1, 3))

    def denormalize(self, x) -> np.ndarray:

        x = np.array(x)

        x = x.reshape(-1, 1)

        m, n = x.shape
        out = np.zeros((m, 3 * n), dtype = x.dtype)
        out[:,::3] = x

        return self.scalar.inverse_transform(out)

    def past_trades(self, size = 1) -> np.ndarray:

        def ymd():
            utc_datetime = datetime.datetime.utcfromtimestamp(time.time())
            return utc_datetime.strftime('%Y-%m-%d')

        try:

            url = os.path.join('http://localhost:8000', self.exchange.id, 'trades', self.exchange.market_id(self.symbol), ymd() + '.csv.xz')
            df: pd.DataFrame = pd.read_csv(url, header = 0, names = ['id', 'side', 'amount', 'price', 'timestamp'])

            df: pd.DataFrame = df.groupby(by=['timestamp', 'side']).agg({ 'price': np.mean, 'amount': np.sum }).reset_index()

            df.set_index('timestamp', inplace = True)
            df.index = pd.to_datetime(df.index, utc = True, unit = 'ms')

            df: pd.DataFrame = df.resample(f'1800S', kind = 'timestamp', convention = 'start').agg(amount=('amount', np.sum), price=('price', np.mean)).dropna()
            df.reset_index(inplace = True)

            # add only older timestamps
            df: pd.DataFrame = df[df['timestamp'] > self.history.loc[-1:,('timestamp')][0]]
            if not df.empty:
                self.history: pd.DataFrame = pd.concat([self.history, df], ignore_index = True)
                self.history.reset_index(inplace = True, drop = True)

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.exchange.id, self.symbol)

        return self.transform(self.history)[-size:]

    def predict(self):

        length = 64

        trades = self.normalize(self.past_trades(length))
        if len(trades) < 64:
            logger.info('{} {} predict - not enough trades. have {} need {}'.format(self.exchange.id, self.symbol, len(trades), 64))
            return 0

        y_pred = self.denormalize(
            self.model.predict(np.expand_dims(trades, axis = 0), workers = mp.cpu_count(), use_multiprocessing = True, verbose = '0')
        )[:, 0][0]

        return y_pred

    def trailing_entry(self) -> str:

        y_pred = self.predict()

        direction = PositionSide.LONG if np.sign(y_pred) == 1 else PositionSide.SHORT
        if direction == PositionSide.LONG and y_pred >= np.square(1 + 0.0005) - 1:

            logger.info('{} {} trailing entry - found long entry {}'.format(
                self.exchange.id, self.symbol, y_pred))

            return PositionSide.LONG

        if direction == PositionSide.SHORT and y_pred <= np.square(1 - 0.0005) - 1:

            logger.info('{} {} trailing entry - found short entry {}'.format(
                self.exchange.id, self.symbol, y_pred))

            return PositionSide.SHORT
            
        logger.info('{} {} trailing entry - no trades found {}'.format(
                        self.exchange.id, self.symbol, y_pred))

        return None

    def process_orders(self, orders):

        for order in orders:

            print('order', [order.get('side'), order.get('filled'), order.get('remaining')])

            if self.cache.insert_order(order):

                if order.get('status') in ['canceled', 'expired', 'rejected']:

                    logger.info('{exchange_id} {symbol} recent order {status}'.format(
                        exchange_id=self.exchange.id, symbol=self.symbol, status=order.get('status')))

                    # remove missed entry 
                    if self.cache.current_position(self.symbol).status is PositionStatus.PENDING and order.get('filled') == 0:
                        logger.info('{id} {symbol} process orders - entry missed removing {position}'.format(id=self.exchange.id, symbol=self.symbol, position=self.cache.current_position(self.symbol).id))

                        self.cache.remove('position', self.cache.current_position(self.symbol).id)

                    if self.cache.current_position(self.symbol).state is PositionState.EXIT and 'buy' in order.get('side') and order.get('filled') > self.cache.current_position(self.symbol).pending_close_amount:

                        order_side = 'sell' if self.cache.current_position(self.symbol).side is PositionSide.LONG else 'buy'
                        price = self.cache.current_position(self.symbol).open_avg * (1 + self.fee)**2 if self.cache.current_position(self.symbol).side is PositionSide.LONG else self.cache.current_position(self.symbol).open_avg * (1 - self.fee)**2
                        
                        args = self.symbol, 'limit', order_side, order.get('filled') - self.cache.current_position(self.symbol).pending_close_amount, price, { 'tdMode': 'cross', 'posSide': self.cache.current_position(self.symbol).side.value }
                        self.queue.put_nowait(('create', PositionState.EXIT, args))

                        logger.info('{id} {symbol} process orders - adjusting for order surplus'.format(id=self.exchange.id, symbol=self.symbol))

                    continue

                # adjust position
                self.adjust_position(order)

    def adjust_position(self, order: dict) -> None:

        try:

            if order.get('filled') == 0:
                return

            type = 'increase' if order.get('side') == 'buy' and self.cache.current_position(self.symbol).side is PositionSide.LONG or order.get('side') == 'sell' and self.cache.current_position(self.symbol).side is PositionSide.SHORT else 'decrease'
            if type == 'increase':

                amount = order.get('filled')

                # fetch previous orders and get the non-cummulative amount
                orders = self.cache.fetch_orders(self.symbol, order.get('id'))
                if len(orders) > 1:
                    amount = amount - orders[len(orders) - 2].get('filled')
           
                self.cache.update(self.cache.current_position(self.symbol).id, 
                    {
                        'open_amount': self.cache.current_position(self.symbol).open_amount + amount, 
                        'open_avg': order.get('average')
                    }
                )

                if self.cache.current_position(self.symbol).status is PositionStatus.PENDING:
                    self.cache.update(self.cache.current_position(self.symbol).id, { 'status': PositionStatus.OPEN })

            if type == 'decrease':

                amount = order.get('filled')

                # fetch previous orders and get the non-cummulative amount
                orders = self.cache.fetch_orders(self.symbol, order.get('id'))
                if len(orders) > 1:
                    amount = amount - orders[len(orders) - 2].get('filled')

                self.cache.update(self.cache.current_position(self.symbol).id, 
                    {
                        'close_amount': self.cache.current_position(self.symbol).close_amount + amount, 
                        'close_avg': order.get('average')
                    }
                )

                if self.cache.current_position(self.symbol).open_amount == self.cache.current_position(self.symbol).close_amount:
                    self.cache.update(self.cache.current_position(self.symbol).id, { 'status': PositionStatus.CLOSED })

                    if self.cache.last_position(self.symbol).pnl(self.leverage) > 0:
                        send_sms('profits', 'position closed\n\n{} +{:.2f}%'.format(self.symbol, self.cache.last_position(self.symbol).pnl(self.leverage)))

            position = self.cache.current_position(self.symbol) if self.cache.current_position(self.symbol).id else self.cache.last_position(self.symbol)

            profits, rmse = '', ''
            if position.status == PositionStatus.CLOSED:
                profits = getattr(position, 'pnl')(self.leverage)

                rmse = f'- RMSE {mean_squared_error([profits / (100 * self.leverage)], [self.predict()], squared = False)}'
                profits = f'- PNL {abs(profits)}'

            logger.info('{} adjusted position - {} {} {} {} {} {} {} {} {} {} {}'.format(
                self.exchange.id, position.id, position.symbol, position.open_avg, position.open_amount, 
                position.close_avg, position.close_amount, position.side, position.status, position.state, 
                profits, rmse))
            
        except Exception as e:
            log_exception(e, self.exchange.id, self.symbol)
  
    async def position_size(self, side: str = 'long') -> float:

        response: dict = None

        try:
            params = { 'instId': self.exchange.market_id(self.symbol), 'tdMode': 'cross', 'ccy': self.exchange.markets.get(self.symbol).get('base'), 'leverage': self.leverage }
            response = await self.exchange.private_get_account_max_size(params)

            data = response.get('data')[0]
            sz = data.get('maxBuy') if 'long' in side else data.get('maxSell')

            return float(sz) 
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.exchange.id, self.symbol)

        return 1 # default size if exception is thrown 
 
    async def consumer(self):

        while True:

            await asyncio.sleep(0)
            if not self.queue.empty():
                try:
                    (command, state, args) = self.queue.get_nowait()
                    if command == 'create':
                        await getattr(self.exchange, 'createOrder')(*args)
                    
                    if command == 'cancel':
                        await getattr(self.exchange, 'cancelOrder')(*args)

                    self.queue.task_done()

                except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
                    
                    # sometimes order completes before it can be canceled
                    if type(e).__name__ == 'OrderNotFound' and command == 'cancel':
                        logger.info('{id} {symbol} consumer - order completed before canceling'.format(id=self.exchange.id, symbol=self.symbol))

                        if self.cache.current_position(self.symbol).state is PositionState.EXIT and self.cache.current_position(self.symbol).pending_open_amount > self.cache.current_position(self.symbol).pending_close_amount:

                            order_side = 'sell' if self.cache.current_position(self.symbol).side is PositionSide.LONG else 'buy'
                            price = self.cache.current_position(self.symbol).open_avg * (1 + self.fee)**2 if self.cache.current_position(self.symbol).side is PositionSide.LONG else self.cache.current_position(self.symbol).open_avg * (1 - self.fee)**2

                            args = self.symbol, 'limit', order_side, self.cache.current_position(self.symbol).pending_open_amount - self.cache.current_position(self.symbol).pending_close_amount, price, { 'tdMode': 'cross', 'posSide': self.cache.current_position(self.symbol).side.value }
                            self.queue.put_nowait(('create', PositionState.EXIT, args))

                            logger.info('{id} {symbol} consumer - adjusting for order surplus'.format(id=self.exchange.id, symbol=self.symbol))

                        self.queue.task_done()
                        
                        continue

                    if type(e).__name__ == 'InsufficientFunds' and state is PositionState.ENTRY and command == 'create':
                        logger.info('{id} {symbol} consumer - insufficient funds removing {position}'.format(id=self.exchange.id, symbol=self.symbol,position=self.cache.current_position(self.symbol).id))
                        self.cache.remove('position', self.cache.current_position(self.symbol).id)

                        self.queue.task_done()

                        continue

                    if type(e).__name__ == 'InvalidOrder' and state is PositionState.ENTRY and command == 'create':
                        logger.info('{id} {symbol} consumer - invalid order removing {position}'.format(id=self.exchange.id, symbol=self.symbol,position=self.cache.current_position(self.symbol).id))
                        self.cache.remove('position', self.cache.current_position(self.symbol).id)

                        self.queue.task_done()

                        continue

                    log_exception(e, self.exchange.id, self.symbol) 

    async def strategy(self) -> None:

        while True:

            await asyncio.sleep(0)
            if self.symbol not in self.exchange.orderbooks:
                continue

            try:

                # exit 
                if self.cache.current_position(self.symbol).status is PositionStatus.OPEN and self.cache.current_position(self.symbol).state is PositionState.ENTRY and self.break_even():

                    self.cache.update(self.cache.current_position(self.symbol).id, { 'state': PositionState.EXIT })

                    order = self.cache.fetch_order(self.symbol)
                    if order.get('status') == 'open':
                        args = order.get('id'), self.symbol
                        self.queue.put_nowait(('cancel', PositionState.EXIT, args))

                    self.cache.update(self.cache.current_position(self.symbol).id, { 'pending_close_amount': self.cache.current_position(self.symbol).open_amount })

                    order_side = 'sell' if self.cache.current_position(self.symbol).side is PositionSide.LONG else 'buy'
                    price = self.cache.current_position(self.symbol).open_avg * (1 + self.fee)**2 if self.cache.current_position(self.symbol).side is PositionSide.LONG else self.cache.current_position(self.symbol).open_avg * (1 - self.fee)**2

                    args = self.symbol, 'limit', order_side, self.cache.current_position(self.symbol).open_amount, price, { 'tdMode': 'cross', 'posSide': self.cache.current_position(self.symbol).side.value }
                    self.queue.put_nowait(('create', PositionState.EXIT, args))

                # entry
                if self.cache.current_position(self.symbol).status is PositionStatus.NONE:

                    # model is trained on 30-minute intervals, which means a new prediction can only be obtained every 30-minutes.
                    await asyncio.sleep(self.seconds_until_30_minute())

                    side = self.trailing_entry()
                    if side:

                        self.cache.insert({ 'id': os.urandom(6).hex(), 'timestamp': int(time.time_ns() / 1000), 'symbol': self.symbol, 'side': side, 'state': PositionState.ENTRY, 'status':  PositionStatus.PENDING })

                        size = await self.position_size(self.cache.current_position(self.symbol).side.value) 
                        if size == 0:
                            logger.info('{id} {symbol} trailing entry - size was zero'.format(id=self.exchange.id,symbol=self.symbol))
                            continue
                        
                        price = 0
                        if self.cache.current_position(self.symbol).side is PositionSide.LONG:
                            price = self.exchange.orderbooks[self.symbol].get('asks')[0][0]

                        if self.cache.current_position(self.symbol).side is PositionSide.SHORT:
                            price = self.exchange.orderbooks[self.symbol].get('bids')[0][0]

                        self.cache.update(self.cache.current_position(self.symbol).id, { 'pending_open_amount': size })

                        order_side = 'buy' if self.cache.current_position(self.symbol).side is PositionSide.LONG else 'sell'
    
                        args = self.symbol, 'limit', order_side, size, price, { 'tdMode': 'cross', 'posSide': self.cache.current_position(self.symbol).side.value }
                        self.queue.put_nowait(('create', PositionState.ENTRY, args))

            except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
                print(e)
                log_exception(e, self.exchange.id, self.symbol)

    async def orders_history(self) -> None:

        order = self.cache.fetch_order(self.symbol)
        if order == {}:
            return

        logger.info('{id} {symbol} re-syncing orders'.format(id=self.exchange.id, symbol=self.symbol))

        response: dict = {}
        try:
            response = await self.exchange.private_get_trade_orders_history({ 'instId': self.exchange.market_id(self.symbol), 'instType': 'SWAP', 'state': 'filled' })
            if response:

                orders = response.get('data')
                for i, o in enumerate(orders):
                    if order.get('id') == o.get('ordId'):
                        orders = orders[:i + 1]
                        break

                orders = self.exchange.parse_orders(orders)
                self.process_orders(orders)

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.exchange.id, self.symbol)

    def set_trading_fee(self) -> float:
        self.fee = self.exchange.markets[self.symbol].get('taker')
        logger.info('{id} {symbol} - set fee to {fee}'.format(id=self.exchange.id, symbol=self.symbol, fee=self.fee))

    async def set_leverage(self):

        params={ 'mgnMode': 'cross' }

        try:
            response: dict = await getattr(self.exchange, 'fetchLeverage')(self.symbol, params)
            if int(response.get('data')[0].get('lever')) is not self.leverage:
                await getattr(self.exchange, 'setLeverage')(self.leverage, self.symbol, params)

                logger.info('{id} {symbol} - set leverage to {leverage}'.format(id=self.exchange.id, symbol=self.symbol,leverage=self.leverage))

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.exchange.id, self.symbol)

    async def watch_order_book(self):

        logger.info(f'{self.exchange.id} {self.symbol} - watching order book')

        while True:
            await asyncio.sleep(0)  

            try: 
                await getattr(self.exchange, 'watchOrderBook')(self.symbol, limit = 20) 
            except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
                log_exception(e, self.exchange.id, self.symbol)

    async def watch_orders(self):

        logger.info(f'{self.exchange.id} {self.symbol} - watching orders')

        while True:
            await asyncio.sleep(0)  

            try:
                orders: list = await getattr(self.exchange, 'watchOrders')(self.symbol)
                print(orders)
                self.process_orders(orders)
            except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
                log_exception(e, self.exchange.id, self.symbol)

    def run(self):

        logger.info(f'{self.exchange.id} {self.symbol} - starting botti')

        try:

            # make sure leverage is updated
            self.exchange.asyncio_loop.run_until_complete(self.set_leverage())

            # set trading fee
            self.set_trading_fee()

            # required to repopulate an already opened position
            self.exchange.asyncio_loop.run_until_complete(self.orders_history())
            return [
                self.watch_orders(),
                self.watch_order_book(),
                self.consumer(),
                self.strategy()
            ]

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.exchange.id, self.symbol)

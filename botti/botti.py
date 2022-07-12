import ccxtpro
import logging
import numpy as np
import pandas as pd
import asyncio
import os
import joblib
import multiprocessing as mp
import datetime
import json
import requests
import time

from keras.models import load_model, Model
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import SimpleExpSmoothing

from botti.exceptions import log_exception
from botti.exchange import Exchange

logger = logging.getLogger(__name__)

class Botti:

    def __init__(self, **kwargs: dict) -> None:

        self.symbol: str = kwargs.get('symbol')
        self.fee: float = 0.0005
        self.leverage: int = kwargs.get('leverage')

        self.queue = asyncio.Queue(maxsize = 1)

        self.exchange: Exchange = ccxtpro.okx

        self.history: pd.DataFrame = kwargs.get('history')
        self.model: Model = load_model('model')
        self.scalar: MinMaxScaler = joblib.load('model/assets/model.scalar')

        self.position = {}

    def __del__(self):
        self.exchange.asyncio_loop.run_until_complete(self.cancel_orders())

    async def cancel_orders(self):
        orders = await self.fetch_open_orders()
        ids = [self.exchange.safe_value(order, 'id') for order in orders]

        if len(ids) > 0:
            await self.exchange.cancel_orders(ids, self.symbol)

    @staticmethod
    def seconds_until_30_minute():
        n = datetime.datetime.utcnow()
        return ((datetime.datetime.min - n) % datetime.timedelta(minutes = 30)).seconds 

    async def best_bid(self):

        orderbooks = await self.fetch_l2_order_book()

        bids = self.exchange.safe_value(orderbooks, 'bids', [])
        best_bid = self.exchange.safe_value(bids, 0, [])

        return self.exchange.safe_value(best_bid, 0, 0)

    async def best_ask(self):

        orderbooks = await self.fetch_l2_order_book()

        asks = self.exchange.safe_value(orderbooks, 'asks', [])
        best_ask = self.exchange.safe_value(asks, 0, [])

        return self.exchange.safe_value(best_ask, 0, 0)

    async def break_even(self, position) -> bool:

        if self.exchange.safe_value(position, 'side', None) == 'long':
            break_even_price = self.exchange.safe_value(position, 'entryPrice', 0) * (1 + self.fee)**2
            return await self.best_bid() > break_even_price

        if self.exchange.safe_value(position, 'side', None) == 'short':
            break_even_price = self.exchange.safe_value(position, 'entryPrice', 0) * (1 - self.fee)**2
            return await self.best_ask() < break_even_price

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

    # TODO: assess why prediction changes even when a new interval hasn't happened yet
    def past_trades(self, size = 1) -> np.ndarray:

        try:

            url = os.path.join('http://localhost:8000', self.exchange.id, 'trades', self.exchange.market_id(self.symbol)) + f'/?since={int(self.history.iloc[-1].timestamp)}'
            trades = json.loads(requests.get(url).text)

            self.history: pd.DataFrame = pd.concat([self.history, pd.DataFrame(trades, columns = ['amount', 'price', 'timestamp'])], ignore_index = True)
            self.history: pd.DataFrame = self.history.astype({ 'amount': float, 'price': float, 'timestamp': int })
            self.history.reset_index(inplace = True, drop = True)

            return self.transform(self.history.copy())[-size:]

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            print(e)
            log_exception(e, self.exchange.id, self.symbol)

    def predict(self):

        size: int = self.model.input_shape[1]

        trades: np.ndarray = self.past_trades(size)
        if trades.shape[0] == size:

            trades: np.ndarray = self.normalize(trades)

            y_pred: np.ndarray = self.denormalize(
                self.model.predict(np.expand_dims(trades, axis = 0), workers = mp.cpu_count(), use_multiprocessing = True, verbose = '0')
            )[:, 0][0]

            return y_pred

        logger.info('{} {} predict - not enough trades. have {} need {}'.format(self.exchange.id, self.symbol, trades.shape[0], size))

        return 0

    def trailing_entry(self) -> str:

        y_pred: np.ndarray = self.predict()

        direction: str = 'long' if np.sign(y_pred) == 1 else 'short'
        if direction == 'long' and y_pred >= np.square(1 + 0.0005) - 1:

            logger.info('{} {} trailing entry - found long entry {}'.format(
                self.exchange.id, self.symbol, y_pred))

            return 'long'

        if direction == 'short' and y_pred <= np.square(1 - 0.0005) - 1:

            logger.info('{} {} trailing entry - found short entry {}'.format(
                self.exchange.id, self.symbol, y_pred))

            return 'short'
            
        logger.info('{} {} trailing entry - no trades found {}'.format(
                        self.exchange.id, self.symbol, y_pred))

        return None
  
    async def used_balance(self):
        balance = await self.fetch_balance()
        return balance['used']

    async def position_size(self, side) -> float:

        try:

            response: dict = await self.exchange.private_get_account_max_size({ 'instId': self.exchange.market_id(self.symbol), 'tdMode': 'cross' })
            if side == 'long':
                return float(response.get('data')[0].get('maxBuy', 0))
            
            if side == 'short':
                return float(response.get('data')[0].get('maxSell', 0))

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.exchange.id, self.symbol)
 
    # FIXME: redo order logic
    async def strategy(self) -> None:

        order = {}

        while True:

            await asyncio.sleep(0)
            try:

                current_position = await self.fetch_position()
                open_orders = await self.fetch_open_orders()

                contracts = self.exchange.safe_value(current_position, 'contracts', 0)

                if contracts > 0:
                    logging.info('{} {} position - {} {} {} {}'.format(self.exchange.id, self.exchange.safe_value(current_position, 'symbol', self.symbol), self.exchange.safe_value(current_position, 'side'), self.exchange.safe_value(current_position, 'entryPrice', 0), self.exchange.safe_value(current_position, 'contracts', 0), self.exchange.safe_value(current_position, 'percentage', 0)))

                if contracts == 0 and len(open_orders) == 0:
                    logging.info('{} {} position - no position'.format(self.exchange.id, self.exchange.safe_value(current_position, 'symbol', self.symbol)))

                # cancel entry order if order has been open longer than 30-minutes and was never filled
                if contracts == 0 and len(open_orders) > 0:

                    order: dict = open_orders[-1]

                    if (int(time.time() * 1000) - int(order.get('timestamp', 0))) > 1800000:
                        await getattr(self.exchange, 'cancelOrder')(order.get('id'), self.symbol)

                # exit 
                if contracts > 0 and await self.break_even(current_position):

                    order_side = 'sell' if self.exchange.safe_value(current_position, 'side') == 'long' else 'buy'
                    price = self.exchange.safe_value(current_position, 'entryPrice', 0) * (1 + self.fee)**2 if self.exchange.safe_value(current_position, 'side') == 'long' else self.exchange.safe_value(current_position, 'entryPrice', 0) * (1 - self.fee)**2

                    args = self.symbol, 'limit', order_side, contracts, price, { 'tdMode': 'cross', 'posSide': self.exchange.safe_value(current_position, 'side') }
                    await getattr(self.exchange, 'createOrder')(*args)
                                        
                # entry
                if contracts == 0 and len(open_orders) == 0 and await self.used_balance() == 0:

                    # model is trained on 30-minute intervals, which means a new prediction can only be obtained every 30-minutes.
                    await asyncio.sleep(self.seconds_until_30_minute())

                    side = self.trailing_entry()
                    if side:

                        size = await self.position_size(side) * 0.25 # use half for trading and other half for hedging.
                        if size == 0:
                            logger.info('{} {} - trailing entry - position size zero'.format(self.exchange.id, self.symbol))
                            continue
              
                        if side == 'long':
                            args = self.symbol, 'limit', 'buy', size, await self.best_ask(), { 'tdMode': 'cross', 'posSide': side }
                            await getattr(self.exchange, 'createOrder')(*args)

                        if side == 'short':
                            args = self.symbol, 'limit', 'sell', size, await self.best_bid(), { 'tdMode': 'cross', 'posSide': side }
                            await getattr(self.exchange, 'createOrder')(*args)

            except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
                
                from ccxt.base.errors import InsufficientFunds, InvalidOrder

                print(e)

                if isinstance(e, InsufficientFunds):
                    continue

                if isinstance(e, InvalidOrder):
                    continue

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

    async def fetch_balance(self):

        try:
            response: dict = await getattr(self.exchange, 'fetchBalance')()
            return response[self.parse_currency()]
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.exchange.id, self.symbol)

    async def fetch_open_orders(self):

        try:
            return await getattr(self.exchange, 'fetchOpenOrders')(self.symbol)
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:

            if isinstance(e, ccxtpro.OnMaintenance):
                return []

            log_exception(e, self.exchange.id, self.symbol)

    # TODO: for some reason BTC/USDT will get returned when BTC/USD has no position
    async def fetch_position(self):

        try:
            position: dict = await getattr(self.exchange, 'fetchPosition')(self.symbol)
            if not position:
                return {}

            if position.get('symbol') == self.symbol:
                return position

            return {}

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:

            if isinstance(e, ccxtpro.OnMaintenance):
                return {}

            print(e)

            log_exception(e, self.exchange.id, self.symbol)

    async def fetch_l2_order_book(self):

        try:
            return await getattr(self.exchange, 'fetchL2OrderBook')(self.symbol, limit = 1)
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.exchange.id, self.symbol)

    def parse_currency(self):
        market = getattr(self.exchange, 'market')(self.symbol)
        currency = market.get('settle')

        if not currency:
            raise Exception('invalid symbol')

        return currency

    async def run(self):

        logger.info(f'{self.exchange.id} {self.symbol} - starting botti')

        try:

            # make sure leverage is updated
            await self.set_leverage()

            # set trading fee
            self.set_trading_fee()

            await self.strategy()

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.exchange.id, self.symbol)

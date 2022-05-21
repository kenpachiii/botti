import time
import ccxtpro
import logging
import numpy as np
import asyncio
import os

from botti.exceptions import log_exception
from botti.exchange import Exchange
from botti.cache import Cache
from botti.enums import SystemState, ServiceType, PositionStatus, PositionState, PositionSide
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
        self.fee: float = 0.0005
        self.leverage: int = kwargs.get('leverage')
        self.upper_limit: int = kwargs.get('upper_limit')
        self.lower_limit: int = kwargs.get('lower_limit')

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

    # def market_depth(self, side: str, price: float, size: float, limit: float = 100) -> int:

    #     orders = np.asarray(self.okx.orderbooks[self.symbol].get(side))[:limit]

    #     # bid window = best bid > price > worst bid
    #     if 'bids' in side and not (orders[0][0] >= price >= orders[-1][0]):
    #         return -1

    #     # ask window = best ask < price < worst ask
    #     if 'asks' in side and not (orders[0][0] <= price <= orders[-1][0]):
    #         return -1

    #     index_arr: np.ndarray
    #     if 'bids' in side:
    #         # find index where price < orders price
    #         index_arr = np.argwhere(price <= orders[:, 0])

    #     if 'asks' in side:
    #         # find index where price > orders price
    #         index_arr = np.argwhere(price >= orders[:, 0])

    #     depth = -1
    #     if index_arr.size > 0:
    #         depth = index_arr[-1][0] 

    #     return depth

    def break_even(self) -> bool:

        if self.cache.position.side is PositionSide.LONG:
            break_even_price = self.cache.position.open_avg * (1 + self.fee)**2
            best_bid = self.okx.orderbooks[self.symbol].get('bids')[0][0]
            return best_bid > break_even_price


        if self.cache.position.side is PositionSide.SHORT:
            break_even_price = self.cache.position.open_avg * (1 - self.fee)**2
            best_ask = self.okx.orderbooks[self.symbol].get('asks')[0][0]
            return best_ask < break_even_price

        return False

    def mid_point(self):
        return (self.okx.orderbooks[self.symbol].get('bids')[0][0] + self.okx.orderbooks[self.symbol].get('asks')[0][0]) / 2

    def trailing_entry(self) -> str:

        fee_spread = ((self.mid_point() * (1 + self.fee)**2) - self.mid_point())
        ask_delta = self.okx.orderbooks[self.symbol].get('asks')[4][0] - self.okx.orderbooks[self.symbol].get('asks')[0][0]
        bid_delta = self.okx.orderbooks[self.symbol].get('bids')[0][0] - self.okx.orderbooks[self.symbol].get('bids')[4][0]
        cum_ask_volume = np.sum(np.asarray(self.okx.orderbooks[self.symbol].get('asks'))[:, 1]) 
        cum_bid_volume = np.sum(np.asarray(self.okx.orderbooks[self.symbol].get('bids'))[:, 1]) 

        if self.okx.trades.get(self.symbol):

            timestamp = 0
            buy_amount, sell_amount = 1, 1
            for trade in self.okx.trades.get(self.symbol)[::-1]:
                if timestamp == 0:
                    timestamp = int(trade.get('timestamp'))

                if timestamp != int(trade.get('timestamp')):
                    break

                if trade.get('side') == 'buy':
                    buy_amount += float(trade.get('amount'))

                if trade.get('side') == 'sell':
                    sell_amount += float(trade.get('amount'))

            if ask_delta > fee_spread and bid_delta < fee_spread * 0.2 and buy_amount / cum_ask_volume >= 0.05 and buy_amount / sell_amount >= 2:

                out = (self.okx.orderbooks[self.symbol].get('asks')[0][0], self.okx.orderbooks[self.symbol].get('asks')[4][0], ask_delta, bid_delta, cum_ask_volume / cum_bid_volume, buy_amount / sell_amount)

                logger.info('{id} trailing entry - no trades found - found long entry {} -> {} {} {} {} {}'.format(
                    id=self.okx.id, *out))

                return PositionSide.LONG

            # if bid_delta > fee_spread and ask_delta < fee_spread * 0.2 and sell_amount / cum_bid_volume >= 0.050 and sell_amount / buy_amount >= 2:

            #     out = (self.okx.orderbooks[self.symbol].get('bids')[0][0], self.okx.orderbooks[self.symbol].get('bids')[4][0], bid_delta, ask_delta, cum_bid_volume / cum_ask_volume, sell_amount / buy_amount)

            #     logger.info('{id} trailing entry - no trades found - found short entry {} -> {} {} {} {} {}'.format(
            #         id=self.okx.id, *out))

            #     return None

            GREEN = '\033[92m'
            RED = '\033[91m'
            BLUE = '\033[96m'
            MAGENTA = '\033[95m'
            PURPLE = '\033[94m'
            END = '\033[0m'

            trade = self.okx.trades.get(self.symbol)[-1]

            ask_history = list(np.asarray(self.okx.orderbooks[self.symbol].get('asks'))[:, 0])
            bid_history = list(np.asarray(self.okx.orderbooks[self.symbol].get('bids'))[:, 0])

            if trade.get('price') in ask_history:
                trade['price'] = GREEN + str(trade.get('price')) + END

            if trade.get('price') in bid_history:
                trade['price'] = RED + str(trade.get('price')) + END

            # cum_ask_volume = '{:09.2f}'.format(cum_ask_volume)
            # cum_bid_volume = '{:09.2f}'.format(cum_bid_volume)
            ask_volume = self.okx.orderbooks[self.symbol].get('asks')[0][1]
            bid_volume = self.okx.orderbooks[self.symbol].get('bids')[0][1]

            if bid_delta > fee_spread and ask_delta < fee_spread * 0.2:
                ask_delta = BLUE + '{:07.2f}'.format(ask_delta) + END
                bid_delta = BLUE + '{:07.2f}'.format(bid_delta) + END
            else:
                ask_delta = '{:07.2f}'.format(ask_delta)
                bid_delta = '{:07.2f}'.format(bid_delta)

            sell_amount_formatted = ''
            if sell_amount / cum_bid_volume >= 0.025:
                sell_amount_formatted = MAGENTA + '{:08.2f}'.format(sell_amount) + END
            else:
                sell_amount_formatted = '{:08.2f}'.format(sell_amount)

            buy_amount_formatted = ''
            if buy_amount / cum_ask_volume >= 0.025:
                buy_amount_formatted = MAGENTA + '{:08.2f}'.format(buy_amount) + END
            else:
                buy_amount_formatted = '{:08.2f}'.format(buy_amount)

            amount = trade.get('amount')
            if amount >= 1000:
                amount = MAGENTA + '{:07.2f}'.format(amount) + END
            else:
                amount = '{:07.2f}'.format(amount)

            cum_ask_volume = '{:09.2f}'.format(cum_ask_volume)
            cum_bid_volume = '{:09.2f}'.format(cum_bid_volume)

            asks = list(np.asarray(self.okx.orderbooks[self.symbol].get('asks'))[:5][:, 0])
            bids = list(np.asarray(self.okx.orderbooks[self.symbol].get('bids'))[:5][:, 0])

            out = '{} {} {} {} {} - {} {} {} {} {} - {} {} - {} {} - {} {} - {} {}'.format(
                asks[4], 
                asks[3], 
                asks[2], 
                asks[1], 
                asks[0], 
                bids[0], 
                bids[1], 
                bids[2], 
                bids[3], 
                bids[4],  
                cum_ask_volume,
                buy_amount_formatted, 
                sell_amount_formatted,
                cum_bid_volume,
                trade.get('price'), 
                amount, 
                ask_delta, 
                bid_delta
            )

            # print(out)

        return None

    def process_orders(self, orders):

        for order in orders:

            self.cache.insert_order(order)

            if order.get('status') in ['canceled', 'expired', 'rejected']:

                logger.info('{exchange_id} recent order {status}'.format(
                    exchange_id=self.okx.id, status=order.get('status')))

                # remove missed entry 
                if self.cache.position.status is PositionStatus.PENDING and order.get('filled') == 0:
                    logger.info('{id} process orders - entry missed removing {position}'.format(id=self.okx.id, position=self.cache.position.id))

                    self.cache.remove('position', self.cache.position.id)

                if self.cache.position.state is PositionState.EXIT and 'buy' in order.get('side') and order.get('filled') > self.cache.position.pending_close_amount:

                    order_side = 'sell' if self.cache.position.side is PositionSide.LONG else 'buy'
                    price = self.cache.position.open_avg * (1 + self.fee)**2 if self.cache.position.side is PositionSide.LONG else self.cache.position.open_avg * (1 - self.fee)**2
                    
                    args = self.symbol, 'limit', order_side, order.get('filled') - self.cache.position.pending_close_amount, price, { 'tdMode': 'cross', 'posSide': self.cache.position.side.value }
                    self.queue.put_nowait(('create', PositionState.EXIT, args))

                    logger.info('{id} process orders - adjusting for order surplus'.format(id=self.okx.id))

                continue

            # adjust position
            self.adjust_position(order)

    def adjust_position(self, order: dict) -> None:

        try:

            if order.get('filled') == 0:
                return

            type = 'increase' if order.get('side') == 'buy' and self.cache.position.side is PositionSide.LONG or order.get('side') == 'sell' and self.cache.position.side is PositionSide.SHORT else 'decrease'
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

            logger.info('{exchange_id} adjusted position - {id} {symbol} {open_avg} {open_amount} {close_avg} {close_amount} {side} {status} {state} {pnl}'.format(
                exchange_id=self.okx.id, id=position.id,symbol=position.symbol, open_avg=position.open_avg, open_amount=position.open_amount,close_avg=position.close_avg, close_amount=position.close_amount,side=position.side,status=position.status,state=position.state,pnl=position.pnl(self.leverage) if position.open_amount == position.close_amount else ''))
            
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
                        await getattr(self.okx, 'createOrder')(*args)
                    
                    if command == 'cancel':
                        await getattr(self.okx, 'cancelOrder')(*args)

                    self.queue.task_done()

                except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
                    
                    # sometimes order completes before it can be canceled
                    if type(e).__name__ == 'OrderNotFound' and command == 'cancel':
                        logger.info('{id} consumer - order completed before canceling'.format(id=self.okx.id))

                        if self.cache.position.state is PositionState.EXIT and self.cache.position.pending_open_amount > self.cache.position.pending_close_amount:

                            order_side = 'sell' if self.cache.position.side is PositionSide.LONG else 'buy'
                            price = self.cache.position.open_avg * (1 + self.fee)**2 if self.cache.position.side is PositionSide.LONG else self.cache.position.open_avg * (1 - self.fee)**2

                            args = self.symbol, 'limit', order_side, self.cache.position.pending_open_amount - self.cache.position.pending_close_amount, price, { 'tdMode': 'cross', 'posSide': self.cache.position.side.value }
                            self.queue.put_nowait(('create', PositionState.EXIT, args))

                            logger.info('{id} consumer - adjusting for order surplus'.format(id=self.okx.id))

                        self.queue.task_done()
                        
                        continue

                    if type(e).__name__ == 'InsufficientFunds' and state is PositionState.ENTRY and command == 'create':
                        logger.info('{id} consumer - insufficient funds removing {position}'.format(id=self.okx.id,position=self.cache.position.id))
                        self.cache.remove('position', self.cache.position.id)

                        self.queue.task_done()

                        continue

                    if type(e).__name__ == 'InvalidOrder' and state is PositionState.ENTRY and command == 'create':
                        logger.info('{id} consumer - invalid order removing {position}'.format(id=self.okx.id,position=self.cache.position.id))
                        self.cache.remove('position', self.cache.position.id)

                        self.queue.task_done()

                        continue

                    print(e)

                    log_exception(e, self.okx.id) 

    async def strategy(self) -> None:

        while True:

            await asyncio.sleep(0)
            if self.symbol not in self.okx.orderbooks:
                continue

            try:

                # # close missed entry
                # # position will be removed when there's confirmation of a canceled order with 0 fills when processing orders
                # if self.cache.position.state is PositionState.ENTRY and self.cache.position.status is PositionStatus.PENDING:
                #     order = self.cache.fetch_order()
                #     if order.get('status') == 'open' and order.get('filled') == 0:
                #         args = order.get('id'), self.symbol
                #         self.queue.put_nowait(('cancel', PositionState.ENTRY, args))

                # exit 
                if self.cache.position.status is PositionStatus.OPEN and self.cache.position.state is PositionState.ENTRY and self.break_even():

                    self.cache.update(self.cache.position.id, { 'state': PositionState.EXIT })

                    order = self.cache.fetch_order()
                    if order.get('status') == 'open':
                        args = order.get('id'), self.symbol
                        self.queue.put_nowait(('cancel', PositionState.EXIT, args))

                    self.cache.update(self.cache.position.id, { 'pending_close_amount': self.cache.position.open_amount })

                    order_side = 'sell' if self.cache.position.side is PositionSide.LONG else 'buy'
                    price = self.cache.position.open_avg * (1 + self.fee)**2 if self.cache.position.side is PositionSide.LONG else self.cache.position.open_avg * (1 - self.fee)**2

                    args = self.symbol, 'limit', order_side, self.cache.position.open_amount, price, { 'tdMode': 'cross', 'posSide': self.cache.position.side.value }
                    self.queue.put_nowait(('create', PositionState.EXIT, args))

                # entry
                if self.cache.position.status is PositionStatus.NONE:

                    side = self.trailing_entry()
                    if side:

                        self.cache.insert({ 'id': os.urandom(6).hex(), 'timestamp': int(time.time_ns() / 1000), 'symbol': self.symbol, 'side': side, 'state': PositionState.ENTRY, 'status':  PositionStatus.PENDING })

                        size = await self.position_size(self.cache.position.side.value) * 0.2
                        if size == 0:
                            logger.info('{id} trailing entry - size was zero'.format(id=self.okx.id))
                            continue
                        
                        price = 0
                        if self.cache.position.side is PositionSide.LONG:
                            price = self.okx.orderbooks[self.symbol].get('asks')[0][0]

                        if self.cache.position.side is PositionSide.SHORT:
                            price = self.okx.orderbooks[self.symbol].get('bids')[0][0]

                        self.cache.update(self.cache.position.id, { 'pending_open_amount': size })

                        order_side = 'buy' if self.cache.position.side is PositionSide.LONG else 'sell'
    
                        args = self.symbol, 'limit', order_side, size, price, { 'tdMode': 'cross', 'posSide': self.cache.position.side.value }
                        self.queue.put_nowait(('create', PositionState.ENTRY, args))

            except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
                print(e)
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

    def set_trading_fee(self) -> float:

        self.fee = self.okx.markets[self.symbol].get('taker')

        logger.info('{id} set fee to {fee}'.format(id=self.okx.id, fee=self.fee))

    async def set_leverage(self):

        params={ 'mgnMode': 'cross' }

        try:
            response: dict = await getattr(self.okx, 'fetchLeverage')(self.symbol, params)
            if int(response.get('data')[0].get('lever')) is not self.leverage:
                await getattr(self.okx, 'setLeverage')(self.leverage, self.symbol, params)

                logger.info('{id} set leverage to {leverage}'.format(id=self.okx.id,leverage=self.leverage))

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)

    async def watch_trades(self):

        try:
            while True:
                await asyncio.sleep(0)
                await getattr(self.okx, 'watchTrades')(self.symbol)
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)

    async def watch_order_book(self):

        try:
            while True:
                await asyncio.sleep(0)   
                await getattr(self.okx, 'watchOrderBook')(self.symbol, limit = 20)  
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)

    # FIXME: ensure ALL new orders only are received
    async def watch_orders(self):
        try:
            while True:
                orders: list = await getattr(self.okx, 'watchOrders')(self.symbol, limit = 1)
                self.process_orders(orders)
        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)

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

    def run(self):

        logger.info('starting botti')

        try:
            self.okx.options
            self.okx = Exchange({
                'asyncio_loop': self.loop,
                'newUpdates': True,
                'apiKey': self.key,
                'secret': self.secret,
                'password': self.password,
                'options': { 'rateLimit': 10, 'watchOrderBook': { 'depth': 'books' }}
            })

            self.loop.run_until_complete(self.system_status())

            self.okx.set_sandbox_mode(self.test)

            self.loop.run_until_complete(self.okx.load_markets(reload=False))

            # make sure leverage is updated
            self.loop.run_until_complete(self.set_leverage())

            # set trading fee
            self.set_trading_fee()

            # required to repopulate an already opened position
            self.loop.run_until_complete(self.orders_history())
            loops = [
                self.watch_orders(),
                self.watch_order_book(),
                self.watch_trades(),
                self.consumer(),
                self.strategy()
            ]

            self.loop.run_until_complete(asyncio.gather(*loops))

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.okx.id)

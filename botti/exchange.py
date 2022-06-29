import ccxtpro
import logging

from botti.enums import SystemStatus
from botti.exceptions import log_exception
from botti.sms import send_sms

logger = logging.getLogger(__name__)

class Exchange(ccxtpro.okx):

    def handle_message(self, client, message):
        if not self.handle_error_message(client, message):
            return

        if message == 'pong':
            return self.handle_pong(client, message)
        # table = self.safe_string(message, 'table')
        # if table is None:
        event = self.safe_string(message, 'event')
        if event is not None:
            methods = {
                # 'info': self.handleSystemStatus,
                # 'book': 'handleOrderBook',
                'login': self.handle_authenticate,
                'subscribe': self.handle_subscription_status,
            }
            method = self.safe_value(methods, event)
            if method is None:
                return message
            else:
                return method(client, message)
        else:
            arg = self.safe_value(message, 'arg', {})
            channel = self.safe_string(arg, 'channel')
            methods = {
                'books': self.handle_order_book, # 400 depth levels will be pushed in the initial full snapshot. Incremental data will be pushed every 100 ms when there is change in order book.
                'books5': self.handle_order_book, # 5 depth levels will be pushed every time. Data will be pushed every 100 ms when there is change in order book.
                'books50-l2-tbt': self.handle_order_book, # 50 depth levels will be pushed in the initial full snapshot. Incremental data will be pushed tick by tick, i.e. whenever there is change in order book.
                'books-l2-tbt': self.handle_order_book, # 400 depth levels will be pushed in the initial full snapshot. Incremental data will be pushed tick by tick, i.e. whenever there is change in order book.
                'tickers': self.handle_ticker,
                'trades': self.handle_trades,
                'account': self.handle_balance,
                # 'margin_account': self.handle_balance,
                'orders': self.handle_orders,

                'positions': self.handle_positions
            }
            method = self.safe_value(methods, channel)
            if method is None:
                if channel.find('candle') == 0:
                    self.handle_ohlcv(client, message)
                else:
                    return message
            else:
                return method(client, message)

    def handle_positions(self, client, message):
        arg = self.safe_value(message, 'arg', {})
        channel = self.safe_string(arg, 'channel')
        positions = self.safe_value(message, 'data', [])

        result = {}
        for position in positions:
            if self.safe_integer(position, 'availPos', 0) > 0:    
                position = getattr(self, 'parse_position')(position)
                result[position.get('symbol')] = position

        client.resolve(result, channel)

    def on_connected(self, client, message = None):
        logger.info('{exchange_id} Connected to {url} - {message}'.format(exchange_id = self.id, url = client.url, message = message))

    async def system_status(self):
        
        try:
            response = await self.fetch_status()
            if response:
                updated = self.iso8601(response.get('updated'))
                status = SystemStatus(response.get('status')).name
                eta = self.iso8601(response.get('eta'))

                logger.warning('system status - {} {} {}'.format(updated, eta, status))
                send_sms('system-status', 'system status\n{}\n{}\n{}'.format(updated, eta, status))

        except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
            log_exception(e, self.exchange.id, self.symbol)
            
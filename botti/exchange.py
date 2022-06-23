import ccxtpro
import logging

from botti.enums import SystemStatus
from botti.exceptions import log_exception
from botti.sms import send_sms

logger = logging.getLogger(__name__)

class Exchange(ccxtpro.okx):

    def nonce(self):
        self.close
        return self.milliseconds()

    def on_connected(self, client, message=None):
        logger.info('{exchange_id} Connected to {url} - {message}'.format(exchange_id=self.id, url=client.url, message=message))

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
            print(e)
            log_exception(e, self.exchange.id, self.symbol)
            
import ccxtpro
import logging

logger = logging.getLogger(__name__)

class Exchange(ccxtpro.okx):

    def nonce(self):
        return self.milliseconds()

    def on_connected(self, client, message=None):
        logger.info('{exchange_id} Connected to {url} - {message}'.format(exchange_id=self.id, url=client.url, message=message))
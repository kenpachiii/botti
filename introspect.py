import logging
import ccxtpro

from botti.cache import Cache

logger = logging.getLogger(__name__)


class Class:

    def __init__(self):

        self.okx = ccxtpro.okx
        self.cache = Cache('botti.db.dump')
        self.p_t = 0

    def trailing_entry(self) -> bool:

        if 'closed' not in self.cache.position.status:
            return False

        if not self.cache.last.id:
            logger.info(
                '{id} trailing entry - no trades found'.format(id=self.okx.id))
            return True

        # upper limit
        if self.p_t > (self.cache.last.close_avg * 1.01):
            logger.info('{id} trailing entry - no trades found - upper limit hit {limit}'.format(
                id=self.okx.id, limit=self.cache.last.close_avg * 1.01))
            return True

        # lower limit
        if self.p_t < (self.cache.last.close_avg * 0.98):
            logger.info('{id} trailing entry - no trades found - lower limit hit {limit}'.format(
                id=self.okx.id, limit=self.cache.last.close_avg * 0.98))
            return True

        return False


test = Class()
test.trailing_entry()

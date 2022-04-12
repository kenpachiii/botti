import asyncio
import ccxtpro
import logging

logger = logging.getLogger(__name__)

API_RETRY_COUNT = 10

def calculate_backoff(retrycount, max_retries):
    """
    Calculate backoff
    """
    return (max_retries - retrycount) ** 2 + 1

def retrier(f):
    async def wrapper(*args, **kwargs):
        count = kwargs.pop('count', API_RETRY_COUNT)
        try:
            return await f(*args, **kwargs)
        except ccxtpro.NetworkError as ex:
            msg = f'{f.__name__}() returned exception: "{ex}". '
            if count > 0:
                msg += f'Retrying still for {count} times.'
                count -= 1
                kwargs['count'] = count
                if isinstance(ex, ccxtpro.DDoSProtection):
                    backoff_delay = calculate_backoff(count + 1, API_RETRY_COUNT)
                    logger.info(f"Applying DDosProtection backoff delay: {backoff_delay}")
                    await asyncio.sleep(backoff_delay)
                if msg:
                    logger.warning(msg)
                return await wrapper(*args, **kwargs)
            else:
                logger.warning(msg + 'Giving up.')
                raise ex
    return wrapper


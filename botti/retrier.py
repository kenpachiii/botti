import ccxtpro
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

def retrier(_func=None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except (ccxtpro.NetworkError) as ex:
                msg = f'{f.__name__}() returned exception: "{ex}". '
                logger.warning(msg + f'Will retry in 3 seconds.')
                time.sleep(3)
                return wrapper(*args, **kwargs)
        return wrapper
    return decorator(_func)
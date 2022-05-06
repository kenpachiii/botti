import logging
import sys
import os
from logging import Formatter
from logging.handlers import TimedRotatingFileHandler

logger = logging.getLogger(__name__)
LOGFORMAT = '%(asctime)s - %(levelname)s - %(message)s'

def _set_loggers() -> None:
    logging.getLogger('asyncio').setLevel(logging.INFO)
    logging.getLogger('ccxtpro.base.exchange').setLevel(logging.INFO)
    logging.getLogger('boto3').setLevel(logging.INFO)

def get_existing_handlers(handlertype):
    """
    Returns Existing handler or None (if the handler has not yet been added to the root handlers).
    """
    return next((h for h in logging.root.handlers if isinstance(h, handlertype)), None)

def setup_logging() -> None:

    path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(path):
        os.mkdir(path)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(Formatter(LOGFORMAT))
    logging.root.addHandler(consoleHandler)


    handler_rf = get_existing_handlers(TimedRotatingFileHandler)
    if handler_rf:
        logging.root.removeHandler(handler_rf)

    handler_rf = TimedRotatingFileHandler('./logs/log',
                                        when='midnight',  
                                        interval=1,
                                        backupCount=0)

    handler_rf.setFormatter(Formatter(LOGFORMAT))
    logging.root.addHandler(handler_rf)

    logging.root.setLevel(logging.INFO)
    _set_loggers()


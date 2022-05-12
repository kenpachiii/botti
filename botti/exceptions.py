import traceback
import os
import logging

from botti.sms import send_sms

logger = logging.getLogger(__name__)

def log_exception(e: Exception, id = None) -> None:

    frame = None

    stack = traceback.extract_tb(e.__traceback__)

    root = os.path.dirname(os.path.abspath(__file__))
    for s in stack:
        if root in s.filename:
            frame = s

    if type(e).__name__ == 'NetworkError':
        logger.warning('{id}{file} - {f} - {error}'.format(id = '{} - '.format(id) if id else '', file = frame.filename, f = frame.name, error = e))
        send_sms('exception', 'network error')
        return

    logger.error('{id}{file} - {f} - {t}'.format(id = '{} - '.format(id) if id else '', file = frame.filename, f = frame.name, t = type(e).__name__))
    send_sms('exception', 'origin: {id}{origin}\n\ntype: {t}'.format(id = '{} '.format(id) if id else '', origin = frame.filename + ' ' + frame.name, t = type(e).__name__))

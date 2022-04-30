import traceback
import os
import logging

from botti.sms import send_sms

logger = logging.getLogger(__name__)

def log_exception(exchange_id, e: Exception) -> None:

    frame = None

    stack = traceback.extract_tb(e.__traceback__)

    root = os.path.dirname(os.path.abspath(__file__))
    for s in stack:
        if root in s.filename:
            frame = s

    if type(e).__name__ == 'NetworkError':
        return

    # TODO: InvalidOrder typically gets thrown when multiple orders go through when only one is needed
    # figure out a way to prevent multiple orders from happening in the first place instead of this temp fix
    if type(e).__name__ == 'InvalidOrder':
        logger.warning('{id} - {file} - {f} - {t}'.format(id=exchange_id, file=frame.filename, f=frame.name, t=type(e).__name__))
        return

    logger.error('{id} - {file} - {f} - {t}'.format(id=exchange_id, file=frame.filename, f=frame.name, t=type(e).__name__))
    send_sms('exception', 'origin: {id} {origin}\n\ntype: {t}'.format(id=exchange_id, origin=frame.filename + ' ' + frame.name, t=type(e).__name__))

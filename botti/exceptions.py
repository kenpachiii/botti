import traceback
import logging

logger = logging.getLogger(__name__)

def log_exception(id: str, e: Exception):

    stack = traceback.extract_tb(e.__traceback__, -1).pop(-1)
    logger.error('{file} - {f} - {id} - {error}'.format(id=id, file=stack.line, f=stack.name, error=str(e)))

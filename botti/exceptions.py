import traceback
import logging

logger = logging.getLogger(__name__)

def handle_exception(id: str, e: Exception):
    stack = traceback.extract_tb(e.__traceback__, -1).pop(-1)
    logger.error('{file} - {f} - {id} - {error}'.format(id=id, file=stack.filename, f=stack.name, error=str(e)))

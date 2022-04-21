import logging

from botti.loggers import setup_logging_pre
from botti.botti import Botti

api_keys = {
    'main': {
        'key': '56956a7f-34ed-48f0-b113-aa00f45a525e',
        'secret': 'F4BAE9A3844E26FC46693A33582C9D8A',
        'password': 'ZGN7WIPReYUzwdsb',
        'test': False
    },
    'botti-api': {
        'key': '34dc5ac4-b78b-4dfa-b0ca-3a69e7cedd07',
        'secret': 'DECA9E6C4CDA7D8C8C97AF5A148FAEEC',
        'password': 'KQMlR1m+g85tJMD2',
        'test': False
    },
    'demo': {
        'key': '696b9c27-0ac6-49fb-ab56-fc8151e70881',
        'secret': '981FB05DD7C5A5C0889D74E547FE6546',
        'password': 'v5pksFMxT7G2uwFy',
        'test': True
    },
    'botti-demo': {
        'key': 'd02f824a-f938-4306-865e-19e3d07b0499',
        'secret': 'C3F9094A6B0912F4F02813DD319403DD',
        'password': '/Bw0fC5iVyHs9FNr',
        'test': True
    }
}

logger = logging.getLogger('botti')

def main():

    setup_logging_pre()

    botti = Botti(symbol='BTC/USDT:USDT', fee = 0.0005, leverage = 2, **api_keys['botti-api'])
    botti.run()

import argparse
import logging

from botti.loggers import setup_logging
from botti.botti import Botti
from botti.exceptions import log_exception

keys = {
    'main': {
        'key': '56956a7f-34ed-48f0-b113-aa00f45a525e',
        'secret': 'F4BAE9A3844E26FC46693A33582C9D8A',
        'password': 'ZGN7WIPReYUzwdsb',
        'test': False
    },
    'botti-api': {
        'key': 'c2369337-b856-44ef-9cb2-9c7a25a0e421',
        'secret': '1F0CA58B3DC17B9C0EB963B99DADB112',
        'password': 'KQMlR1m+g85tJMD2',
        'test': False
    },
    'demo': {
        'key': '696b9c27-0ac6-49fb-ab56-fc8151e70881',
        'secret': '981FB05DD7C5A5C0889D74E547FE6546',
        'password': 'v5pksFMxT7G2uwFy',
        'test': True
    },
}

logger = logging.getLogger('botti')

def main():

    setup_logging()

    parser = argparse.ArgumentParser(description='Botti trading bot.')
    parser.add_argument('--symbol', type=str, help='symbol to trade', required = True)
    parser.add_argument('--leverage', type=int, help='leverage to use', required = True)
    parser.add_argument('--keys', type=int, help='keys to use', required = True)

    args = parser.parse_args()

    botti = Botti(symbol=args.symbol, leverage = args.leverage, upper_limit = 1.005, lower_limit = 0.995, **keys[args.keys])
    botti.run()













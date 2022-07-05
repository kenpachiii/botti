import argparse
import itertools
import requests
import logging
import os
import numpy as np
import datetime
import time
import json
import pandas as pd 

import dask.dataframe as dd
import ccxtpro
import asyncio

from botti.exchange import Exchange
from botti.loggers import setup_logging
from botti.botti import Botti
from botti.exceptions import log_exception

keys = {
    'main': {
        'apiKey': '56956a7f-34ed-48f0-b113-aa00f45a525e',
        'secret': 'F4BAE9A3844E26FC46693A33582C9D8A',
        'password': 'ZGN7WIPReYUzwdsb',
        'test': False
    },
    'botti-api': {
        'apiKey': 'c2369337-b856-44ef-9cb2-9c7a25a0e421',
        'secret': '1F0CA58B3DC17B9C0EB963B99DADB112',
        'password': 'KQMlR1m+g85tJMD2',
        'test': False
    },
    'demo': {
        'apiKey': '696b9c27-0ac6-49fb-ab56-fc8151e70881',
        'secret': '981FB05DD7C5A5C0889D74E547FE6546',
        'password': 'v5pksFMxT7G2uwFy',
        'test': True
    },
}

logger = logging.getLogger('botti')

def fetch_history(exchange: Exchange, symbol: str) -> pd.DataFrame:
    
    url = os.path.join('http://localhost:8000', exchange.id, 'trades', exchange.market_id(symbol))
    trades = json.loads(requests.get(url).text)

    df: pd.DataFrame = pd.DataFrame(trades, columns = ['amount', 'price', 'timestamp'])
    df = df.astype({ 'amount': float, 'price': float, 'timestamp': int })

    logger.info(f'{exchange.id} {symbol} - loaded - {df.shape[0]}')

    return df

async def symbol_loop(exchange: Exchange, symbol: str, leverage: int):

    history: dd.DataFrame = fetch_history(exchange, symbol)
 
    botti: Botti = Botti(symbol = symbol, leverage = leverage, history = history)
    setattr(botti, 'exchange', exchange)

    await botti.run()

async def main():

    setup_logging()

    try:

        parser = argparse.ArgumentParser(description = 'Botti trading bot.')
        parser.add_argument('--symbols', type = str, nargs = '+', help = 'symbol to trade', required = True)
        parser.add_argument('--leverage', type = int, help = 'leverage to use', required = True)
        parser.add_argument('--keys', type = str, help = 'keys to use', required = True)

        args = parser.parse_args()

        loop = asyncio.get_event_loop()

        exchange: Exchange = Exchange({
            'asyncio_loop': loop,
            'newUpdates': True,
            'options': { 'enableRateLimit': True, 'watchOrderBook': { 'depth': 'books' }}
        })

        for attr, value in keys[args.keys].items():
            setattr(exchange, attr, value)

        exchange.set_sandbox_mode(keys[args.keys].get('test'))

        await exchange.system_status()
        await exchange.load_markets(reload = False)
        
        loops = [symbol_loop(exchange, symbol, args.leverage) for symbol in args.symbols]

        await asyncio.gather(*loops)
        await exchange.close()

    except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
        
        log_exception(e, exchange.id)

        await exchange.close()


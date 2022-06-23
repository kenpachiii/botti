import argparse
import itertools
import requests
import logging
import os
import numpy as np
import datetime
import time

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

def read_file(file: str) -> dd.DataFrame:
    ddf: dd.DataFrame = dd.read_csv(file, header = 0, names = ['id', 'side', 'amount', 'price', 'timestamp'], blocksize = None)
    ddf = ddf.astype({ 'id': str,'side': str, 'amount': float, 'price': float, 'timestamp': int })
    return ddf

def build_file_urls(exchange: Exchange, symbol: str) -> tuple:
    url = os.path.join('http://localhost:8000', exchange.id, 'trades', exchange.market_id(symbol))
    req = requests.get(os.path.join(url, 'index'))
    return (url, [line.decode() for line in req.iter_lines()])

def fetch_history(exchange: Exchange, symbol: str, days: int = 2) -> dd.DataFrame:

    ddf: dd.DataFrame = None
    
    url, files = build_file_urls(exchange, symbol)
    files.sort()

    files: list = files[-days:]

    ddf: dd.DataFrame = dd.concat([read_file(os.path.join(url, file)) for file in files])
    ddf: dd.DataFrame = ddf.groupby(by=['timestamp', 'side']).agg({ 'price': np.mean, 'amount': np.sum }, split_out = len(files)).reset_index()

    ddf: dd.DataFrame = ddf.persist()

    ddf: dd.DataFrame = ddf.set_index('timestamp').repartition(npartitions = len(files))
    ddf.index = dd.to_datetime(ddf.index, utc = True, unit = 'ms')

    ddf: dd.DataFrame = ddf.persist()

    ddf_buy: dd.DataFrame = ddf[ddf.side == 'BUY']
    ddf_sell: dd.DataFrame = ddf[ddf.side == 'SELL']

    ddf_buy: dd.DataFrame = ddf_buy.resample(f'1800S').agg({ 'price': np.mean, 'amount': np.sum }).dropna()
    ddf_sell: dd.DataFrame = ddf_sell.resample(f'1800S').agg({ 'price': np.mean, 'amount': np.sum }).dropna()

    ddf: dd.DataFrame = ddf.persist()

    ddf: dd.DataFrame = dd.concat([ddf_buy, ddf_sell])
    ddf: dd.DataFrame = dd.from_pandas(ddf.compute().sort_index(), npartitions = ddf.npartitions)

    logger.info(f'{exchange.id} {symbol} - loaded {[os.path.basename(file) for file in files]} - {ddf.compute().shape[0]}')

    return ddf

def symbol_loop(exchange: Exchange, symbol: str, leverage: int) -> list:

    history: dd.DataFrame = fetch_history(exchange, symbol, days = 4).compute().reset_index()
 
    botti: Botti = Botti(symbol = symbol, leverage = leverage, history = history)
    setattr(botti, 'exchange', exchange)

    return botti.run()

def main():

    setup_logging()

    try:

        parser = argparse.ArgumentParser(description='Botti trading bot.')
        parser.add_argument('--symbols', type=str, nargs='+', help='symbol to trade', required = True)
        parser.add_argument('--leverage', type=int, help='leverage to use', required = True)
        parser.add_argument('--keys', type=str, help='keys to use', required = True)

        args = parser.parse_args()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_debug(True)

        exchange: Exchange = Exchange({
            # 'asyncio_loop': loop,
            'newUpdates': True,
            'options': { 'rateLimit': 10, 'watchOrderBook': { 'depth': 'books' }}
        })

        for attr, value in keys[args.keys].items():
            setattr(exchange, attr, value)

        exchange.set_sandbox_mode(keys[args.keys].get('test'))

        loop.run_until_complete(exchange.system_status())
        loop.run_until_complete(exchange.load_markets(reload=False))
        
        loops = list(itertools.chain(*[symbol_loop(exchange, symbol, args.leverage) for symbol in args.symbols]))
        loop.run_until_complete(asyncio.gather(*loops))
        loop.run_until_complete(exchange.close())

    except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
        print(e)
        log_exception(e, exchange.id)

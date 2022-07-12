import argparse
import ccxtpro
import asyncio
import numpy as np

from botti.exchange import Exchange

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

async def main():

    try:

        parser = argparse.ArgumentParser(description = 'print account statistics.')
        parser.add_argument('--account', type = str, help = 'account to use', required = True)

        args = parser.parse_args()

        loop = asyncio.get_event_loop()

        exchange: Exchange = Exchange({
            'asyncio_loop': loop,
            'newUpdates': True,
            'options': { 'enableRateLimit': True, 'watchOrderBook': { 'depth': 'books' }}
        })

        for attr, value in keys[args.account].items():
            setattr(exchange, attr, value)

        exchange.set_sandbox_mode(keys[args.account].get('test'))

        await exchange.system_status()
        await exchange.load_markets(reload = False)
        
        currencies = {}

        orders = await exchange.private_get_trade_orders_history_archive({ 'instType': 'SPOT', 'state': 'filled' })
        for order in orders.get('data'):

            if order.get('instId') not in currencies.keys():
                currencies[order.get('instId')] = []

            if order.get('side') == 'buy':

                currencies[order.get('instId')].append([float(order.get('fillPx')), float(order.get('fillSz'))])

        for ccy, v in currencies.items():
            v = np.asarray(v)

            price, size = v[:, 0], v[:, 1]
            average = np.sum(price * size) / np.sum(size)

            print(ccy, average, np.sum(size), average * np.sum(size))

        await exchange.close()

    except (ccxtpro.NetworkError, ccxtpro.ExchangeError, Exception) as e:
        
        print(e)

        await exchange.close()

asyncio.run(main())


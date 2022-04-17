import math
import numpy as np
import ccxtpro
import asyncio

def market_depth(self, side: str, price: float, size: float) -> float:

    orders = np.asarray(self.order_book.get(side))

    # if worst bid price is greater than price, then return price
    # implies order book window is out of range
    if 'bids' in side and orders[-1][0] > price:
        return price

    # if worst ask price is less than price, then return price
    # implies order book window is out of range
    if 'asks' in side and orders[-1][0] < price:
        return price

    index_arr: np.ndarray = np.ndarray()
    if 'bids' in side:
        # find index where price < orders price
        index_arr = np.argwhere(price < orders[:, 0])

    if 'asks' in side:
        # find index where price > orders price
        index_arr = np.argwhere(price > orders[:, 0])

    if index_arr.size > 0:
        # create orders slice and reverse
        orders = orders[:index_arr[-1][0]+1][::-1]

        # find index where position.open_amount <= cummulative contracts
        cum_index = np.argwhere(size <= np.cumsum(orders[:, 1]))

        if cum_index.size > 0:
            return orders[cum_index[0][0]][0]

    return price

# import traceback
# import logging
# import ccxtpro
# import os

# logger = logging.getLogger(__name__)

# def something(okx: ccxtpro.okx):
#     okx.throw_exactly_matched_exception({ 'string': ccxtpro.InvalidOrder }, 'string', 'message')

# try:

#     okx = ccxtpro.okx()
#     something(okx)

# except Exception as e:

#     frame = None

#     stack = traceback.extract_tb(e.__traceback__)

#     root = os.path.dirname(os.path.abspath(__file__))
#     for s in stack:
#         if root in s.filename:
#             frame = s

#     logger.error('{id} - {file} - {f} - {t}'.format(id='okx', file=frame.filename, f=frame.name, t=type(e).__name__))

# # price < orders price
# [[40382.9, 177.0], [40382.6, 1.0], [40382.5, 1.0], [40380.1, 1.0], [40380.0, 5.0], [40379.9, 3.0], [40379.5, 3.0], [40378.9, 1.0], [40378.8, 351.0]]

# # price > orders price
# [[40383.0, 1694.0], [40383.1, 216.0], [40383.2, 381.0], [40383.3, 124.0], [40383.4, 762.0], [40384.1, 64.0], [40384.2, 145.0], [40384.5, 4.0]]
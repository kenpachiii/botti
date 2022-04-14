# # import asyncio
# # # # # import logging
# # # # # import inspect
# # # # # import traceback
import ccxtpro
import numpy as np

# # # # # from loggers import *

# # # # # logger = logging.getLogger('scrath')

# # class Async:
# #     def __init__(self):
# #         self.counter = 0
# #         self._lock = asyncio.Lock()

# #         self.queue = asyncio.Queue()

# #     async def async_call(self):

# #         while True:

# #             await asyncio.sleep(2)

# #             print('async call')

# #             while self.queue.qsize() > 0:

# #                 n = await self.queue.get()

# #                 async with self._lock:
# #                     await self.create_call(n)

# #     async def create_call(self, n):
# #         # await asyncio.sleep(2)
# #         print(n)
        
# #     async def watch_call(self):

        
# #         while True:

# #             await asyncio.sleep(1)

# #             print('watch call')

# #             self.counter += 1
# #             await self.queue.put(self.counter)


# #     def run(self):

# #         loop = asyncio.new_event_loop()
# #         asyncio.set_event_loop(loop)

# #         loops = [
# #             self.watch_call(),
# #             self.async_call()
# #         ]

# #         loop.run_until_complete(asyncio.gather(*loops))

# # Async().run()
# # # import sqlite3
# # # import uuid


# # # class Cache:

# # #     def __init__(self) -> None:

# # #         try: 
# # #             self.con: sqlite3.Connection = sqlite3.connect('./test.db')
# # #             self.con.row_factory = sqlite3.Row
# # #             self.cur: sqlite3.Cursor = self.con.cursor()

# # #             # self.init()
# # #         except Exception as e:
# # #             print(e)

# # #     def init(self) -> None:
# # #         try:
# # #             self.cur.execute('''CREATE TABLE if not exists position (id TEXT DEFAULT NULL, timestamp INTEGER DEFAULT 0, symbol TEXT DEFAULT NULL, side TEXT, open_amount REAL DEFAULT 0, open_avg REAL DEFAULT 0, close_amount REAL DEFAULT 0, close_avg REAL DEFAULT 0, status TEXT DEFAULT 1, triggered INTEGER DEFAULT 0);''')
# # #             self.con.commit()
# # #         except Exception as e:
# # #             print(e)

# # # class Position(Cache):

# # #     def __init__(self):
# # #         self._id = uuid.uuid4().hex
# # #         super().__init__()

# # #     @property
# # #     def id(self):
# # #         return self._id

# # #     def update(self, object):
# # #         cmd = '''UPDATE position set timestamp = ?, symbol = ?, side = ?, open_amount = ?, open_avg = ?, close_amount = ?, close_avg = ?, status = ?, triggered = ? WHERE id = ?;'''
# # #         self.cur.execute(cmd, object)
# # #         self.con.commit()
        

        


# # # import numpy as np

orders = np.asarray([[4.33255e+04, 3.64000e+02],
 [4.33250e+04, 1.17000e+02],
 [4.33240e+04, 1.10000e+02],
 [4.33234e+04, 1.39000e+02],
 [4.33233e+04, 1.01000e+02],
 [4.33231e+04, 1.58000e+02],
 [4.33230e+04, 1.26000e+02],
 [4.33229e+04, 3.16000e+02],
 [4.33228e+04, 3.13000e+02],
 [4.33225e+04, 4.25000e+02],
 [4.33218e+04, 2.44000e+02],
 [4.33212e+04, 6.47000e+02],
 [4.33211e+04, 1.40000e+02],
 [4.33209e+04, 3.17000e+02],
 [4.33207e+04, 2.76000e+02],
 [4.33205e+04, 4.23000e+02],
 [4.33201e+04, 1.03000e+02], 
 [4.32517e+04, 1.00000e+00], 
 [4.32493e+04, 1.00000e+00], 
 [4.32460e+04, 2.80000e+01], 
 [4.32287e+04, 1.00000e+00],
 [4.32199e+04, 1.00000e+00],
 [4.31995e+04, 1.50000e+01],
 [4.31986e+04, 2.00000e+00],
 [4.31878e+04, 1.00000e+00],
 [4.31804e+04, 2.00000e+00],
 [4.31778e+04, 1.00000e+00],
 [4.31431e+04, 3.00000e+00],
 [4.31312e+04, 1.00000e+00],
 [4.31100e+04, 2.00000e+00],
 [4.31011e+04, 1.00000e+00],
 [4.30846e+04, 5.00000e+00],
 [4.30833e+04, 1.00000e+00],
 [4.30574e+04, 7.00000e+02],
 [4.30228e+04, 3.00000e+00],
 [4.30219e+04, 8.00000e+00],
 [4.30150e+04, 1.00000e+00],
 [4.30144e+04, 1.00000e+00],
 [4.30000e+04, 1.00000e+02],
 [4.29552e+04, 1.30000e+01],
 [4.29500e+04, 4.00000e+01],
 [4.29466e+04, 1.00000e+00],
 [4.29277e+04, 1.00000e+00],
 [4.29000e+04, 5.50000e+02],
 [4.28880e+04, 1.35000e+02],
 [4.28839e+04, 2.10000e+01],
 [4.28800e+04, 8.42000e+02],
 [4.28500e+04, 2.90000e+01],
 [4.28410e+04, 1.00000e+00],
 [4.28400e+04, 2.00000e+00],
 [4.28332e+04, 1.00000e+00],
 [4.28173e+04, 5.00000e+00],
 [4.28099e+04, 1.00000e+00],
 [4.28020e+04, 3.00000e+01],
 [4.28000e+04, 5.77000e+02],
 [4.27989e+04, 5.71000e+02],
 [4.27600e+04, 5.45000e+02],
 [4.27543e+04, 7.00000e+02],
 [4.27483e+04, 5.80000e+01],
 [4.27177e+04, 1.00000e+00],
 [4.27168e+04, 2.00000e+00],
 [4.27163e+04, 2.00000e+00],
 [4.27108e+04, 1.60000e+01],
 [4.27000e+04, 3.26100e+03],
 [4.26733e+04, 1.00000e+00],
 [4.26612e+04, 1.00000e+00],
 [4.26586e+04, 5.00000e+00],
 [4.26500e+04, 3.00000e+02],
 [4.26020e+04, 3.00000e+01],
 [4.26000e+04, 1.36000e+02],
 [4.25990e+04, 1.00000e+02],
 [4.25800e+04, 5.37000e+02],
 [4.25700e+04, 2.00000e+00],
 [4.25500e+04, 9.59000e+02],
 [4.25366e+04, 1.00000e+00],
 [4.25168e+04, 5.00000e+01],
 [4.25050e+04, 3.00000e+01],
 [4.25000e+04, 3.30300e+03],
 [4.24049e+04, 1.00000e+00],
 [4.24020e+04, 3.00000e+01],
 [4.24000e+04, 6.00000e+00],
 [4.23660e+04, 1.00000e+01],
 [4.23320e+04, 8.00000e+01],
 [4.23200e+04, 2.33000e+02],
 [4.23000e+04, 2.36000e+02],
 [4.22540e+04, 1.00000e+00],
 [4.22500e+04, 1.30200e+03],
 [4.22166e+04, 2.00000e+01],
 [4.22020e+04, 3.00000e+01],
 [4.22000e+04, 3.79000e+02],
 [4.21630e+04, 5.00000e+01],
 [4.21500e+04, 1.43700e+03],
 [4.21280e+04, 2.00000e+02],
 [4.21000e+04, 1.00000e+02],
 [4.20010e+04, 8.90000e+01],
 [4.20000e+04, 1.58200e+03],
 [4.19850e+04, 5.90000e+01],
 [4.19440e+04, 3.00000e+02],
 [4.19006e+04, 4.31000e+02],
 [4.19000e+04, 4.72000e+02]])

# import ccxtpro
# import asyncio
# import time
# from functools import wraps
# import logging

# logger = logging.getLogger(__name__)

# def retrier(_func=None):
#     def decorator(f):
#         @wraps(f)
#         def wrapper(*args, **kwargs):
#             try:
#                 return f(*args, **kwargs)
#             except (ccxtpro.NetworkError) as ex:
#                 msg = f'{f.__name__}() returned exception: "{ex}". '
#                 logger.warning(msg + f'Will retry in 10 seconds.')
#                 time.sleep(10)
#                 return wrapper(*args, **kwargs)
#         return wrapper
#     return decorator(_func)


# @retrier
# def retry_me():
#     raise ccxtpro.NetworkError

# async def test():
#     while True:
#         await asyncio.sleep(1)
#         retry_me()

# loop = asyncio.get_event_loop()
# loop.run_until_complete(test())

# # class Async:

# #     def __init__(self):
# #         pass

# #     async def watch_call(self):
# #             raise ccxtpro.NetworkError
        
# #     @retrier
# #     def run(self):

# #         loop = asyncio.new_event_loop()
# #         asyncio.set_event_loop(loop)

# #         loops = [
# #             self.watch_call()
# #         ]

# #         loop.run_until_complete(asyncio.gather(*loops))

# # Async().run()

def safe_index(arr, i, default = None):

    if type(i).__name__ == 'int':
        i = [i]

    for ii in i:

        if ii == -1 and len(arr) > 0:
            arr = arr[-1]
            continue

        n = 0

        while (n < len(arr)):
            if n == ii:
                arr = arr[n] 
                break
            n += 1

    return arr 

def market_depth(orders, side, price, size) -> float:

    orders = ccxtpro.okx.safe_value(orders, side, [])

    # if worst bid price is greater than price, then return price 
    # implies order book window is out of range
    if 'bids' in side and orders[-1][0] > price:
        return price

    # if worst ask price is less than price, then return price 
    # implies order book window is out of range
    if 'asks' in side and orders[-1][0] < price:
        return price

    # find index where price < orders price
    index_arr = np.argwhere(price < orders[:,0])
    if index_arr.size > 0:
        # create orders slice and reverse
        orders = orders[:index_arr[-1][0]+1][::-1]

        # find index where position.open_amount <= cummulative contracts
        cum_index = np.argwhere(size <= np.cumsum(orders[:,1]))

        if cum_index.size > 0:
            return orders[cum_index[0][0]][0]
                
    return price



order_book = { 'bids': [], 'asks': [] }

# market_depth(order_book, 'bids', 4.30833e+04, 30)

print(safe_index(orders, [0], 0))
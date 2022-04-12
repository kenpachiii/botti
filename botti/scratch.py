# # import asyncio
# # import logging
# # import inspect
# # import traceback
# # import ccxtpro

# # from loggers import *

# # logger = logging.getLogger('scrath')

# # def calculate_backoff(retrycount, max_retries):
# #     """
# #     Calculate backoff
# #     """
# #     return (max_retries - retrycount) ** 2 + 1

# # def retrier_async(f):
# #     async def wrapper(*args, **kwargs):
# #         count = kwargs.pop('count', 10)
# #         try:
# #             return await f(*args, **kwargs)
# #         except Exception as ex:
# #             msg = f'{f.__name__}() returned exception: "{ex}". '
# #             if count > 0:
# #                 msg += f'Retrying still for {count} times.'
# #                 count -= 1
# #                 kwargs['count'] = count
# #                 if isinstance(ex, Exception):
# #                     backoff_delay = calculate_backoff(count + 1, 10)
# #                     logger.info(f"Applying DDosProtection backoff delay: {backoff_delay}")
# #                     await asyncio.sleep(backoff_delay)
# #                 if msg:
# #                     logger.warning(msg)
# #                 return await wrapper(*args, **kwargs)
# #             else:
# #                 logger.warning(msg + 'Giving up.')
# #                 raise ex
# #     return wrapper


# # class Something:
# #     def func(self):
# #         raise Exception('something')


# # class Asnyc:
# #     def __init__(self):

# #         self.property = None

# #         return

# #     @retrier_async
# #     async def async_call(self):
# #         await asyncio.sleep(1)
# #         something = Something()
# #         something.func()
          

# #     def sync_call(self):
# #         something = Something()
# #         something.func()
# #         # try:
# #         #     raise Exception('something')
# #         # except Exception as e:
# #         #     logger.error(str(e))


# # async def run():

# #     setup_logging_pre()

# #     f = Asnyc()

# #     try:
# #         await f.async_call()
# #     except Exception as e:
# #         stack = traceback.extract_tb(e.__traceback__, -1).pop(-1)
# #         logger.error('{file} - {f} - {id} - {error}'.format(id='okx', file=stack.filename, f=stack.name, error=str(e)))






# # loop = asyncio.new_event_loop()
# # loop.run_until_complete(run())

import numpy as np

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
 [4.33201e+04, 1.03000e+02], # 103
 [4.32517e+04, 1.00000e+00], # 2
 [4.32493e+04, 1.00000e+00], # 1
 [4.32460e+04, 2.80000e+01], # here
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

def market_depth(order_book, side, price, size) -> float:

    orders = np.asarray(order_book[side])

    # if worst bid price is greater than price, then return price 
    # implies order book window is out of range
    if 'bids' in side and orders[-1][0] > price:
        return price

    # if worst ask price is less than price, then return price 
    # implies order book window is out of range
    if 'asks' in side and orders[-1][0] < price:
        return price

    # find index where price < orders price
    index = np.argwhere(price < orders[:,0])
    if index.size > 0:
        # create orders slice and reverse
        orders = orders[:index[-1][0]+1][::-1]

        # find index where position.open_amount <= cummulative contracts
        cum_index = np.argwhere(size <= np.cumsum(orders[:,1]))

        if cum_index.size > 0:
            return orders[cum_index[0][0]][0]
                
    return price

# print(market_depth({'bids':orders}, 'bids', 4.25800e+04, 30))


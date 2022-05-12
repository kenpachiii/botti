# class Position:

#     def __init__(self, object: dict = {}):
#         self.id = object.get('id')
        
#     @property
#     def __dict__(self):
#         return {k: vars(self)[k] for k in vars(self).keys()}

#     @property
#     def id(self):
#         return self.__id

#     @id.setter
#     def id(self, id):
#         self.__id = id


# position = Position({ 'id': 1 })

# position.id = 2


# print('{id}'.format(**position.__dict__))
import numpy as np

def market_depth(order_book, side: str, price: float, size: float, limit: float = 100) -> float:

    orders = np.asarray(order_book.get(side))[:limit]

    # bid window = best bid > price > worst bid
    if 'bids' in side and not (orders[0][0] >= price >= orders[-1][0]):
        return price

    # ask window = best ask < price < worst ask
    if 'asks' in side and not (orders[0][0] <= price <= orders[-1][0]):
        return price

    index_arr: np.ndarray
    if 'bids' in side:
        # find index where price < orders price
        index_arr = np.argwhere(price <= orders[:, 0])

    if 'asks' in side:
        # find index where price > orders price
        index_arr = np.argwhere(price >= orders[:, 0])

    if index_arr.size > 0:
        orders = orders[:index_arr[-1][0]+1]

    return np.sum(orders[:, 1]) >= size


order_book = {
    'bids': [[30558.0, 420.0], [30557.7, 142.0], [30557.3, 438.0], [30556.7, 628.0], [30555.9, 463.0], [30555.5, 678.0], [30555.1, 283.0], [30554.7, 112.0], [30554.1, 593.0], [30553.5, 632.0], [30553.1, 579.0], [30552.9, 491.0], [30543.0, 3.0], [30540.0, 1.0], [30539.5, 15.0], [30539.4, 5.0], [30539.3, 8.0], [30539.2, 6.0], [30539.1, 11.0], [30536.8, 3.0]],
    'asks': [[30558.9, 8.0], [30559.0, 5.0], [30559.1, 8.0], [30559.2, 12.0], [30559.3, 6.0], [30565.3, 445.0], [30565.5, 466.0], [30565.8, 131.0], [30567.0, 515.0], [30568.3, 671.0], [30568.5, 524.0], [30569.0, 173.0], [30569.6, 478.0], [30569.9, 277.0], [30570.1, 29.0], [30570.2, 644.0], [30570.6, 301.0], [30570.7, 175.0], [30575.3, 496.0], [30577.1, 3.0]]
}

price = 30558.0 
size = 420

print(market_depth(order_book, 'bids', price, size))
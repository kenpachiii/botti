import numpy as np
import json

def market_depth(order_book: dict, side: str, price: float, size: float) -> float:

    orders = np.asarray(order_book.get(side))

    # if worst bid price is greater than price, then return price
    # implies order book window is out of range
    if 'bids' in side and orders[-1][0] > price:
        return price

    # if worst ask price is less than price, then return price
    # implies order book window is out of range
    if 'asks' in side and orders[-1][0] < price:
        return price

    index_arr: np.ndarray
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

order_book = json.loads(open('order_book_dump').read())['BTC/USDT:USDT']

break_even_price = 39781.5 * (1 + 0.0005)**2

print(break_even_price)
print(market_depth(order_book, 'bids', break_even_price, 605))
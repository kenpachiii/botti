# open position
#   - pending order
#   - partial order
#   - completed order

# close position
#   - pending order
#   - partial order
#   - completed order
import time
contracts = 0
timestamp = time.time() * 1000

open_orders = [{ 'timestamp': timestamp - 1800001}, { 'timestamp': (timestamp) - (1800000 * 2) }]

# cancel orders if order has been open longer than 30-minutes
if contracts == 0 and len(open_orders) > 0:
    timestamps = [o.get('timestamp', 0) for o in open_orders]
    timestamps.sort()

    if ((timestamp) - timestamps[-1]) > 1800000:
        print('canceling')
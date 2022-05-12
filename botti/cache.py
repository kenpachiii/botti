import sqlite3
import logging
import json

from botti.position import Position
from botti.sms import send_sms
from botti.exceptions import log_exception

logger = logging.getLogger(__name__)

class Cache:

    def __init__(self, path = './botti.db') -> None:

        try: 
            self.con: sqlite3.Connection = sqlite3.connect(path)
            self.con.row_factory = sqlite3.Row
            self.cur: sqlite3.Cursor = self.con.cursor()

            self.init()
        except Exception as e:
            log_exception(e)

    def init(self) -> None:
        try:
            self.cur.execute('''CREATE TABLE if not exists orders (id TEXT DEFAULT NULL, clientOrderId TEXT, datetime TEXT, timestamp INTEGER DEFAULT 0, lastTradeTimestamp INTEGER DEFAULT 0, status TEXT, symbol TEXT, type TEXT, timeInForce TEXT, postOnly INTEGER, side TEXT, price REAL DEFAULT 0, stopPrice REAL DEFAULT 0, average REAL DEFAULT 0, amount REAL DEFAULT 0, filled REAL DEFAULT 0, remaining REAL DEFAULT 0, cost REAL DEFAULT 0, info TEXT, fees TEXT, fee TEXT, trades TEXT);''')
            self.cur.execute('''CREATE TABLE if not exists position (id TEXT DEFAULT NULL, timestamp INTEGER DEFAULT 0, symbol TEXT DEFAULT NULL, side TEXT, open_amount REAL DEFAULT 0, pending_open_amount REAL DEFAULT 0, open_avg REAL DEFAULT 0, close_amount REAL DEFAULT 0, pending_close_amount REAL DEFAULT 0, close_avg REAL DEFAULT 0, status INTEGER, state INTEGER, triggered INTEGER DEFAULT 0);''')
            self.con.commit()
        except Exception as e:
            log_exception(e)

    def insert_order(self, order: dict) -> None:

        try:

            # make copy otherwise it edits the actual cache
            order = order.copy()

            # FIXME: ccxt isnt adding this itself
            # this also may be wrong from ccxt perspective
            trade = order.get('info').get('tradeId')
            if trade.isnumeric():
                order['trades'] = [trade]
                
            for k in order.keys():
                if type(order[k]) == dict or type(order[k]) == list:
                    order[k] = json.dumps(order[k])

            columns = ', '.join("`" + str(x).replace('/', '_') + "`" for x in order.keys())
            values = ', '.join("'" + str(x).replace('/', '_') + "'" for x in order.values())

            insert = 'INSERT INTO %s ( %s ) VALUES ( %s );' % ('orders', columns, values)

            self.cur.execute(insert)
            self.con.commit()
        except Exception as e:
            log_exception(e)

    def fetch_order(self) -> dict:
        try: 

            order = {}

            values = self.cur.execute('''SELECT * FROM orders ORDER BY timestamp DESC, CASE WHEN status = 'open' THEN 1 WHEN status = 'closed' THEN 2 END DESC;''').fetchone()
            if values is None:
                return order

            for k in values.keys():

                try:
                    order[k] = json.loads(values[k])
                except Exception:
                    order[k] = values[k]

            return order

        except Exception as e:
            log_exception(e)

    def fetch_orders(self, id) -> dict:
        try: 

            orders = []

            select = '''SELECT * FROM orders WHERE id = '{id}' ORDER BY timestamp DESC, filled ASC;'''.format(id=id)

            values = self.cur.execute(select).fetchall()
            if values is None:
                return []

            for i in range(0, len(values)):

                order = {}

                for k in values[i].keys():

                    try:
                        order[k] = json.loads(values[i][k])
                    except Exception:
                        order[k] = values[i][k]

                orders.append(order)

            return orders

        except Exception as e:
            log_exception(e)

    def insert(self, position: dict) -> None:

        try:

            position = position.copy()
                
            for k in position.keys():
                if type(position[k]) == dict or type(position[k]) == list:
                    position[k] = json.dumps(position[k])

                if k == 'status' or k == 'state':
                    position[k] = position.get(k).value

            columns = ', '.join("`" + str(x) + "`" for x in position.keys())
            values = ', '.join("'" + str(x) + "'" for x in position.values())

            insert = 'INSERT INTO %s ( %s ) VALUES ( %s );' % ('position', columns, values)

            self.cur.execute(insert)
            self.con.commit()
        except Exception as e:
            log_exception(e)

    def update(self, id: str, position: dict) -> None:

        try:

            keys = position.keys()
        
            args: list = []
            for k in keys:

                if k == 'id':
                    continue

                if k == 'status' or k == 'state':
                    args.append(k + ' = ' + "'{}'".format(str(position.get(k).value)))

                    continue

                args.append(k + ' = ' + "'{}'".format(str(position.get(k))))

            update = '''UPDATE position SET {values} WHERE id = '{id}';'''.format(values=', '.join(args), id=id)

            self.cur.execute(update)
            self.con.commit()
        except Exception as e:
            log_exception(e)

    def remove(self, table: str, id: str) -> None:
        try:

            remove = '''DELETE from {table} WHERE id = '{id}';'''.format(table=table,id=id)
 
            self.cur.execute(remove)
            self.con.commit()
        except Exception as e:
            log_exception(e)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception as e:
            log_exception(e)

    def close(self) -> None:
        try:
            self.con.commit()
            self.con.close()
        except Exception as e:
            log_exception(e)

        logger.info('flushed and closed sqlite connection')

    @property
    # returns currently opened trade if any
    def position(self) -> Position:
        try: 
            values = self.cur.execute('''SELECT * FROM position WHERE status = 1 OR status = 2;''').fetchone()
            return Position({k: values[k] for k in values.keys()}) if values else Position({})
        except Exception as e:
            log_exception(e)

    @property
    # returns last open trade excluding currectly open trade if any
    def last(self) -> Position:
        try: 
            values = self.cur.execute('''SELECT * FROM position WHERE status = 3 ORDER BY timestamp DESC;''').fetchone()
            return Position({k: values[k] for k in values.keys()}) if values else Position({})
        except Exception as e:
            log_exception(e)









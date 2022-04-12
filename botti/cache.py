import sqlite3
import logging
from botti.position import Position

logger = logging.getLogger(__name__)

class Cache:

    # FIXME: how to make this 'safer' has no unique contraints 
    # FIXME: position is actually an order so should be refactored
    def __init__(self) -> None:

        try: 
            self.con: sqlite3.Connection = sqlite3.connect('./botti.db')
            self.con.row_factory = sqlite3.Row
            self.cur: sqlite3.Cursor = self.con.cursor()

            self.init()
        except Exception as e:
            logger.error('init - {error}'.format(error=e))

    def init(self) -> None:
        try:
            self.cur.execute('''CREATE TABLE if not exists position (id TEXT DEFAULT NULL, timestamp INTEGER DEFAULT 0, symbol TEXT DEFAULT NULL, side TEXT, open_amount REAL DEFAULT 0, open_avg REAL DEFAULT 0, close_amount REAL DEFAULT 0, close_avg REAL DEFAULT 0, status TEXT DEFAULT 1, triggered INTEGER DEFAULT 0);''')
            self.con.commit()
        except Exception as e:
            logger.error('inti - {error}'.format(error=e))

    def insert(self, position: Position) -> None:

        try:

            args = (position.id, position.timestamp, position.symbol, position.side, position.open_amount, position.open_avg, position.close_amount, position.close_avg, position.status, position.triggered)

            self.cur.execute('''INSERT INTO position VALUES (?,?,?,?,?,?,?,?,?,?);''', args)
            self.con.commit()
        except Exception as e:
            logger.error('insert - {error}'.format(error=e))

    def update(self, position: Position) -> None:

        try:

            args = (position.timestamp, position.symbol, position.side, position.open_amount, position.open_avg, position.close_amount, position.close_avg, position.status, position.triggered, position.id)

            self.cur.execute('''UPDATE position set timestamp = ?, symbol = ?, side = ?, open_amount = ?, open_avg = ?, close_amount = ?, close_avg = ?, status = ?, triggered = ? WHERE id = ?;''', args)
            self.con.commit()
        except Exception as e:
            logger.error('update - {error}'.format(error=e))

    def clear(self) -> None:
        try:
            self.cur.execute('''DELETE from position;''')
            self.con.commit()
        except Exception as e:
            logger.error('clear - {error}'.format(error=e))

    def __del__(self) -> None:
        try:
            self.close()
        except Exception as e:
            logger.error('{error}'.format(error=e))

    def close(self) -> None:
        try:
            self.con.commit()
            self.con.close()
        except Exception as e:
            logger.error('close - {error}'.format(error=e))

        logger.info('flushed and closed sqlite connection')

    def all(self) -> Position:
        try: 
            values = self.cur.execute('''SELECT * FROM position;''').fetchall()
            for value in values:
                print(vars(Position({k: value[k] for k in value.keys()})))
        except Exception as e:
            logger.error('position - {error}'.format(error=e))

    @property
    # returns currently opened trade if any
    def position(self) -> Position:
        try: 
            values = self.cur.execute('''SELECT * FROM position WHERE status = 'open' OR status = 'pending';''').fetchone()
            return Position({k: values[k] for k in values.keys()}) if values else Position({})
        except Exception as e:
            logger.error('position - {error}'.format(error=e))

    @property
    # returns last open trade excluding currectly open trade if any
    def last(self) -> Position:
        try: 
            values = self.cur.execute('''SELECT * FROM position WHERE status = 'closed';''').fetchone()
            return Position({k: values[k] for k in values.keys()}) if values else Position({})
        except Exception as e:
            logger.error('position - {error}'.format(error=e))

cache = Cache()

# position = Position({ 'id': '2f6f0bf63798', 'timestamp': 1649558505, 'symbol': 'BTC/USDT:USDT', 'side': 'buy', 'open_amount': 30.0, 'open_avg': 42735.7, 'close_amount': 0, 'close_avg': 0, 'status': 'pending', 'triggered': 0 })

# cache.insert(position)

# position.update({ 'triggered': 1, 'status': 'open' })

# cache.update(position)

# print(vars(cache.position))

cache.all()






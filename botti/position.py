from enum import Enum
from decimal import DivisionByZero
from typing import Any

class PositionStatus(Enum):
    PENDING = 0
    OPEN = 1
    CLOSED = 2

    def __str__(self):
        return self.name

class Position:

    def __init__(self, object: dict = {}) -> None:

        self._id: str = object.get('id')
        self._timestamp: int = object.get('timestamp') or 0
        self._symbol: str = object.get('symbol')
        self._side: str = object.get('side')
        self._open_amount: float = object.get('open_amount') or 0
        self._open_avg: float = object.get('open_avg') or 0
        self._close_amount: float = object.get('close_amount') or 0
        self._close_avg: float = object.get('close_avg') or 0
        self._status: PositionStatus = object.get('status') 
        self._triggered: bool = object.get('triggered') or 0

    @property
    def id(self) -> str:
        return self._id

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def side(self) -> str:
        return self._side

    @property
    def open_amount(self) -> float:
        return self._open_amount

    @property
    def open_avg(self) -> float: 
        return self._open_avg

    @property
    def close_amount(self) -> float:
        return self._close_amount

    @property
    def close_avg(self) -> float: 
        return self._close_avg

    @property
    def status(self) -> PositionStatus: 
        return PositionStatus(self._status) if self._status is not None else None

    @property
    def triggered(self) -> int: 
        return self._triggered

    def update(self, object: dict) -> None:
        for key, value in object.items():
            setattr(self, '_{}'.format(key), value)

    def get(self, key: str) -> Any:
        return vars(self).get('_{}'.format(key))

    def position_avg(self, type: str, order: dict) -> float:

        avg = 'open_avg' if type == 'open' else 'close_avg'
        amount = 'open_amount' if type == 'open' else 'close_amount'

        x1: float = self.get(avg) * self.get(amount)
        x2: float = order.get('average') * order.get('filled')
        return (x1 + x2) / (self.get(amount) + order.get('filled'))

    def pnl(self, leverage: float, close: float = 0) -> float:

        try:
            close = close if close > 0 else self.close_avg
            return (((close - self.open_avg) / self.open_avg) * 100) * leverage
        except DivisionByZero:
            return 0.0

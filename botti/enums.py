from enum import Enum, unique, auto

@unique
class PositionStatus(Enum):
    PENDING = auto()
    OPEN = auto()
    CLOSED = auto()
    NONE = None

    def __str__(self):
        return self.name

@unique
class PositionState(Enum):
    ENTRY = auto()
    EXIT = auto()
    NONE = None

    def __str__(self):
        return self.name

@unique
class PositionSide(Enum):
    LONG = 'long'
    SHORT = 'short'
    NONE = None

    def __str__(self):
        return self.name

@unique
class SystemStatus(Enum):
    OK = 'ok'
    SHUTDOWN = 'shutdown'
    ERROR = 'error'
    MAITENANCE = 'maintenance'

    def __str__(self):
        return self.name

@unique
class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'

    def __str__(self):
        return self.name
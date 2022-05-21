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
class SystemState(Enum):
    SCHEDULED = 'scheduled'
    ONGOING = 'ongoing'
    COMPLETED = 'completed'
    CANCELED = 'canceled'

    def __str__(self):
        return self.name

@unique
class ServiceType(Enum):
    WEBSOCKET = '0'
    SPOTMARGIN = '1'
    FUTURES = '2'
    PERPETUAL = '3'
    OPTIONS = '4'
    TRADING = '5'

    def __str__(self):
        return self.name

@unique
class OrderSide(Enum):
    BUY = 'buy'
    SELL = 'sell'

    def __str__(self):
        return self.name
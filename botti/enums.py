from enum import Enum, unique, auto

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
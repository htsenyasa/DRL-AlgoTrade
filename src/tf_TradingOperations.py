import numpy as np

signal = {"Buy": True, "Sell": False}

class StockHandler():
    """ 
    Handles stock
        GetFunc    : The function to download/get the stock in question
        UpdateFunc : The function to update the stock in question.
        horizon    : namedtuple("Horizon", ["start", "end", "interval"])
        
        The main reason for these two functions to be different is that there may be a need to
        use different update function for simulation purposes.
    """
    def __init__(self, stockCode, GetFunc, UpdateFunc, horizon, interpolate=True):
        self.stockCode = stockCode
        self.GetFunc = GetFunc
        self.UpdateFunc = UpdateFunc
        self.horizon = horizon

        self.dataFrame = GetFunc(self.stockCode, start=horizon.start, end=horizon.end, interval=horizon.interval)
        
        if interpolate is True:
            self.dataFrame.replace(0.0, np.nan, inplace=True)
            self.dataFrame.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
            self.dataFrame.fillna(method='ffill', inplace=True)
            self.dataFrame.fillna(method='bfill', inplace=True)
            self.dataFrame.fillna(0, inplace=True)


class PositionHandler():
    """ Creates a position for a given stock """
    def __init__(self, stock, initialCash = 100_000, initialLots = 0, initialCost = 0, currentPosition = 0, tradingFee = 0.02):
        self.stock = stock
        self.cash = float(initialCash)
        self.lots = float(initialLots)
        self.cost = float(initialCost)
        self.currentPosition = currentPosition # Long or Short or No Position

        self.value = self.lots * self.cost
        self.returns = 0

        self.LONG = 1
        self.NO_POSITION = 0
        self.SHORT = -1       

        self.tradingFee = tradingFee

    def IsLong(self):
        return self.currentPosition == self.LONG
    def IsShort(self):
        return self.currentPosition == self.SHORT
    def GetPosition(self):
        return self.currentPosition 

    def GoLong(self):
        if self.IsLong():
            ...

class DummyPosition():
    """ Creates an empty dummy position for a given stock for simulation purposes"""
    def __init__(self, stock, initialCash = 100_000, tradingFee = 0.02, epsilon=0.1):
        self.stock = stock

        self.LONG = 1
        self.NO_POSITION = 0
        self.SHORT = -1
        self.NO_ACTION = 0
        self.tradingFee = tradingFee
        self.epsilon = epsilon
        self.initialCash = float(initialCash)

        self.dataFrame = stock.dataFrame.copy()        
        self.dataFrame["Cash"] = float(initialCash)
        self.dataFrame["Position"] = self.NO_POSITION
        self.dataFrame["Action"] = self.NO_ACTION
        self.dataFrame["Lots"] = 0
        self.dataFrame["Holdings"] = 0
        self.dataFrame["Value"] = self.dataFrame["Holdings"] + self.dataFrame["Cash"]
        self.dataFrame["Returns"] = 0


    def IsLong(self, tick):
        return self.dataFrame["Position"][tick] == self.LONG

    def IsShort(self, tick):
        return self.dataFrame["Position"][tick] == self.SHORT

    def GetPosition(self, tick):
        return self.dataFrame["Position"][tick]

    def ClosePosition(self, tick):
        ...

    def ResetPosition(self):
        self.dataFrame = self.stock.dataFrame.copy()        
        self.dataFrame["Cash"] = self.initialCash
        self.dataFrame["Position"] = self.NO_POSITION
        self.dataFrame["Action"] = self.NO_ACTION
        self.dataFrame["Lots"] = 0
        self.dataFrame["Holdings"] = 0
        self.dataFrame["Value"] = self.dataFrame["Holdings"] + self.dataFrame["Cash"]
        self.dataFrame["Returns"] = 0


    def GoLong(self, tick):
        prev = tick-1
        self.dataFrame["Position"][tick] = self.LONG
        if self.IsLong(prev):
            self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][prev]
            self.dataFrame["Holdings"][tick] = self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick]
        elif self.IsShort(prev):
            self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][prev] - self.dataFrame["Lots"][prev] * self.dataFrame["Close"][tick] * (1 + self.tradingFee)
            self.dataFrame["Lots"][tick] = int(self.dataFrame["Cash"][tick] // (self.dataFrame["Close"][tick] * (1 + self.tradingFee))) 
            self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][tick] - self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick] * (1 + self.tradingFee)
            self.dataFrame["Holdings"][tick] = self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick]
            self.dataFrame["Action"][tick] = self.LONG
        else: #NO_POSITION
            self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][prev]
            self.dataFrame["Lots"][tick] = int(self.dataFrame["Cash"][tick] // (self.dataFrame["Close"][tick] * (1 + self.tradingFee)))
            self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][tick] - self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick] * (1 + self.tradingFee)
            self.dataFrame["Holdings"][tick] = self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick]
            self.dataFrame["Action"][tick] = self.LONG

    def GoShort(self, tick):
        prev = tick - 1
        self.dataFrame["Position"][tick] = self.SHORT
        if self.IsLong(prev):
            
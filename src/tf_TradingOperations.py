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
        self.startDate = horizon.start
        self.endDate = horizon.end
        self.interval = horizon.interval

        self.dataFrame = GetFunc(self.stockCode, start=self.startDate, end=self.endDate, interval=self.interval)
        
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

        self.long = 1
        self.noPosition = 0
        self.short = -1       

        self.tradingFee = tradingFee

    def IsLong(self):
        return self.currentPosition == self.long
    def IsShort(self):
        return self.currentPosition == self.short
    def GetPosition(self):
        return self.currentPosition 

    def GoLong(self):
        if self.IsLong():
            ...
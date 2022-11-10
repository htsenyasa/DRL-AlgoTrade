import numpy as np

class Position():
    """ Creates a position for a given stock """
    def __init__(self, stockCode, initialLots = 0, initialCost = 0, currentPosition = None):
        self.stockCode = stockCode
        self.lots = initialLots
        self.initialCost = initialCost
        self.currentPosition = currentPosition # Long or Short
        self.history = []

    def SetPositionHorizon(self, startDate, endDate, interval="1d"):
        """
        horizonLength: horizonLength*interval
        """
        self.startDate = startDate
        self.endDate = endDate
        self.interval = interval

    def GetPositionHorizon(self):
        return (self.startDate, self.endDate, self.interval)


class StockHandler():
    """ 
    Handles stock
        GetFunc    : The function to download/get the stock in question
        UpdateFunc : The function to update the stock in question.
        horizon    : namedtuple("Horizon", ["start", "end", "interval"])
        
        The main reason for these two functions to be different is that there may be a need to
        use different update function for simulation purposes.
    """
    def __init__(self, stockCode, GetFunc, UpdateFunc, horizon):
        self.stockCode = stockCode
        self.GetFunc = GetFunc
        self.UpdateFunc = UpdateFunc
        self.startDate = horizon.start
        self.endDate = horizon.end
        self.interval = horizon.interval

        self.dataFrame = GetFunc(self.startDate, self.endDate, self.interval)
        self.dataFrame.replace(0.0, np.nan, inplace=True)
        self.dataFrame.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        self.dataFrame.fillna(method='ffill', inplace=True)
        self.dataFrame.fillna(method='bfill', inplace=True)
        self.dataFrame.fillna(0, inplace=True)
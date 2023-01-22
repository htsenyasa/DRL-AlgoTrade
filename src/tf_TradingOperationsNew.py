import numpy as np
import pandas as pd
from collections import namedtuple
import mplfinance as mpf
import matplotlib.pyplot as plt


signal = {"Buy": True, "Sell": False}
Horizon = namedtuple("Horizon", ["start", "end", "interval"])

class StockHandler():
    """ 
    Handles stock
        GetFunc    : The function to download/get the stock in question
        UpdateFunc : The function to update the stock in question.
        horizon    : namedtuple("Horizon", ["start", "end", "interval"])
        
        The main reason for these two functions (get and update) to be different is that there may be a need to
        use different update function for simulation purposes.
    """
    def __init__(self, stockCode, GetFunc, UpdateFunc, horizon, interpolate=True):
        self.stockCode = stockCode
        self.GetFunc = GetFunc
        self.UpdateFunc = UpdateFunc
        self.horizon = horizon

        self.dataFrame = GetFunc(self.stockCode, start=horizon.start, end=horizon.end, interval=horizon.interval, progress=False)
        
        if interpolate is True:
            self.dataFrame.replace(0.0, np.nan, inplace=True)
            self.dataFrame.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
            self.dataFrame.fillna(method='ffill', inplace=True)
            self.dataFrame.fillna(method='bfill', inplace=True)
            self.dataFrame.fillna(0, inplace=True)
        
        self.dataFrame["Close"] = self.dataFrame["Adj Close"]
        self.dataFrame = self.dataFrame.drop(columns=["Adj Close"])

        self.SetPrecision(2)

    def Update(self):
        newData = self.UpdateFunc(self.stockCode)
        if newData is None:
            return None
        self.dataFrame = pd.concat([self.dataFrame, newData])
        return self.dataFrame


    def SetPrecision(self, precision):
        self.dataFrame.Open = self.dataFrame.Open.round(precision)
        self.dataFrame.High = self.dataFrame.High.round(precision)
        self.dataFrame.Low = self.dataFrame.Low.round(precision)
        self.dataFrame.Close = self.dataFrame.Close.round(precision)



class DummyPosition():
    """ Creates an empty dummy position for a given stock for simulation purposes"""
    def __init__(self, stock, t = 1, initialCash = 100_000, tradingFee = 0.0, shortMargin = 0.8):
        self.stock = stock
        self.LONG = 1
        self.NO_POSITION = 0
        self.SHORT = -1
        self.NO_ACTION = 0
        self.tradingFee = tradingFee
        self.initialCash = float(initialCash)
        self.shortMargin = shortMargin
        self.__t = t # To be able reset position to initial version.
        self.t = t  # Current candle. Its unit (interval e.g., "1d", "1m" etc) is dictated by the stock.dataFrame attribute
        self.ResetPosition() # Initialize various variables including OHLCV data.



    def ResetPosition(self):
        self.dataFrame = self.stock.dataFrame.copy()        
        self.Length = len(self.dataFrame.index)
        self.dateTimeIndex = self.dataFrame.index 
        self.open = self.dataFrame.Open.values
        self.high = self.dataFrame.High.values
        self.low = self.dataFrame.Low.values
        self.close = self.dataFrame.Close.values
        self.volume = self.dataFrame.Volume.values
        self.cash = np.full(self.Length, float(self.initialCash), dtype=float)
        self.position = np.full(self.Length, self.NO_POSITION, dtype=int)
        self.action = np.full(self.Length, self.NO_ACTION, dtype=int)
        self.lots = np.full(self.Length, 0, dtype=int)
        self.holdings = np.full(self.Length, 0.0, dtype=float)
        self.value = self.holdings + self.cash
        self.returns = np.full(self.Length, 0.0, dtype=float)
        self.buyHistory = np.full(self.Length, np.nan, dtype=float)
        self.sellHistory = np.full(self.Length, np.nan, dtype=float)
        self.t = self.__t



    def SetStartingPoint(self, t):
        self.__t = t
        self.t = t


    def IsLong(self):
        return self.position[self.t] == self.LONG


    def IsShort(self):
        return self.position[self.t] == self.SHORT


    def GetPosition(self):
        return self.position[self.t]


    def __PreIteration(self):
        t = self.t
        self.cash[t] = self.cash[t-1]
        self.lots[t] = self.lots[t-1]
        self.position[t] = self.position[t-1]
        self.holdings[t] = self.lots[t] * self.close[t]
        self.value[t] = self.holdings[t] + self.cash[t]
        self.returns[t] = (self.value[t] - self.value[t-1])/self.value[t-1]



    def __PostIteration(self):
        t = self.t
        self.holdings[t] = self.lots[t] * self.close[t]
        self.value[t] = self.holdings[t] + self.cash[t]
        self.returns[t] = (self.value[t] - self.value[t-1])/self.value[t-1]



    @staticmethod
    def __Iteration(f):
        def Iteration(*args):
            self = args[0]
            self.__PreIteration()
            f(*args)
            self.__PostIteration()
            self.t += 1
        return Iteration



    def __Buy(self, lots):
        t = self.t
        price = self.close[t] * (1 + self.tradingFee)
        self.lots[t] += lots
        self.cash[t] -= lots * price



    def __Sell(self, lots):
        t = self.t
        price = self.close[t] * (1 - self.tradingFee)
        self.cash[t] += lots * price
        self.lots[t] -= lots



    def GetNumberOfLots(self, shortFlag = False):
        t = self.t
        price = self.close[t] * (1+ self.tradingFee)
        if shortFlag:
            if self.value[t] * self.shortMargin > abs(self.holdings[t]):
                return int(((self.value[t] + self.holdings[t]) * self.shortMargin) // price)
            else:
                return 0
        return int(self.cash[t] // price)


    @__Iteration
    def GoLong(self):
        t = self.t

        if self.IsShort():
            self.CloseShortPosition()

        lots = self.GetNumberOfLots()

        if lots > 0:
            self.action[t] = self.LONG
            self.__Buy(lots)

        self.position[t] = self.LONG


    @__Iteration
    def GoShort(self):
        t = self.t

        if self.IsLong():
            self.CloseLongPosition()

        lots = self.GetNumberOfLots(shortFlag=True)

        if lots > 0:
            self.action[t] = self.SHORT
            self.__Sell(lots)

        self.position[t] = self.SHORT


    def CloseLongPosition(self):
        t = self.t
        self.__Sell(abs(self.lots[t]))
        self.__PostIteration()


    def CloseShortPosition(self):
        t = self.t
        self.__Buy(abs(self.lots[t]))
        self.__PostIteration()



    def ToDataFrame(self):
        self.dataFrame["Cash"] = self.cash
        self.dataFrame["Position"] = self.position
        self.dataFrame["Action"] = self.action
        self.dataFrame["Lots"] = self.lots
        self.dataFrame["Holdings"] = self.holdings
        self.dataFrame["Value"] = self.value
        self.dataFrame["Returns"] = self.returns


    def NewActionBranch(self):
        t = self.t
        self.__tempCash = self.cash[t-1:t+1]
        self.__tempPosition = self.position[t-1:t+1]
        self.__tempAction = self.action[t-1:t+1]
        self.__tempLots = self.lots[t-1:t+1]
        self.__tempHoldings = self.holdings[t-1:t+1]
        self.__tempValue = self.value[t-1:t+1]
        self.__tempReturns = self.returns[t-1:t+1]


    def MergeBranches(self):
        t = self.t
        self.cash[t-1:t+1] = self.__tempCash
        self.position[t-1:t+1] = self.__tempPosition
        self.action[t-1:t+1] = self.__tempAction
        self.lots[t-1:t+1] = self.__tempLots
        self.holdings[t-1:t+1] = self.__tempHoldings
        self.value[t-1:t+1] = self.__tempValue
        self.returns[t-1:t+1] = self.__tempReturns



    def ParseActions(self):
        actions = self.position
        index = []

        buyHistory = np.full(self.dataFrameLength, np.nan)
        sellHistory = np.full(self.dataFrameLength, np.nan)

        for i in range(len(actions)-1):
            if actions[i] == actions[i+1]:
                index.append(i+1)

        actions[index] = 0

        for i, item in enumerate(actions):
            if item == self.LONG:
                buyHistory[i] = self.close[i]
            elif item == self.SHORT:
                sellHistory[i] = self.close[i]

        return buyHistory, sellHistory



    def PlotActionsCapital(self, saveFileName, showFlag = False):
        buyHistory, sellHistory = self.ParseActions()
        idx = np.argwhere(~np.isnan(buyHistory))
        idx = idx.flatten()
        buys = self.value[idx]

        idxSell = np.argwhere(~np.isnan(sellHistory))
        idxSell = idxSell.flatten()
        sells = self.value[idxSell]

        # plt.plot(self.dataFrame.index, self.dataFrame["Value"], "r")
        # plt.scatter(self.dataFrame.index[idx], buys, s=200, marker="^")
        # plt.scatter(self.dataFrame.index[idxSell], sells, s=200, marker="v")

        fig, ax1 = plt.subplots()
        ax1.plot(self.dataFrame.index, self.dataFrame["Value"], "r", label="Capital")
        ax1.scatter(self.dataFrame.index[idx], buys, s=200, marker="^", label="Long")
        ax1.scatter(self.dataFrame.index[idxSell], sells, s=200, marker="v", label="Short")
        ax1.set_xlabel("Date", fontsize=20)
        ax1.set_ylabel("Capital", fontsize=20)
        ax1.legend(loc="upper left", fontsize=14)
        ax1.tick_params(labelsize=15)
        # ax1.set_aspect()
        figure = plt.gcf()
        figure.set_size_inches(16,9)
        plt.tight_layout()

        plt.savefig(saveFileName + ".png", format = "png", dpi=300)
        if showFlag == True:
            plt.show()



    def PlotActionsCandle(self, saveFileName, showFlag = False):
        buyHistory, sellHistory = self.ParseActions()
        idx = np.argwhere(~np.isnan(buyHistory))
        idx = idx.flatten()
        buys = self.dataFrame["Close"][idx]

        idxSell = np.argwhere(~np.isnan(sellHistory))
        idxSell = idxSell.flatten()
        sells = self.dataFrame["Close"][idxSell]

        apd = [mpf.make_addplot(buyHistory,type='scatter', markersize=50,marker='^'), mpf.make_addplot(sellHistory,type='scatter', markersize=50,marker='v')]
        saveFileName = saveFileName + "-candle" + ".png"
        mpf.plot(self.dataFrame, addplot=apd, type="candle", savefig=saveFileName)



    def PlotActionsPrice(self, saveFileName, showFlag = False):
        buyHistory, sellHistory = self.ParseActions()
        idx = np.argwhere(~np.isnan(buyHistory))
        idx = idx.flatten()
        buys = self.dataFrame["Close"][idx]

        idxSell = np.argwhere(~np.isnan(sellHistory))
        idxSell = idxSell.flatten()
        sells = self.dataFrame["Close"][idxSell]

        fig, ax1 = plt.subplots()
        ax1.plot(self.dataFrame.index, self.dataFrame["Close"], "r", label="Price")
        ax1.scatter(self.dataFrame.index[idx], buys, s=200, marker="^", label="Long")
        ax1.scatter(self.dataFrame.index[idxSell], sells, s=200, marker="v", label="Short")
        ax1.set_xlabel("Date", fontsize=20)
        ax1.set_ylabel("Price", fontsize=20)
        ax1.legend(loc="upper left", fontsize=14)
        ax1.tick_params(labelsize=15)
        # ax1.set_aspect()
        figure = plt.gcf()
        figure.set_size_inches(16,9)
        plt.tight_layout()

        saveFileName = saveFileName + "-price" + ".png"
        plt.savefig(saveFileName, format = "png", dpi=300)
        if showFlag == True:
            plt.show()
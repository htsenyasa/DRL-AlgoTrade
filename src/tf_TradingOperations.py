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
        
    def Update(self):
        newData = self.UpdateFunc(self.stockCode)
        if newData is None:
            return None
        self.dataFrame = pd.concat([self.dataFrame, newData])
        return self.dataFrame


class DummyPosition():
    """ Creates an empty dummy position for a given stock for simulation purposes"""
    def __init__(self, stock, tick = 30, initialCash = 100_000, tradingFee = 0.1/100, epsilon=0.1):
        self.stock = stock
        self.LONG = 1
        self.NO_POSITION = 0
        self.SHORT = -1
        self.NO_ACTION = 0
        self.tradingFee = tradingFee
        self.epsilon = epsilon
        self.initialCash = float(initialCash)
        self.__tick = tick # To be able reset position to initial version.
        self.tick = tick  # Current candle. Its unit (interval e.g., "1d", "1m" etc) is dictated by the stock.dataFrame attribute

        self.dataFrame = stock.dataFrame.copy()        
        self.dataFrameLength = len(self.dataFrame.index)
        self.dataFrame["Cash"] = float(initialCash)
        self.dataFrame["Position"] = self.NO_POSITION
        self.dataFrame["Action"] = self.NO_ACTION
        self.dataFrame["Lots"] = 0
        self.dataFrame["Holdings"] = 0
        self.dataFrame["Value"] = self.dataFrame["Holdings"] + self.dataFrame["Cash"]
        self.dataFrame["Returns"] = 0

        # For plotting during testing
        self.buyHistory = np.full(self.dataFrameLength, np.nan)
        self.sellHistory = np.full(self.dataFrameLength, np.nan)



    def IsLong(self, tick = False):
        if tick:
            return self.dataFrame["Position"][tick] == self.LONG
        return self.dataFrame["Position"][self.tick] == self.LONG



    def IsShort(self, tick):
        if tick:
            return self.dataFrame["Position"][tick] == self.SHORT
        return self.dataFrame["Position"][self.tick] == self.SHORT



    def GetPosition(self, tick):
        if tick:
            return self.dataFrame["Position"][tick]
        return self.dataFrame["Position"][self.tick]



    def ResetPosition(self):
        self.dataFrame = self.stock.dataFrame.copy()        
        self.dataFrame["Cash"] = self.initialCash
        self.dataFrame["Position"] = self.NO_POSITION
        self.dataFrame["Action"] = self.NO_ACTION
        self.dataFrame["Lots"] = 0
        self.dataFrame["Holdings"] = 0
        self.dataFrame["Value"] = self.dataFrame["Holdings"] + self.dataFrame["Cash"]
        self.dataFrame["Returns"] = 0
        self.tick = self.__tick



    def ComputeLowerBound(self, cash, lots, price):
        deltaValues = - cash - lots * price * (1 + self.epsilon) * (1 + self.tradingFee)
        if deltaValues < 0:
            lowerBound = int(deltaValues // (price * (2 * self.tradingFee + (self.epsilon * (1 + self.tradingFee)))))
        else:
            lowerBound = int(deltaValues // (price * self.epsilon * (1 + self.tradingFee)))
        return lowerBound



    def GoLong2(self, tick):
        prev = tick-1
        self.dataFrame["Position"][tick] = self.LONG
        if self.IsLong(prev):
            self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][prev]
            self.dataFrame["Lots"][tick] = self.dataFrame["Lots"][prev]
            self.dataFrame["Holdings"][tick] = self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick]
        elif self.IsShort(prev):
            #Close short position, to do that obtain necessary amount of shares first to return borrowed shares.
            self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][prev] - self.dataFrame["Lots"][prev] * self.dataFrame["Close"][tick] * (1 + self.tradingFee)
            self.dataFrame["Lots"][tick] = int(self.dataFrame["Cash"][tick] // (self.dataFrame["Close"][tick] * (1 + self.tradingFee))) 
            self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][tick] - self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick] * (1 + self.tradingFee)
            self.dataFrame["Holdings"][tick] = self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick]
            self.dataFrame["Action"][tick] = self.LONG
        else: #NO_POSITION
            self.dataFrame["Lots"][tick] = int(self.dataFrame["Cash"][prev] // (self.dataFrame["Close"][tick] * (1 + self.tradingFee)))
            self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][prev] - self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick] * (1 + self.tradingFee)
            self.dataFrame["Holdings"][tick] = self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick]
            self.dataFrame["Action"][tick] = self.LONG

        self.dataFrame["Value"][tick] = self.dataFrame["Holdings"][tick] + self.dataFrame["Cash"][tick]
        self.dataFrame['Returns'][tick] = (self.dataFrame['Value'][tick] - self.dataFrame['Value'][prev])/self.dataFrame['Value'][prev]



    def GoShort2(self, tick):
        prev = tick - 1
        self.dataFrame["Position"][tick] = self.SHORT
        if self.IsLong(prev):
            self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][prev] + self.dataFrame["Lots"][prev] * self.dataFrame["Close"][tick] * (1 - self.tradingFee)
            self.dataFrame["Lots"][tick] = int(self.dataFrame["Cash"][tick] // (self.dataFrame["Close"][tick] * (1 + self.tradingFee)))
            self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][tick] + self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick] * (1 - self.tradingFee)
            self.dataFrame["Holdings"][tick] = -self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick]
            self.dataFrame["Action"][tick] = self.SHORT
        elif self.IsShort(prev):
            lowerBound = self.ComputeLowerBound(self.dataFrame["Cash"][prev], self.dataFrame["Lots"]
            [prev], self.dataFrame["Close"][prev])
            if lowerBound <= 0:
                self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][prev]
                self.dataFrame["Lots"][tick] = self.dataFrame["Lots"][prev]
                self.dataFrame["Holdings"][tick] = -self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick]
            else:
                numberOfSharesToBuy = min(lowerBound, self.dataFrame["Lots"][prev])
                self.dataFrame["Lots"][tick] = self.dataFrame["Lots"][prev] - numberOfSharesToBuy
                self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][prev] - numberOfSharesToBuy * self.dataFrame["Close"][tick] * (1 + self.tradingFee)
                self.dataFrame["Holdings"][tick] = -self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick]
                self.dataFrame["Action"][tick] = self.SHORT
        else:
            self.dataFrame["Lots"][tick] = int(self.dataFrame["Cash"][tick] // (self.dataFrame["Close"][tick] * (1 + self.tradingFee)))
            self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][prev] + self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick]
            self.dataFrame["Holdings"][tick] = -self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick]
            self.dataFrame["Action"][tick] = self.SHORT


        self.dataFrame["Value"][tick] = self.dataFrame["Holdings"][tick] + self.dataFrame["Cash"][tick]
        self.dataFrame['Returns'][tick] = (self.dataFrame['Value'][tick] - self.dataFrame['Value'][prev])/self.dataFrame['Value'][prev]



    def GoLong(self, tick):
        prev = tick-1
        self.dataFrame["Position"][tick] = self.LONG
        self.dataFrame["Lots"][tick] = self.dataFrame["Lots"][prev]
        self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][prev]
        lotsToBuy = int(self.dataFrame["Cash"][tick] // (self.dataFrame["Close"][tick] * (1 + self.tradingFee)))
        self.dataFrame["Lots"][tick] += lotsToBuy
        self.dataFrame["Cash"][tick] -= lotsToBuy * self.dataFrame["Close"][tick] * (1 + self.tradingFee)
        self.dataFrame["Holdings"][tick] = self.dataFrame["Lots"][tick] * self.dataFrame["Close"][tick]
        self.dataFrame["Action"][tick] = self.LONG
        self.dataFrame["Value"][tick] = self.dataFrame["Holdings"][tick] + self.dataFrame["Cash"][tick]
        self.dataFrame['Returns'][tick] = (self.dataFrame['Value'][tick] - self.dataFrame['Value'][prev])/self.dataFrame['Value'][prev]



    def GoShort(self, tick):
        prev = tick - 1
        self.dataFrame["Position"][tick] = self.SHORT
        self.dataFrame["Cash"][tick] = self.dataFrame["Cash"][prev]
        self.dataFrame["Lots"][tick] = self.dataFrame["Lots"][prev]
        price = self.dataFrame["Close"][tick] * (1 - self.tradingFee)
        self.dataFrame["Cash"][tick] += self.dataFrame["Lots"][tick] * price
        self.dataFrame["Lots"][tick] = 0
        self.dataFrame["Holdings"][tick] = 0
        self.dataFrame["Action"][tick] = self.SHORT
        self.dataFrame["Value"][tick] = self.dataFrame["Holdings"][tick] + self.dataFrame["Cash"][tick]
        self.dataFrame['Returns'][tick] = (self.dataFrame['Value'][tick] - self.dataFrame['Value'][prev])/self.dataFrame['Value'][prev]


    def ParseActions(self):
        actions = self.dataFrame["Position"].to_numpy(dtype=int)
        index = []

        buyHistory = np.full(self.dataFrameLength, np.nan)
        sellHistory = np.full(self.dataFrameLength, np.nan)

        for i in range(len(actions)-1):
            if actions[i] == actions[i+1]:
                index.append(i+1)

        actions[index] = 0

        for i, item in enumerate(actions):
            if item == self.LONG:
                buyHistory[i] = self.dataFrame["Close"][i]
            elif item == self.SHORT:
                sellHistory[i] = self.dataFrame["Close"][i]

        return buyHistory, sellHistory



    def PlotActionsCapital(self, saveFileName, showFlag = False):
        buyHistory, sellHistory = self.ParseActions()
        idx = np.argwhere(~np.isnan(buyHistory))
        idx = idx.flatten()
        buys = self.dataFrame["Value"][idx]

        idxSell = np.argwhere(~np.isnan(sellHistory))
        idxSell = idxSell.flatten()
        sells = self.dataFrame["Value"][idxSell]

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
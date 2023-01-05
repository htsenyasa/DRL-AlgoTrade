import multiprocessing as mp
import yfinance as yf
import tf_TradingOperations as to
import ai_TradingEnvironment as te
import ai_TDQN as tdqn
import ai_Network as network
import pandas as pd
pd.options.mode.chained_assignment = None
import torch
import numpy as np
import mplfinance as mpf
import tf_DataAugmentation as da
import matplotlib.pyplot as plt
import random

device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


networkSettings = tdqn.networkSettings_(inputLayerSize=117, hiddenLayerSize=512, outputLayerSize=2, dropout=0.2)

tdqnSettings = tdqn.tdqnSettings_(gamma=0.4, 
                                  epsilonStart=1.0, 
                                  epsilonEnd=0.01,
                                  epsilonDecay=10000,
                                  capacity=100000, 
                                  learningRate=0.0001,
                                  targetUpdateFrequency=1000, 
                                  batchSize=32, 
                                  gradientClipping=1,
                                  targetNetworkUpdate=1000, 
                                  alpha=0.1, 
                                  numberOfEpisodes = 100, 
                                  rewardClipping = 1
                                  )

optimSettings = tdqn.optimSettings_(L2Factor=0.000001)


startingDate = '2012-1-1'
splittingDate = '2018-1-1'
endingDate = '2020-1-1'


trainingHorizon = te.Horizon(startingDate, splittingDate, "1d")
testingHorizon = te.Horizon(splittingDate, endingDate, "1d")

def InitializeProcess(stockName, identifierString, verbose = False):
    fileName = stockName + identifierString

    if verbose == True:
        print("Stock Name: " + stockName)
    
    StockTraining = to.StockHandler(stockName, yf.download, yf.download, trainingHorizon)
    StockTesting = to.StockHandler(stockName, yf.download, yf.download, testingHorizon)
    PositionTraining = to.DummyPosition(StockTraining)
    PositionTesting = to.DummyPosition(StockTesting)
    TrainingEnvironment = te.TradingEnvironment(PositionTraining)
    TestingEnvironment = te.TradingEnvironment(PositionTesting)
    Agent = tdqn.TDQNAgent(TrainingEnvironment, TestingEnvironment, tdqnSettings, networkSettings, optimSettings)
    Agent.Training()
    Agent.SaveModel(fileName)
    Agent.Testing()
    PositionTesting.PlotActionsCapital(fileName + "-Capital", showFlag=True)
    Agent.PlotLoss(fileName + "-Loss", showFlag=True)



if __name__ == "__main__":
    listOfStocks = ["AAPL", "ISCTR.IS"]
    processList = [mp.Process(target=InitializeProcess, args = (listOfStocks[i], "1820", True)) for i in range(2)]


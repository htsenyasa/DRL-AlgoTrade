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
torch.manual_seed(33)
random.seed(33)
np.random.seed(33)


networkSettings = tdqn.networkSettings_(inputLayerSize=117, hiddenLayerSize=512, outputLayerSize=2, dropout=0.2)

tdqnSettings = tdqn.tdqnSettings_(gamma=0.4, 
                                  epsilonStart=1.0, 
                                  epsilonEnd=0.01,
                                  epsilonDecay=10000,
                                  capacity=100000, 
                                  learningRate=0.0001,
                                  targetUpdateFrequency=500, 
                                  batchSize=64, 
                                  gradientClipping=1,
                                  targetNetworkUpdate=1000, 
                                  alpha=0.1, 
                                  numberOfEpisodes = 80, 
                                  rewardClipping = 1
                                  )

optimSettings = tdqn.optimSettings_(L2Factor=0.000001)


startingDate = '2012-1-1'
splittingDate = '2018-1-1'
endingDate = '2020-1-1'


trainingHorizon = te.Horizon(startingDate, splittingDate, "1d")
testingHorizon = te.Horizon(splittingDate, endingDate, "1d")


def ReadFromFile(stockName, start, end, interval, progress):
    data = pd.read_csv("./Data/" + stockName, index_col=0)
    data = data.loc[start:end]
    data.index = pd.to_datetime(data.index)
    return data


def InitializeTrainingTesting(stockName, identifierString, verbose = False):
    fileName = stockName + identifierString

    if verbose == True:
        print("Stock Name: " + stockName)
    
    StockTraining = to.StockHandler(stockName, ReadFromFile, ReadFromFile, trainingHorizon)
    StockTesting = to.StockHandler(stockName, ReadFromFile, ReadFromFile, testingHorizon)
    PositionTraining = to.DummyPosition(StockTraining)
    PositionTesting = to.DummyPosition(StockTesting)
    TrainingEnvironment = te.TradingEnvironment(PositionTraining)
    TestingEnvironment = te.TradingEnvironment(PositionTesting)
    Agent = tdqn.TDQNAgent(TrainingEnvironment, TestingEnvironment, tdqnSettings, networkSettings, optimSettings)
    Agent.Training(verbose=False)
    Agent.SaveModel(fileName)
    Agent.Testing()
    PositionTesting.PlotActionsCapital(fileName + "-Capital", showFlag=False)
    PositionTesting.PlotActionsPrice(fileName + "-Price", showFlag=False)
    Agent.PlotLoss(fileName + "-Loss", showFlag=False)


def InitializeTesting(stockName, identifierString, verbose = False):
    fileName = stockName + identifierString

    if verbose == True:
        print("Stock Name: " + stockName)
    
    StockTesting = to.StockHandler(stockName, ReadFromFile, ReadFromFile, testingHorizon)
    PositionTesting = to.DummyPosition(StockTesting)
    TestingEnvironment = te.TradingEnvironment(PositionTesting)
    Agent = tdqn.TDQNAgent(TestingEnvironment, TestingEnvironment, tdqnSettings, networkSettings, optimSettings)
    Agent.LoadModel(fileName)
    Agent.Testing()
    PositionTesting.PlotActionsCapital(fileName + "-Capital", showFlag=False)
    PositionTesting.PlotActionsPrice(fileName + "-Price", showFlag=False)
    Agent.PlotLoss(fileName + "-Loss", showFlag=False)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    listOfStocks = ["AAPL", "ISCTR.IS"]
    # listOfStocks = ["ISCTR.IS"]
    identifier = "-1820-{}E-{}B".format(tdqnSettings.numberOfEpisodes, tdqnSettings.batchSize)
    processList = [mp.Process(target=InitializeTrainingTesting, args = (listOfStocks[i], identifier, True)) for i in range(len(listOfStocks))]
    
    for process in processList:
        process.start()

    for process in processList:
        process.join()



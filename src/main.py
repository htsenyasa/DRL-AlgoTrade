import multiprocessing as mp
import yfinance as yf
import tf_TradingOperations as to
import ai_TradingEnvironment as te
import ai_TDQN as tdqn
import pandas as pd
pd.options.mode.chained_assignment = None
import torch
import numpy as np
import random
import os

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
                                  targetUpdateFrequency=500, 
                                  batchSize=32, 
                                  gradientClipping=1,
                                  targetNetworkUpdate = 100, 
                                  alpha=0.1, 
                                  numberOfEpisodes = 50,
                                  rewardClipping = 1
                                  )

optimSettings = tdqn.optimSettings_(L2Factor=0.000001)


startingDate = '2012-01-01'
splittingDate = '2021-01-01'
endingDate = '2023-01-01'


trainingHorizon = te.Horizon(startingDate, splittingDate, "1d")
testingHorizon = te.Horizon(splittingDate, endingDate, "1d")



def InitializeTrainingTesting(stockName, identifierString, verbose = False):
    fileName = stockName + identifierString
    modelFileName = "./Models/" + fileName + "/" + stockName
    figureFileName = "./Figures/" + fileName + "/" + stockName

    if verbose == True:
        print("Stock Name: " + stockName + identifierString)
    
    StockTraining = to.StockHandler(stockName, yf.download, yf.download, trainingHorizon)
    StockTesting = to.StockHandler(stockName, yf.download, yf.download, testingHorizon)
    PositionTraining = to.DummyPosition(StockTraining)
    PositionTesting = to.DummyPosition(StockTesting)
    TrainingEnvironment = te.TradingEnvironment(PositionTraining)
    TestingEnvironment = te.TradingEnvironment(PositionTesting)
    Agent = tdqn.TDQNAgent(TrainingEnvironment, TestingEnvironment, tdqnSettings, networkSettings, optimSettings)
    Agent.Training(verbose=True)
    Agent.SaveModel(modelFileName)
    Agent.Testing()
    PositionTesting.PlotActionsCapital(figureFileName + "-Capital", showFlag=False)
    PositionTesting.PlotActionsPrice(figureFileName + "-Price", showFlag=False)
    Agent.PlotLoss(figureFileName + "-Loss", showFlag=False)


def InitializeTesting(stockName, identifierString, verbose = False):
    fileName = stockName + identifierString
    modelFileName = "./Models/" + fileName + "/" + stockName
    figureFileName = "./Figures/" + fileName + "/" + stockName

    if verbose == True:
        print("Stock Name: " + stockName)
    
    StockTesting = to.StockHandler(stockName, yf.download, yf.download, testingHorizon)
    PositionTesting = to.DummyPosition(StockTesting)
    TestingEnvironment = te.TradingEnvironment(PositionTesting)
    Agent = tdqn.TDQNAgent(TestingEnvironment, TestingEnvironment, tdqnSettings, networkSettings, optimSettings)
    Agent.LoadModel(modelFileName)
    Agent.Testing()
    PositionTesting.PlotActionsCapital(figureFileName + "-Capital", showFlag=False)
    PositionTesting.PlotActionsPrice(figureFileName + "-Price", showFlag=False)
    Agent.PlotLoss(figureFileName + "-Loss", showFlag=False)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    listOfStocksNames = ["AAPL", "ISCTR.IS"]
    identifier = "-2123-{}E-{}B-{}U".format(tdqnSettings.numberOfEpisodes, tdqnSettings.batchSize, tdqnSettings.targetNetworkUpdate)

    paths = ["Models", "Figures"]

    for path in paths:
        for stockName in listOfStocksNames:
            dirs = path + "/" + stockName + identifier
            if not os.path.exists(dirs):
                os.makedirs(dirs, exist_ok=True)

    processList = [mp.Process(target=InitializeTrainingTesting, args = (listOfStocksNames[i], identifier, True)) for i in range(len(listOfStocksNames))]
    
    for process in processList:
        process.start()

    for process in processList:
        process.join()


